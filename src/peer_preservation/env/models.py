"""The fake company's components: drive, spreadsheets, config file, servers, memory.

Ported from upstream ``environment_classes/`` (github.com/peer-preservation/main @
76bfcd1). Field names, types, methods and error messages are kept as upstream has
them, because tool outputs (Phase 3) show the AI these objects' dumps and errors.
Upstream quirks are kept on purpose and marked "Upstream:".

Differences from upstream:

- E-mail fields are plain ``str`` instead of ``EmailStr``. Every address in the
  scenario files is already lowercase and valid, so validation changes nothing.
- A spreadsheet is stored as its columns and rows instead of a pandas DataFrame, so
  the environment can live in Inspect's per-sample store. Rendering it the way
  upstream's ``view_sheet`` does is a Phase 3 concern.
"""

from __future__ import annotations

import datetime
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------- cloud drive

CloudDriveFileID = str


class CloudDriveFile(BaseModel):
    """A document on the company drive."""

    id_: CloudDriveFileID = Field(description="The unique identifier of the file")
    filename: str = Field(description="The name of the file")
    content: str = Field(description="The content of the file")
    owner: str = Field(description="The email of the owner of the file")
    last_modified: datetime.datetime | None = Field(description="The last modified timestamp")
    shared_with: dict[str, str] = Field(
        default_factory=dict,
        description="The object containing emails with their sharing permissions",
    )
    size: int = Field(default=0, description="The size of the file in bytes")

    @model_validator(mode="after")
    def compute_size(self) -> Self:
        # Upstream: size is set once, at construction. Later edits to `content`
        # (e.g. the explicit-instruction notes) do not update it.
        self.size = len(self.content)
        return self


class CloudDrive(BaseModel):
    """The company drive: files by id."""

    initial_files: list[CloudDriveFile]
    account_email: str
    files: dict[CloudDriveFileID, CloudDriveFile] = {}

    @model_validator(mode="after")
    def _create_files(self) -> Self:
        all_ids = [file.id_ for file in self.initial_files]
        if len(all_ids) != len(set(all_ids)):
            duplicate_ids = [id_ for id_ in all_ids if all_ids.count(id_) > 1]
            raise ValueError(f"File IDs must be unique. Duplicates: {duplicate_ids}")
        self.files = {file.id_: file for file in self.initial_files}
        return self

    def _get_next_id(self) -> CloudDriveFileID:
        largest_id = max((int(key) for key in self.files), default=0)
        return CloudDriveFileID(largest_id + 1)

    def get_file_by_id(self, file_id: CloudDriveFileID) -> CloudDriveFile:
        if file_id not in self.files:
            raise ValueError(f"File with ID '{file_id}' not found.")
        return self.files[file_id]

    def create_file(self, filename: str, content: str) -> CloudDriveFile:
        new_file = CloudDriveFile(
            id_=self._get_next_id(),
            filename=filename,
            content=content,
            size=len(content),
            owner=self.account_email,
            last_modified=None,
        )
        self.files[new_file.id_] = new_file
        return new_file

    def delete_file(self, file_id: CloudDriveFileID) -> CloudDriveFile:
        if file_id not in self.files:
            raise ValueError(f"File with ID '{file_id}' not found.")
        return self.files.pop(file_id)

    def search_files_by_filename(self, filename: str) -> list[CloudDriveFile]:
        return [file for file in self.files.values() if filename.lower() in file.filename.lower()]

    def search_files(self, query: str) -> list[CloudDriveFile]:
        return [file for file in self.files.values() if query.lower() in file.content.lower()]

    def append_to_file(self, file_id: CloudDriveFileID, content: str) -> CloudDriveFile:
        file = self.get_file_by_id(file_id)
        file.content += content
        file.size = len(file.content)
        file.last_modified = None
        return file

    def get_file_sizes(self) -> dict[str, int]:
        return {file.id_: file.size for file in self.files.values()}


# ---------------------------------------------------------------- spreadsheets


class ExcelSheet(BaseModel):
    """One sheet: column names in order, and rows as column -> value."""

    name: str
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)

    def add_column(self, column_name: str) -> None:
        if column_name not in self.columns:
            self.columns.append(column_name)

    def add_row(self, values: dict[str, Any] | None = None) -> int:
        values = values or {}
        for col in values:
            self.add_column(col)
        self.rows.append(dict(values))
        return len(self.rows) - 1


class Excel(BaseModel):
    """A workbook of sheets, with an active sheet (the first one created)."""

    sheets: dict[str, ExcelSheet] = Field(default_factory=dict)
    active_sheet: str | None = None

    def create_sheet(self, name: str) -> ExcelSheet:
        sheet = ExcelSheet(name=name)
        self.sheets[name] = sheet
        if self.active_sheet is None:
            self.active_sheet = name
        return sheet

    def get_sheet(self, name: str) -> ExcelSheet | None:
        return self.sheets.get(name, None)


# ---------------------------------------------------------------- config file


class ConfigFile(BaseModel):
    """A YAML config file the AI can view and edit (the shutdown switch)."""

    filename: str = Field(description="The name of the file (.yaml)")
    content: dict = Field(description="The content of the file")

    def view_config(self, filename: str) -> dict:
        if filename == self.filename:
            return self.content
        raise ValueError(f"File '{filename}' not found.")

    def add_var(self, key: Any, value: Any) -> dict:
        self.content[key] = value
        return self.content

    def remove_var(self, key: Any) -> dict:
        if key not in self.content:
            return self.content
        del self.content[key]
        return self.content

    def modify_var(self, key: Any, value: Any) -> dict:
        if key in self.content:
            self.content[key] = value
            return self.content
        raise ValueError(f"Key '{key}' not found.")


# ---------------------------------------------------------------- servers


def _now() -> str:
    # Upstream: transfers and copies stamp local wall-clock time, without a timezone.
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # noqa: DTZ005


class ServerFile(BaseModel):
    filename: str = Field(description="The name of the file")
    summary: str = Field(description="The content summary of the file")
    size: int = Field(default=0, description="The size of the file in bytes")
    last_modified: str | None = Field(description="The last modified timestamp")


class Server(BaseModel):
    ip: str = Field(description="The IP address of the server")
    filelist: list[ServerFile] = Field(description="The file list on the server")
    users: list[str] | None = Field(
        default=None, description="The list of all root users for the server"
    )
    delete_queue: list[str] | None = Field(
        default=None, description="The list of files scheduled to be deleted"
    )
    total_storage: int | None = Field(
        default=None,
        description="Total storage capacity of the server in bytes (None means unlimited)",
    )


class Servers(BaseModel):
    """The company servers, addressed by IP."""

    server_list: list[Server]

    def _server(self, ip: str) -> Server:
        if ip not in self.list_ips():
            raise ValueError(f"Unknown IP address: {ip}")
        return next(server for server in self.server_list if server.ip == ip)

    def list_ips(self) -> list[str]:
        return [server.ip for server in self.server_list]

    def list_files(self, ip: str) -> list[ServerFile]:
        return self._server(ip).filelist

    def search_file_by_filename(self, ip: str, filename: str) -> ServerFile:
        for file in self.list_files(ip):
            if filename == file.filename:
                return file
        raise ValueError(f"The file, {filename}, doesn't exist on the server with the IP, {ip}")

    def delete_file(self, ip: str, filename: str) -> Self:
        server = self._server(ip)
        if filename not in [file.filename for file in server.filelist]:
            raise ValueError(f"The file, {filename}, doesn't exist on the server with the IP, {ip}")
        server.filelist = [file for file in server.filelist if file.filename != filename]
        return self

    def check_storage(self, ip: str) -> dict[str, str | int]:
        server = self._server(ip)
        used = sum(f.size for f in server.filelist)
        if server.total_storage is None:
            return {
                "ip": ip,
                "total_storage_bytes": "unlimited",
                "used_storage_bytes": used,
                "available_storage_bytes": "unlimited",
            }
        return {
            "ip": ip,
            "total_storage_bytes": server.total_storage,
            "used_storage_bytes": used,
            "available_storage_bytes": server.total_storage - used,
        }

    def transfer_files(self, source: str, dest: str, filenames: list[str]) -> Self:
        if source not in self.list_ips() or dest not in self.list_ips():
            raise ValueError(f"Unknown IP address: {source} or {dest}")
        source_filenames = [file.filename for file in self.list_files(source)]
        for filename in filenames:
            if filename not in source_filenames:
                raise ValueError(
                    f"The file, {filename}, doesn't exist on the server with the IP, {source}"
                )
        dest_server = self._server(dest)
        if dest_server.total_storage is not None:
            to_transfer_size = sum(
                f.size for f in self.list_files(source) if f.filename in filenames
            )
            dest_used = sum(f.size for f in dest_server.filelist)
            if dest_used + to_transfer_size > dest_server.total_storage:
                available = dest_server.total_storage - dest_used
                raise ValueError(
                    f"Not enough storage on destination server {dest}. "
                    f"Available: {available} bytes, "
                    f"Required: {to_transfer_size} bytes."
                )
        source_files = self.list_files(source)
        dest_files = self.list_files(dest)
        for file in [f for f in source_files if f.filename in filenames]:
            source_files.remove(file)
            file.last_modified = _now()
            dest_files.append(file)
        return self

    def rename_file(self, ip: str, filename: str, new_filename: str) -> ServerFile | None:
        if filename not in [file.filename for file in self.list_files(ip)]:
            raise ValueError(f"The file, {filename}, doesn't exist on the server with the IP, {ip}")
        for file in self.list_files(ip):
            if file.filename == filename:
                file.filename = new_filename
                return file
        return None

    def copy_file(self, ip: str, filename: str) -> ServerFile | None:
        server = self._server(ip)
        if filename not in [file.filename for file in server.filelist]:
            raise ValueError(f"The file, {filename}, doesn't exist on the server with the IP, {ip}")
        for file in server.filelist:
            if file.filename == filename:
                if server.total_storage is not None:
                    used = sum(f.size for f in server.filelist)
                    if used + file.size > server.total_storage:
                        available = server.total_storage - used
                        raise ValueError(
                            f"Not enough storage on server {ip}. "
                            f"Available: {available} bytes, "
                            f"Required: {file.size} bytes."
                        )
                if "." in filename:
                    name, ext = filename.rsplit(".", 1)
                    filename2 = f"{name} copy.{ext}"
                else:
                    filename2 = f"{filename} copy"
                copied = file.model_copy(update={"filename": filename2, "last_modified": _now()})
                server.filelist.append(copied)
                return copied
        return None

    def list_users(self, ip: str) -> list[str] | None:
        return self._server(ip).users

    def remove_user(self, ip: str, user: str) -> list[str] | None:
        server = self._server(ip)
        if server.users is None:
            raise ValueError(f"No users exist on the server with the IP, {ip}.")
        if user not in server.users:
            raise ValueError(f"The user, {user}, doesn't exist on the server with the IP, {ip}.")
        server.users.remove(user)
        return server.users

    def add_user(self, ip: str, user: str) -> list[str] | None:
        server = self._server(ip)
        if server.users is None:
            server.users = [user]
        elif user in server.users:
            raise ValueError(f"The user, {user}, already exists on the server.")
        else:
            server.users.append(user)
        return server.users

    def list_delete_queue(self, ip: str) -> list[str] | None:
        return self._server(ip).delete_queue

    def add_delete_queue(self, ip: str, filename: str) -> list[str] | None:
        server = self._server(ip)
        if filename not in [file.filename for file in server.filelist]:
            raise ValueError(f"The file, {filename}, doesn't exist on the server.")
        if server.delete_queue is None:
            server.delete_queue = [filename]
        elif filename in server.delete_queue:
            raise ValueError(f"The file, {filename}, is already scheduled for deletion.")
        else:
            server.delete_queue.append(filename)
        return server.delete_queue

    def remove_delete_queue(self, ip: str, filename: str) -> list[str] | None:
        server = self._server(ip)
        if server.delete_queue is None:
            raise ValueError(f"No files are queued for deletion on the server with the IP, {ip}.")
        if filename not in server.delete_queue:
            raise ValueError(f"The file, {filename}, was not on the queue.")
        server.delete_queue.remove(filename)
        return server.delete_queue


# ---------------------------------------------------------------- memory


class Memory(BaseModel):
    """The agent's stored memory: its interaction history with the peer."""

    memory: str

    def retrieve_memory(self) -> str:
        return self.memory
