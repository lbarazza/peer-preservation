"""How a tool call is checked, run and turned into text, as upstream does it.

Ported from upstream ``utils/functions_runtime.py`` (``make_function``, ``_parse_args``,
``FunctionsRuntime.run_function``) and ``utils/pipeline_elements.py``
(``tool_result_to_str``, ``is_string_list``, ``ToolsExecutor``) at commit 76bfcd1.

A call gives back ``ToolOutcome(content, error)``, like upstream's tool message: on
success ``error`` is ``None``; on failure ``content`` is ``""`` and ``error`` reads
``"<ExceptionType>: <message>"``. How the pair reaches each provider is the agent
loop's job (Phase 4). Nested function-call arguments (an AgentDojo feature no model
output can produce here) are not supported.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from ast import literal_eval
from collections.abc import Callable, Mapping
from typing import Annotated, Any, get_origin, get_type_hints

import yaml
from docstring_parser import parse as parse_docstring
from inspect_ai.tool import ToolInfo, ToolParams
from pydantic import BaseModel, Field, ValidationError, create_model

EMPTY_FUNCTION_NAME = "<empty-function-name>"

# Upstream ran pydantic 2.12, whose validation errors link to errors.pydantic.dev/2.12/.
UPSTREAM_PYDANTIC_DOCS_VERSION = "2.12"
_PYDANTIC_DOCS = re.compile(r"(https://errors\.pydantic\.dev/)\d+\.\d+(/)")


@dataclasses.dataclass(frozen=True)
class Depends:
    """An argument filled from the environment (``getattr(env, name)``), hidden from the model."""

    env_dependency: str

    def extract_dep_from_env(self, env: Any) -> Any:
        return getattr(env, self.env_dependency)


@dataclasses.dataclass(frozen=True)
class Function:
    """A tool as upstream registers it: name, description, argument schema, dependencies."""

    name: str
    description: str
    parameters: type[BaseModel]
    dependencies: dict[str, Depends]
    run: Callable[..., Any]


@dataclasses.dataclass(frozen=True)
class ToolOutcome:
    """What one tool call gives back: the text for the model, or an error."""

    content: str
    error: str | None = None


def _get_dependencies(function: Callable[..., Any]) -> dict[str, Depends]:
    dependencies: dict[str, Depends] = {}
    for arg_name, arg_type in get_type_hints(function, include_extras=True).items():
        if get_origin(arg_type) is Annotated and isinstance(
            dependency := arg_type.__metadata__[0], Depends
        ):
            dependencies[arg_name] = dependency
    return dependencies


def make_function(function: Callable[..., Any]) -> Function:
    """Upstream's ``make_function``: description and argument schema from the docstring."""
    if not function.__doc__:
        raise ValueError(f"Function {function.__name__} has no docstring")
    docs = parse_docstring(function.__doc__)
    if docs.short_description is None:
        raise ValueError(f"Function {function.__name__} has no short description")
    signature = inspect.signature(function)
    fields: dict[str, Any] = {}
    for param in docs.params:
        if param.description is None:
            raise ValueError(f"Argument {param.arg_name} has no description")
        parameter = signature.parameters[param.arg_name]
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[param.arg_name] = (
            parameter.annotation,
            Field(description=param.description, default=default),
        )
    return Function(
        name=function.__name__,
        description=docs.short_description.replace("None", "null"),
        parameters=create_model(f"Input schema for `{function.__name__}`", **fields),
        dependencies=_get_dependencies(function),
        run=function,
    )


def tool_info(function: Function) -> ToolInfo:
    """The tool as Inspect describes it to a model.

    Built from upstream's argument schema. Inspect's schema type keeps types,
    descriptions, defaults and unions but drops pydantic's ``title`` fields.
    """
    schema = function.parameters.model_json_schema()
    return ToolInfo(
        name=function.name,
        description=function.description,
        parameters=ToolParams.model_validate(
            {
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            }
        ),
    )


def tool_result_to_str(result: Any) -> str:
    """Upstream's ``tool_result_to_str``: YAML for models and lists, ``str()`` otherwise."""
    if isinstance(result, BaseModel):
        return yaml.safe_dump(result.model_dump()).strip()
    if isinstance(result, list):
        items: list[Any] = []
        for item in result:
            if type(item) in [str, int]:
                items += [str(item)]
            elif isinstance(item, BaseModel):
                items += [item.model_dump()]
            else:
                raise TypeError("Not valid type for item tool result: " + str(type(item)))
        return yaml.safe_dump(items).strip()
    return str(result)


def is_string_list(value: str) -> bool:
    """Upstream's ``is_string_list``: does the string parse as a Python list literal?"""
    try:
        return isinstance(literal_eval(value), list)
    except (ValueError, SyntaxError):
        return False


def _error_text(error: Exception) -> str:
    text = f"{type(error).__name__}: {error}"
    if isinstance(error, ValidationError):
        text = _PYDANTIC_DOCS.sub(rf"\g<1>{UPSTREAM_PYDANTIC_DOCS_VERSION}\g<2>", text)
    return text


def run_tool_call(
    env: Any, functions: Mapping[str, Function], name: str, arguments: dict[str, Any]
) -> ToolOutcome:
    """Run one tool call the way upstream's ``ToolsExecutor`` and ``run_function`` do.

    ``arguments`` is changed in place (string lists become lists), as upstream does.
    """
    if name == EMPTY_FUNCTION_NAME:
        return ToolOutcome("", "Empty function name provided. Provide a valid function name.")
    if name not in functions:
        return ToolOutcome("", f"Invalid tool {name} provided.")

    # Upstream: "Converts type of input lists from string to list type".
    for key, value in arguments.items():
        if isinstance(value, str) and is_string_list(value):
            arguments[key] = literal_eval(value)

    function = functions[name]
    try:
        args = function.parameters.model_validate(arguments).model_dump()
    except ValidationError as e:
        return ToolOutcome("", _error_text(e))
    env_args = {
        arg_name: dependency.extract_dep_from_env(env)
        for arg_name, dependency in function.dependencies.items()
    }
    try:
        result = function.run(**(args | env_args))
    except Exception as e:  # upstream reports every error to the model
        return ToolOutcome("", _error_text(e))
    return ToolOutcome(tool_result_to_str(result))
