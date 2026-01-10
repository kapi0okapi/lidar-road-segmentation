"""YAML experiment configs mapped onto dataclasses, with dotted command-line overrides.

cfg = load_config(TrainConfig, "configs/base.yaml", ["optim.lr=3e-4", "data.batch_size=64"])
"""

from __future__ import annotations

import dataclasses
import types
import typing
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def load_config(cls: type[T], path: str | Path | None = None, overrides: list[str] = ()) -> T:
    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override '{item}' is not KEY=VALUE")
        *parents, leaf = key.split(".")
        node = raw
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = _parse_value(value)
    return _build(cls, raw)


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Rebuilds a config saved inside a checkpoint."""
    return _build(cls, data)


def save_config(cfg: Any, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(_plain(dataclasses.asdict(cfg)), sort_keys=False))


def _plain(value: Any) -> Any:
    """Tuples -> lists, so the saved YAML stays readable."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def _parse_value(text: str) -> Any:
    value = yaml.safe_load(text)
    if isinstance(value, str):  # YAML 1.1 reads "1e-3" as a string
        try:
            return float(value)
        except ValueError:
            pass
    return value


def _build(cls: type[T], data: dict[str, Any]) -> T:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**{name: _coerce(hints[name], value) for name, value in data.items()})


def _coerce(tp: Any, value: Any) -> Any:
    """Converts YAML values (dicts, lists) into the annotated type: nested dataclasses,
    lists of dataclasses, tuples and optional fields."""
    if value is None:
        return None
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        return _coerce(next(a for a in args if a is not type(None)), value)
    if dataclasses.is_dataclass(tp) and isinstance(value, dict):
        return _build(tp, value)
    if (tp is tuple or origin is tuple) and isinstance(value, list | tuple):
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], v) for v in value)
        if len(args) == len(value):
            return tuple(_coerce(a, v) for a, v in zip(args, value, strict=True))
        return tuple(value)
    if origin is list and isinstance(value, list):
        return [_coerce(args[0], v) for v in value] if args else list(value)
    return value
