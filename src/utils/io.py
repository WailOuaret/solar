from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def _parse_scalar(value: str) -> Any:
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none", "~"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _simple_yaml_load(text: str) -> dict[str, Any]:
    lines = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        lines.append(raw_line.rstrip())

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(lines):
            return {}, index

        current = lines[index]
        current_indent = len(current) - len(current.lstrip(" "))
        if current_indent < indent:
            return {}, index

        is_list = current.lstrip().startswith("- ")
        if is_list:
            items = []
            while index < len(lines):
                line = lines[index]
                line_indent = len(line) - len(line.lstrip(" "))
                if line_indent < indent:
                    break
                if not line.lstrip().startswith("- "):
                    break
                item_text = line.lstrip()[2:].strip()
                index += 1
                if item_text:
                    items.append(_parse_scalar(item_text))
                else:
                    nested, index = parse_block(index, indent + 2)
                    items.append(nested)
            return items, index

        mapping: dict[str, Any] = {}
        while index < len(lines):
            line = lines[index]
            line_indent = len(line) - len(line.lstrip(" "))
            if line_indent < indent:
                break
            if line_indent > indent:
                nested, index = parse_block(index, line_indent)
                if mapping:
                    last_key = list(mapping.keys())[-1]
                    mapping[last_key] = nested
                continue

            content = line.strip()
            if ":" not in content:
                index += 1
                continue
            key, value = content.split(":", 1)
            key = key.strip()
            value = value.strip()
            index += 1
            if value:
                mapping[key] = _parse_scalar(value)
            else:
                if index < len(lines):
                    next_line = lines[index]
                    next_indent = len(next_line) - len(next_line.lstrip(" "))
                    if next_indent > line_indent:
                        nested, index = parse_block(index, next_indent)
                        mapping[key] = nested
                    else:
                        mapping[key] = {}
                else:
                    mapping[key] = {}
        return mapping, index

    parsed, _ = parse_block(0, 0)
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text) or {}
    return _simple_yaml_load(text)


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        if yaml is not None:
            yaml.safe_dump(data, handle, sort_keys=False)
        else:
            json.dump(data, handle, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
