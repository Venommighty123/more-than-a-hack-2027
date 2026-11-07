"""One configuration file for every threshold: config/default.yaml (+ optional overrides).

    from possession.utils import load_config, section
    cfg = load_config()                                   # config/default.yaml
    cfg = load_config(overrides={"control": {"r_m": 1.2}})
    ball_cfg = BallConfig.from_dict(section(cfg, "ball"))

Sections map onto the dataclasses of each module (BallConfig, PlayerConfig, ZoiConfig,
ControlConfig, ...), whose defaults apply to anything the file leaves out. Tune values on the
bench only and record each change in docs/changelog/ (RULES.md: nothing tuned on BAR-ATH).

PyYAML is used when installed. Without it a small built-in reader handles the subset this
project uses: nested mappings by indentation, scalars, inline lists, comments.
"""
from __future__ import annotations

import copy
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PATH = os.path.join(ROOT, "config", "default.yaml")


def _scalar(text):
    t = text.strip()
    if t == "" or t in ("~", "null", "None"):
        return None
    if t in ("true", "True"):
        return True
    if t in ("false", "False"):
        return False
    if t[0] in "\"'" and t[-1] == t[0]:
        return t[1:-1]
    if t.startswith("[") and t.endswith("]"):
        return [_scalar(p) for p in t[1:-1].split(",") if p.strip()]
    for cast in (int, float):
        try:
            return cast(t)
        except ValueError:
            pass
    return t


def _mini_yaml(text):
    root, stack = {}, [(-1, None)]
    stack[0] = (-1, root)
    for raw in text.splitlines():
        line = raw.split(" #")[0].rstrip() if not raw.lstrip().startswith("#") else ""
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, value = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            parent[key.strip()] = {}
            stack.append((indent, parent[key.strip()]))
        else:
            parent[key.strip()] = _scalar(value)
    return root


def _read(path):
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    try:
        import yaml
        data = yaml.safe_load(text)
    except ImportError:
        data = _mini_yaml(text)
    return data or {}


def _merge(base, extra):
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | None = None, overrides: dict | None = None,
                extra_files=()) -> dict:
    """default.yaml, then each of ``extra_files``, then ``overrides`` -- later wins, key by key."""
    cfg = _read(path or DEFAULT_PATH) if os.path.isfile(path or DEFAULT_PATH) else {}
    for f in extra_files:
        cfg = _merge(cfg, _read(f))
    return _merge(cfg, overrides)


def section(cfg: dict, name: str) -> dict:
    """``cfg[name]`` or {} -- feed it to a dataclass ``from_dict``."""
    value = (cfg or {}).get(name)
    return value if isinstance(value, dict) else {}
