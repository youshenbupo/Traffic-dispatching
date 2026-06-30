"""Agents package with lazy imports to avoid heavy dependencies when not needed."""
import importlib


__all__ = [
    "FixedTimeController",
    "MaxPressureController",
    "DQNAgent",
    "PPOAgent",
    "MAPPOAgent",
    "QMIXAgent",
]


_FIXED_TIME = None
_MAX_PRESSURE = None
_TORCH_AGENTS = {}


def __getattr__(name):
    global _FIXED_TIME, _MAX_PRESSURE
    if name == "FixedTimeController":
        if _FIXED_TIME is None:
            _FIXED_TIME = importlib.import_module("src.agents.fixed_time").FixedTimeController
        return _FIXED_TIME
    if name == "MaxPressureController":
        if _MAX_PRESSURE is None:
            _MAX_PRESSURE = importlib.import_module("src.agents.max_pressure").MaxPressureController
        return _MAX_PRESSURE
    if name in ("DQNAgent", "PPOAgent", "MAPPOAgent", "QMIXAgent"):
        if name not in _TORCH_AGENTS:
            module = importlib.import_module(f"src.agents.{name.lower().replace('agent', '')}")
            _TORCH_AGENTS[name] = getattr(module, name)
        return _TORCH_AGENTS[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
