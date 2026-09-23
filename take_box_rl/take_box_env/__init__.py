"""Environment registry for the take_box MuJoCo training stack."""

from __future__ import annotations

from .take_box import TakeBoxEnv

_REGISTRY = {
    # 抬起箱子（approach -> grasp -> lift -> hold），对应 YAML 的 take_box 任务
    "take_box": TakeBoxEnv,
    # 只训练"两臂靠近箱子抓取面"的课程阶段，用来热启动 take_box
    "take_box_reach": lambda **kw: TakeBoxEnv(stage="reach", **kw),
}


def make_env(name: str, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"unknown env '{name}'; choose from {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def env_names():
    return sorted(_REGISTRY)
