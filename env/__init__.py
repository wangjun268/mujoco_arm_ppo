"""Environment registry used by the training / eval / viewer scripts."""

from __future__ import annotations

from .two_joint_reacher import TwoJointReacher
from .three_joint_reacher import ThreeJointReacher
from .rokae_reacher import RokaeReacher, RokaePro7Reacher, RokaePro7RealReacher
from .rokae_pro7_pick import RokaePro7Pick, RokaePro7PickReal
from .rokae_pro7_pick_place import RokaePro7PickPlace

_REGISTRY = {
    "two_joint": TwoJointReacher,
    "three_joint": ThreeJointReacher,
    "six_joint": RokaeReacher,
    "pro7_joint": RokaePro7Reacher,
    "pro7_urdf": RokaePro7RealReacher,
    "pro7_pick": RokaePro7Pick,
    "pro7_pick_urdf": RokaePro7PickReal,
    "pro7_pick_place": RokaePro7PickPlace,
}


def make_env(name: str, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"unknown env '{name}'; choose from {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def env_names():
    return sorted(_REGISTRY)
