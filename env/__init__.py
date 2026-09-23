"""Environment registry used by the training / eval / viewer scripts."""

from __future__ import annotations

from .rokae_reacher import RokaePro7RealReacher
from .rokae_pro7_pick import RokaePro7Pick, RokaePro7PickReal
from .rokae_pro7_pick_place import RokaePro7PickPlace
from .dual_arm_reacher import DualArmCoopReach, DualArmReach

_REGISTRY = {
    # Every registered environment now runs on the real-URDF mesh models: the
    # Pro7 reach, the vision-driven grasp (its old alias `pro7_pick_urdf` is
    # kept) and the whole pick & place episode.
    "pro7_urdf": RokaePro7RealReacher,
    "pro7_pick": RokaePro7Pick,
    "pro7_pick_urdf": RokaePro7PickReal,
    "pro7_pick_place": RokaePro7PickPlace,
    # Dual-arm (lkwy73_o1) 14-DoF tasks: independent targets per arm, and a
    # cooperative variant where the two targets form one rigid bar.
    "dual_arm_reach": DualArmReach,
    "dual_arm_coop": DualArmCoopReach,
}


def make_env(name: str, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"unknown env '{name}'; choose from {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def env_names():
    return sorted(_REGISTRY)
