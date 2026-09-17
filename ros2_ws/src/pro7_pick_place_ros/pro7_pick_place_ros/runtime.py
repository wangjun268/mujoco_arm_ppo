"""Make the node run on an interpreter that has both ``rclpy`` and ``mujoco``.

On this machine ROS 2 Jazzy's ``rclpy`` belongs to the *system* Python, while
the MuJoCo project lives in a Conda environment -- and that Conda environment
ships a ``libstdc++`` too old for rclpy's extension modules.  A console script
installed by ``colcon`` therefore starts on an interpreter that cannot import
one of the two, so :func:`ensure_runtime` re-executes the process on a python
that can import both (preloading the system ``libstdc++`` if needed).

Override the interpreter with the ``PRO7_PYTHON`` environment variable.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from typing import Iterable, List, Optional

_MARKER = "PRO7_RUNTIME_REEXEC"
#: Where distributions normally keep the ``libstdc++`` that rclpy needs.
_SYSTEM_LIBSTDCPP = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"


def _importable(module: str, python: Optional[str] = None, preload: Optional[str] = None) -> bool:
    """Can ``module`` be imported, here or in a child interpreter?"""
    if python is None:
        try:
            importlib.import_module(module)
            return True
        except Exception:
            return False
    env = dict(os.environ)
    if preload:
        env["LD_PRELOAD"] = preload
    try:
        done = subprocess.run(
            [python, "-c", f"import {module}"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=60, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def _candidates() -> List[str]:
    found = []
    for candidate in (os.environ.get("PRO7_PYTHON"), shutil.which("python3"), sys.executable):
        if candidate and candidate not in found:
            found.append(candidate)
    return found


def _resolve(modules: Iterable[str]):
    """Return ``(python, preload)`` able to import ``modules``, or ``None``."""
    modules = tuple(modules)
    if all(_importable(module) for module in modules):
        return sys.executable, None  # nothing to do: we are already there
    for python in _candidates():
        if all(_importable(module, python) for module in modules):
            return python, None
        if all(_importable(module, python, _SYSTEM_LIBSTDCPP) for module in modules):
            return python, _SYSTEM_LIBSTDCPP
    return None


def ensure_runtime(
    modules: Iterable[str] = ("rclpy", "mujoco"),
    *,
    command: Optional[str] = None,
    argv: Optional[Iterable[str]] = None,
) -> None:
    """Re-exec on a suitable interpreter; raise if there is none.

    ``command`` is replayed as the entry point (``node`` / ``client``) so the
    new process lands on the same console script, with the same arguments.
    """
    modules = tuple(modules)
    if os.environ.get(_MARKER):  # already re-exec'ed once: don't loop
        return
    if all(_importable(module) for module in modules):
        return

    resolved = _resolve(modules)
    if resolved is None:
        raise RuntimeError(
            "no python interpreter can import "
            + ", ".join(modules)
            + "; set PRO7_PYTHON to one that can (e.g. the Conda env of the "
            "mujoco_arm_ppo project)"
        )
    python, preload = resolved
    env = dict(os.environ)
    env[_MARKER] = "1"
    if preload:
        env["LD_PRELOAD"] = preload
    if python == sys.executable:
        # Same interpreter, but it needs the preload: start it over with it.
        if not preload:
            return
    replay = [python, "-m", "pro7_pick_place_ros.entry"]
    if command:
        replay.append(command)
    replay.extend(sys.argv[1:] if argv is None else argv)
    os.execve(python, replay, env)
