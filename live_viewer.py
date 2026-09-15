"""Reusable real-time MuJoCo viewer for the training and demo scripts.

``mujoco.viewer.launch_passive`` renders the *current* state of an ``MjModel`` /
``MjData`` pair from its own thread, so a loop that steps the physics thousands
of times per second only has to publish that state with ``sync()``.  Every
caller in this project needs the same three things around that handle:

* a **throttle**, so "sync after every step" does not flood the UI,
* a fixed camera and an on-screen status block (step, reward, success rate),
* the safe shutdown dance - ``close()`` then a short pause, so the viewer's UI
  thread finishes tearing GLFW down before the interpreter exits (otherwise the
  process can segfault, see the notes in the README).

Importing this module never opens a window, and ``launch`` can be injected, so
the class is unit-testable without a display.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional, Sequence

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

#: GLFW keycode for 'T' (used by the demos to re-randomise the task).
KEY_T = 84

#: Pause after ``close()`` so the UI thread can finish tearing GLFW down before
#: the interpreter exits (0 in tests).
CLOSE_GRACE = 0.6

#: Where status lines land in the Simulate text overlay, in order.
_GRID_POSITIONS = (
    mujoco.mjtGridPos.mjGRID_TOPLEFT,
    mujoco.mjtGridPos.mjGRID_TOPRIGHT,
    mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
    mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
)


class LiveViewer:
    """A throttled, self-closing wrapper around the passive MuJoCo viewer."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        camera: Optional[str] = None,
        fps: float = 60.0,
        key_callback: Optional[Callable[[int], None]] = None,
        launch: Optional[Callable] = None,
    ):
        self.model = model
        self.data = data
        self.fps = max(1.0, float(fps))
        self._status: Sequence[str] = ()
        self._next_sync = 0.0
        self._closed = False
        self.close_grace = CLOSE_GRACE
        self._launch = mujoco.viewer.launch_passive if launch is None else launch
        self.handle = self._launch(model, data, key_callback=key_callback)
        if camera is not None:
            self.set_camera(camera)

    # ------------------------------------------------------------------ #
    # window configuration
    # ------------------------------------------------------------------ #
    def set_camera(self, *names: str) -> bool:
        """Fix the viewer camera to the first of ``names`` present in the model."""
        for name in names:
            cam_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, name
            )
            if cam_id >= 0:
                self.handle.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self.handle.cam.fixedcamid = cam_id
                return True
        return False

    def set_status(self, lines: Sequence[str]) -> None:
        """Set the overlay text (up to four lines, one per screen corner)."""
        self._status = tuple(lines)

    # ------------------------------------------------------------------ #
    # per-step publishing
    # ------------------------------------------------------------------ #
    @property
    def running(self) -> bool:
        return not self._closed and bool(self.handle.is_running())

    def sync(self, *, force: bool = False) -> bool:
        """Publish the current ``data`` state; returns ``False`` once closed.

        Calls are throttled to ``fps`` so a fast training loop can call this on
        every step without stalling the UI thread.
        """
        if not self.running:
            self.close()
            return False
        now = time.time()
        if not force and now < self._next_sync:
            return True
        self._next_sync = now + 1.0 / self.fps
        if self._status:
            self.handle.set_texts(
                [
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        _GRID_POSITIONS[i % len(_GRID_POSITIONS)],
                        line,
                        None,
                    )
                    for i, line in enumerate(self._status[: len(_GRID_POSITIONS)])
                ]
            )
        self.handle.sync()
        return True

    def close(self) -> None:
        """Close the window and let the UI thread finish before returning."""
        if self._closed:
            return
        self._closed = True
        self.handle.close()
        time.sleep(self.close_grace)

    def __enter__(self) -> "LiveViewer":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class Pacer:
    """Hold a simulation loop to a watchable rate.

    The physics timestep is fixed (0.02 s for every model here), so ``speed=1``
    means "one simulated second per wall-clock second" - i.e. the arm moves at
    the speed you would see on a real robot instead of the blur you get from a
    loop that steps thousands of times per second.  ``speed <= 0`` disables
    pacing entirely (train as fast as the CPU allows).
    """

    def __init__(self, speed: float = 1.0, sim_dt: float = 0.02):
        self.interval = 0.0 if speed <= 0 else sim_dt / float(speed)
        self._next = time.time()

    @property
    def paced(self) -> bool:
        return self.interval > 0.0

    def tick(self) -> float:
        """Sleep just enough to keep the target rate; returns the slept time."""
        if not self.paced:
            return 0.0
        self._next += self.interval
        delay = self._next - time.time()
        if delay <= 0:  # we fell behind: resynchronise instead of racing ahead
            self._next = time.time()
            return 0.0
        time.sleep(delay)
        return delay
