"""Wall-clock driver for the MuJoCo cell and the scripted pick & place expert.

The node never touches MuJoCo itself: this module owns

* one **plant thread** that advances physics (or the running expert) at
  ``sim_hz`` and keeps the vision estimate fresh, and
* at most **one** :class:`PickPlaceJob` at a time, so two goals can never fight
  over the same arm.

A job is the step-wise expert from :func:`grasp.common.iter_pick_and_place`,
advanced one control step per plant tick.  That means a goal plays out at
wall-clock speed (watchable, and the ROS timers keep publishing while it runs)
instead of blocking one long call -- see ``sim_hz`` to speed it up.

The plant thread is also the *owner of the MuJoCo renderers*: it builds them,
renders the published RGB-D frames and executes scene resets (which rebuild
them).  MuJoCo's renderer wraps an OpenGL context that must not be created or
destroyed from a thread while another one renders with it -- doing that
segfaults, which is why every GL touch point funnels through this thread.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .scene import GraspScene

#: Keep the last control state and stop stepping physics while idle.
IDLE_FREEZE = "freeze"
#: Keep stepping physics with the last controls while idle.
IDLE_STEP = "step"
IDLE_MODES = (IDLE_FREEZE, IDLE_STEP)


@dataclass
class PickPlaceRequest:
    """What one goal asks for; every field is optional."""

    place_target: Optional[np.ndarray] = None
    cube_hint: Optional[np.ndarray] = None
    seed: Optional[int] = None
    randomize_cube: bool = True
    pick_only: bool = False
    settle_steps: Optional[int] = None


@dataclass(frozen=True)
class TaskSnapshot:
    """Thread-safe view of the active (or last) job."""

    phase: str
    running: bool
    finished: bool
    steps: int
    holding: bool
    ever_holding: bool
    target: np.ndarray
    distance: float
    place_error: float
    result: Any = None


class PickPlaceJob:
    """The scripted expert, resumable one control step at a time."""

    def __init__(self, scene: GraspScene, request: PickPlaceRequest):
        self.scene = scene
        self.request = request
        self.phase = "approach"
        self.steps = 0
        self.holding = False
        self.ever_holding = False
        self.target = np.zeros(3)
        self.distance = float("inf")
        self.finished = False
        self.result: Any = None
        with scene.lock:
            gc = scene.gc
            kwargs = {"place": not request.pick_only}
            if request.cube_hint is not None:
                kwargs["cube_hint"] = request.cube_hint
            if request.settle_steps is not None:
                kwargs["settle_steps"] = int(request.settle_steps)
            if request.place_target is not None:
                kwargs["place_target"] = request.place_target
            self._episode = gc.iter_pick_and_place(
                scene.model, scene.data, scene.ids, scene.detect_renderer, **kwargs
            )

    # ------------------------------------------------------------------ #
    def advance(self) -> bool:
        """Run one control step (one ``mj_step``); False once the job is over."""
        with self.scene.lock:
            if self.finished:
                return False
            try:
                step = next(self._episode)
            except StopIteration as done:
                self.result = done.value
                self.finished = True
                self.phase = "done"
                return False
            except Exception as exc:  # never kill the plant thread
                self.result = self.scene.gc.PickPlaceResult(
                    self.ever_holding, False, self.steps, f"expert error: {exc}"
                )
                self.finished = True
                self.phase = "error"
                return False
            self.phase = step.phase
            self.steps = int(step.step)
            self.holding = bool(step.holding)
            self.ever_holding = self.ever_holding or self.holding
            self.target = np.array(step.target, dtype=float)
            self.distance = float(step.distance)
            return True

    def cancel(self, reason: str = "cancelled") -> bool:
        """Stop the expert where it stands; keeps whatever the jaws hold."""
        with self.scene.lock:
            if self.finished:
                return False
            self._episode.close()
            self.result = self.scene.gc.PickPlaceResult(
                self.ever_holding, False, self.steps, reason
            )
            self.finished = True
            self.phase = "cancelled"
            return True

    def snapshot(self) -> TaskSnapshot:
        with self.scene.lock:
            return TaskSnapshot(
                phase=self.phase,
                running=not self.finished,
                finished=self.finished,
                steps=self.steps,
                holding=self.holding,
                ever_holding=self.ever_holding,
                target=np.array(self.target),
                distance=float(self.distance),
                place_error=float(
                    np.linalg.norm(
                        self.scene.gc.cube_world(self.scene.data, self.scene.ids)
                        - self.scene.place_target
                    )
                ),
                result=self.result,
            )


class PickPlaceSimulator:
    """Runs :class:`GraspScene` on its own thread and serialises expert jobs."""

    def __init__(
        self,
        scene: GraspScene,
        *,
        sim_hz: float = 240.0,
        idle_mode: str = IDLE_FREEZE,
        vision_hz: float = 5.0,
        image_hz: float = 0.0,
        name: str = "pro7_pick_place_sim",
    ):
        if idle_mode not in IDLE_MODES:
            raise ValueError(f"idle_mode must be one of {IDLE_MODES}, got {idle_mode!r}")
        self.scene = scene
        self.sim_hz = float(sim_hz)
        self.idle_mode = idle_mode
        self.vision_hz = float(vision_hz)
        self.image_hz = float(image_hz)
        #: The node clears this while nobody subscribes to the camera topics.
        self.image_enabled = True
        self._name = name
        self._job: Optional[PickPlaceJob] = None
        self._last_result: Any = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last_vision = 0.0
        self._last_image = 0.0
        self._images = None
        self._pending_reset = None
        self._reset_event = threading.Event()
        self._reset_result: Optional[int] = None
        self._ready = threading.Event()
        self._startup_error: Optional[BaseException] = None
        self._submit_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def start(self, timeout: float = 120.0) -> bool:
        """Start the plant thread; returns True once the scene is usable."""
        if self._thread is not None:
            return self._ready.is_set()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()
        return self._ready.wait(timeout) and self._startup_error is None

    def shutdown(self, timeout: float = 2.0) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout)

    def __enter__(self) -> "PickPlaceSimulator":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.shutdown()

    # ------------------------------------------------------------------ #
    # jobs
    # ------------------------------------------------------------------ #
    @property
    def startup_error(self) -> Optional[BaseException]:
        """Why :meth:`start` failed (``None`` when the plant thread came up)."""
        return self._startup_error

    @property
    def busy(self) -> bool:
        with self.scene.lock:
            return self._job is not None

    def submit(self, request: PickPlaceRequest) -> Optional[PickPlaceJob]:
        """Start a job, or return ``None`` when one is already running."""
        with self._submit_lock:
            if self.busy:
                return None
            if request.seed is not None or request.randomize_cube:
                if self.reset(
                    seed=request.seed, randomize_cube=request.randomize_cube
                ) is None:
                    return None
            with self.scene.lock:
                if self._job is not None:
                    return None
                if request.place_target is not None:
                    self.scene.set_place_target(request.place_target)
                job = PickPlaceJob(self.scene, request)
                self._job = job
                self._last_result = None
                return job

    def cancel(self) -> bool:
        """Cancel the running job (if any); True when one was cancelled."""
        with self.scene.lock:
            if self._job is None:
                return False
            return self._job.cancel()

    def task(self) -> Optional[TaskSnapshot]:
        with self.scene.lock:
            return None if self._job is None else self._job.snapshot()

    def last_result(self) -> Any:
        with self.scene.lock:
            return self._last_result

    def reset(self, timeout: float = 30.0, **kwargs) -> Optional[int]:
        """Reset the scene; ``None`` (and no change) while a job is running.

        The rebuild runs on the plant thread, because it recreates the GL
        renderers.
        """
        with self.scene.lock:
            if self._job is not None:
                return None
            thread = self._thread
            if thread is None or not thread.is_alive():
                return self.scene.reset(**kwargs)  # not started: do it here
            self._reset_event.clear()
            self._reset_result = None
            self._pending_reset = kwargs
        if not self._reset_event.wait(timeout):
            return None
        with self.scene.lock:
            return self._reset_result

    def images(self):
        """Last ``(colour, depth)`` frame the plant thread rendered, or ``None``."""
        with self.scene.lock:
            return self._images

    # ------------------------------------------------------------------ #
    # the plant thread
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        try:
            with self.scene.lock:
                if not self.scene.renderers_ready:
                    self.scene.build_renderers()
        except BaseException as exc:  # pragma: no cover - startup failure
            self._startup_error = exc
            self._ready.set()
            return
        self._ready.set()

        period = 1.0 / self.sim_hz if self.sim_hz > 0.0 else 0.0
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            self._tick()
            if period <= 0.0:  # unthrottled: run as fast as the CPU allows
                continue
            next_tick += period
            delay = next_tick - time.perf_counter()
            if delay > 0.0:
                self._stop.wait(delay)
            else:  # fell behind (loaded machine): re-base instead of spinning
                next_tick = time.perf_counter()
        with self.scene.lock:  # GL objects die on the thread that made them
            self.scene.close_renderers()

    def _tick(self) -> None:
        with self.scene.lock:
            if self._pending_reset is not None:
                kwargs, self._pending_reset = self._pending_reset, None
                try:
                    self._reset_result = self.scene.reset(**kwargs)
                except Exception:  # pragma: no cover - reset must not kill the thread
                    self._reset_result = None
                finally:
                    self._reset_event.set()
                return
            job = self._job
            if job is not None:
                if not job.advance():
                    self._last_result = job.result
                    self._job = None
            elif self.idle_mode == IDLE_STEP:
                self.scene.step()
        self._tick_vision()
        self._tick_images()

    def _tick_vision(self) -> None:
        if self.vision_hz <= 0.0:
            return
        now = time.perf_counter()
        if now - self._last_vision < 1.0 / self.vision_hz:
            return
        self._last_vision = now
        try:
            self.scene.detect()
        except Exception:  # a failed detection must not kill the plant thread
            pass

    def _tick_images(self) -> None:
        """Render the published RGB-D frame -- on the plant thread, always."""
        if self.image_hz <= 0.0 or not self.image_enabled:
            return
        now = time.perf_counter()
        if now - self._last_image < 1.0 / self.image_hz:
            return
        self._last_image = now
        try:
            images = self.scene.rgbd()
        except Exception:
            return
        with self.scene.lock:
            self._images = images
