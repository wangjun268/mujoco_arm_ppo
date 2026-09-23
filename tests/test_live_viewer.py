"""Tests for the throttled live-viewer wrapper (no window is ever opened)."""

import types
import time

import pytest

import grasp.common as gc
from live_viewer import LiveViewer, Pacer
from train_ppo import LiveViewerCallback


class FakeHandle:
    """Stands in for ``mujoco.viewer.Handle``."""

    def __init__(self):
        self.cam = types.SimpleNamespace(type=None, fixedcamid=None)
        self.running = True
        self.syncs = 0
        self.texts = None
        self.closed = False

    def is_running(self) -> bool:
        return self.running

    def sync(self) -> None:
        self.syncs += 1

    def set_texts(self, texts) -> None:
        self.texts = texts

    def close(self) -> None:
        self.closed = True


def make_viewer(**kwargs):
    """A LiveViewer wired to a fake window over a real MuJoCo model."""
    model, data, _ = gc.make_scene(seed=0)
    handle = FakeHandle()

    def launch(model, data, key_callback=None):
        return handle

    viewer = LiveViewer(model, data, launch=launch, **kwargs)
    viewer.close_grace = 0.0
    return viewer, handle


def test_sync_is_throttled_but_the_first_call_goes_through():
    viewer, handle = make_viewer(fps=0.5)  # 1 update per 2 s
    assert viewer.sync() is True
    assert handle.syncs == 1
    for _ in range(5):  # way too soon -> dropped by the throttle
        assert viewer.sync() is True
    assert handle.syncs == 1
    viewer.close()


def test_force_bypasses_the_throttle():
    viewer, handle = make_viewer(fps=0.5)
    viewer.sync()
    viewer.sync(force=True)
    assert handle.syncs == 2
    viewer.close()


def test_set_camera_fixes_the_first_matching_camera():
    viewer, handle = make_viewer()
    assert viewer.set_camera("nope", "cam_iso") is True
    assert handle.cam.fixedcamid >= 0
    assert viewer.set_camera("nope") is False
    viewer.close()


def test_status_lines_are_published_to_the_window():
    viewer, handle = make_viewer()
    viewer.set_status(["step 100", "fps 4000"])
    viewer.sync()
    assert handle.texts is not None
    assert [entry[2] for entry in handle.texts] == ["step 100", "fps 4000"]
    viewer.close()


def test_sync_reports_false_and_closes_once_the_window_is_gone():
    viewer, handle = make_viewer()
    handle.running = False
    assert viewer.sync() is False
    assert handle.closed is True
    assert viewer.running is False
    # Closing twice must not touch the handle again.
    viewer.close()
    viewer.close()


def test_status_is_limited_to_the_four_screen_corners():
    viewer, handle = make_viewer()
    viewer.set_status([f"line{i}" for i in range(9)])
    viewer.sync()
    assert len(handle.texts) == 4
    viewer.close()


@pytest.mark.parametrize("fps", [0.0, -3.0])
def test_silly_fps_values_are_clamped(fps):
    viewer, _handle = make_viewer(fps=fps)
    assert viewer.fps >= 1.0
    viewer.close()


def test_callback_status_is_safe_without_a_model():
    """Regression: an un-attached callback must not blow up mid-training.

    ``status_lines`` used to read ``self.logger`` through SB3's property, which
    raises when no model is attached; the resulting exception inside the
    training loop left the viewer thread alive and the process hanging.
    """
    viewer, _handle = make_viewer()
    callback = LiveViewerCallback(viewer)
    lines = callback.status_lines()
    assert len(lines) == 4
    assert all(isinstance(line, str) for line in lines)
    assert callback._on_step() is True
    viewer.close()


def test_callback_stops_syncing_once_the_window_is_closed():
    viewer, handle = make_viewer()
    callback = LiveViewerCallback(viewer)
    callback._on_step()
    handle.running = False
    callback._on_step()  # notices the window is gone
    assert callback._closed is True
    syncs = handle.syncs
    callback._on_step()  # no further work after the window closes
    assert handle.syncs == syncs


def test_pacer_is_disabled_at_zero_speed():
    """``--live-speed 0`` must not slow training down at all."""
    pacer = Pacer(0.0)
    assert pacer.paced is False
    started = time.time()
    for _ in range(50):
        pacer.tick()
    assert time.time() - started < 0.05


def test_pacer_interval_follows_the_requested_speed():
    assert Pacer(1.0, sim_dt=0.02).interval == pytest.approx(0.02)
    assert Pacer(4.0, sim_dt=0.02).interval == pytest.approx(0.005)


def test_pacer_holds_the_target_rate():
    pacer = Pacer(speed=0.5, sim_dt=0.02)  # half real time -> 40 ms per step
    started = time.time()
    for _ in range(4):
        pacer.tick()
    assert time.time() - started >= 0.12


def test_rollout_reports_every_step():
    """The evaluation rollout must publish frames so the window never freezes."""
    from env import make_env
    from grasp.policy import Policy, rollout

    env = make_env("pro7_pick", obs_target="true")
    ticks = []
    ok, dist = rollout(env, Policy(), seed=0, on_step=lambda: ticks.append(1))
    env.close()
    assert 0 < len(ticks) <= env.max_steps
    assert isinstance(ok, bool) and dist >= 0.0
