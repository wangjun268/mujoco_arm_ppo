"""The wrist adapter has to stay the part the scenes mount, and stay harmless.

``tools/make_wrist_flange.py`` generates the adapter that fills the 60 mm between
the Pro7 wrist and the LinkerHand L20; the pick cell and the reach training model
(``pro7_urdf``) all carry it.  These tests pin the two properties that make it
safe to have added: it is regenerable, and it is *visual only* -- removing it
from the XML changes neither the mass nor one step of the trajectory.
"""

import os
import re

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from paths import ASSETS_DIR, asset_path  # noqa: E402
from tools.make_wrist_flange import (  # noqa: E402
    PROFILE,
    SEGMENTS,
    revolve,
    write_stl,
)

#: ``model -> geom name`` of every scene that carries an adapter.
SCENES = (
    ("rokae_xmate_pro7_pick_real.xml", "wrist_flange"),
    ("rokae_xmate_pro7_real.xml", "wrist_flange"),
)


def test_profile_fills_the_wrist_to_hand_gap():
    """60 mm stand-off, never wider than the 30 mm flange face it sits on."""
    assert PROFILE[0][1] == 0.0
    assert PROFILE[-1][1] == pytest.approx(0.06)
    assert max(radius for radius, _ in PROFILE) <= 0.030
    assert all(0.0 < radius <= 0.030 for radius, _ in PROFILE)
    # z must be monotonic, or the revolved surface self-intersects.
    assert all(a[1] < b[1] for a, b in zip(PROFILE, PROFILE[1:]))

    triangles = revolve(PROFILE, SEGMENTS)
    vertices = triangles.reshape(-1, 3)
    assert vertices[:, 2].min() == pytest.approx(0.0)
    assert vertices[:, 2].max() == pytest.approx(0.06)
    assert np.hypot(vertices[:, 0], vertices[:, 1]).max() == pytest.approx(0.030)
    # A closed surface with outward normals encloses a positive volume.
    volume = np.einsum(
        "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
    ).sum() / 6.0
    assert volume > 0.1e-3  # > 100 cm^3: a solid turned part, not a shell


def test_committed_stl_is_regenerable(tmp_path):
    """Re-running the generator must reproduce the checked-in mesh."""
    out = tmp_path / "flange.stl"
    write_stl(revolve(PROFILE, SEGMENTS), str(out))
    with open(asset_path("meshes/pro7_l20_flange.stl"), "rb") as fh:
        assert out.read_bytes() == fh.read()


@pytest.mark.parametrize("model, geom", SCENES)
def test_adapter_is_visual_only(model, geom):
    """Removing the adapter may not change the mass or a single step."""
    text = open(asset_path(model)).read()
    pattern = re.compile(r'\s*<geom name="%s".*?/>' % geom, re.S)
    assert pattern.search(text), f"{model} no longer mounts {geom}?"
    # The stripped copy has to live in assets/, or its relative meshes miss.
    probe = os.path.join(ASSETS_DIR, f".strip_{geom}_{os.getpid()}.xml")
    with open(probe, "w") as fh:
        fh.write(pattern.sub("", text))
    try:
        with_adapter = mujoco.MjModel.from_xml_path(asset_path(model))
        without = mujoco.MjModel.from_xml_path(probe)
        assert float(with_adapter.body_mass.sum()) == pytest.approx(
            float(without.body_mass.sum()), abs=1e-12
        )
        a, b = mujoco.MjData(with_adapter), mujoco.MjData(without)
        controls = np.linspace(-1.0, 1.0, 5 * with_adapter.nu).reshape(5, with_adapter.nu)
        for row in controls:
            a.ctrl[:] = row
            b.ctrl[:] = row
            mujoco.mj_step(with_adapter, a)
            mujoco.mj_step(without, b)
        assert np.array_equal(a.qpos, b.qpos)
    finally:
        os.remove(probe)
