"""The exported URDF has to stay identical to the MuJoCo cell it came from.

``tools/convert_arm_urdf.py`` re-emits ``assets/rokae_xmate_pro7_pick_real.urdf``
from the compiled MJCF, and its ``--check`` re-loads the result in MuJoCo to diff
the geometry, the kinematic chain and the masses.  Both halves are covered here,
so editing the model without re-running the converter (or breaking the
converter) fails the suite instead of quietly shipping a stale URDF.
"""

import os
import xml.etree.ElementTree as ET

import pytest

from tools.convert_arm_urdf import (
    DEFAULT_MODEL,
    GRIPPER_BODY,
    ROOT_BODY,
    CellModel,
    UrdfWriter,
    asset_path,
)
from tools.urdf_selfcheck import check


def _urdf_path() -> str:
    return os.path.splitext(asset_path(DEFAULT_MODEL))[0] + ".urdf"


def _writer_kwargs() -> dict:
    """The defaults ``main()`` writes the shipped URDF with."""
    return dict(mesh_uri="file", urdf_dir=os.path.dirname(_urdf_path()), hand="merged")


def test_committed_urdf_is_regenerable():
    """Re-running the converter must reproduce the checked-in file, byte for byte."""
    with open(_urdf_path()) as fh:
        committed = fh.read()
    assert UrdfWriter(CellModel(asset_path(DEFAULT_MODEL)), **_writer_kwargs()).xml() == committed


def test_committed_urdf_is_valid_xml():
    """MuJoCo is lenient; rviz, xacro and ElementTree are not."""
    with open(_urdf_path()) as fh:
        root = ET.fromstring(fh.read().encode("utf-8"))
    assert root.tag == "robot"
    assert len(root.findall("link")) == 9


def test_urdf_round_trips_through_mujoco():
    """MuJoCo has to re-load the URDF as the same robot: geoms, chain, mass."""
    cell = CellModel(asset_path(DEFAULT_MODEL))
    assert check(cell, _writer_kwargs(), poses=2)


@pytest.mark.parametrize("hand", ("merged", "articulated", "skip"))
def test_every_hand_mode_round_trips(hand):
    """``--hand skip`` / ``articulated`` have to re-load with the same mass too.

    The mass is where a massless link bites: without an explicit <inertial> a
    URDF consumer guesses one from the geometry, and the adapter on the wrist
    became 0.14 kg of imaginary metal in the two exports that do not bake the
    hand (and its inertia) into ``gripper``.
    """
    cell = CellModel(asset_path(DEFAULT_MODEL))
    assert check(cell, {**_writer_kwargs(), "hand": hand}, poses=1)


def test_urdf_names_match_the_topics_the_node_publishes():
    """The display is driven by the node's TF and joint states, so names matter."""
    cell = CellModel(asset_path(DEFAULT_MODEL))
    text = UrdfWriter(cell, **_writer_kwargs()).xml()
    # TF chain of pro7_pick_place_ros.scene: world -> base -> link1..7 -> gripper.
    assert cell.body_name(cell.body(ROOT_BODY)) == "base"
    for link in ["base", *[f"link{i}" for i in range(1, 8)], GRIPPER_BODY]:
        assert f'<link name="{link}">' in text
    # ... driven by the 7 arm joints of /pro7_pick_place/joint_states.
    for joint in range(1, 8):
        assert f'<joint name="joint{joint}"' in text


def test_articulated_hand_adds_every_l20_joint():
    """``--hand articulated`` keeps the full 22-link hand instead of baking it."""
    cell = CellModel(asset_path(DEFAULT_MODEL))
    text = UrdfWriter(cell, **{**_writer_kwargs(), "hand": "articulated"}).xml()
    assert text.count("<link ") == 31  # 9 arm links + 22 hand links
    assert text.count("<joint ") == 30  # 7 arm + 21 L20 + the two fixed ones
    for joint in ("middle_pip", "index_mcp_roll", "thumb_dip"):
        assert f'<joint name="{joint}"' in text
