"""Model tooling: regenerate the MJCF / URDF assets from their sources.

* :mod:`tools.build_dual_arm_model` -- lkwy73_o1 description -> ``assets/dual_arm_reach.xml``
* :mod:`tools.convert_arm_urdf` -- compiled MJCF cell -> URDF for rviz2
* :mod:`tools.convert_hand_urdf` -- vendor LinkerHand L20 URDF -> MJCF fragments
* :mod:`tools.make_wrist_flange` -- Pro7 wrist -> L20 adapter (revolved STL)
* :mod:`tools.urdf_selfcheck` -- re-load an export in MuJoCo and diff it

The assets under ``assets/`` are build products of these scripts; the vendor
URDF / cleaned robot descriptions are the sources, so every generator is safe
to re-run (that is what the tests check).
"""

from __future__ import annotations
