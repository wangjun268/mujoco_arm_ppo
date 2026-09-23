"""Vision grasp / pick & place stack for the Pro7 + LinkerHand L20 cell.

The package collects everything that belongs to the vision-driven grasp cell:

* :mod:`grasp.common` -- scene constants, scripted expert, pick & place plan
* :mod:`grasp.detect` -- red-cube segmentation and RGB-D localisation
* :mod:`grasp.policy` -- the 128-128-Tanh MLP shared by training and replay
* :mod:`grasp.demo` / :mod:`grasp.pick_place_demo` -- scripted end-to-end runs
* :mod:`grasp.montage` / :mod:`grasp.overlay` / :mod:`grasp.visualize`
* :mod:`grasp.supervised` / :mod:`grasp.train_live` -- DAgger training
* :mod:`grasp.view` -- live MuJoCo window for the grasp / pick & place task

Importing the package deliberately imports nothing else: ``import grasp`` must
stay cheap (no MuJoCo context, no torch), so the entry points stay testable.
Every entry point also runs as a plain script -- ``python3 grasp/demo.py`` --
which is why each of them puts the checkout root on ``sys.path`` itself.
"""

from __future__ import annotations
