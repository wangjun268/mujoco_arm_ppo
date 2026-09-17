"""Console-script entry points.

These exist so the interpreter is fixed *before* ``rclpy`` / ``mujoco`` are
imported; ``python -m pro7_pick_place_ros.entry node`` is the re-exec target of
:func:`pro7_pick_place_ros.runtime.ensure_runtime`.
"""

from __future__ import annotations

import sys

from .runtime import ensure_runtime


def run_node(argv=None) -> int:
    """Entry point of ``pick_place_node``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    ensure_runtime(("rclpy", "mujoco"), command="node", argv=argv)
    from .node import main

    main(argv)
    return 0


def run_client(argv=None) -> int:
    """Entry point of ``pick_place_client``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    ensure_runtime(("rclpy",), command="client", argv=argv)
    from .client import main

    return main(argv) or 0


def main(argv=None) -> int:
    """Dispatch ``python -m pro7_pick_place_ros.entry <node|client>``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    what = argv[0] if argv and argv[0] in ("node", "client") else "node"
    rest = argv[1:] if argv and argv[0] in ("node", "client") else argv
    return run_node(rest) if what == "node" else run_client(rest)


if __name__ == "__main__":
    sys.exit(main())
