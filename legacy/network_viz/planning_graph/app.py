"""Application entry point for the LiSNN U1 alpha planning graph."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from PySide6.QtWidgets import QApplication

from .rendering import configure_runtime, install_runtime_patches
from .theme import apply_dark_theme


def _parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--software-rendering",
        action="store_true",
        help="Use the Qt raster viewport instead of the default OpenGL viewport.",
    )
    parser.add_argument(
        "--performance-mode",
        action="store_true",
        help="Start with more aggressive level-of-detail thresholds.",
    )
    return parser.parse_known_args(list(argv[1:]))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv if argv is None else argv)
    args, qt_arguments = _parse_args(arguments)
    configure_runtime(
        software_rendering=args.software_rendering,
        performance_mode=args.performance_mode,
    )

    app = QApplication([arguments[0], *qt_arguments])
    app.setApplicationName("LiSNN Planning Graph U1 Alpha")
    apply_dark_theme(app)

    # Import and patch after runtime preferences are known, but before any
    # PlanningView or graphics item is instantiated.
    from .window import PlanningGraphWindow

    install_runtime_patches()
    window = PlanningGraphWindow()
    window.show()
    return app.exec()
