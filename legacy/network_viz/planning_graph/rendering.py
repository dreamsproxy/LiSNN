"""Public runtime facade for the planning graph rendering layer.

Qt platform plugins differ in how long borrowed native-menu wrappers remain
valid. This module installs the renderer while giving the performance toggle
its own Python-owned menu, avoiding transient QMenu wrappers on offscreen and
native-menu platforms. Navigation, the expanding world, and overview group
titles remain independent of the OpenGL/software backend.
"""

from __future__ import annotations

from . import group_titles as _group_titles
from . import navigation as _navigation
from . import performance as _performance
from . import world as _world

RenderProfile = _performance.RenderProfile
BALANCED_PROFILE = _performance.BALANCED_PROFILE
PERFORMANCE_PROFILE = _performance.PERFORMANCE_PROFILE
active_profile = _performance.active_profile
configure_runtime = _performance.configure_runtime
detail_tier = _performance.detail_tier
grid_step_for_lod = _performance.grid_step_for_lod
performance_mode_enabled = _performance.performance_mode_enabled
set_performance_mode = _performance.set_performance_mode

_INSTALLED = False


def install_runtime_patches() -> None:
    """Install renderer, navigation, expanding world, titles, and UI controls."""

    global _INSTALLED
    if _INSTALLED:
        return

    # The original installer attempted to borrow the existing View menu through
    # QMenuBar.actions(). Some platform plugins invalidate that transient QMenu
    # wrapper. Returning None lets the core installer create the action and
    # status indicator without touching a borrowed menu.
    _performance._find_menu = lambda window, title: None
    _performance.install_runtime_patches()

    # Install navigation after the renderer so it wraps the final
    # OpenGL/software viewport initializer. Event handling itself remains
    # backend-independent.
    _navigation.install_navigation_patches()

    # The world wrapper must follow navigation because it extends zoom and pan
    # with scene-rect expansion. The scene rectangle remains finite at any one
    # moment but grows in stable chunks whenever content or the viewport moves.
    _world.install_world_patches()

    # Install after the LOD painter so the title banner only supplements the
    # overview tiers where the normal group label has been culled.
    _group_titles.install_group_title_patches()

    from .window import PlanningGraphWindow

    original_window_init = PlanningGraphWindow.__init__

    def window_init(self, *args, **kwargs):
        original_window_init(self, *args, **kwargs)
        self.rendering_menu = self.menuBar().addMenu("Rendering")
        self.rendering_menu.addAction(self.performance_action)

    PlanningGraphWindow.__init__ = window_init
    _INSTALLED = True
