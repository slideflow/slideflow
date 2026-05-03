"""
This module is responsible for loading all the plugins that are installed in the system.
"""

import pkg_resources

from slideflow.util import log


def load_plugins():
    for entry_point in pkg_resources.iter_entry_points('slideflow.plugins'):
        try:
            register = entry_point.load()
        except Exception as e:
            log.warning(f"Failed to load plugin {entry_point.name}: {e}")
            continue
        if not callable(register):
            log.error(
                f"Plugin {entry_point.name} did not export a callable; "
                "skipping."
            )
            continue
        try:
            register()
        except Exception as e:
            log.warning(
                f"Plugin {entry_point.name} register() raised: {e}"
            )

load_plugins()
