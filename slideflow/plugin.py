"""
This module is responsible for loading all the plugins that are installed in the system.
"""

import pkg_resources

def load_plugins():
    for entry_point in pkg_resources.iter_entry_points('slideflow.plugins'):
        register = entry_point.load()
        register()

load_plugins()
