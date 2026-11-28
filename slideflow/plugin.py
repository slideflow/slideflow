"""
This module is responsible for loading all the plugins that are installed in the system.
"""

def _plugin_entry_points():
    try:
        from importlib.metadata import entry_points
    except ImportError:  # Python 3.7
        from pkg_resources import iter_entry_points
        return iter_entry_points('slideflow.plugins')
    eps = entry_points()
    if hasattr(eps, 'select'):  # Python 3.10+
        return eps.select(group='slideflow.plugins')
    return eps.get('slideflow.plugins', [])

def load_plugins():
    for entry_point in _plugin_entry_points():
        register = entry_point.load()
        register()

load_plugins()
