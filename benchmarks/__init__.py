import pkgutil
import importlib

# Discover and import all benchmark runners
for _, name, _ in pkgutil.walk_packages(__path__):
    importlib.import_module(f'.{name}', __name__)
