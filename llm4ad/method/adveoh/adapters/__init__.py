"""Auto-import all adapter modules so their classes are accessible as
``from llm4ad.method.adveoh.adapters import AdvTSPEvaluation`` etc.
"""
import os
import importlib
import inspect

__all__ = []

_package_dir = os.path.dirname(__file__)
for _filename in sorted(os.listdir(_package_dir)):
    if _filename.endswith('.py') and _filename != '__init__.py' and not _filename.startswith('_'):
        _module_name = _filename[:-3]
        try:
            _module = importlib.import_module(f'.{_module_name}', package=__name__)
            for _attr_name in dir(_module):
                _attr = getattr(_module, _attr_name)
                if inspect.isclass(_attr) and _attr.__module__ == _module.__name__:
                    globals()[_attr_name] = _attr
                    if _attr_name not in __all__:
                        __all__.append(_attr_name)
                elif isinstance(_attr, str) and _attr_name.endswith('_template_program'):
                    globals()[_attr_name] = _attr
                    if _attr_name not in __all__:
                        __all__.append(_attr_name)
                elif isinstance(_attr, str) and _attr_name.endswith('_task_description'):
                    globals()[_attr_name] = _attr
                    if _attr_name not in __all__:
                        __all__.append(_attr_name)
        except (ImportError, ModuleNotFoundError) as e:
            pass
