# Alias module — allows `from data_loader import ...` to work
# with the numbered source file 02_data_loader.py
import importlib
_mod = importlib.import_module("02_data_loader")
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith('_')})
