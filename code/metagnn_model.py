# Alias module — allows `from metagnn_model import ...` to work
# with the numbered source file 01_metagnn_model.py
import importlib
_mod = importlib.import_module("01_metagnn_model")
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith('_')})
