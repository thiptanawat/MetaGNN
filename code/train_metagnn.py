# Alias module — allows `from train_metagnn import ...` to work
# with the numbered source file 03_train_metagnn.py
import importlib
_mod = importlib.import_module("03_train_metagnn")
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith('_')})
