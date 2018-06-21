import os
import pathlib


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
