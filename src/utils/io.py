import os
import pathlib


def ensure_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
