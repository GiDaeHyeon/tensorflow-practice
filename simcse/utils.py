"""SimCSE Utils"""
from pathlib import Path

import yaml


def config(file_path: str):
    with Path(file_path).open("r", encoding="utf8") as cf:
        _config = yaml.load(cf, yaml.FullLoader)
    return _config
