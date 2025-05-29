from munch import Munch, munchify
from typing import Dict, Tuple, TypeVar, Union
from pathlib import path
class MunchConfig(Munch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __getattr__(self,item):
        if item not in self:
            return None
        return self[item]
Config = TypeVar("ConfigInput", MunchConfig, Dict, str, Path, None)
