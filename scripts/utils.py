from munch import Munch, munchify
from typing import Dict, Tuple, TypeVar, Union
from pathlib import Path
from logger import Logger
import torch 

from pathlib import Path
import numpy as np
import struct
import matplotlib.pyplot as plt

BASEDIR = Path(__file__).parent.parent
SEARCH_DIRS = [BASEDIR]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRONT_LEFT_UP_T_OPENCV = torch.tensor(
    [
        [+0.0, +0.0, +1.0, +0.0],
        [-1.0, +0.0, +0.0, +0.0],
        [+0.0, -1.0, +0.0, +0.0],
        [+0.0, +0.0, +0.0, +1.0],
    ],
    device="cpu",
)
OPENCV_T_FRONT_LEFT_UP = torch.linalg.inv(FRONT_LEFT_UP_T_OPENCV)
class MunchConfig(Munch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __getattr__(self,item):
        if item not in self:
            return None
        return self[item]
Config = TypeVar("ConfigInput", MunchConfig, Dict, str, Path, None)
def load_config(config: Config) -> Config:
    if config is None:
        return None

    # Type check
    if isinstance(config, Dict):
        config = munchify(config, factory=MunchConfig)
    if isinstance(config, str):
        config = Path(config)
    if not isinstance(config, Path) and not isinstance(config, MunchConfig):
        raise ValueError(f"Invalid config: {config}")

    # Load config
    if isinstance(config, Path):
        # Add to search paths
        if (
            config.is_absolute()
            and config.exists()
            and config.is_file()
            and config not in SEARCH_DIRS
        ):
            SEARCH_DIRS.append(config.parent)

        # Check if path has a key
        parts = str(config).split("::")
        if len(parts) > 1:
            key = parts[1]
            config = parts[0]
        else:
            key = None

        # Find absolute path
        abs_path = None
        for basedir in SEARCH_DIRS[::-1]:
            if (basedir / config).exists():
                abs_path = basedir / config
                break

        if abs_path is None:
            raise ValueError(f"Config not found: {config}")

        # Load config
        with open(abs_path, "r") as f:
            all_cfgs = munchify(yaml.safe_load(f), factory=MunchConfig)

        if key is not None:
            cfg = all_cfgs[key]
        else:
            all_keys = list(all_cfgs.keys())
            if len(all_keys) != 1:
                raise ValueError(f"Config must contain exactly one key, got {len(all_keys)}")
            key = all_keys[0]
            cfg = all_cfgs[key]

        Logger.info(f"Loaded {key} from {abs_path}", name="Config")
    else:
        key = None
        cfg = config
        all_cfgs = {}

    # Assign default name if not provided
    if "name" not in cfg:
        cfg["name"] = key



def read_pfm(filename):
    with Path(filename).open('rb') as pfm_file:

        line1, line2, line3 = (pfm_file.readline().decode('latin-1').strip() for _ in range(3))
        assert line1 in ('PF', 'Pf')
        
        channels = 3 if "PF" in line1 else 1
        width, height = (int(s) for s in line2.split())
        scale_endianess = float(line3)
        bigendian = scale_endianess > 0
        scale = abs(scale_endianess)

        buffer = pfm_file.read()
        samples = width * height * channels
        assert len(buffer) == samples * 4
        
        fmt = f'{"<>"[bigendian]}{samples}f'
        decoded = struct.unpack(fmt, buffer)
        shape = (height, width, 3) if channels == 3 else (height, width)
        return np.flipud(np.reshape(decoded, shape)) * scale



def field_of_view2focal_length(
    res: Union[int, Tuple[int, int]], fov: Union[float, Tuple[float, float]]
) -> Tuple[float, float]:
    if isinstance(res, int):
        return (res / 2) / np.tan(fov / 2)
    else:
        return (
            (res[0] / 2) / np.tan(fov[0] / 2),
            (res[1] / 2) / np.tan(fov[1] / 2),
        )
    
def main():
    # image = read_pfm('memorial.pfm')
    # plt.imshow(image)
    # plt.show()