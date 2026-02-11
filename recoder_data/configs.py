from dataclasses import dataclass
from typing import List

@dataclass
class Configs:
    model_path: str = "../assets/franka_emika_panda/mjx_single_cube.xml"
    output_path: str = "data"
    camera_height: int = 480
    camera_width: int = 480
    teledex_port: int = 8888
    render_every: int = 20  # render every N steps to save computation

@dataclass
class DataMetadata:
    cube_position: List[float]
    bin_position: List[float]
    system_frequency: float
    camera_height: int
    camera_width: int
    total_timesteps: int