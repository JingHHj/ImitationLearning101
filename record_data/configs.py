from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Configs:
    model_path: str = "../assets/franka_emika_panda/mjx_single_cube.xml"
    output_path: str = "data"
    camera_height: int = 480
    camera_width: int = 480
    teledex_port: int = 8888
    render_every: int = 10  # render every N steps to save computation

@dataclass
class DataMetadata:
    cube_position: Optional[float] = None
    bin_position: Optional[float] = None
    system_frequency: float = 0.0
    camera_height: int = 480
    camera_width: int = 480
    total_timesteps: int = 0
    eps_completed: bool = False
    name: str = 'episode-0'

  