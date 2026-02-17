import h5py
import numpy as np
import os
import time
from typing import List
from configs import Configs, DataMetadata
import mujoco

class Dataset:
    """
        The structure of the dataset for one episode, includes main data list and metadata.

        main data list looks like:
        - step 0: [side_image, ee_image, joint_positions]
        - step 1: [side_image, ee_image, joint_positions]
        - ...
        metadata includes:
        - cube_position
        - bin_position
        - system_frequency
        - camera_height
        - camera_width
        - total_timesteps
        - eps_completed
        - name 

    """
    def __init__(self, configs: Configs, data: mujoco.MjData, model: mujoco.MjModel):
        self.side_camera_images: List[np.ndarray] = []
        self.ee_camera_images: List[np.ndarray] = []
        self.joint_positions: List[List[float]] = []
        
        self.metadata: DataMetadata = DataMetadata(
            cube_position=data.body("box").xpos.copy(),
            bin_position=data.body("bin").xpos.copy(),
            system_frequency=1.0 / (model.opt.timestep * configs.render_every),
            camera_height=configs.camera_height,
            camera_width=configs.camera_width,
            total_timesteps=0,
            eps_completed=False,    
            name = f"episode_{int(time.time())}"
        ) 

        os.makedirs(configs.output_path, exist_ok=True)
        self.filepath = os.path.join(configs.output_path, f"{self.metadata.name}.hdf5")

    def insert(self, 
               side_image: np.ndarray, 
               ee_image: np.ndarray, 
               joint_positions: List[float]):
        """
            Add a new data pair into the dataset
        """
        self.side_camera_images.append(side_image)
        self.ee_camera_images.append(ee_image)
        self.joint_positions.append(joint_positions)
        self.metadata.total_timesteps += 1  

    def save(self):
        """
            Save the this episode's dataset to given path:
            - Every episode is saved as a HDF5 file
            - Inside of each HDF5 file, each row is a timestep, with following columns: side_image, ee_image, joint_positions
            - Metadata is saved in the attributes of the HDF5 file. 
        """
        with h5py.File(self.filepath, "w") as f:
            f.create_dataset("side_images", data=np.array(self.side_camera_images, dtype=np.uint8))
            f.create_dataset("ee_images", data=np.array(self.ee_camera_images, dtype=np.uint8))
            f.create_dataset("joint_positions", data=np.array(self.joint_positions, dtype=np.float64))

            f.attrs["cube_position"] = self.metadata.cube_position
            f.attrs["bin_position"] = self.metadata.bin_position
            f.attrs["system_frequency"] = self.metadata.system_frequency
            f.attrs["camera_height"] = self.metadata.camera_height
            f.attrs["camera_width"] = self.metadata.camera_width
            f.attrs["total_timesteps"] = self.metadata.total_timesteps
            f.attrs["eps_completed"] = self.metadata.eps_completed
            f.attrs["name"] = self.metadata.name

        print(f"Saved episode to {self.filepath} ({self.metadata.total_timesteps} timesteps)")

        