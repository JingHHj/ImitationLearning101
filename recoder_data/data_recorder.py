import os
os.environ.setdefault("MUJOCO_GL", "egl")  # GPU-accelerated offscreen rendering

import mujoco
import mujoco.viewer
import numpy as np
import cv2
from loop_rate_limiters import RateLimiter
import mink
import teledex
from utils import key_callback
from typing import List, Dict, Any
from configs import Configs, DataMetadata
from utils import randomization, get_camera_pos
import time


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
    def __init__(self, configs: Configs):
        self.side_camera_images: List[np.ndarray] = []
        self.ee_camera_images: List[np.ndarray] = []
        self.joint_positions: List[List[float]] = []
        self.metadata: DataMetadata = DataMetadata(
            cube_position=[],
            bin_position=[],
            system_frequency=0.0,
            camera_height=configs.camera_height,
            camera_width=configs.camera_width,
            total_timesteps=0,
            eps_completed=False,    
            name = f"episode_{int(time.time())}"
        )  

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

    def save(self, path: str):
        """
            Save the this episode's dataset to given path
        """
        self.metadata.total_timesteps = len(self.joint_positions)
        # TODO: implement saving logic, e.g. save images to disk and metadata to a JSON file.
        pass

class DataRecorder:
    def __init__(self, configs: Configs):
        self.configs = configs
        # --- MuJoCo setup ---
        self.model = mujoco.MjModel.from_xml_path(configs.model_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=configs.camera_height, width=configs.camera_width)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False # Disable shadows
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False # Disable reflections
        self.viewer = None  # Will be set when launching the viewer

        # --- Mink setup ---
        self.configuration = mink.Configuration(self.model)
        self.end_effector_task = mink.FrameTask(
            frame_name="gripper",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        # Posture regularization — biases joints toward the home configuration
        self.posture_task = mink.PostureTask(model=self.model, cost=1e-2)
        self.tasks = [self.end_effector_task, self.posture_task]
        self.limits = [mink.ConfigurationLimit(model=self.model)]

        # --- TeleDex setup ---
        self.session = teledex.Session(port=configs.teledex_port)
        self.handler = teledex.MujocoHandler(model=self.model, data=self.data)
        self.session.add_handler(self.handler)

        # --- Other setup ---
        self.rate = RateLimiter(frequency=200.0, warn=False) # Rate limiter for rendering
        self.dataset: Dataset = Dataset(configs) # For data recording
        self.gripper_closed: bool = False # For TeleDex toggle state
        self.step_count: int = 0  # For controlling render frequency
        self.prev_time = self.data.time # For detecting resets (time going backwards or becoming 0)
        self.session.start() # Start receiving phone data.
        self.recording = True
        

    def run(self):
        with mujoco.viewer.launch_passive(
            model=self.model, data=self.data, show_left_ui=False, show_right_ui=False, 
            key_callback=lambda keycode: self.key_callback(keycode=keycode)
        ) as self.viewer:
        
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # Initialize to pickup keyframe.
            self.configuration.update(self.data.qpos)
            self.posture_task.set_target_from_configuration(self.configuration)
            mujoco.mj_forward(self.model, self.data)  # Run forward kinematics so site positions are populated.

            # Place the mocap target at the current end-effector position.
            mink.move_mocap_to_frame(self.model, self.data, "mocap_target", "gripper", "site")

            self.handler.link_body(
                name="mocap_target",
                scale=2.0,
                position_origin=self.data.mocap_pos[0].copy(),
                rotation_origin=self.data.site("gripper").xmat.copy().reshape(3, 3),
                toggle_fn=lambda: self.toggle_gripper(self.gripper_closed),
            )

            while self.viewer.is_running():
                # Render camera views at 20 Hz (every 10th physics step).
                if self.step_count % self.configs.render_every == 0:
                    side_image, ee_image = self.render()
                    # Record data
                    if self.recording: # If recording, insert data into dataset
                        self.dataset.insert( 
                            qpos=self.data.qpos.copy(),
                            side_image=side_image,
                            ee_image=ee_image,
                        )
                    elif self.dataset.metadata.total_timesteps > 0: # If not recording and dataset not empty
                        self.dataset.metadata.eps_completed = self.check_task_completed() # Check if task completed
                        self.dataset.save(self.configs.output_path) # Save the dataset
                        self.dataset = Dataset(self.configs) # Reset the dataset for next episode
                    # If not recording and dataset empty, do nothing
    
                self.update()

            self.close()

    def key_callback(self, keycode):
        """
            Hit r, the environment will reset:
            1. Reset arm configuration to keyframe 0
            2. Position of the cube and the bin will be randomized. 

            Hit p, the current camera pose will be printed in the terminal. 
        """

        if keycode == ord("S") or keycode == ord("s"): # Set
            # Reset arm to keyframe 0 (sets data.time=0, triggering main loop reset detection)
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            # Randomize cube and bin positions
            x_cube, y_cube, x_bin, y_bin = randomization()
            # Set cube position
            box_adr = self.model.jnt_qposadr[self.model.body("box").jntadr[0]]
            self.data.qpos[box_adr:box_adr + 3] = [x_cube, y_cube, 0.03]

            bin_adr = self.model.jnt_qposadr[self.model.body("bin").jntadr[0]]
            self.data.qpos[bin_adr:bin_adr + 3] = [x_bin, y_bin, 0.035125]
            mujoco.mj_forward(self.model, self.data)
        
        if keycode == ord("P") or keycode == ord("p"): # Print
            get_camera_pos(self.model, self.data, self.viewer)

        if keycode == ord("S") or keycode == ord("s"): # Record
            self.recording = not self.recording
            print(f"Recording: {'ON' if self.recording else 'OFF'}")


    def update(self):
        """
            The update step of the main loop, including:
            1. Read target pose from the mocap body (updated by TeleDex).
            2. Solve IK — computes joint velocities that move EE toward target.
            3. Integrate velocities into joint positions.
            4. Set actuator controls.
            5. Step physics.
            6. Sync IK configuration back from physics state.
        """
        # TODO: Check if this following if is necessary
        # Detect reset (time goes backwards or becomes 0)
        if self.data.time < self.prev_time or self.data.time == 0:
            print("Reset detected! Updating IK configuration to pickup keyframe")
            # mujoco.mj_resetDataKeyframe(model, data, 0)
            self.configuration.update(self.data.qpos)
            self.posture_task.set_target_from_configuration(self.configuration)
            mujoco.mj_forward(self.model, self.data)
            mink.move_mocap_to_frame(self.model, self.data, "mocap_target", "gripper", "site")

        self.prev_time = self.data.time

        # 1. Read target pose from the mocap body (updated by TeleDex).
        self.end_effector_task.set_target(
            mink.SE3.from_mocap_name(self.model, self.data, "mocap_target")
        )

        # 2. Solve IK — computes joint velocities that move EE toward target.
        vel = mink.solve_ik(
            configuration=self.configuration,
            tasks=self.tasks,
            dt=self.rate.dt,
            solver="quadprog",
            damping=1e-3,
            limits=self.limits,
        )
        # 3. Integrate velocities into joint positions.
        self.configuration.integrate_inplace(vel, self.rate.dt)
        # 4. Set actuator controls.
        self.data.ctrl = np.concatenate(
            [self.configuration.q[:7], 
             [0.0 if self.gripper_closed else 0.04]]
        )
        # 5. Step physics.
        mujoco.mj_step(self.model, self.data)
        # 6. Sync IK configuration back from physics state.
        self.configuration.update(self.data.qpos)
        self.step_count += 1
        self.viewer.sync()
        self.rate.sleep()

    def render(self)-> List[np.ndarray, np.ndarray]:
        """
            Rendering the two camera views and display them side by side using OpenCV.
        """
        self.renderer.update_scene(self.data, camera="side_camera")
        side_image = self.renderer.render()

        self.renderer.update_scene(self.data, camera="ee_camera")
        ee_image = self.renderer.render()
        cv2.imshow("Camera Views", cv2.cvtColor(
            np.hstack([side_image, ee_image]), 
            cv2.COLOR_RGB2BGR)
        )
        # cv2.waitKey(1)
        return side_image, ee_image

    def close(self):
        """
            Clean up resources when done.
        """
        self.renderer.close()
        cv2.destroyAllWindows()
        self.session.stop()
        
    def check_task_completed(self) -> bool:
        """
            Check if the current episode's task is completed, e.g. by checking the position of the cube.
        """
        # TODO: Implement task completion check, e.g. check if the cube is in the bin.
        return False


    def toggle_gripper(self, gripper_closed: bool):
        """
            Hitting the toggle button in the iOS app, will call this function to open/close the gripper.
        """
        gripper_closed = not gripper_closed
        print(f"Gripper: {'CLOSED' if gripper_closed else 'OPEN'}")

                
if __name__ == "__main__":
    cfg = Configs(
        model_path="../assets/franka_emika_panda/mjx_single_cube.xml",
        output_path="data",
        camera_height=480,      
        camera_width=480,
        teledex_port=8888,
        render_every=20
    )
    recorder = DataRecorder(configs=cfg)
    recorder.run()