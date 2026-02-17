import os
os.environ.setdefault("MUJOCO_GL", "egl")  # GPU-accelerated offscreen rendering

import mujoco
import mujoco.viewer
import numpy as np
import cv2

from loop_rate_limiters import RateLimiter
import mink
import teledex
from configs import Configs
from utils import randomization, get_camera_pos
from dataset import Dataset


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
        self.dataset: Dataset = Dataset(configs, self.data, self.model) # For data recording
        
        self.gripper_closed: bool = False # For TeleDex toggle state
        self.step_count: int = 0  # For controlling render frequency
        self.session.start() # Start receiving phone data.
        self.recording = False    

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
                toggle_fn=lambda: self.toggle_gripper(),
            )

            while self.viewer.is_running():
                # Render camera views at 20 Hz (every 10th physics step).
                if self.step_count % self.configs.render_every == 0:
                    side_image, ee_image = self.render()
                    self.dataset.insert( 
                        joint_positions=self.data.qpos[:7].tolist(), # Record only the 7 arm joints, not the gripper
                        side_image=side_image,
                        ee_image=ee_image,
                    )
                self.update()
            self.close()

    def key_callback(self, keycode):
        """
            Hit r, the environment will reset:
            1. Reset arm configuration to keyframe 0
            2. Position of the cube and the bin will be randomized. 

            Hit p, the current camera pose will be printed in the terminal. 
        """

        if keycode == ord("S") or keycode == ord("s"): # RESET
            ## 1. Reset arm to keyframe 0 
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            ## 2. Randomize cube and bin positions
            x_cube, y_cube, x_bin, y_bin = randomization()
            # Cube
            box_adr = self.model.jnt_qposadr[self.model.body("box").jntadr[0]]
            self.data.qpos[box_adr:box_adr + 3] = [x_cube, y_cube, 0.03]

            # Bin
            bin_adr = self.model.jnt_qposadr[self.model.body("bin").jntadr[0]]
            self.data.qpos[bin_adr:bin_adr + 3] = [x_bin, y_bin, 0.035125]
            mujoco.mj_forward(self.model, self.data)

            # 3. Update IK configuration and mocap target
            self.configuration.update(self.data.qpos)
            self.posture_task.set_target_from_configuration(self.configuration)
            mink.move_mocap_to_frame(self.model, self.data, "mocap_target", "gripper", "site")

            # 4. Reset dataset for new episode
            self.dataset = Dataset(self.configs, self.data, self.model) # Reset the dataset for next episode
        
        if keycode == ord("P") or keycode == ord("p"): # Print
            get_camera_pos(self.model, self.data, self.viewer)

        if keycode == ord("R") or keycode == ord("r"): # Record
            if self.recording:
            # Recording --> stop and save 
                self.dataset.metadata.eps_completed = self.check_task_completed() # Check if task completed
                self.dataset.save() # Save the dataset
                
            # Not recording --> start recording
            self.recording = not self.recording # Toggle recording state anyways
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

    def render(self) -> tuple[np.ndarray, np.ndarray]:
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
        cv2.waitKey(1)
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
            Check if the cube's center is inside the bin.
            Bin bounds are derived from geom sizes/positions in the XML.
        """
        # Derive inner bounds from model geoms (all in bin-local frame)
        bottom_size = self.model.geom("bin_bottom").size
        right_wall = self.model.geom("bin_right")
        front_wall = self.model.geom("bin_front")

        inner_x = right_wall.pos[0] - right_wall.size[0]  # wall offset - wall thickness
        inner_y = front_wall.pos[1] - front_wall.size[1]
        z_min = bottom_size[2]                              # top of bottom plate
        z_max = right_wall.pos[2] + right_wall.size[2]     # top of walls

        cube_pos = self.data.body("box").xpos
        bin_pos = self.data.body("bin").xpos
        rel = cube_pos - bin_pos

        in_x = -inner_x < rel[0] < inner_x
        in_y = -inner_y < rel[1] < inner_y
        in_z = z_min < rel[2] < z_max

        return in_x and in_y and in_z


    def toggle_gripper(self):
        """
            Hitting the toggle button in the iOS app, will call this function to open/close the gripper.
        """
        self.gripper_closed = not self.gripper_closed
        print(f"Gripper: {'CLOSED' if self.gripper_closed else 'OPEN'}")

                
if __name__ == "__main__":
    cfg = Configs(
        model_path="../assets/franka_emika_panda/mjx_single_cube.xml",
        output_path="data",
        camera_height=360,      
        camera_width=360,
        teledex_port=8888,
        render_every=4 # Render every 4 steps, f = 50 HZ, dt = 20ms
    )
    recorder = DataRecorder(configs=cfg)
    recorder.run()