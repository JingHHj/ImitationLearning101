#!/usr/bin/env python3
"""
Teleoperate Franka Panda in MuJoCo with real-time camera display.

Same as teleop.py but also renders side_camera and ee_camera views
in an OpenCV window during teleoperation.

Usage:
  1. Run this script: python teleop_with_cameras.py
  2. Open the MuJoCo AR iOS app
  3. Enter the IP and port printed in the terminal
  4. Move your phone to control the end-effector
  5. Use the toggle button on the app to open/close the gripper

Dependencies:
  pip install mink teledex loop-rate-limiters quadprog opencv-python
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")  # GPU-accelerated offscreen rendering

import mujoco
import mujoco.viewer
import numpy as np
import cv2
from loop_rate_limiters import RateLimiter
import mink
import teledex
from recoder_data.utils import key_callback

CAMERA_WIDTH = 480
CAMERA_HEIGHT = 480



def main():
    model = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/mjx_single_cube.xml")
    data = mujoco.MjData(model)

    # Offscreen renderer for camera views
    renderer = mujoco.Renderer(model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

    configuration = mink.Configuration(model)
    end_effector_task = mink.FrameTask(
        frame_name="gripper",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )

    # Posture regularization — biases joints toward the home configuration
    posture_task = mink.PostureTask(model=model, cost=1e-2)

    tasks = [end_effector_task, posture_task]
    limits = [mink.ConfigurationLimit(model=model)]
    gripper_closed = False

    def toggle_gripper():
        nonlocal gripper_closed
        gripper_closed = not gripper_closed
        print(f"Gripper: {'CLOSED' if gripper_closed else 'OPEN'}")

    # --- TeleDex setup ---
    session = teledex.Session(port=8888)
    handler = teledex.MujocoHandler(model=model, data=data)
    session.add_handler(handler)

    


    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False, 
        key_callback=lambda keycode: key_callback(keycode, model, data)
    ) as viewer:
        
        mujoco.mj_resetDataKeyframe(model, data, 0)  # Initialize to pickup keyframe.
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)

        mujoco.mj_forward(model, data)  # Run forward kinematics so site positions are populated.

        # Place the mocap target at the current end-effector position.
        mink.move_mocap_to_frame(model, data, "mocap_target", "gripper", "site")

        initial_pos = data.mocap_pos[0].copy()
        initial_mat = data.site("gripper").xmat.copy().reshape(3, 3)

        handler.link_body(
            name="mocap_target",
            scale=2.0,
            position_origin=initial_pos,
            rotation_origin=initial_mat,
            toggle_fn=toggle_gripper,
        )

        # Start receiving phone data.
        session.start()
        rate = RateLimiter(frequency=200.0, warn=False)
        render_every = 10  # Render cameras at 20 Hz (200 Hz / 10)
        step_count = 0
        prev_time = data.time
        while viewer.is_running():
            # Detect reset (time goes backwards or becomes 0)
            if data.time < prev_time or data.time == 0:
                print("Reset detected! Updating IK configuration to pickup keyframe")
                # mujoco.mj_resetDataKeyframe(model, data, 0)
                configuration.update(data.qpos)
                posture_task.set_target_from_configuration(configuration)
                mujoco.mj_forward(model, data)
                mink.move_mocap_to_frame(model, data, "mocap_target", "gripper", "site")
            prev_time = data.time

            # 1. Read target pose from the mocap body (updated by TeleDex).
            T_wt = mink.SE3.from_mocap_name(model, data, "mocap_target")
            end_effector_task.set_target(T_wt)

            # 2. Solve IK — computes joint velocities that move EE toward target.
            vel = mink.solve_ik(
                configuration=configuration,
                tasks=tasks,
                dt=rate.dt,
                solver="quadprog",
                damping=1e-3,
                limits=limits,
            )

            # 3. Integrate velocities into joint positions.
            configuration.integrate_inplace(vel, rate.dt)

            # 4. Set actuator controls.
            data.ctrl[:7] = configuration.q[:7]
            data.ctrl[7] = 0.0 if gripper_closed else 0.04

            # 5. Step physics.
            mujoco.mj_step(model, data)

            # 6. Sync IK configuration back from physics state.
            configuration.update(data.qpos)

            # 7. Render camera views at 20 Hz (every 10th physics step).
            if step_count % render_every == 0:
                renderer.update_scene(data, camera="side_camera")
                side_image = renderer.render()

                renderer.update_scene(data, camera="ee_camera")
                ee_image = renderer.render()

                combined = np.hstack([side_image, ee_image])
                cv2.imshow("Camera Views", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0.01)

            step_count += 1
            viewer.sync()
            rate.sleep()

    renderer.close()
    cv2.destroyAllWindows()
    session.stop()


if __name__ == "__main__":
    main()
