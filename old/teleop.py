#!/usr/bin/env python3
"""
Teleoperate Franka Panda in MuJoCo using iPhone (TeleDex) + inverse kinematics (mink).

Architecture:
  iPhone AR → TeleDex → mocap_target body → mink IK → joint ctrl → mj_step → arm moves

Usage:
  1. Run this script: python teleop_mink.py
  2. Open the MuJoCo AR iOS app
  3. Enter the IP and port printed in the terminal
  4. Move your phone to control the end-effector
  5. Use the toggle button on the app to open/close the gripper

Dependencies:
  pip install mink teledex loop-rate-limiters quadprog
"""

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import mink
import teledex


def main():
    model = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/mjx_single_cube.xml")
    data = mujoco.MjData(model)

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
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:

        mujoco.mj_resetDataKeyframe(model, data, 0) # Initialize to pickup keyframe.
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)

        mujoco.mj_forward(model, data) # Run forward kinematics so site positions are populated.

        # Place the mocap target at the current end-effector position.
        mink.move_mocap_to_frame(model, data, "mocap_target", "gripper", "site")

        initial_pos = data.mocap_pos[0].copy()
        initial_mat = data.site("gripper").xmat.copy().reshape(3, 3)

        handler.link_body(
            name="mocap_target",
            scale=2.0, # how much phone movement maps to sim movement (tune to taste)
            position_origin=initial_pos, # offset so AR origin = current EE position
            rotation_origin=initial_mat, # orient AR frame to match current EE orientation
            toggle_fn=toggle_gripper, # called when the iOS app toggle changes
        )

        # Start receiving phone data.
        session.start()
        rate = RateLimiter(frequency=200.0, warn=False)
        prev_time = data.time
        while viewer.is_running():
            # Detect reset (time goes backwards or becomes 0)
            if data.time < prev_time or data.time == 0:
                print("Reset detected! Updating IK configuration to pickup keyframe")
                mujoco.mj_resetDataKeyframe(model, data, 0)  # Reset to pickup keyframe
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
                configuration = configuration,
                tasks = tasks,
                dt = rate.dt,
                solver = "quadprog",
                damping = 1e-3,
                limits = limits
            )

            # 3. Integrate velocities into joint positions.
            configuration.integrate_inplace(vel, rate.dt)

            # 4. Set actuator controls.
            data.ctrl[:7] = configuration.q[:7]          # Arm joints from IK
            data.ctrl[7] = 0.0 if gripper_closed else 0.04  # Gripper open/close

            # 5. Step physics (handles contacts, gravity, box dynamics, etc.)
            mujoco.mj_step(model, data)

            # 6. Sync IK configuration back from physics state so the IK
            configuration.update(data.qpos)

            viewer.sync()
            rate.sleep()

    session.stop()


if __name__ == "__main__":
    main()
