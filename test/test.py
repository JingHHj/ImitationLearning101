import mujoco
import mujoco.viewer

"""
    Test run the MuJoCo simulation
"""

def main():
    model = mujoco.MjModel.from_xml_path("../assets/franka_emika_panda/mjx_single_cube.xml")
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
