import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def get_camera_pos(model, data, viewer_cam):
    """Get camera pos/forward/up directly from mjv_cameraInRoom."""
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=1000)
    mujoco.mjv_updateScene(model, data, opt, None, viewer_cam,
                           mujoco.mjtCatBit.mjCAT_ALL, scene)

    pos = np.zeros(3)
    forward = np.zeros(3)
    up = np.zeros(3)
    mujoco.mjv_cameraInRoom(pos, forward, up, scene)

    # rotation matrix from camera axes
    right = np.cross(forward, up)
    R = np.column_stack([right, up, -forward])
    quat = Rotation.from_matrix(R).as_quat(scalar_first=True)

    print(f"\npos:     {np.round(pos, 2)}")
    print(f"quat:    {np.round(quat, 2)}")

    

def randomization(r_min=0.25, r_max=0.60, min_dist=0.3):
    """
        Randomize the location of the cube and the bin.

        Randomize rules: 
        Area the half circle in front of the robot within r_max out of r_min (to avoid being too close to the robot):
        1. Randomize the x_cube within (r_min, r_max)
        2. Randomize the y_cube to make sure it's within the half circle. 
        3. Randomize the x_bin and y_bin to be within the half circle while keep a distance at least min_dist away from the cube
    """

    # Randomize cube position in the half circle
    x_cube = np.random.uniform(r_min, r_max)
    y_max_cube = np.sqrt(r_max**2 - x_cube**2)
    y_cube = np.random.uniform(-y_max_cube, y_max_cube)

    # Randomize bin position, keeping min_dist from cube
    while True:
        x_bin = np.random.uniform(r_min, r_max)
        y_max_bin = np.sqrt(r_max**2 - x_bin**2)
        y_bin = np.random.uniform(-y_max_bin, y_max_bin)
        if np.sqrt((x_bin - x_cube)**2 + (y_bin - y_cube)**2) >= min_dist:
            break

    return x_cube, y_cube, x_bin, y_bin

def key_callback(keycode, model, data, viewer):
    """
        Hit r, the environment will reset:
        1. Reset arm configuration to keyframe 0
        2. Position of the cube and the bin will be randomized. 

        Hit p, the current camera pose will be printed in the terminal. 
    """
    
    if keycode == ord("P") or keycode == ord("p"): # Print
        get_camera_pos(model, data, viewer)


    
    