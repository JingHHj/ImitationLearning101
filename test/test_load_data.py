import h5py
import cv2
from typing import List, Optional
import numpy as np


file_path = "data/episode_1771284357.hdf5"  
f = h5py.File(file_path, "r")

# import pdb; pdb.set_trace()

# Load datasets
ee_images = f["ee_images"]
joint_positions = f["joint_positions"]
side_images = f["side_images"]

print(f"ee_images shape: {ee_images.shape}")
print(f"joint_positions shape: {joint_positions.shape}")
print(f"side_images shape: {side_images.shape}")


for k,v in f.attrs.items():
    print(f"{k}: {v}")


def play_video(images:List[Optional[np.ndarray]], delay=30):
    """
        Play a sequence of images as a video.

        Args:
            images (array-like): Sequence of images to display.
            delay (int): Delay between frames in milliseconds.
    """

    for i, frame in enumerate(images):
        cv2.imshow("Camera Views", cv2.cvtColor(
            np.hstack(frame) if isinstance(frame, list) else frame,
            cv2.COLOR_RGB2BGR)
        )
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

img_list = [
    [side, ee] for side, ee in zip(side_images, ee_images)
]

play_video(img_list)  # Show the first frame for 1 second


f.close()
cv2.destroyAllWindows()