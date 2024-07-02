# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import cv2
import numpy as np
from matplotlib import pyplot as plt
import zarr
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
# from skimage.io import imread
register_codecs()
from umi.common.cv_util import get_image_transform

# %%
zarr_path = 'data/replay_buffer.zarr.zip'
with zarr.ZipStore(zarr_path, mode='r') as zip_store:
    replay_buffer = ReplayBuffer.copy_from_store(
        src_store=zip_store, store=zarr.MemoryStore())
#%%
replay_buffer['robot0_eef_rot_axis_angle'].shape
gripper_widths = replay_buffer['robot0_gripper_width']
positions = replay_buffer['robot0_eef_pos']

old_positions = replay_buffer['robot0_eef_pos']

n1, n2 = 0, 600
plt.title("z offset")
plt.plot(np.arange(len(positions[n1:n2, 2])), positions[n1:n2, 2], label='new')
plt.plot(np.arange(len(old_positions[n1:n2, 2])), old_positions[n1:n2, 2], label='old')
plt.legend()

x, y, z = positions[n1:n2, 0], positions[n1:n2, 1], positions[n1:n2, 2]

# Plot the trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, marker='o', linestyle='-', color='b')

# Annotate start and end points
ax.text(x[0], y[0], z[0], 'Start', color='red', fontsize=10)
ax.text(x[-1], y[-1], z[-1], 'End', color='red', fontsize=10)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('trajectory')

plt.show()

#%%
plt.plot(np.arange(len(z)), z, linestyle='-', color='b')
plt.show()
print(np.min(z), np.max(z))

# %%
plt.imshow(replay_buffer['camera0_rgb'][n1])

#%%
plt.imshow(replay_buffer['camera0_rgb'][n2])

#%%
assert len(replay_buffer['camera0_rgb']) == len(replay_buffer['mic_0']) == len(replay_buffer['robot0_eef_pos'])

# %%
contact_audio = replay_buffer['mic_0'][n1:n2]
env_audio = replay_buffer['mic_1'][n1:n2]

channel_l = np.reshape(contact_audio, -1)
channel_r = np.reshape(env_audio, -1)
audio_l = np.stack([channel_l, channel_l], axis=1)
audio_r = np.stack([channel_r, channel_r], axis=1)

import imageio
rgb_images = np.array(replay_buffer['camera0_rgb'][n1:n2])
video_path = "test_dataloader.mp4"
with imageio.get_writer(video_path, mode='I', fps=60) as writer:
    for rgb_image in rgb_images:
        writer.append_data(np.array(rgb_image))

import IPython.display as ipd
ipd.Audio(channel_l, rate=48000)

# import soundfile as sf
# sf.write('test_dataloader_l.wav', audio_l, samplerate=48000)
# sf.write('test_dataloader_r.wav', audio_r, samplerate=48000)
