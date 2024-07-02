import os
import ray
import zarr
import time
import numpy as np
import librosa
import matplotlib.pyplot as plt

from imagecodecs import imread
from moviepy.editor import VideoFileClip, ImageSequenceClip

import cv2
import os
import torch
import torchaudio
import noisereduce as nr
import soundfile as sf
from moviepy.editor import VideoFileClip
from moviepy.editor import AudioFileClip

AUDIO_FPS = 48000

def convert_to_video_with_sound(video_path, audio_path, output_path, channel_name):
    video_clip = VideoFileClip(video_path)
    print("video length:", video_clip.duration)
    audio_clip = AudioFileClip(audio_path, fps=AUDIO_FPS)
    
    if channel_name == 'contact':
        audio = np.stack([audio_clip.to_soundarray()[:,0], audio_clip.to_soundarray()[:,0]], axis=1)
        audio_path = audio_path[:audio_path.find('audio.wav')] + 'contact.wav'
        sf.write(audio_path, audio, samplerate=AUDIO_FPS)
    elif channel_name == 'env':
        audio = np.stack([audio_clip.to_soundarray()[:,1], audio_clip.to_soundarray()[:,1]], axis=1)
        audio_path = audio_path[:audio_path.find('audio.wav')] + 'env.wav'
        sf.write(audio_path, audio, samplerate=AUDIO_FPS)
    
    audio_clip = AudioFileClip(audio_path, fps=AUDIO_FPS)
    print("audio length:", audio_clip.duration)
    video_clip = video_clip.set_audio(audio_clip)
    
    video_clip.write_videofile(output_path, codec="libx264")

    video_clip.close()
    audio_clip.close()


def plot_audio(raw_data_path, audio_data, sample_rate, current_time, window_size, modality):
    plt.clf()

    plt.figure(figsize=(8, 8))
    start_sample = max(0, int((current_time - window_size / 2) * sample_rate))
    end_sample = min(int((current_time + window_size / 2) * sample_rate), len(audio_data))
    windowed_audio = audio_data[start_sample:end_sample]
    # windowed_audio = nr.reduce_noise(y=windowed_audio, sr=48000, thresh_n_mult_nonstationary=1, stationary=False)
    time_axis = np.linspace(
        start_sample / sample_rate, end_sample / sample_rate, len(windowed_audio)
    )

    # Plot the waveform in the top subplot
    sr = 16000
    plt.subplot(2, 1, 1)
    audio_transform = torch.nn.Sequential(
        torchaudio.transforms.Resample(48000, sr),
    )
    downsample_waveform = audio_transform(torch.from_numpy(windowed_audio).float())
    plt.plot(np.arange(len(downsample_waveform)), downsample_waveform, color='#ed5c9b')
    plt.xlabel('')
    plt.ylabel('Amplitude')
    plt.ylim(np.min(audio_data), np.max(audio_data))
    plt.axis('off')

    audio_transform = torch.nn.Sequential(
        torchaudio.transforms.Resample(48000, sr),
        torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=int(sr * 0.025), hop_length=int(sr * 0.01), n_mels=64)
    )
    mel_spectrogram = audio_transform(torch.from_numpy(windowed_audio).float())
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=0.5)

    # Plot the spectrogram in the bottom subplot
    plt.subplot(2, 1, 2)
    librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sr, hop_length=int(sr * 0.01), cmap='magma')
    plt.clim(-40, 20)
    plt.colorbar(format='%+2.0f dB')

    # Adjust spacing between subplots
    plt.tight_layout()
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    # plt.xticks([])
    # plt.yticks([])
    plt.tight_layout()
    # plt.axis('off')
    plt.savefig(f"{raw_data_path}/{modality}/{current_time}.png", transparent=True)


def convert_images_to_video(raw_data_path, duration, modality):
    image_folder = f'{raw_data_path}/{modality}'
    images = []
    for current_time in np.arange(0, duration, 0.5):
        images.append(f"{image_folder}/{round(current_time, 1)}.png")

    # Create a video clip from the images
    clip = ImageSequenceClip(images, fps=2)
    video_filename = f'{raw_data_path}/{modality}.mp4'
    clip.write_videofile(video_filename, codec='libx264')


# @ray.remote
def process_data(data_idx, raw_data_path):
    sr = 48000
    video_clip = VideoFileClip(f'{raw_data_path}/raw_video.mp4')

    audio_data = video_clip.audio.to_soundarray(fps=sr)
    duration = video_clip.audio.duration
    window_size = 2

    for channel_idx in [0, 1]:
        # left channel contact data, right channel environment data
        if channel_idx == 0:
            modality = 'contact_audio'
        if channel_idx == 1:
            modality = 'env_audio'
        os.system(f"mkdir -p {raw_data_path}/{data_idx}/{modality}")
        for current_time in np.arange(0, duration, 0.5):
            plot_audio(f'{raw_data_path}/{data_idx}', audio_data[:, channel_idx], sr, round(current_time, 1), window_size=window_size, modality=modality)
        convert_images_to_video(f'{raw_data_path}/{data_idx}', duration=duration, modality=modality)


if __name__ == '__main__':
    s_time = time.time()
    ray.init(num_cpus=8)

    raw_data_paths = []
    data_folders = ['data/20240504_flipping/demos']
    for data_folder in data_folders:
        raw_data_paths += [os.path.join(data_folder, f) for f in os.listdir(data_folder)]
    
    ray_ids = []
    for data_idx, raw_data_path in enumerate(raw_data_paths):
        ray_ids.append(process_data.remote(data_idx, raw_data_path))
    ray.get(ray_ids)

    e_time = time.time()
    print("[INFO] process time:", round(e_time - s_time), " seconds.")
