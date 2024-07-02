import os
import sys
import time
import argparse
import queue
import numpy as np
import tkinter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import sounddevice as sd
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.mic import Microphone
from umi.real_world.audio_recorder import AudioRecorder
from umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'channels', type=int, default=[1, 2], nargs='*', metavar='CHANNEL',
    help='input channels to plot (default: the first channel)')
parser.add_argument(
    '-d', '--device', type=int_or_str, default=0,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-w', '--window', type=float, default=200, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
parser.add_argument(
    '-i', '--interval', type=float, default=30,
    help='minimum time between plot updates (default: %(default)s ms)')
parser.add_argument(
    '-b', '--blocksize', type=int, default=10, help='block size (in samples)')
parser.add_argument(
    '-r', '--samplerate', type=float, help='sampling rate of audio device')
parser.add_argument(
    '-n', '--downsample', type=int, default=10, metavar='N',
    help='display every Nth sample (default: %(default)s)')
args = parser.parse_args(remaining)
if any(c < 1 for c in args.channels):
    parser.error('argument CHANNEL: must be >= 1')
mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1
q = queue.Queue()

if args.samplerate is None:
    device_info = sd.query_devices(args.device, 'input')
    args.samplerate = device_info['default_samplerate']
        
length = int(args.window * args.samplerate / (1000 * args.downsample))
plotdata = np.zeros((length, len(args.channels)))

fig, ax = plt.subplots()
lines = ax.plot(plotdata)
if len(args.channels) > 1:
    ax.legend([f'channel {c}' for c in args.channels],
            loc='lower left', ncol=len(args.channels))
    ax.axis((0, len(plotdata), -1, 1))
    ax.set_yticks([0])
    ax.yaxis.grid(True)
    ax.tick_params(bottom=False, top=False, labelbottom=False,
                right=False, left=False, labelleft=False)
    fig.tight_layout(pad=0)


def update_plot(frame):
    """This is called by matplotlib for each plot update.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global plotdata
    while True:
        try:
            data_dict = q.get_nowait()
            data = data_dict['audio_block']
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])
    return lines

def test():
    # Find and reset all Elgato capture cards.
    # Required to workaround a firmware bug.
    reset_all_elgato_devices()
    v4l_paths = get_sorted_v4l_paths()
    time.sleep(4) # sleep for 4 seconds for audio stream to initialize
    
    with SharedMemoryManager() as shm_manager:
        audio_recorder = AudioRecorder(
            shm_manager=shm_manager,
            put_fps=120,
            sr=48000, 
            num_channel=2,
            codec="aac",
            input_audio_fmt="fltp"
        )
        with Microphone(
            shm_manager=shm_manager,
            device_id=1,
            num_channel=2,
            block_size=800,
            audio_sr=48000,
            audio_recorder=audio_recorder
        ) as mic:
            audio_path = 'data_local/test.wav'
            print("Start audio recording...")
            # delay recording by 2 seconds
            rec_start_time = time.time() + 2
            mic.start_recording(audio_path, start_time=rec_start_time)
            # ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=True, cache_frame_data=False)

            data = None
            while True:
                data = mic.get(out=data)
                # q.put(data)
                # plt.show()
                
                t = time.time()
                # print('capture_latency', data['receive_timestamp']-data['capture_timestamp'], 'receive_latency', t - data['receive_timestamp'])
                # print('receive', t - data['receive_timestamp'])

                dt = time.time() - data['timestamp']
                
                time.sleep(1/60)
                # record audio for 10 seconds
                if time.time() > (rec_start_time + 10.0):
                    break


if __name__ == "__main__":
    test()