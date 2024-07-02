from typing import List, Optional, Union, Dict, Callable
import numbers
import copy
import time
import pathlib
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from umi.real_world.mic import Microphone
from umi.real_world.audio_recorder import AudioRecorder

class MultiMicrophone:
    def __init__(self,
            # v4l2 device file path
            # e.g. /dev/video0
            # or /dev/v4l/by-id/usb-Elgato_Elgato_HD60_X_A00XB320216MTR-video-index0
            dev_video_paths: List[str],
            shm_manager: Optional[SharedMemoryManager]=None,
            get_max_k=30,
            receive_latency=0.0,
            transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
            audio_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
            device_id='0',
            num_channel=2,
            put_fps=None,
            block_size=800,
            audio_sr=48000,
            put_downsample=True,
            audio_recorder: Optional[Union[AudioRecorder, List[AudioRecorder]]]=None,
            verbose=False
        ):
        super().__init__()

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        mics = dict()
        for i, path in enumerate(dev_video_paths):
            mics[path] = Microphone(            
                shm_manager=shm_manager,
                get_max_k=get_max_k,
                receive_latency=receive_latency,
                device_id=int(device_id[i]),
                num_channel=num_channel,
                block_size=block_size,
                audio_sr=audio_sr,
                put_downsample=put_downsample,
                audio_recorder=audio_recorder[i])

        self.mics = mics
        self.shm_manager = shm_manager

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    @property
    def n_mics(self):
        return len(self.mics)
    
    @property
    def is_ready(self):
        is_ready = True
        for mic in self.mics.values():
            if not mic.is_ready:
                is_ready = False
        return is_ready
    
    def start(self, wait=True, put_start_time=None):
        if put_start_time is None:
            put_start_time = time.time()
        for mic in self.mics.values():
            mic.start(wait=False, put_start_time=put_start_time)
        
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        for mic in self.mics.values():
            mic.stop(wait=False)
        
        if wait:
            self.stop_wait()

    def start_wait(self):
        for mic in self.mics.values():
            mic.start_wait()

    def stop_wait(self):
        for mic in self.mics.values():
            mic.join()

    def get(self, k=None, out=None) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Return order T,H,W,C
        {
            0: {
                'rgb': (T,H,W,C),
                'timestamp': (T,)
            },
            1: ...
        }
        """
        if out is None:
            out = dict()
        for _, mic in enumerate(self.mics.values()):
            i = mic.device_id
            this_out = None
            if i in out:
                this_out = out[i]
            this_out = mic.get(k=k, out=this_out)
            out[i] = this_out
        return out

    def start_recording(self, audio_path: Union[str, List[str]], start_time: float):
        if isinstance(audio_path, str):
            # directory
            video_dir = pathlib.Path(audio_path)
            assert video_dir.parent.is_dir()
            video_dir.mkdir(parents=True, exist_ok=True)
            audio_path = list()
            for i in range(self.n_mics):
                audio_path.append(
                    str(video_dir.joinpath(f'{i}.wav').absolute()))
        assert len(audio_path) == self.n_mics

        for i, mic in enumerate(self.mics.values()):
            mic.start_recording(audio_path[i], start_time)
    
    def stop_recording(self):
        for i, mic in enumerate(self.mics.values()):
            mic.stop_recording()
    
    def restart_put(self, start_time):
        for mic in self.mics.values():
            mic.restart_put(start_time)


def repeat_to_list(x, n: int, cls):
    if x is None:
        x = [None] * n
    if isinstance(x, cls):
        x = [copy.deepcopy(x) for _ in range(n)]
    assert len(x) == n
    return x
