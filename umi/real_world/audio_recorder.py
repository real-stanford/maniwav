import sys
import time
import av
import numpy as np
import sounddevice as sd
import soundfile as sf
import multiprocessing as mp

from diffusion_policy.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from umi.common.timestamp_accumulator import get_accumulate_timestamp_idxs


class AudioEncoderProcess(mp.Process):
    def __init__(self, 
            shm_manager, 
            data_example: np.ndarray,
            file_path,
            codec, sr, num_channel, input_audio_fmt, 
            **kwargs):
        super().__init__()

        self.file_path = file_path
        self.codec = codec
        self.sr = sr
        self.num_channel = num_channel
        self.input_audio_fmt = input_audio_fmt
        self.kwargs = kwargs
        self.shape = None
        self.dtype = None

        self.audio_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples={'audio_block': data_example},
            buffer_size=128
        )
        self.stop_event = mp.Event()

    def stop(self, wait=True):
        # wake up thread waiting on queue
        self.stop_event.set()
        if wait:
            self.join()
    
    def put_audio_block(self, audio_block: np.ndarray):
        assert audio_block is not None
        self.audio_queue.put({'audio_block': audio_block})

    def run(self):
        with sf.SoundFile(self.file_path, mode='w', samplerate=self.sr, channels=self.num_channel) as file:
            data = None
            while not self.stop_event.is_set():
                try:
                    data = self.audio_queue.get(out=data)
                    file.write(data['audio_block'])
                except Empty:
                    time.sleep(0.5/60)


class AudioRecorder():
    def __init__(self, shm_manager, sr, num_channel, codec, input_audio_fmt, **kwargs):
        self.shm_manager = shm_manager
        self.sr = sr
        self.num_channel = num_channel
        self.codec = codec
        self.input_audio_fmt = input_audio_fmt
        self.kwargs = kwargs

        self._reset_state()

    def _reset_state(self):
        self.file_path = None
        self.enc_thread = None
        self.start_time = None
        self.next_global_idx = 0
        
    def __del__(self):
        self.stop()

    def is_ready(self):
        return self.start_time is not None

    def start(self, file_path, start_time=None):
        if self.is_ready():
            # if still recording, stop first and start anew.
            self.stop()
        
        self.file_path = file_path
        self.start_time = start_time
    
    def write_frame(self, audio_block: np.ndarray, frame_time):
        if not self.is_ready():
            raise RuntimeError('Must run start() before writing!')
        
        # create encode threads if not already
        if self.enc_thread is None:
            self.enc_thread = AudioEncoderProcess(
                shm_manager=self.shm_manager,
                data_example=audio_block,
                file_path=self.file_path,
                codec=self.codec,
                sr=self.sr,
                num_channel=self.num_channel,
                input_audio_fmt=self.input_audio_fmt,
                **self.kwargs
            )
            self.enc_thread.start()

        n_repeats = 1
        # if self.start_time is not None:
        #     local_idxs, global_idxs, self.next_global_idx \
        #         = get_accumulate_timestamp_idxs(
        #         # only one timestamp
        #         timestamps=[frame_time],
        #         start_time=self.start_time,
        #         dt=1/self.put_fps,
        #         next_global_idx=self.next_global_idx
        #     )
        #     # number of apperance means repeats
        #     n_repeats = len(local_idxs)
        
        # print(n_repeats)
        if self.start_time is not None and frame_time >= self.start_time:
            self.enc_thread.put_audio_block(audio_block)
        
    def stop(self):        
        if not self.is_ready():
            return
        self.enc_thread.stop(wait=True)
        # reset runtime parameters
        self._reset_state()
