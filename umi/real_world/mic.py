import sys
import enum
import time
import queue
import numpy as np
import sounddevice as sd
from typing import Optional, Callable, Dict

import multiprocessing as mp
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from umi.real_world.audio_recorder import AudioRecorder
from umi.common.timestamp_accumulator import get_accumulate_timestamp_idxs

class Command(enum.Enum):
    RESTART_PUT = 0
    START_RECORDING = 1
    STOP_RECORDING = 2

class Microphone(mp.Process):
    MAX_PATH_LENGTH = 4096 # linux path has a limit of 4096 bytes
    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        get_max_k=30,
        receive_latency=0.0,
        transform=None,
        audio_transform=None,
        device_id=0,
        num_channel=2,
        put_fps=None,
        block_size=800,
        audio_sr=48000,
        put_downsample=True,
        audio_recorder: Optional[AudioRecorder]=None,
        verbose=False
    ):
        super().__init__()
        
        if put_fps is None:
            put_fps = (audio_sr // block_size) * 5
        examples = {
            'audio_block': np.empty(shape=(block_size, num_channel), dtype=np.float32)
        }
        # examples['mic_capture_timestamp'] = 0.0
        # examples['mic_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0
        
        audio_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if audio_transform is None 
                else audio_transform(dict(examples)),
            get_max_k=1,
            get_time_budget=0.2,
            put_desired_frequency=put_fps
        )

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None
                else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps
        )
        
        examples = {
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': 0.0,
            'audio_path': np.array('b'*self.MAX_PATH_LENGTH),
            'recording_start_time': 0.0,
        }
        
        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            buffer_size=128
        )
        
        if audio_recorder is None:
            audio_recorder = AudioRecorder(
                shm_manager=shm_manager,
                sr=audio_sr,
                num_channel=num_channel,
                codec='aac',
                input_audio_fmt='fltp'
            )
    
        self.transform = transform
        self.receive_latency = receive_latency
        self.device_id = device_id
        self.num_channel = num_channel
        self.put_fps = put_fps
        self.audio_sr = audio_sr
        self.block_size = block_size
        self.put_downsample = put_downsample
        self.audio_recorder = audio_recorder
        self.audio_transform = audio_transform
        self.verbose = verbose
        self.put_start_time = None

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.audio_ring_buffer = audio_ring_buffer
        self.command_queue = command_queue
        
        self.q = queue.Queue()

        # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        super().start()
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()
    
    def end_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)
    
    def get_audio(self, out=None):
        return self.audio_ring_buffer.get(out=out)
    
    def start_recording(self, audio_path: str, start_time: float=-1):
        path_len_audio = len(audio_path.encode('utf-8'))
        if path_len_audio > self.MAX_PATH_LENGTH:
            raise RuntimeError('audio_path too long.')
        self.command_queue.put({
            'cmd': Command.START_RECORDING.value,
            'audio_path': audio_path,
            'recording_start_time': start_time
        })
        
    def stop_recording(self):
        self.command_queue.put({
            'cmd': Command.STOP_RECORDING.value
        })
    
    def restart_put(self, start_time):
        self.command_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': start_time
        })
        
    
    def callback(self, indata, frames, tm, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        # indata: block_size x channel
        self.q.put({'timestamp':time.time(), 'data': indata.copy()})

    # ========= interval API ===========
    def run(self):
        # limit threads
        threadpool_limits(1)

        if self.audio_recorder:
            audio_stream = sd.InputStream(
                samplerate=self.audio_sr,
                blocksize=self.block_size,
                device=self.device_id, 
                channels=self.num_channel, 
                callback=self.callback)
            device_info = sd.query_devices(self.device_id, 'input')
            print("[INFO] microphone device:", device_info)
            audio_stream.start()
        
        try:
            # put frequency regulation
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            # reuse frame buffer
            frame = None
            iter_idx = 0
            t_start = time.time()
            while not self.stop_event.is_set():
                data = dict()
                try:
                    rec = self.q.get_nowait()              
                    data['audio_block'] = rec['data']
                    data['timestamp'] = rec['timestamp']
                    t_cal = data['timestamp'] - self.receive_latency # calibrated audio latency

                    # apply transform
                    put_data = data
                    if self.transform is not None:
                        put_data = self.transform(dict(data))

                    if self.put_downsample:
                        # put frequency regulation
                        local_idxs, global_idxs, put_idx \
                            = get_accumulate_timestamp_idxs(
                                timestamps=[t_cal],
                                start_time=put_start_time,
                                dt=1/self.put_fps,
                                # this is non in first iteration
                                # and then replaced with a concrete number
                                next_global_idx=put_idx,
                                # continue to pump frames even if not started.
                                # start_time is simply used to align timestamps.
                                allow_negative=True
                            )

                        for step_idx in global_idxs:
                            put_data['step_idx'] = step_idx
                            put_data['timestamp'] = t_cal
                            self.ring_buffer.put(put_data, wait=False)
                    else:
                        step_idx = int((data['timestamp'] - put_start_time) * self.put_fps)
                        put_data['step_idx'] = step_idx
                        put_data['timestamp'] = t_cal
                        self.ring_buffer.put(put_data, wait=False)

                    # signal ready
                    if iter_idx == 0:
                        self.ready_event.set()
                        
                    # put to vis
                    audio_data = data
                    if self.audio_transform == self.transform:
                        audio_data = put_data
                    elif self.audio_transform is not None:
                        audio_data = self.audio_transform(dict(data))
                    self.audio_ring_buffer.put(audio_data, wait=False)
                    
                    # # record frame
                    # rec_data = data
                    # if self.recording_transform == self.transform:
                    #     rec_data = put_data
                    # elif self.recording_transform is not None:
                    #     rec_data = self.recording_transform(dict(data))
                    
                    if self.audio_recorder.is_ready():
                        self.audio_recorder.write_frame(
                            audio_block=audio_data['audio_block'],
                            frame_time=audio_data['timestamp'])

                    # # perf
                    # t_end = time.time()
                    # duration = t_end - t_start
                    # frequency = np.round(1 / duration, 1)
                    # t_start = t_end
                    # if self.verbose:
                    #     print(f'[Microphone {self.dev_video_path}] FPS {frequency}')

                    # fetch command from queue
                    try:
                        commands = self.command_queue.get_all()
                        n_cmd = len(commands['cmd'])
                    except Empty:
                        n_cmd = 0

                    # execute commands
                    for i in range(n_cmd):
                        command = dict()
                        for key, value in commands.items():
                            command[key] = value[i]
                        cmd = command['cmd']
                        if cmd == Command.RESTART_PUT.value:
                            put_idx = None
                            put_start_time = command['put_start_time']
                        elif cmd == Command.START_RECORDING.value:
                            audio_path = str(command['audio_path'])
                            start_time = command['recording_start_time']
                            if start_time < 0:
                                start_time = None
                            self.audio_recorder.start(audio_path, start_time=start_time)
                        elif cmd == Command.STOP_RECORDING.value:
                            self.audio_recorder.stop()
                            # stop need to flush all in-flight frames to disk, which might take longer than dt.
                            # soft-reset put to drop frames to prevent ring buffer overflow.
                            put_idx = None

                    iter_idx += 1
                    
                except Empty:
                    pass
        finally:
            self.audio_recorder.stop()
            audio_stream.close()
