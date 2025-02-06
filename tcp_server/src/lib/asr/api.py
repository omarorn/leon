import os
import pyaudio
import audioop
import time
import torch
import numpy as np
from openwakeword.model import Model as WakeWordModel
from faster_whisper import WhisperModel

from ..constants import ASR_MODEL_PATH, IS_WAKE_WORD_ENABLED, WAKE_WORD_MODEL_FOLDER_PATH
from ..utils import ThrottledCallback, is_macos, get_settings


class ASR:
    def __init__(self,
                 # @see https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py
                 # auto, cpu, cuda
                 device='auto',
                 interrupt_leon_speech_callback=None,
                 transcribed_callback=None,
                 end_of_owner_speech_callback=None,
                 active_listening_disabled_callback=None):
        tic = time.perf_counter()
        self.log('Loading model...')

        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                self.log('Using CUDA (Compute Unified Device Architecture)')

        if 'cuda' in device:
            assert torch.cuda.is_available()

        self.log(f'Device: {device}')

        compute_type = 'float16'
        if is_macos():
            compute_type = 'int8_float32'

        if device == 'cpu':
            compute_type = 'int8_float32'

        self.wake_word = None
        self.wake_word_listening_thread = None
        self.compute_type = compute_type
        self.is_recording = False

        """
        Thottle the interrupt Leon's speech callback to avoid sending too many messages to the client
        """
        self.interrupt_leon_speech_callback = ThrottledCallback(
            interrupt_leon_speech_callback, 0.8
        )
        self.transcribed_callback = transcribed_callback
        self.end_of_owner_speech_callback = end_of_owner_speech_callback
        self.active_listening_disabled_callback = active_listening_disabled_callback

        self.device = device
        self.is_voice_activity_detected = False
        self.silence_start_time = 0
        self.is_active_listening_enabled = False
        self.complete_text = ''

        self.audio_format = pyaudio.paInt16
        self.buffer = bytearray()
        self.silence_frames_count = 0
        self.channels = 1
        self.rate = 16000
        self.frames_per_buffer = 1024
        self.rms_threshold = get_settings('asr')['rms_mic_threshold']
        # Duration of silence after which the audio data is considered as a new utterance (in seconds)
        self.silence_duration = get_settings('asr')['silence_duration']
        """
        Duration of silence after which the active listening is stopped (in seconds).
        Once stopped, the active listening can be resumed by starting a new recording event
        """
        self.base_active_listening_duration = get_settings('asr')['active_listening_duration']
        self.active_listening_duration = self.base_active_listening_duration

        self.audio = pyaudio.PyAudio()
        self.mic_stream = None
        self.model = None
        self.is_wake_word_enabled = False

        model_params = {
            'model_size_or_path': ASR_MODEL_PATH,
            'device': self.device,
            'compute_type': self.compute_type,
            'local_files_only': True
        }
        if self.device == 'cpu':
            model_params['cpu_threads'] = 4

        should_init_wake_word = True
        wake_word_model_name = get_settings('wake_word')['model_file_name']
        wake_word_model_path = os.path.join(WAKE_WORD_MODEL_FOLDER_PATH, wake_word_model_name)

        if not IS_WAKE_WORD_ENABLED:
            self.log('Wake word is disabled')
            should_init_wake_word = False
        if not os.path.exists(wake_word_model_path):
            self.log(f'Wake word model not found at {wake_word_model_path}')
            should_init_wake_word = False

        self.open_mic_stream()

        self.model = WhisperModel(**model_params)

        self.log('Model loaded')
        toc = time.perf_counter()

        self.log(f"Time taken to load model: {toc - tic:0.4f} seconds")

        if should_init_wake_word:
            self.wake_word = WakeWord(
                asr=self,
                model_path=wake_word_model_path,
                device=get_settings('wake_word')['device'],
                detection_threshold=get_settings('wake_word')['detection_threshold']
            )
            self.wake_word.start_listening()

    def open_mic_stream(self):
        try:
            self.mic_stream = self.audio.open(format=self.audio_format,
                                              channels=self.channels,
                                              rate=self.rate,
                                              frames_per_buffer=self.frames_per_buffer,
                                              input=True,
                                              input_device_index=self.audio.get_default_input_device_info()["index"])  # Use the default input device
        except Exception as e:
            self.log('Error to open mic stream:', e)

    def start_recording(self):
        self.is_recording = True
        # Convert the silence duration to the number of audio frames required to detect the silence
        silence_threshold = int(self.silence_duration * self.rate / self.frames_per_buffer)

        try:
            self.log("Recording...")

            while self.is_recording:
                data = self.mic_stream.read(self.frames_per_buffer, exception_on_overflow=False)
                rms = audioop.rms(data, 2)  # width=2 for format=paInt16

                if rms >= self.rms_threshold:
                    if not self.is_voice_activity_detected:
                        self.is_active_listening_enabled = True
                        self.is_voice_activity_detected = True

                    self.interrupt_leon_speech_callback()

                    self.buffer.extend(data)
                    self.silence_frames_count = 0
                else:
                    if self.is_voice_activity_detected:
                        self.silence_start_time = time.time()
                        self.is_voice_activity_detected = False

                    if self.silence_frames_count < silence_threshold:
                        self.silence_frames_count += 1
                    else:
                        if len(self.buffer) > 0:
                            self.log('Silence detected')

                            audio_data = np.frombuffer(self.buffer, dtype=np.int16)
                            if self.compute_type == 'int8_float32':
                                audio_data = audio_data.astype(np.float32) / 32768.0
                            transcribe_params = {
                                'beam_size': 5,
                                'language': 'en',
                                'task': 'transcribe',
                                'condition_on_previous_text': False,
                                'hotwords': 'talking to Leon'
                            }
                            if self.device == 'cpu':
                                transcribe_params['temperature'] = 0
                            segments, info = self.model.transcribe(audio_data, **transcribe_params)

                            for segment in segments:
                                self.log("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                                self.complete_text += segment.text

                            self.transcribed_callback(self.complete_text)
                            time.sleep(0.1)
                            # Notify the end of the owner's speech
                            self.end_of_owner_speech_callback(self.complete_text)

                            self.complete_text = ''
                            self.buffer = bytearray()

                        should_stop_active_listening = self.is_active_listening_enabled and time.time() - self.silence_start_time > self.active_listening_duration
                        if should_stop_active_listening:
                            self.log('Active listening disabled')
                            self.is_active_listening_enabled = False
                            self.stop_recording()
                            self.active_listening_disabled_callback()
        except Exception as e:
            self.log('Error:', e)

    def stop_recording(self):
        if self.wake_word:
            self.wake_word.reset_model_state()

        self.is_recording = False
        # self.mic_stream.stop_stream()
        # self.mic_stream.close()
        # self.log('Stream closed, recording stopped')
        self.log('Recording stopped')

        if self.is_wake_word_enabled:
            self.wake_word.start_listening()

    @staticmethod
    def log(*args, **kwargs):
        print('[ASR]', *args, **kwargs)

class WakeWord:
    def __init__(self, asr, model_path, device='cpu', detection_threshold=0.5):
        tic = time.perf_counter()
        self.log('Loading model...')

        self.log(f'Device: {device}')

        self.asr = asr
        self.model_path = model_path
        self.device = device
        self.detection_threshold = detection_threshold
        self.chunk_size = 1280
        self.audio = None
        self.is_listening = False

        # @see https://github.com/dscripka/openWakeWord/blob/main/openwakeword/model.py#L38
        # @see https://github.com/dscripka/openWakeWord/blob/main/openwakeword/utils.py#L38
        self.model = WakeWordModel(
            device=self.device,
            wakeword_models=[self.model_path],
            melspec_model_path=os.path.join(WAKE_WORD_MODEL_FOLDER_PATH, 'melspectrogram.onnx'),
            embedding_model_path=os.path.join(WAKE_WORD_MODEL_FOLDER_PATH, 'embedding.onnx'),
            ncpu=1,
            inference_framework='onnx'
        )

        self.log('Model loaded')
        toc = time.perf_counter()

        self.log(f"Time taken to load model: {toc - tic:0.4f} seconds")

        self.asr.is_wake_word_enabled = True

    def reset_model_state(self):
        """Reset the wake word model's prediction buffer to avoid false triggers."""
        for mdl in self.model.prediction_buffer.keys():
            self.model.prediction_buffer[mdl] = []

    def start_listening(self):
        self.is_listening = True
        self.audio = None

        self.reset_model_state()

        try:
            self.log("Listening...")

            while self.is_listening:
                # Get audio
                # Reuse the shared mic audio stream with ASR
                self.audio = np.frombuffer(self.asr.mic_stream.read(self.chunk_size), dtype=np.int16)

                # Feed to openWakeWord model
                prediction = self.model.predict(self.audio)

                for mdl in self.model.prediction_buffer.keys():
                    scores = list(self.model.prediction_buffer[mdl])

                    if scores[-1] > self.detection_threshold:
                        self.log(f"Wakeword Detected! ({mdl})")
                        self.is_listening = False
                        self.asr.start_recording()

        except Exception as e:
            self.is_listening = False
            self.log('Error:', e)

    @staticmethod
    def log(*args, **kwargs):
        print('[Wake word]', *args, **kwargs)
