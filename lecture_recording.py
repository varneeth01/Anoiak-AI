# lecture_recording.py
import cv2
import os
import wave
import pyaudio

class LectureRecorder:
    def __init__(self, video_resolution=(640, 480), fps=30, audio_format=(1, 2, 16000, 0, 'WAVE', 'not compressed')):
        self.video_resolution = video_resolution
        self.fps = fps
        self.audio_format = audio_format
        self.video_writer = None
        self.audio_writer = None

    def start_recording(self, video_path, audio_path):
        # Start video recording
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, self.video_resolution)

        # Start audio recording
        self.audio_writer = wave.open(audio_path, 'wb')
        self.audio_writer.setparams(self.audio_format)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                 channels=self.audio_format[1],
                                 rate=self.audio_format[2],
                                 input=True,
                                 frames_per_buffer=1024)

    def record_frame(self, frame):
        self.video_writer.write(frame)

    def record_audio(self):
        data = self.stream.read(1024)
        self.audio_writer.writeframes(data)

    def stop_recording(self):
        self.video_writer.release()
        self.audio_writer.close()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
