import librosa
from typing import *

def load_audio(filename: str, block_len=256, frame_len=2048):
    sr = librosa.get_samplerate(filename)
    stream = librosa.stream(
        filename,
        block_length=block_len,
        frame_length=frame_len,
        hop_length=frame_len,
    )
    return stream, sr
