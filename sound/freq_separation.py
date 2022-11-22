import numpy as np
from typing import *

ONE_OCTAVE_BANDS = \
    (11.049, 22.097, 44.194, 88.388, 176.777, 353.553, 707.107,
     1414.214, 2828.427, 5656.854, 11313.708, 22627.417)

ONE_THIRD_OCTAVE_BANDS = \
    (13.920, 17.538, 22.097, 27.841, 35.077, 44.194, 55.681, 70.154,
     88.388, 111.362, 140.308, 176.777, 222.725, 280.616, 353.553,
     445.449, 561.231, 707.107, 890.899, 1122.462, 1414.214, 1781.797,
     2244.924, 2828.427, 3563.595, 4489.848, 5656.854, 7127.190,
     8979.696, 11313.708, 14254.379, 17959.393, 22627.417)


# 푸리에 변환후 주파수와 각 주파수에 해당하는 값 계산
def get_fft_and_freq(
    data: np.array,
    sample_rate: int,
):
    fft = np.fft.rfft(data)
    freq = np.fft.rfftfreq(len(data), d=1./sample_rate)
    return fft, freq


# split frequencies (위에 있는 BANDS 들)에 따라 fft 값들을 분할하기 위한 범위들을 계산
def get_split_ranges(
    freq: np.array,
    split_frequencies: Iterable[float]
):
    last_point = np.argmin(np.abs(freq - next(split_frequencies)))
    ranges = []
    for f in split_frequencies:
        idx = np.argmin(np.abs(freq - f))
        if freq[idx] - f > 0.5:
            ranges.append((last_point, len(freq)))
            return ranges
        ranges.append((last_point, idx+1))
        last_point = idx
    return ranges


# 특정 구간 (idx_rng) 의 fft 값들을 이용해 오디오로 재구성
def get_filtered_audio(
    original_fft: np.array,
    idx_rng: Tuple[int, int]
):
    fft = np.array(original_fft, copy=True)
    fft[range(0, idx_rng[0])] = 0
    fft[range(idx_rng[1], len(original_fft))] = 0
    fft = np.fft.irfft(fft)
    return fft


if __name__ == "__main__":
    import librosa
    import soundfile as sf
    sig, sr = librosa.load('test-audio.wav', sr=None)

    fft, freq = get_fft_and_freq(sig, sr)
    ranges = get_split_ranges(freq, iter(ONE_OCTAVE_BANDS))

    for rng in ranges:
        filtered = get_filtered_audio(fft, rng)
        sf.write(f'out({freq[rng[0]]:.3f}-{freq[rng[1]-1]:.3f}).wav', filtered, sr, 'PCM_24')


