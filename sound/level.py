import numpy as np
# from numpy.typing import *
import librosa
from scipy.signal import find_peaks


# raw data -> rms values 변환하는 함수
def calculate_rms(data, sr) :
    s, phase = librosa.magphase(librosa.stft(data, center=False))
    rms = librosa.feature.rms(S=s)
    return rms

# rms values -> dB values 변환하는 함수
def rms2db(rms) :
    ref = 2 * 10 ** (-5)
    spl = 10 * np.log10((rms ** 2) / (ref ** 2))
    return spl

# LAmax 값 보정용 함수
def lamax_adjustment(db) :
    return db + 10.5006

# LAeq 값 보정용 함수
def laeq_adjustment(db) :
    return db + 17.3279

# 주어진 dB values 에서 [frame_beg, frame_end) 범위의 LAmax 계산
def calculate_lamax(dba, frames_beg, frames_end) :
    dba = lamax_adjustment(dba)
    return np.max(dba[frames_beg:frames_end]).item()

# 주어진 dB values 에서 [frame_beg, frame_end) 범위의 LAeq 계산
def calculate_laeq(dba, frames_beg, frames_end) :
    dba = lamax_adjustment(dba)
    return np.mean(dba[frames_beg:frames_end]).item()

# dB values 를 받아서 피크 인덱스들을 반환 threshold 이상의 피크들만 반환
def find_spike_indices(dba, threshold) :
    peaks, _ = find_peaks(dba, height=threshold, prominence=10)
    return peaks


