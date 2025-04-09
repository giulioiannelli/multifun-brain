from .shared import *

def hello_brain(name):
    return f"Hello, {name}! Welcome to multifun-brain."

def bandpass_filter(data, low, high, fs=1, order=4):
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = butter(order, [low_norm, high_norm], btype='band')
    return filtfilt(b, a, data)

