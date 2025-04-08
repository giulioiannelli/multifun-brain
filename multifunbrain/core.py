from .shared import *

def hello_brain(name):
    return f"Hello, {name}! Welcome to multifun-brain."

def bandpass_filter(data, low, high, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data)




