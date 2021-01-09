import torchaudio
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from fastai.data.transforms import get_files
import torch.nn.functional as F
import pandas as pd
import numpy as np
from configparser import ConfigParser

cfg = ConfigParser()
cfg.read('config.ini')

BASE_DATAPATH = Path(cfg.get('base','data_dir'))
BASE_PROJECTPATH = Path(cfg.get('base','project_dir'))

def get_wingbeat_files(dsname):
    datadir = Path(BASE_DATAPATH/dsname)
    return get_files(datadir, extensions='.wav', recurse=True, folders=None, followlinks=True)

def open_wingbeat(fname, plot=False, verbose=False):
    waveform, sample_rate = torchaudio.load(str(fname))

    if verbose:
        print(f"Shape of waveform: {waveform.size()}")
        print(f"Sample rate of waveform: {sample_rate}")

    if plot:
        plt.figure()
        plt.plot(waveform.t().numpy())
        plt.show()

    return waveform

def label_func(fname):
    # TODO: Use regular expressions instead
    return str(fname).split('/')[len(BASE_DATAPATH.parts)+1]

def make_dataset_df(dsname, verbose=False):

    datadir = Path(BASE_DATAPATH/dsname)

    files = get_wingbeat_files(dsname)
    labels = pd.Series(files).apply(lambda x: label_func(x)).tolist() #list(files.map(label_func))
    if verbose:
        print(f"Found {len(list(files))} in dataset: {dsname}, ", end="")
        print(f"and {len(set(labels))} label(s): {np.unique(labels)}")
    
    lbl2files = {l: [f for f in files if label_func(f) ==l] for l in list(set(labels))}

    return files, labels, lbl2files

def butter_bandpass_filter(data, lowcut=120, highcut=1500, fs=8000, order=4):
    from scipy.signal import butter, lfilter

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='bandpass')
    signal_filtered = lfilter(b, a, data)
    return signal_filtered