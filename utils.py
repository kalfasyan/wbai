import torchaudio
import torch
torch.manual_seed(42)
from pathlib import Path
import matplotlib.pyplot as plt
from fastai.data.transforms import get_files
import torch.nn.functional as F
import pandas as pd
import numpy as np
from configparser import ConfigParser
import sys
sys.setrecursionlimit(10000)

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
    dsname = str(fname).split('/')[len(BASE_DATAPATH.parts)]
    if (dsname.startswith(("Suzukii_RL", "Melanogaster_RL"))):
        if dsname == 'Suzukii_RL':
            return 'D. suzukii'
        elif dsname == 'Melanogaster_RL':
            return 'D. melanogaster'
    else:
        return str(fname).split('/')[len(BASE_DATAPATH.parts)+1]

def butter_bandpass_filter(data, lowcut=120., highcut=1500., fs=8000., order=4):
    from scipy.signal import butter, lfilter

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='bandpass')
    signal_filtered = lfilter(b, a, data)
    return signal_filtered

def calc_mean_std_1D(loader=None):
    from tqdm import tqdm


    channels_sum, channels_sqrd_sum, num_batches = 0,0,0

    for data in tqdm(loader, desc='Calculating mean and std..\t'):
        data = data[0]
        channels_sum += torch.mean(data,dim=[0,2]) 
        channels_sqrd_sum += torch.mean(data**2, dim=[0,2])
        num_batches += 1 

    mean = (channels_sum/num_batches)
    std = (channels_sqrd_sum/num_batches - mean**2)**0.5
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    return mean, std


@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for x_batch,y_batch,path_batch,idx_batch in loader:

        y_batch = torch.as_tensor(y_batch).type(torch.LongTensor)
        x_batch,y_batch = x_batch.cuda(), y_batch.cuda()


        preds = model(x_batch.float())
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )
    return all_preds

def np_hist(df, col, res=0.1, rot=45, fs=12):
    import matplotlib.pyplot as plt
    import numpy as np
    np.random.seed(42)
    values = df[col]
    _bins, _edges = np.histogram(values, np.arange(df[col].min(), df[col].max(), res))
    plt.plot(_edges[:len(_edges)-1], _bins)
    plt.ylabel('counts'); plt.xlabel(col)
    plt.xticks(rotation=rot, fontsize=fs);
    plt.yticks(fontsize=fs);
    plt.show()

def wingbeat_duration(sig, rolling_window=150, fs=44100.):
    sig = pd.Series(sig.squeeze())
    sig = sig.abs().rolling(rolling_window).mean()
    return (sig > 0.0025).sum() / fs * 1000. # in ms

def get_WBduration_from_loader(loader, fs=44100.):
    """
    Returns the average Wingbeat duration given a dataloader
    in milliseconds
    """
    from tqdm import tqdm

    durations = []
    for x,y,p,i in tqdm(loader):
        durations += list(map(lambda z: wingbeat_duration(z, fs=fs), x))

    return durations, np.mean(durations), np.median(durations), np.std(durations)

def get_medianWBDset_psd_from_loader(loader):
    from tqdm import tqdm
    psds = []
    for x,y,p,i in tqdm(loader):
        psds += x
    df_psds = pd.DataFrame(np.stack(psds).squeeze())
    return df_psds.median()

def get_datestr_range(start='',end=''):
    """
    Function to create a list of ordered date strings "%Y%m%d"
    """
    import datetime

    start = datetime.datetime.strptime(f"{start}", "%Y%m%d")
    end = datetime.datetime.strptime(f"{end}", "%Y%m%d")
    date_array = \
        (start + datetime.timedelta(days=x) for x in range(0, (end-start).days+1))

    datestrlist = [] 
    for date_object in date_array:
        datestrlist.append(date_object.strftime("%Y%m%d"))
    return datestrlist

