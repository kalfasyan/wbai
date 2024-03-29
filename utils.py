from unicodedata import normalize

import torch
import torchaudio

torch.manual_seed(42)
import shutil
import sys
from configparser import ConfigParser
from pathlib import Path
from shutil import copy2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
from fastai.data.transforms import get_files
from torch.autograd import Variable
from tqdm import tqdm

sys.setrecursionlimit(10000)

cfg = ConfigParser()
cfg.read('/home/kalfasyan/projects/wbai/config.ini')

BASE_DATAPATH = Path(cfg.get('base','data_dir'))
BASE_PROJECTPATH = Path(cfg.get('base','project_dir'))
BASE_DATACREATEDDIR = Path(cfg.get('base','datacreated_dir'))

def get_wingbeat_files(dsname):
    datadir = Path(BASE_DATAPATH/dsname)
    return get_files(datadir, extensions='.wav', recurse=True, folders=None, followlinks=True)

def open_wingbeat(fname, plot=False, verbose=False, rate=False, rpiformat=False):
    from scipy import signal
    waveform, sample_rate = torchaudio.load(str(fname))

    if rpiformat:
        # print(sample_rate)
        # assert waveform.shape[1] == 72000, print(f"{waveform.shape[1]}") 
        assert sample_rate == 48000
        waveform = waveform[0,:30000]
        resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=8000, lowpass_filter_width=1)
        waveform = resample(waveform)
        # waveform = torch.from_numpy(signal.resample(waveform.T, 5000).squeeze())

    if verbose:
        print(f"Shape of waveform: {waveform.size()}")
        print(f"Sample rate of waveform: {sample_rate}")

    if plot:
        plt.figure()
        plt.plot(waveform.t().numpy())
        plt.show()

    if rate:
        return waveform, sample_rate
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


# @torch.no_grad()
# def get_all_preds(model, loader):
#     all_preds = torch.tensor([])
#     for x_batch,y_batch,path_batch,idx_batch in loader:

#         y_batch = torch.as_tensor(y_batch).type(torch.LongTensor)
#         x_batch,y_batch = x_batch.cuda(), y_batch.cuda()


#         preds = model(x_batch.float())
#         all_preds = torch.cat(
#             (all_preds, preds)
#             ,dim=0
#         )
#     return all_preds

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

def get_wbt_duration(fname, lowcut=140., highcut=1500., order=4, window=150, th=0.0025, plot=False):
    sig, rate = open_wingbeat(fname, rate=True)
    sig = butter_bandpass_filter(sig, lowcut=lowcut, highcut=highcut, fs=rate, order=order)
    sig = pd.Series(sig.squeeze()).abs().rolling(window).mean()
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(sig)
        plt.plot(((sig > th)).astype(int)*.02)
    return (sig > th).sum() / rate * 1000. # in ms

def get_wbt_duration_loader(loader, th=0.0025):
    """
    Returns the average Wingbeat duration given a dataloader
    in milliseconds
    """
    from tqdm import tqdm

    durations = []
    for x,y,p,i,r in tqdm(loader):
        durations += list(map(lambda z: get_wbt_duration(z), x))

    return durations, np.mean(durations), np.median(durations), np.std(durations)

def wingbeat_duration(sig, rolling_window=150, th=0.0025, fs=44100., plot=False):
    sig = pd.Series(sig.squeeze())
    sig = sig.abs().rolling(rolling_window).mean()
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(sig)
    return (sig > th).sum() / fs * 1000. # in ms

def get_WBduration_from_loader(loader, th=0.0025, fs=44100.):
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

def test_model(model, loader, dataset):
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from tqdm import tqdm

    model.eval()
    correct = 0
    y_pred,y_true = [],[]
    for x_batch,y_batch,path_batch,idx_batch in tqdm(loader, desc='Testing..\t'):
        y_batch = torch.as_tensor(y_batch).type(torch.LongTensor)
        x_batch,y_batch = x_batch.cuda(), y_batch.cuda()
        pred = model(x_batch)
        _, preds = torch.max(pred, 1)
        y_pred.extend(preds.detach().cpu().numpy())
        y_true.extend(y_batch.detach().cpu().numpy())
        correct += (pred.argmax(axis=1) == y_batch).float().sum().item()
    accuracy = correct / len(dataset) * 100.
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_pred=y_pred, y_true=y_true)*100.:.2f}")
    print(f"Confusion matrix: \n{confusion_matrix(y_pred=y_pred, y_true=y_true, normalize='true')}")

def test_model_binary(model, loader, dataset):
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from tqdm import tqdm

    model.eval()
    correct = 0
    y_pred,y_true = [],[]
    for x_batch,y_batch,path_batch,idx_batch in tqdm(loader, desc='Testing..\t'):
        y_batch = torch.as_tensor(y_batch).type(torch.LongTensor)
        x_batch,y_batch = x_batch.cuda(), y_batch.cuda()
        pred = model(x_batch)
        preds = (pred>0.5).type(torch.IntTensor).squeeze()
        y_pred.extend(preds.detach().cpu().numpy())
        y_true.extend(y_batch.detach().cpu().numpy())
        correct += ((pred>0.5).type(torch.IntTensor).squeeze().cuda() == y_batch.squeeze()).float().sum().item()
    accuracy = correct / len(dataset) * 100.
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_pred=y_pred, y_true=y_true)*100.:.2f}")
    print(f"Confusion matrix: \n{confusion_matrix(y_pred=y_pred, y_true=y_true, normalize='true')}")
    return y_pred, y_true

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

@torch.no_grad()
def get_all_preds(model, loader, dataframe=False, final_nodes=2):
    all_preds = torch.tensor([]).cuda()
    all_labels = torch.tensor([]).cuda()
    all_paths = []
    all_idx = torch.tensor([]).cuda()
    for x_batch, y_batch, path_batch, idx_batch in tqdm(loader):

        preds = model(x_batch.cuda())
        all_preds = torch.cat((all_preds, preds), dim=0)
        all_labels = torch.cat((all_labels, y_batch.cuda()), dim=0)
        all_paths.extend(path_batch)
        all_idx = torch.cat((all_idx, idx_batch.cuda()), dim=0)
    
    out = all_preds,all_labels,all_paths,all_idx

    if not dataframe:
        return out
    else:
        if final_nodes==1:
            df_out = pd.DataFrame(out[0], columns=['pred'])
            df_out['softmax'] = df_out.pred.apply(lambda x: 1 if x>0 else 0)
        else:
            df_out = pd.DataFrame(out[0], columns=[f'pred{i}' for i in range(final_nodes)])
            df_out['softmax'] = torch.argmax(F.softmax(out[0], dim=1), dim=1).detach().cpu()
        df_out['y'] = out[1].cpu()
        df_out['fnames'] = out[2]
        df_out['idx'] = out[3].cpu()
        return df_out

def plot_wingbeat(dataset, idx=None):
    from IPython.display import Audio
    if idx is None:
        idx = int(torch.randint(0, len(dataset), (1,)))
    sig = dataset[idx][0]
    plt.plot(sig.T); plt.ylim(-.04,.04)
    Audio(sig, rate=8000, autoplay=True)

def plot_wingbeat_spectrogram(dataset, idx=None):
    if idx is None:
        idx = int(torch.randint(0, len(dataset), (1,)))
    sig = dataset[idx][0]
    plt.imshow(sig[0])

def save_checkpoint(state, is_best, filename=''):
    filename = f'{BASE_DATACREATEDDIR}/{filename}.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f"{filename.split('.')[0]}_best.pth.tar")

def load_checkpoint(filename, model, optimizer):
    assert isinstance(filename, str) and filename.endswith('pth.tar'), "Only works with a pth.tar file."
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

def copy_files(filelist, destination):
    for f in tqdm(filelist, total=len(filelist), desc="Copying files.."):
        copy2(f, destination)

def show_peaks(sig, height=0.04, prominence=0.001, width=1, distance=5):
    from scipy.signal import find_peaks
    
    plt.plot(sig)
    p, _ = find_peaks(sig.squeeze(), height=height, prominence=prominence, width=width, distance=distance)
    plt.plot(p, sig[p], 'x')
    plt.title(len(p))

def resize2d(img, size):
    with torch.no_grad():
        return (F.adaptive_avg_pool2d(Variable(img), size)).data

def scale(A):
    return (A-np.min(A))/(np.max(A) - np.min(A))

def get_bad_first_chunk_score(x, plot=False):
    x = scale(x)

    winsize = 5
    part = x[620:712+winsize]
    part = pd.Series(part).rolling(winsize).std().abs()
    if plot:
        part.plot()

    partA = part.iloc[:46]
    partB = part.iloc[46:]
    
    if plot:
        partA.plot(color='b')
        partB.plot(color='k')

    scoreA = partA.sum()
    scoreB = partB.sum()

    score = scoreB - scoreA
    return score