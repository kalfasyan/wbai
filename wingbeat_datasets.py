import librosa
from librosa.core.spectrum import amplitude_to_db
import numpy as np
import psutil
from scipy.signal.spectral import spectrogram
import torch
from torch._C import Value
import torch.nn.functional as F
from scipy import signal as sg
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import os
from torchaudio.transforms import Spectrogram, AmplitudeToDB
import pandas as pd

from utils import (BASE_DATAPATH, butter_bandpass_filter, make_dataset_df,
                   open_wingbeat)

num_workers = psutil.cpu_count()
print(f"Available workers: {num_workers}")

class WingbeatsDataset(Dataset):
    """Wingbeats dataset."""

    def __init__(self, dsname, custom_label=[], transform=None, verbose=True):
        """
        Args:
            dsname (string): Dataset Name.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dsname = dsname
        self.verbose = verbose
        self.files, self.labels, self.lbl2files = make_dataset_df(dsname, verbose=True)
        if len(custom_label) == 1:
            self.labels = [custom_label[0] for _ in self.labels]
            if self.verbose:
                print(f"Label(s) changed to {custom_label}")
        else:
            if self.verbose:
                print("No custom label applied.")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fname = self.files[idx]
        label = self.labels[idx]
        wbt = open_wingbeat(fname, plot=False)
        sample = {'x': wbt, 'y': label, 'path': str(fname), 'idx': idx}

        if self.transform:
            sample = self.transform(sample)

        return sample['x'], sample['y'], sample['path'], sample['idx']

    def onehotlabels(self):
        "To be used when selecting a dataset with more than one species"
        le = preprocessing.LabelEncoder()

        self.labels = le.fit_transform(self.labels)
        self.labels = F.one_hot(torch.as_tensor(self.labels))

    def clean(self,threshold=25, batch_size=32, num_workers=num_workers):
        """
        Given a WingbeatsDataset, it will return a clean Subset after applying 
        a threshold on the sum of l2-normalized PSDs (which has shown to separate valid from invalid wingbeats).

        See 'clean_wingbeatsdataset_inds' definition below.
        """
        dloader = DataLoader(self, batch_size=batch_size, num_workers=num_workers)

        inds, fnames = clean_wingbeatsdataset_inds(name=self.dsname, threshold=threshold,batch_size=batch_size, num_workers=num_workers)
        
        inds = [int(i) for i in inds]
        self.clean_subset = torch.utils.data.Subset(self, inds)
        self.clean_inds = inds
        self.clean_fnames = fnames

        if self.verbose:
            print(f"Nr. of valid wingbeats: {len(inds)}")
        return self.clean_subset

class FilterWingbeat(object):
    "Class to apply a signal processing filter to a dataset"

    def __init__(self, setting='bandpass'):
        self.setting = setting

    def __call__(self, sample):
        wbt, label = sample['x'], sample['y']

        if self.setting == 'bandpass':
            wbt = torch.from_numpy(butter_bandpass_filter(wbt)).float()
        else:
            raise NotImplementedError('!')

        return {'x': wbt, 'y': label, 'path': sample['path'], 'idx': sample['idx']}

class NormalizeWingbeat(object):
    "Class to normalize a wbt."
    def __call__(self, sample): 
        wbt, label = sample['x'], sample['y']
        wbt = (wbt-wbt.mean()) / wbt.std()

        return {'x': wbt, 'y': label, 'path': sample['path'], 'idx': sample['idx']}

class TransformWingbeat(object):
    "Class to transform wingbeat datasets."
    """
    Args:
        setting (string): Data type to transform to.
        Available settings:
        - stft: Spectrograms
        - psd: Power Spectral density
            -psdl1: PSD normalized with l1 norm
            -psdl2: PSD normalized with l2 norm
    """
    def __init__(self, setting=''):
        self.setting = setting
        assert len(self.setting), "Please provide a transformation setting."

    def __call__(self, sample):
        wbt = sample['x']
        label = sample['y']
        
        if self.setting.startswith('stft'):
            spec = Spectrogram(n_fft=256, hop_length=42)(wbt)
            spec = AmplitudeToDB()(spec)
            spec = torch.from_numpy(np.repeat(spec.numpy()[...,np.newaxis],3,0))
            spec = spec[:,:,:,0]
        
            if self.setting == 'stftraw':
                return {'x': (wbt,spec), 'y': label, 'path': sample['path'], 'idx': sample['idx']}
            else:
                return {'x': spec, 'y': label, 'path': sample['path'], 'idx': sample['idx']}

        elif self.setting.startswith('psd'):
            _, psd = sg.welch(wbt.numpy().squeeze(), fs=8000, scaling='density', window='hanning', nfft=8192, nperseg=256, noverlap=128+64)
            if self.setting == 'psdl1':
                psd = preprocessing.normalize(psd.reshape(1,-1), norm='l1')
            elif self.setting == 'psdl2':
                psd = preprocessing.normalize(psd.reshape(1,-1), norm='l2')
            return {'x': psd, 'y': label, 'path': sample['path'], 'idx': sample['idx']}

def clean_wingbeatsdataset_inds(name="Melanogaster_RL/Y", threshold=25, batch_size=32, num_workers=num_workers):
    """
    Helper function to clean a WingbeatsDataset. It is used in its 'clean' method.
    """

    # Checking if the indice are alreaddy available for the given dataset and threshold
    fname = f"./data_created/{name.replace('/','-').replace(' ', '')}_th{threshold}_cleaned"
    if os.path.isfile(f"{fname}.npy") and os.path.isfile(f"{fname}.csv"):
        return np.load(f"{fname}.npy").tolist(), pd.read_csv(f"{fname}.csv")['fnames'].tolist()

    # Dataset of l2-normalized Power-Spectral-Densities
    dataset = WingbeatsDataset(name, transform=TransformWingbeat(setting='psdl2'), verbose=False)
    dloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    # Creating empty tensor to populate with valid wingbeat indice
    all_inds_valid = torch.tensor([])
    all_paths_valid = []

    for i, (x_batch,y_batch,path_batch,idx) in enumerate(tqdm(dloader, desc="Cleaning dataset..\t")):
        # Using the sums of l2-normalized PSDs
        xb_sums = x_batch.squeeze().sum(axis=1)
        # We set a threshold to get valid wingbeats, all invalid set to zero
        xb_valid = torch.where(xb_sums<threshold,xb_sums,torch.zeros(xb_sums.shape))
        # Retrieving the indice of nonzero (valid wingbeat) sums
        inds_valid = idx[xb_valid.nonzero().squeeze()]
        paths_valid = np.array(path_batch)[xb_valid.nonzero().squeeze()].tolist()
        # Concatenating to indice to 'all_inds_valid'
        if inds_valid.dim() > 0:
            all_inds_valid = torch.cat((all_inds_valid, inds_valid),0)
            all_paths_valid += paths_valid
        elif inds_valid.dim() == 0: # in case the torch tensor is 0-dim
            all_inds_valid = torch.cat((all_inds_valid, inds_valid.view(1)))
            all_paths_valid += [paths_valid] 

    list_all_inds_valid = list(all_inds_valid)

    if len(all_paths_valid):
        # Saving indice to avoid recalculating the above all the time
        np.save(fname, list_all_inds_valid)
        pd.DataFrame({"fnames": all_paths_valid}).to_csv(f"{fname}.csv", index=False)
    else:
        raise ValueError("List of clean wingbeats returned empty.")

    return list_all_inds_valid, all_paths_valid

def get_clean_wingbeatsdataset_filenames(dset_names=[]):
    all_fnames = []
    for dsname in dset_names:
        d = WingbeatsDataset(dsname=dsname)
        d.clean()
        all_fnames += d.clean_fnames
    return all_fnames
