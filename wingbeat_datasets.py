import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from sklearn import preprocessing
from utils import BASE_PATH, open_wingbeat, make_dataset_df, butter_bandpass_filter
import librosa
import numpy as np
from scipy import signal as sg
from tqdm import tqdm

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
        self.files, self.labels, self.lbl2files = make_dataset_df(dsname, verbose=True)
        if len(custom_label) == 1:
            self.labels = [custom_label[0] for _ in self.labels]
            if verbose:
                print(f"Label(s) changed to {custom_label}")
        else:
            if verbose:
                print("No custom label applied.")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fname = self.files[idx]
        label = self.labels[idx]
        wingbeat = open_wingbeat(fname, plot=False)
        sample = {'x': wingbeat, 'y': label, 'path': str(fname), 'idx': idx}

        if self.transform:
            sample = self.transform(sample)

        return sample['x'], sample['y'], sample['path'], sample['idx']

    def onehotlabels(self):
        le = preprocessing.LabelEncoder()

        self.labels = le.fit_transform(self.labels)
        self.labels = F.one_hot(torch.as_tensor(self.labels))

    def clean(self,threshold=25, batch_size=16, num_workers=16):
        dloader = DataLoader(self, batch_size=batch_size, num_workers=num_workers)

        inds = clean_wingbeatsdataset_inds(name=self.dsname, threshold=threshold,batch_size=batch_size, num_workers=num_workers)
        
        self.clean_subset = torch.utils.data.Subset(super(), inds)
        return self.clean_subset

class ValidateWingbeat(object):
    def __init__(self, setting='default'):
        self.setting = setting

    def __call__(self, sample):
        wingbeat = sample['x']
        label = sample['y']

        if self.setting == 'default':
            wingbeat = torch.from_numpy(butter_bandpass_filter(wingbeat)).float()
            _, wingbeat = sg.welch(wingbeat, fs=8000, scaling='density', window='hanning', nfft=8192, nperseg=256, noverlap=128+64)
            wingbeat = preprocessing.normalize(wingbeat.reshape(1,-1), norm='l2')
            score = np.sum(wingbeat)
        else:
            raise NotImplementedError('!')

        return {'x': wingbeat, 'y': label, 'path': sample['path'], 
                'idx': sample['idx'], 'score': score}

class FilterWingbeat(object):
    def __init__(self, setting='bandpass'):
        self.setting = setting

    def __call__(self, sample):
        wingbeat = sample['x']
        label = sample['y']

        if self.setting == 'bandpass':
            wingbeat = torch.from_numpy(butter_bandpass_filter(wingbeat)).float()
        else:
            raise NotImplementedError('!')

        return {'x': wingbeat, 'y': label, 'path': sample['path'], 'idx': sample['idx']}

class TransformWingbeat(object):
    def __init__(self, setting=''):
        self.setting = setting
        assert len(self.setting), "Please provide a transformation setting."

    def __call__(self, sample):
        
        wingbeat = sample['x']
        label = sample['y']
        
        if self.setting == 'stft':
            wingbeat = librosa.stft(wingbeat.numpy().squeeze(), n_fft = 256, hop_length = int(256/6))
            wingbeat = librosa.amplitude_to_db(np.abs(wingbeat))
            wingbeat = np.flipud(wingbeat)
        elif self.setting.startswith('psd'):
            _, wingbeat = sg.welch(wingbeat.numpy().squeeze(), fs=8000, scaling='density', window='hanning', nfft=8192, nperseg=256, noverlap=128+64)
            if self.setting == 'psdl1':
                wingbeat = preprocessing.normalize(wingbeat.reshape(1,-1), norm='l1')
            elif self.setting == 'psdl2':
                wingbeat = preprocessing.normalize(wingbeat.reshape(1,-1), norm='l2')

        return {'x': wingbeat, 'y': label, 'path': sample['path'], 'idx': sample['idx']}

def clean_wingbeatsdataset_inds(name="Melanogaster_RL/Y", threshold=25, batch_size=16, num_workers=16):
    import os
    fname = f"./data_created/{name.replace('/','-').replace(' ', '')}_th{threshold}_cleaninds.npy"
    if os.path.isfile(fname):
        return np.load(fname).tolist()

    dataset = WingbeatsDataset(name, transform=TransformWingbeat(setting='psdl2'), verbose=False)
    dloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    all_inds_valid = torch.tensor([])
    for i, (x_batch,y_batch,path_batch,idx) in enumerate(tqdm(dloader)):
        xb_sums = x_batch.squeeze().sum(axis=1)
        xb_valid = torch.where(xb_sums<threshold,xb_sums,torch.zeros(xb_sums.shape))
        inds_valid = idx[xb_valid.nonzero().squeeze()]
        if inds_valid.dim() > 0:
            all_inds_valid = torch.cat((all_inds_valid, inds_valid),0)
        elif inds_valid.dim() == 0:
            all_inds_valid = torch.cat((all_inds_valid, inds_valid.view(1)))

    list_all_inds_valid = list(all_inds_valid.numpy().astype(int))
    np.save(fname, list_all_inds_valid)
    return list_all_inds_valid
