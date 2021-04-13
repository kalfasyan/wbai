import torch
from librosa.core.spectrum import amplitude_to_db
from scipy.signal.spectral import spectrogram
from utils import get_wingbeat_files, label_func
from fastai.data.transforms import get_files
import librosa
from sklearn import preprocessing
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram, Spectrogram
from datasets import SR
import numpy as np

torch.manual_seed(42)


from utils import (BASE_DATAPATH, BASE_PROJECTPATH, butter_bandpass_filter, open_wingbeat)

class Focus(object):
    def __init__(self):
        super().__init__()

    def __call__(self, sample):
        wbtold, label = sample['x'], sample['y']
        wbt = torch.abs(wbtold)
        wbt = wbt.reshape(wbt.shape[0], -1, 250).mean(dim=2)
        wbt = torch.where(wbt.T> 0.0025, 1, 0).repeat_interleave(250)
        wbt = torch.mul(wbt,wbtold)
        return {'x': wbt, 'y': label, 'path': sample['path'], 'idx': sample['idx']}

class RandomRoll(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        wbt, label = sample['x'], sample['y']
        if torch.rand(1) < self.p:
            wbt = torch.roll(wbt, shifts=np.random.randint(500,4500))
        return {'x': wbt, 'y': label, 'path': sample['path'], 'idx': sample['idx']}                                                        

class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        wbt, label = sample['x'], sample['y']
        if torch.rand(1) < self.p:
            wbt = torch.flip(wbt, [1])
        return {'x': wbt, 'y': label, 'path': sample['path'], 'idx': sample['idx']}                                                        

class RandomNoise(object):
    def __init__(self, p=0.5, width_min=500, width_max=1000):
        self.p = p
        self.width_min = width_min
        self.width_max = width_max
        assert 1000 >= width_max > width_min, "Ensure that 1000 >= width_max > width_min"
    
    def __call__(self, sample):
        wbt, label = sample['x'], sample['y']
        if torch.rand(1) < self.p:
            rdm = np.random.choice(range(0,4000))
            rdm_width = np.random.choice(range(self.width_min, self.width_max))
            noise = np.random.normal(0, 0.002, rdm_width)
            wbt[:, rdm:rdm+rdm_width] = torch.from_numpy(noise).reshape(1,-1)
        return {'x': wbt, 'y': label, 'path': sample['path'], 'idx': sample['idx']}                                                        


class Bandpass(object):
    "Class to apply a signal processing filter to a dataset"

    def __init__(self, lowcut=120., highcut=1500., fs=8000., order=4):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order

    def __call__(self, sample):
        wbt, label = sample['x'], sample['y']
        wbt = torch.from_numpy(butter_bandpass_filter(wbt, 
                                                    lowcut=self.lowcut,
                                                    highcut=self.highcut,
                                                    fs=self.fs,
                                                    order=self.order)).float()
        return {'x': wbt, 'y': label, 'path': sample['path'], 'idx': sample['idx']}                                                        

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
            if self.setting == 'stftcrop': spec = spec[:,5:70,:]
            spec = AmplitudeToDB()(spec)
            spec = torch.from_numpy(np.repeat(spec.numpy()[...,np.newaxis],3,0))
            spec = spec[:,:,:,0]
        
            if self.setting == 'stftraw':
                return {'x': (wbt,spec), 'y': label, 'path': sample['path'], 'idx': sample['idx']}
            else:
                return {'x': spec, 'y': label, 'path': sample['path'], 'idx': sample['idx']}

        elif self.setting.startswith('melstft'):
            # TODO: Something wrong in the output spec. Missing frequencies.
            spec = MelSpectrogram(n_fft=256, hop_length=42)(wbt)
            spec = AmplitudeToDB()(spec)
            spec = torch.from_numpy(np.repeat(spec.numpy()[...,np.newaxis],3,0))
            spec = spec[:,:,:,0]
        
            if self.setting == 'melstftraw':
                return {'x': (wbt,spec), 'y': label, 'path': sample['path'], 'idx': sample['idx']}
            else:
                return {'x': spec, 'y': label, 'path': sample['path'], 'idx': sample['idx']}

        elif self.setting == 'reassigned_stft':
            freqs, times, mags = librosa.reassigned_spectrogram(y=wbt.numpy().squeeze(), sr=SR, 
                                                                n_fft=256, hop_length=42, center=False)
            mags_db = librosa.power_to_db(mags)
            mags_db = np.expand_dims(librosa.power_to_db(mags),axis=0)
            mags_db = torch.from_numpy(np.repeat(mags_db[...,np.newaxis],3,0))
            mags_db = mags_db[:,:,:,0]
            return {'x': mags_db, 'y': label, 'path': sample['path'], 'idx': sample['idx']}

        elif self.setting.startswith('psd'):
            _, psd = sg.welch(wbt.numpy().squeeze(), fs=SR, scaling='density', window='hanning', nfft=8192, nperseg=256, noverlap=128+64)
            if self.setting == 'psdl1':
                psd = preprocessing.normalize(psd.reshape(1,-1), norm='l1')
            elif self.setting == 'psdl2':
                psd = preprocessing.normalize(psd.reshape(1,-1), norm='l2')
            return {'x': psd, 'y': label, 'path': sample['path'], 'idx': sample['idx']}