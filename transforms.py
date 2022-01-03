import torch
from librosa.core.spectrum import amplitude_to_db
from scipy.signal.spectral import spectrogram
from utils import get_wingbeat_files, label_func, resize2d
from fastai.data.transforms import get_files
import librosa
from sklearn import preprocessing
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram, Spectrogram
from datasets import SR
import numpy as np
from scipy import signal as sg

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
        sample['x'] = wbt
        return sample
class RandomRoll(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        wbt, label = sample['x'], sample['y']
        if torch.rand(1) < self.p:
            wbt = torch.roll(wbt, shifts= int(torch.randint(500,4500,(1,))))
        sample['x'] = wbt
        return sample
class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        wbt, label = sample['x'], sample['y']
        if torch.rand(1) < self.p:
            wbt = torch.flip(wbt, [1])
        sample['x'] = wbt
        return sample
class RandomNoise(object):
    def __init__(self, p=0.5, width_min=500, width_max=1000):
        self.p = p
        self.width_min = width_min
        self.width_max = width_max
        assert 1000 >= width_max > width_min, "Ensure that 1000 >= width_max > width_min"
    
    def __call__(self, sample):
        wbt, label = sample['x'], sample['y']
        if torch.rand(1) < self.p:
            rdm = torch.randint(0, 4000, (1,)) 
            rdm_width = torch.randint(self.width_min, self.width_max, (1,)) 
            noise = torch.normal(0, 0.002, (rdm_width,))
            wbt[:, rdm:rdm+rdm_width] = noise.reshape(1,-1)
        sample['x'] = wbt
        return sample
class Bandpass(object):
    "Class to apply a signal processing filter to a dataset"

    def __init__(self, lowcut=140., highcut=1500., order=4):
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

    def __call__(self, sample):
        wbt, label, rate = sample['x'], sample['y'], sample['rate']
        wbt = torch.from_numpy(butter_bandpass_filter(wbt, 
                                                    lowcut=self.lowcut,
                                                    highcut=self.highcut,
                                                    fs=rate,
                                                    order=self.order)).float()
        sample['x'] = wbt
        return sample

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

        sample['x'] = wbt
        return sample

class NormalizeWingbeat(object):
    "Class to normalize a wbt."
    def __call__(self, sample): 
        wbt, label = sample['x'], sample['y']
        wbt = (wbt-wbt.mean()) / wbt.std()
        sample['x'] = wbt
        return sample

class NormalizedPSD(object):

    def __init__(self, norm='l2', scaling='density', window='hanning', nfft=8192, nperseg=5000, noverlap=2500):
        self.norm = norm
        self.scaling = scaling
        self.window = window
        self.nfft = nfft
        self.nperseg = nperseg
        self.noverlap = noverlap
        assert self.norm in ['l1','l2'], "Please provide a valid norm: ['l1','l2']"

    def __call__(self, sample):
        wbt, label, rate = sample['x'], sample['y'], sample['rate']
        _, psd = sg.welch(wbt.numpy().squeeze(), 
                            fs=rate, 
                            scaling=self.scaling, 
                            window=self.window, 
                            nfft=self.nfft, 
                            nperseg=self.nperseg, 
                            noverlap=self.noverlap)
        psd = preprocessing.normalize(psd.reshape(1,-1), norm=self.norm)
        sample['x'] = psd
        return sample

class NormalizedPSDSums(object):

    def __init__(self, norm='l2', scaling='density', window='hanning', nfft=8192, nperseg=5000, noverlap=2500):
        self.norm = norm
        self.scaling = scaling
        self.window = window
        self.nfft = nfft
        self.nperseg = nperseg
        self.noverlap = noverlap
        assert self.norm in ['l1','l2'], "Please provide a valid norm: ['l1','l2']"

    def normalized_psd_sum(self, wbt, rate):
        _, psd = sg.welch(wbt.numpy().squeeze(), 
                    fs=rate, 
                    scaling=self.scaling, 
                    window=self.window, 
                    nfft=self.nfft, 
                    nperseg=self.nperseg, 
                    noverlap=self.noverlap)

        psd = preprocessing.normalize(psd.reshape(1,-1), norm=self.norm)
        return psd.sum()

    def __call__(self, sample):
        wbt, label, rate = sample['x'], sample['y'], sample['rate']

        score1 = self.normalized_psd_sum(wbt.T[:2501], rate)
        score2 = self.normalized_psd_sum(wbt.T[2500:], rate)
        score3 = self.normalized_psd_sum(wbt.T[1250:3750], rate) 

        sample['x'] = min([score1,score2,score3])
        return sample



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
        wbt, label, rate = sample['x'], sample['y'], sample['rate']
        
        if self.setting.startswith('stft'):
            spec = Spectrogram(n_fft=8192, hop_length=5, win_length=600, window_fn=torch.hann_window, power=2, normalized=True)(wbt) # , win_length=20
            if self.setting.startswith('stftcrop'): 
                spec = spec[:,140:1500,:]
            spec = AmplitudeToDB()(spec)
            spec = torch.from_numpy(np.repeat(spec.numpy()[...,np.newaxis],3,0))
            spec = spec[:,:,:,0]
            if self.setting == 'stftcropresize':
                spec = resize2d(spec, (400,295))
        
            if self.setting == 'stftraw':
                sample['x'] = (wbt,spec)
                return sample
            else:
                sample['x'] = spec
                return sample

        elif self.setting.startswith('melstft'):
            # TODO: Something wrong in the output spec. Missing frequencies.
            spec = MelSpectrogram(sample_rate=rate, n_fft=8192, hop_length=25, f_max=3500, win_length=50, window_fn=torch.hann_window, power=2, normalized=True)(wbt)
            spec = AmplitudeToDB()(spec)
            spec = torch.from_numpy(np.repeat(spec.numpy()[...,np.newaxis],3,0))
            spec = spec[:,:,:,0]
        
            if self.setting == 'melstftraw':
                sample['x'] = (wbt,spec)
                return sample
            else:
                sample['x'] = spec
                return sample

        elif self.setting == 'reassigned_stft':
            freqs, times, mags = librosa.reassigned_spectrogram(y=wbt.numpy().squeeze(), sr=rate, 
                                                                n_fft=256, hop_length=42, center=False)
            mags_db = librosa.power_to_db(mags)
            mags_db = np.expand_dims(librosa.power_to_db(mags),axis=0)
            mags_db = torch.from_numpy(np.repeat(mags_db[...,np.newaxis],3,0))
            mags_db = mags_db[:,:,:,0]
            sample['x'] = mags_db
            return sample

        elif self.setting.startswith('psd'):
            sig = wbt.numpy().squeeze()

            nfft = 8192 if len(sig)<=5000 else 65536

            _, psd = sg.welch(sig, fs=rate, scaling='density', window='hanning', nfft=nfft, nperseg=len(sig), noverlap=len(sig)//2)
            if self.setting == 'psdl1':
                psd = preprocessing.normalize(psd.reshape(1,-1), norm='l1')
            elif self.setting == 'psdl2':
                psd = preprocessing.normalize(psd.reshape(1,-1), norm='l2')
            sample['x'] = psd[:,140:1500]
            return sample