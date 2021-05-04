import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transforms import Bandpass, NormalizedPSDSums, TransformWingbeat
import pandas as pd
from datasets import WingbeatsDataset
from utils import open_wingbeat, get_wbt_duration
import matplotlib.pyplot as plt

class WingbeatDatasetProfiler(object):

    def __init__(self, dsname, bandpass_high=1500., rollwindow=150, noisethresh=0.003):
        self.dsname = dsname
        self.bandpass_high =bandpass_high
        self.rollwindow = rollwindow
        self.noisethresh = noisethresh
        self.wbts = WingbeatsDataset(dsname=self.dsname, 
                                                verbose=False,
                                                custom_label=[0], 
                                                clean=False, 
                                                transform=transforms.Compose([Bandpass(highcut=self.bandpass_high)]))
        self.psds = WingbeatsDataset(dsname=self.dsname, 
                                                verbose=False,
                                                custom_label=[0], 
                                                clean=False, 
                                                transform=transforms.Compose([Bandpass(highcut=self.bandpass_high), 
                                                                            TransformWingbeat(setting='psdl2')]))
        self.stfts = WingbeatsDataset(dsname=self.dsname, 
                                                verbose=False,
                                                custom_label=[0], 
                                                clean=False, 
                                                transform=transforms.Compose([Bandpass(highcut=self.bandpass_high), 
                                                                            TransformWingbeat(setting='stft')]))

    def get_dataset_df(self):

        d = WingbeatsDataset(dsname=self.dsname, custom_label=[1], clean=False, transform=transforms.Compose([Bandpass(highcut=self.bandpass_high), NormalizedPSDSums(norm='l2')]))
        dloader = DataLoader(d, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        sums,paths,labels,idx = [],[],[],[]
        for s,l,p,i,_ in tqdm(dloader, total=len(d)//32):
            sums.extend(s)
            paths.extend(p)
            idx.extend(i)
            labels.extend(l)
        df = pd.DataFrame({"x": paths, "score": torch.tensor(sums)})
        df['duration'] = df.x.apply(lambda x: get_wbt_duration(x, window=self.rollwindow, th=self.noisethresh))
        df['sum'] = df.x.apply(lambda x: open_wingbeat(x).abs().sum())
        df['idx'] = idx
        df['y'] = labels
        self.df = df
        return df

    def plot_random_psds(self, df=pd.DataFrame(), noaxis=True):
        if not len(df):
            df = self.df

        plt.figure(figsize=(20,12))
        for i in tqdm(range(20)):
            plt.subplot(4,5,i+1)
            sig = self.psds[df.iloc[i].name][0].squeeze()[:1500]
            plt.plot(sig.T);
            plt.title(df.loc[df.iloc[i].name].score)
            if noaxis:
                plt.axis('off')
            plt.ylim(0,.18)
