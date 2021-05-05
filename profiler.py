import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transforms import Bandpass, NormalizedPSDSums, TransformWingbeat
import pandas as pd
from datasets import WingbeatsDataset
from utils import open_wingbeat, get_wbt_duration
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
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
        self.get_dataset_df();

    def get_dataset_df(self, batch_size=16):

        d = WingbeatsDataset(dsname=self.dsname, custom_label=[1], clean=False, transform=transforms.Compose([Bandpass(highcut=self.bandpass_high), NormalizedPSDSums(norm='l2')]))
        dloader = DataLoader(d, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        sums,paths,labels,idx = [],[],[],[]
        for s,l,p,i,_ in tqdm(dloader, total=len(d)//batch_size, desc='Collecting all data from the dataloader..'):
            sums.extend(s)
            paths.extend(p)
            idx.extend(i)
            labels.extend(l)
        print("Creating a pandas Dataframe with file-paths, clean-scores, duration, sums of abs values, indice and labels..")        
        df = pd.DataFrame({"x": paths, "score": torch.tensor(sums)})
        df['duration'] = df.x.apply(lambda x: get_wbt_duration(x, window=self.rollwindow, th=self.noisethresh))
        df['sum'] = df.x.apply(lambda x: open_wingbeat(x).abs().sum())
        df['max'] = df.x.apply(lambda x: open_wingbeat(x).abs().max())
        df['idx'] = idx
        df['y'] = labels
        df['fname'] = df.x.apply(lambda x: x.split('/')[-1][:-4])
        df['date'] = df.fname.apply(lambda x: pd.to_datetime(''.join(x.split('_')[0:2]), format='F%y%m%d%H%M%S'))
        df['datestr'] = df['date'].apply(lambda x: x.strftime("%Y%m%d"))
        df['datehourstr'] = df['date'].apply(lambda x: x.strftime("%y%m%d_%H"))
        self.df = df
        print('Finished.')
        return df

    def wbts_inclean_range(self, low=8, high=22):
        assert hasattr(self, 'df'), "Create the dataset dataframe first"
        sub = self.df[(self.df.score > low) & (self.df.score < high)]
        print(len(sub))
        return sub

    def boxplot(self, col='score', df=pd.DataFrame(), showfliers=False, ylim=()):
        if not len(df):
            df = self.df.sample(len(self.df), replace=False)
        assert col in df.columns, f"{col} not in df.columns"

        sub = pd.DataFrame(df[col], columns=[col])
        plt.figure()
        sub.boxplot(showfliers=showfliers)
        if len(ylim):
            plt.ylim(ylim)
        plt.show()
        return sub

    def plot_random_wbts(self, df=pd.DataFrame(), noaxis=True, ylim=.06):
        if not len(df):
            df = self.df.sample(len(self.df), replace=False)

        plt.figure(figsize=(20,12))
        for i in tqdm(range(20)):
            plt.subplot(4,5,i+1)
            sig = self.wbts[df.iloc[i].name][0].squeeze()
            plt.plot(sig.T);
            plt.title(df.loc[df.iloc[i].name].score, y=0.9)
            if noaxis:
                plt.axis('off')
            plt.ylim(-1*ylim, ylim)

    def plot_random_psds(self, df=pd.DataFrame(), noaxis=True):
        if not len(df):
            df = self.df.sample(len(self.df), replace=False)

        plt.figure(figsize=(20,12))
        for i in tqdm(range(20)):
            plt.subplot(4,5,i+1)
            sig = self.psds[df.iloc[i].name][0].squeeze()[:1500]
            plt.plot(sig.T);
            plt.title(df.loc[df.iloc[i].name].score, y=0.9)
            if noaxis:
                plt.axis('off')
            plt.ylim(0,.18)
    
    def plot_random_stfts(self, df=pd.DataFrame(), noaxis=True):
        if not len(df):
            df = self.df.sample(len(self.df), replace=False)

        plt.figure(figsize=(20,12))
        for i in tqdm(range(20)):
            plt.subplot(4,5,i+1)
            stft = torch.flipud(self.stfts[df.iloc[i].name][0][0])
            plt.imshow(stft);
            plt.title(df.loc[df.iloc[i].name].score, y=0.98)
            if noaxis:
                plt.axis('off')

    def plot_daterange(self, start='', end='', figx=8, figy=26, linewidth=4):
        """
        Method to plot a histogram within a date range (starting from earliest datapoint to latest)
        """

        import matplotlib.pyplot as plt

        from utils import get_datestr_range

        if '' in {start, end}:
            start = self.df.datestr.sort_values().iloc[0]
            end = self.df.datestr.sort_values().iloc[-1] 
        all_dates = get_datestr_range(start=start,end=end)

        hist_dict = self.df.datestr.value_counts().to_dict()
        mydict = {}
        for d in all_dates:
            if d not in list(hist_dict.keys()):
                mydict[d] = 0
            else:
                mydict[d] = hist_dict[d]

        series = pd.Series(mydict)
        ax = series.sort_index().plot(xticks=range(0,series.shape[0]), figsize=(figy,figx), rot=90, linewidth=linewidth)
        ax.set_xticklabels(series.index);