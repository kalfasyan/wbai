import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transforms import Bandpass, NormalizedPSD, NormalizedPSDSums, TransformWingbeat
import pandas as pd
from datasets import WingbeatsDataset
from utils import open_wingbeat, get_wbt_duration, show_peaks
import time
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
class WingbeatDatasetProfiler(object):

    def __init__(self, dsname, bandpass_low=140., bandpass_high=1500., rollwindow=150, noisethresh=0.003, rpiformat=False, custom_label=[0],
                        height=0.04, prominence=0.001, width=1, distance=5):
        self.dsname = dsname
        self.bandpass_high =bandpass_high
        self.rollwindow = rollwindow
        self.noisethresh = noisethresh
        self.rpiformat = rpiformat
        self.custom_label = custom_label
        self.height = height
        self.prominence = prominence
        self.width = width
        self.distance = distance
        self.transforms = []
        self.wbts = WingbeatsDataset(dsname=self.dsname, 
                                                verbose=False,
                                                custom_label=self.custom_label, 
                                                clean=False, 
                                                transform=transforms.Compose([Bandpass(lowcut=bandpass_low, highcut=self.bandpass_high)]),
                                                rpiformat=self.rpiformat)
        self.psds = WingbeatsDataset(dsname=self.dsname, 
                                                verbose=False,
                                                custom_label=self.custom_label, 
                                                clean=False, 
                                                transform=transforms.Compose([Bandpass(lowcut=bandpass_low, highcut=self.bandpass_high), 
                                                                            TransformWingbeat(setting='psdl2')]),
                                                rpiformat=self.rpiformat)
        self.stfts = WingbeatsDataset(dsname=self.dsname, 
                                                verbose=False,
                                                custom_label=self.custom_label, 
                                                clean=False, 
                                                transform=transforms.Compose([Bandpass(lowcut=bandpass_low, highcut=self.bandpass_high), 
                                                                            TransformWingbeat(setting='stftcropresize')]),
                                                rpiformat=self.rpiformat)
        self.get_dataset_df(height=self.height, prominence=self.prominence, width=self.width, distance=self.distance);

    def get_dataset_df(self, batch_size=16, height=0.04, prominence=0.001, width=1, distance=5):

        d = WingbeatsDataset(dsname=self.dsname, 
                            custom_label=self.custom_label, 
                            clean=False, 
                            transform=transforms.Compose([Bandpass(highcut=self.bandpass_high), 
                                                            NormalizedPSD(norm='l2', scaling='density', window='hanning', nfft=8192, nperseg=5000, noverlap=2500)]))

        dloader = DataLoader(d, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        sums,paths,labels,idx,peaks,peaksxtra = [],[],[],[],[], []
        time.sleep(.5)
        for x,l,p,i,_ in tqdm(dloader, total=len(d)//batch_size, desc='Collecting all data from the dataloader..'):
            paths.extend(p)
            idx.extend(i.numpy())
            labels.extend(l.numpy())
            for sig in x:
                sums.extend([sig.squeeze().sum()])
                p, _ = find_peaks(sig.squeeze(), height=height, prominence=prominence, width=width, distance=distance)
                peaks.extend([len(p)])
                peaksxtra.extend([p])


        print("Creating a pandas Dataframe with file-paths, clean-scores, duration, sums of abs values, indice and labels..")        
        df = pd.DataFrame({"x": paths, "y": labels, "idx": idx, "score": torch.tensor(sums), "peaks": peaks, "peaksxtra": peaksxtra})
        print("Duration")
        df['duration'] = df.x.apply(lambda x: get_wbt_duration(x, window=self.rollwindow, th=self.noisethresh))
        print("Sum..")
        df['sum'] = df.x.apply(lambda x: open_wingbeat(x).abs().sum().numpy())
        print("Max..")
        df['max'] = df.x.apply(lambda x: open_wingbeat(x).abs().max().numpy())
        print("Filename..")
        df['fname'] = df.x.apply(lambda x: x.split('/')[-1][:-4])
        print("Date..")
        if self.rpiformat:
            df['date'] = df.fname.apply(lambda x: pd.to_datetime(''.join(x.split('_')[0:2])))
        else:
            df['date'] = df.fname.apply(lambda x: pd.to_datetime(''.join(x.split('_')[0:2]), format=f'F%y%m%d%H%M%S'))
        print("Date string..")
        df['datestr'] = df['date'].apply(lambda x: x.strftime("%Y%m%d"))
        print("Datehour string..")
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

    def plot_random_wbts(self, df=pd.DataFrame(), noaxis=True, ylim=.06, title=''):
        if not len(df):
            df = self.df.sample(len(self.df), replace=False)

        if self.rpiformat:
            plt.figure(figsize=(30,12))
        else:
            plt.figure(figsize=(20,12))

        for i in tqdm(range(20)):
            plt.subplot(4,5,i+1)
            sig = self.wbts[df.iloc[i].name][0].squeeze()
            plt.plot(sig.T);
            score = df.loc[df.iloc[i].name].score
            duration = df.loc[df.iloc[i].name].duration
            filename = df.loc[df.iloc[i].name].x.split('/')[-1]
            if title == 'filename':
                plt.title(filename)
            else:
                plt.title(f"score:{score:.0f}, duration:{duration:.0f}", y=0.9)
            if noaxis:
                plt.axis('off')
            plt.ylim(-1*ylim, ylim)

    def plot_random_psds(self, df=pd.DataFrame(), noaxis=True, title='score'):
        if not len(df):
            df = self.df.sample(len(self.df), replace=False)

        if self.rpiformat:
            plt.figure(figsize=(30,12))
        else:
            plt.figure(figsize=(20,12))

        for i in tqdm(range(20)):
            plt.subplot(4,5,i+1)
            sig = self.psds[df.iloc[i].name][0].squeeze()

            # plt.plot(sig.T);
            score = df.loc[df.iloc[i].name].score
            duration = df.loc[df.iloc[i].name].duration
            filename = df.loc[df.iloc[i].name].x.split('/')[-1]
            peaks = df.loc[df.iloc[i].name].peaks

            show_peaks(sig);

            if title == 'filename':
                plt.title(filename)
            else:
                plt.title(f"score:{score:.0f}, duration:{duration:.0f}, peaks:{peaks}", y=0.9)
            if noaxis:
                plt.axis('off')
            plt.ylim(0,.5)
    
    def plot_random_stfts(self, df=pd.DataFrame(), noaxis=True):
        if not len(df):
            df = self.df.sample(len(self.df), replace=False)

        if self.rpiformat:
            plt.figure(figsize=(30,12))
        else:
            plt.figure(figsize=(20,12))
        for i in tqdm(range(20)):
            plt.subplot(4,5,i+1)
            stft = torch.flipud(self.stfts[df.iloc[i].name][0][0])
            plt.imshow(stft,interpolation='nearest', aspect='auto');
            score = df.loc[df.iloc[i].name].score
            duration = df.loc[df.iloc[i].name].duration
            plt.title(f"score:{score:.0f}, duration:{duration:.0f}", y=0.9)
            if noaxis:
                plt.axis('off')

    def plot_daterange(self, df=pd.DataFrame(), start='', end='', figx=8, figy=26, linewidth=4):
        """
        Method to plot a histogram within a date range (starting from earliest datapoint to latest)
        """

        import matplotlib.pyplot as plt

        from utils import get_datestr_range
    
        if not len(df):
            df = self.df.sample(len(self.df), replace=False)

        if '' in {start, end}:
            start = df.datestr.sort_values().iloc[0]
            end = df.datestr.sort_values().iloc[-1] 
        all_dates = get_datestr_range(start=start,end=end)

        hist_dict = df.datestr.value_counts().to_dict()
        mydict = {}
        for d in all_dates:
            if d not in list(hist_dict.keys()):
                mydict[d] = 0
            else:
                mydict[d] = hist_dict[d]

        series = pd.Series(mydict)
        ax = series.sort_index().plot(xticks=range(0,series.shape[0]), figsize=(figy,figx), rot=90, linewidth=linewidth)
        ax.set_xticklabels(series.index);
