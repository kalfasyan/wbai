from pathlib import Path

import numpy as np
import psutil
import torch

from utils import get_wingbeat_files, label_func

torch.manual_seed(42)
import os
from configparser import ConfigParser

import pandas as pd
import torch.nn.functional as F
from scipy import signal as sg
from sklearn import preprocessing
from torch._C import Value
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from utils import BASE_DATAPATH, BASE_PROJECTPATH, open_wingbeat

cfg = ConfigParser()
cfg.read(f'{BASE_PROJECTPATH}/config.ini')

clean_lowthresh = float(cfg.get('cleanthresholds','low'))
clean_highthresh = float(cfg.get('cleanthresholds','high'))

num_workers = psutil.cpu_count()
print(f"Available workers: {num_workers}")

SR = 8000
class DataFrameset(Dataset):
    """
    Dataset class that can take a pandas.DataFrame as input.
    """
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.df.loc[idx]
        fname = sample["x"]
        label = sample["y"]
        wbt,rate = open_wingbeat(fname, plot=False, rate=True)
        sample = {'x': wbt, 'y': label, 'path': str(fname), 'idx': idx, 'rate': rate}

        if self.transform:
            sample = self.transform(sample)

        return sample['x'], sample['y'], sample['path'], sample['idx']

class WingbeatsDataset(Dataset):
    """Wingbeats dataset."""

    def __init__(self, dsname, custom_label=[], clean=True, sample=0, transform=None, verbose=True, rpiformat=False):
        """
        Args:
            dsname (string): Dataset Name.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dsname = dsname
        self.verbose = verbose
        self.transform = transform
        self.clean = clean
        self.sample = sample
        self.rpiformat = rpiformat

        if self.clean:
            self.files, self.labels, self.lbl2files, self.paths, self.sums = make_dataset_df(dsname, sample=self.sample, clean=self.clean, verbose=self.verbose)
        else:
            self.files, self.labels, self.lbl2files = make_dataset_df(dsname, sample=self.sample, clean=self.clean, verbose=self.verbose)

        if len(custom_label) == 1:
            self.labels = [custom_label[0] for _ in self.labels]
            if self.verbose:
                print(f"Label(s) changed to {custom_label}")
        else:
            if self.verbose:
                print("No custom label applied.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fname = self.files[idx]
        label = self.labels[idx]
        wbt,rate = open_wingbeat(fname, plot=False, rate=True, rpiformat=self.rpiformat)
        sample = {'x': wbt, 'y': label, 'path': str(fname), 'idx': idx, 'rate': rate}

        if self.transform:
            sample = self.transform(sample)

        return sample['x'], sample['y'], sample['path'], sample['idx'], sample['rate']

    def onehotlabels(self):
        "To be used when selecting a dataset with more than one species"
        le = preprocessing.LabelEncoder()

        self.labels = le.fit_transform(self.labels)
        self.labels = F.one_hot(torch.as_tensor(self.labels))


    def parse_filenames(self, version='1',temp_humd=False, hist_temp=False, hist_humd=False, hist_date=False):
        """
        Since the stored fnames contain metadata, this function gets all these features and 
        constructs a pandas Dataframe with them.
        """
        from utils import np_hist
        
        self.df = pd.concat([pd.Series(list(self.files)), pd.Series(self.labels)], axis=1)
        self.df.columns = ['fnames', 'labels']
        df = self.df
        df.fnames = df.fnames.astype(str)
        df.labels = df.labels.astype(str)
        df['wavnames'] = df['fnames'].apply(lambda x: x.split('/')[-1][:-4])
        # LightGuide sensor version
        if version=='1':                        
            df['date'] = df['wavnames'].apply(lambda x: pd.to_datetime(''.join(x.split('_')[0:2]), 
                                                                        format='F%y%m%d%H%M%S'))
            df['datestr'] = df['date'].apply(lambda x: x.strftime("%Y%m%d"))
            df['date_day'] = df['date'].apply(lambda x: x.day)
            df['date_hour'] = df['date'].apply(lambda x: x.hour)
            df['gain'] = df['wavnames'].apply(lambda x: x.split('_')[3:][1])
            if temp_humd:
                df['temperature'] = pd.to_numeric(df['wavnames'].apply(lambda x: x.split('_')[3:][3] if len(x.split('_')[3:])>=3 else np.nan))
                df['humidity'] = pd.to_numeric(df['wavnames'].apply(lambda x: x.split('_')[3:][5] if len(x.split('_')[3:])>=4 else np.nan))
            if hist_temp:
                np_hist(df, 'temperature')
            if hist_humd:
                np_hist(df, 'humidity')
            if hist_date:
                import matplotlib.pyplot as plt
                df.datestr.sort_values().value_counts()[df.datestr.sort_values().unique()].plot(kind='bar', figsize=(22,10))
                plt.ylabel('Counts of signals')
            self.df_info = df
        # Fresnel sensor version
        elif version=='2':
            print('VERSION 2')
            df['date'] = df['wavnames'].apply(lambda x: pd.to_datetime(''.join(x.split('_')[1]), 
                                                                        format='%Y%m%d%H%M%S'))
            df['datestr'] = df['date'].apply(lambda x: x.strftime("%Y%m%d"))
            df['date_day'] = df['date'].apply(lambda x: x.day)
            df['date_hour'] = df['date'].apply(lambda x: x.hour)
            df['index'] = df['wavnames'].apply(lambda x: x.split('_')[2])
            if temp_humd:
                df['temperature'] = pd.to_numeric(df['wavnames'].apply(lambda x: x.split('_')[3][4:]))
                df['humidity'] = pd.to_numeric(df['wavnames'].apply(lambda x: x.split('_')[4][3:]))
            if hist_temp:
                np_hist(df, 'temperature')
            if hist_humd:
                np_hist(df, 'humidity')
            if hist_date:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10,6))
                df.date.hist(xrot=45)
                plt.ylabel('Counts of signals')
            self.df_info = df
        else:
            print("No sensor features collected. Select valid version")
        self.sensor_features = True

    def plot_daterange(self, start='', end='', figx=8, figy=26, linewidth=4):
        """
        Method to plot a histogram within a date range (starting from earliest datapoint to latest)
        """
        assert hasattr(self, 'sensor_features'), "Parse filenames first to generate features."

        import matplotlib.pyplot as plt

        from utils import get_datestr_range

        if '' in {start, end}:
            start = self.df_info.datestr.sort_values().iloc[0]
            end = self.df_info.datestr.sort_values().iloc[-1] 
        all_dates = get_datestr_range(start=start,end=end)

        hist_dict = self.df_info.datestr.value_counts().to_dict()
        mydict = {}
        for d in all_dates:
            if d not in list(hist_dict.keys()):
                mydict[d] = 0
            else:
                mydict[d] = hist_dict[d]

        series = pd.Series(mydict)
        ax = series.sort_index().plot(xticks=range(0,series.shape[0]), figsize=(figy,figx), rot=90, linewidth=linewidth)
        ax.set_xticklabels(series.index);



def clean_wingbeatsdataset_inds(name="Melanogaster_RL/Y", filtered=True, low_thresh=clean_lowthresh, high_thresh=clean_highthresh, batch_size=30, num_workers=num_workers):
    """
    Helper function to clean a WingbeatsDataset. It is used in its 'clean' method.
    """
    from transforms import Bandpass, TransformWingbeat

    # Checking if the indice are alreaddy available for the given dataset and threshold
    fname = f"{BASE_PROJECTPATH}/data_created/{name.replace('/','-').replace(' ', '')}_thL{low_thresh}_thH{high_thresh}_cleaned"
    if os.path.isfile(f"{fname}.npy") and os.path.isfile(f"{fname}.csv"):
        return np.load(f"{fname}.npy").tolist(), \
                pd.read_csv(f"{fname}.csv")['fnames'].tolist(), \
                np.load(f"{fname}_sums.npy").tolist()

    # Dataset of l2-normalized Power-Spectral-Densities
    if filtered:
        dataset = WingbeatsDataset(name, transform=transforms.Compose([Bandpass(), TransformWingbeat(setting='psdl2')]), clean=False, verbose=False)
    else:
        dataset = WingbeatsDataset(name, transform=TransformWingbeat(setting='psdl2'), clean=False, verbose=False)
    dloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    # Creating empty tensor to populate with valid wingbeat indice
    all_inds_valid = torch.tensor([])
    all_sums_valid = torch.tensor([])
    all_paths_valid = []

    for i, (x_batch,y_batch,path_batch,idx,rate) in enumerate(tqdm(dloader, desc=f"Cleaning dataset {name}..\t")):
        # Using the sums of l2-normalized PSDs
        xb_sums = x_batch.squeeze().sum(axis=1)
        # We set a threshold to get valid wingbeats, all invalid set to zero
        xb_valid = torch.where((low_thresh<xb_sums) & (xb_sums<high_thresh),xb_sums,torch.zeros(xb_sums.shape))
        inds_valid_batch = xb_valid.nonzero().squeeze()
        # Retrieving the indice, sums and paths of nonzero (valid wingbeats)
        sums_valid = xb_valid[inds_valid_batch]
        inds_valid = idx[inds_valid_batch]
        paths_valid = np.array(path_batch)[inds_valid_batch].tolist()
        # Concatenating to indice to 'all_inds_valid'
        if inds_valid.dim() > 0:
            all_inds_valid = torch.cat((all_inds_valid, inds_valid),0)
            all_sums_valid = torch.cat((all_sums_valid, sums_valid),0)
            all_paths_valid += paths_valid
        elif inds_valid.dim() == 0: # in case the torch tensor is 0-dim
            all_inds_valid = torch.cat((all_inds_valid, inds_valid.view(1)))
            all_sums_valid = torch.cat((all_sums_valid, sums_valid.view(1)))
            all_paths_valid += [paths_valid] 

    list_all_inds_valid = list(all_inds_valid)
    list_all_sums_valid = list(all_sums_valid)

    if len(all_paths_valid):
        # Saving indice to avoid recalculating the above all the time
        np.save(fname, list_all_inds_valid)
        np.save(f"{fname}_sums", list_all_sums_valid)
        pd.DataFrame({"fnames": all_paths_valid}).to_csv(f"{fname}.csv", index=False)
    else:
        raise ValueError("List of clean wingbeats returned empty.")

    return list_all_inds_valid, all_paths_valid, all_sums_valid


def make_dataset_df(dsname, clean=False, sample=0, verbose=False):
    datadir = Path(BASE_DATAPATH/dsname)

    files = get_wingbeat_files(dsname)

    if clean:
        inds, paths, sums = clean_wingbeatsdataset_inds(name=dsname)
        files = files[inds]

    if sample > 0:
        sampled_inds = np.random.choice(range(len(files)), sample, replace=True)
        if sample > len(files):
            print("Asked to sample more than the number of files found.")
        files = files[sampled_inds]

    labels = pd.Series(files).apply(lambda x: label_func(x)).tolist() #list(files.map(label_func))
    if verbose:
        print(f"Found {len(list(files))} in dataset: {dsname}, ", end="")
        print(f"and {len(set(labels))} label(s): {np.unique(labels)}")
    
    lbl2files = {l: [f for f in files if label_func(f) ==l] for l in list(set(labels))}

    if clean:
        return files, labels, lbl2files, paths, sums 
    else: 
        return files,labels, lbl2files


def get_clean_wingbeatsdataset_filenames(dset_names=[]):
    all_fnames = []
    for dsname in dset_names:
        d = WingbeatsDataset(dsname=dsname)
        d.clean()
        all_fnames += d.clean_fnames
    return all_fnames

def normalized_psd_sum(sig):
    _,p = sg.welch(sig, fs=SR, scaling='density', window='hanning', nfft=8192, nperseg=256, noverlap=128+64)
    p = preprocessing.normalize(p.reshape(1,-1), norm='l2').T.squeeze()
    return p.sum()

def evaluate_data(df, model, batch_size=32, transforms_list=[]):
    from utils import test_model, worker_init_fn
    if not len(transforms_list):
        from transforms import Bandpass
        transforms_list = [Bandpass(lowcut=140, highcut=1500)]
    X_test, y_test = df.iloc[:,0], df.iloc[:,1]
    test_dataset = DataFrameset(pd.concat([X_test, y_test], axis=1), transform=transforms.Compose(transforms_list))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    y_true, y_pred, x_batch = test_model(model,test_dataloader, test_dataset)

    print(pd.Series(y_pred).value_counts())