import torch
from torch.utils.data import Dataset, ConcatDataset
import torch.nn.functional as F
from sklearn import preprocessing
from utils import open_wingbeat, make_dataset_df, butter_bandpass_filter

class WingbeatsDataset(Dataset):
    """Wingbeats dataset."""

    def __init__(self, dsname, custom_label='', transform=None):
        """
        Args:
            dsname (string): Dataset Name.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dsname = dsname
        self.files, self.labels, self.lbl2files = make_dataset_df(dsname, verbose=True)
        if custom_label is not None:
            self.labels = [custom_label for _ in self.labels]  
            print(f"Label(s) changed to {custom_label}")
        else: 
            pass
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fname = self.files[idx]
        label = self.labels[idx]
        wingbeat = open_wingbeat(fname, plot=False)
        sample = {'x': wingbeat, 'y': label, 'path': str(fname)}

        if self.transform:
            sample = self.transform(sample)

        return sample['x'], sample['y'], sample['path']

    def onehotlabels(self):
        le = preprocessing.LabelEncoder()
        self.labels = le.fit_transform(self.labels)
        self.labels = F.one_hot(torch.as_tensor(self.labels))

class FilterWingbeat(object):
    def __init__(self, filter_type='bandpass'):
        self.filter_type = filter_type

    def __call__(self, sample):
        wingbeat = sample['x']
        label = sample['y']
        if self.filter_type == 'bandpass':
            wingbeat = torch.from_numpy(butter_bandpass_filter(wingbeat)).float()
        else:
            raise NotImplementedError('!')
        return {'x': wingbeat, 'y': label, 'path': sample['path']}
