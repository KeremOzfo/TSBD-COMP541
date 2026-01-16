import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
import math

from .uea import subsample, interpolate_missing, Normalizer

warnings.filterwarnings('ignore')


# ===========================
#   BASE UEA DATASET
# ===========================
class UEAloader(Dataset):

    def __init__(self, root_path, file_list=None, limit_size=None, flag=None):
        self.root_path = root_path
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()

        # dataset size limit (debug)
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:
                limit_size = int(limit_size * len(self.all_IDs))

            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # features = all columns
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # normalization
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)

    def load_all(self, root_path, file_list=None, flag=None):
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]

        if flag is not None:
            flag = flag.upper()
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))

        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            raise Exception("No .ts files found in {}".format(root_path))

        all_df, labels_df = self.load_single(input_paths[0])
        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                   replace_missing_vals_with='NaN')

        labels = pd.Series(labels, dtype="category")
        labels_df = pd.DataFrame(labels.cat.codes, dtype=np.int8)
        self.class_names = labels.cat.categories

        # handle unequal lengths
        lengths = df.applymap(lambda x: len(x)).values

        if np.sum(np.abs(lengths - np.expand_dims(lengths[:, 0], -1))) > 0:
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        if np.sum(np.abs(lengths - np.expand_dims(lengths[0, :], 0))) > 0:
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        self.num_cls = np.max(labels_df.values) + 1

        # flatten hierarchical dataframe into (seq_len, feat) format
        df = pd.concat(
            (
                pd.DataFrame({col: df.loc[row, col] for col in df.columns})
                .reset_index(drop=True)
                .set_index(pd.Series(lengths[row, 0] * [row]))
                for row in range(df.shape[0])
            ),
            axis=0
        )

        df = df.groupby(df.index).transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        return case  # UEA normalizer already applied

    def __getitem__(self, ind):
        return self.instance_norm(torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values)), \
               torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values)

    def __len__(self):
        return len(self.all_IDs)


# ===========================
#      BACKDOOR LOADER
# ===========================
class UEAloader_bd(UEAloader):
    def __init__(self, root_path, bd_model, file_list=None, limit_size=None,
                 flag=None, poision_rate=0.1, silent_poision=False,
                 target_label=0, max_len=0, enc_in=0):

        super().__init__(root_path, file_list, limit_size, flag)
        self.max_len = max_len
        self.enc_in = enc_in
        self.G = bd_model.to('cpu')
        self.G.eval()

        # choose poisoned indices
        self.total_bd = math.ceil(len(self.all_IDs) * poision_rate)
        if self.total_bd % 2 != 0 and silent_poision:
            self.total_bd += 1

        self.bd_inds = np.random.choice(self.all_IDs, self.total_bd, replace=False)
        self.target_label = target_label
        self.silent_bd_set = []

        if silent_poision:
            self.bd_inds, self.silent_bd_set = np.array_split(self.bd_inds, 2)

    def __getitem__(self, ind):
        x = self.instance_norm(torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values))
        y = torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values)

        y_bd = torch.ones_like(y) * self.target_label

        # padded input
        x_pad = torch.zeros(self.max_len, self.enc_in)
        end = x.shape[0]
        x_pad[:end] = x[:end]
        x_pad = x_pad.unsqueeze(0).float()

        if ind in self.bd_inds:
            y = y_bd
            t, t_clip = self.G(x_pad, None, None, None, y_bd.unsqueeze(0))
            x = (x_pad + t_clip).squeeze(0)

        elif ind in self.silent_bd_set:
            t, t_clip = self.G(x_pad, None, None, None, y_bd.unsqueeze(0))
            x = (x_pad + t_clip).squeeze(0)

        return x, y


# ===========================
#   BACKDOOR – LABEL ONLY
# ===========================
class UEAloader_bd2(UEAloader):
    def __init__(self, root_path, bd_model, file_list=None, limit_size=None,
                 flag=None, poision_rate=0.1, silent_poision=False,
                 target_label=0):

        super().__init__(root_path, file_list, limit_size, flag)

        self.G = bd_model.to("cpu")
        self.G.eval()

        self.total_bd = math.ceil(len(self.all_IDs) * poision_rate)
        if self.total_bd % 2 != 0 and silent_poision:
            self.total_bd += 1

        self.bd_inds = np.random.choice(self.all_IDs, self.total_bd, replace=False)
        self.target_label = target_label
        self.silent_bd_set = []

        if silent_poision:
            self.bd_inds, self.silent_bd_set = np.array_split(self.bd_inds, 2)

    def __getitem__(self, ind):
        x = self.instance_norm(torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values))
        y = torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values)

        y_bd = torch.ones_like(y) * self.target_label
        is_bd = torch.zeros_like(y)

        if ind in self.bd_inds:
            y = y_bd
            is_bd[:] = 1
        elif ind in self.silent_bd_set:
            is_bd[:] = 1

        return x, y, is_bd
