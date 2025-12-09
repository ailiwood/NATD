import os.path
from glob import iglob
from tools.file_name_cic2018 import parse_cic_filename
from tools.center_crop import device
from tools.Dst_ip import pross_dst_src_ip
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tools.file_name_cic2019 import parse_dataset_name
from tools.split_flow_id import split_flow_id


class Cic_2018_Dataset(Dataset):
    def __init__(self, data_dir="Data", data_name="cic2019_NetBIOS", split='train', test_size=0.1, random_state=42, fold_indices=None):
        assert data_name in ["cic2018_02_14_2018", "cic2018_02_15_2018", "cic2018_02_16_2018","cic2018_02_20_2018", "cic2018_02_21_2018",
                             "cic2018_02_22_2018", "cic2018_02_23_2018", "cic2018_02_28_2018", "cic2018_03_01_2018", "cic2018_03_02_2018"], "Data_name Error!"
        name, csv_name = parse_cic_filename(data_name)
        data_dir = os.path.join(data_dir, name, csv_name)
        print("=============== Data loading 🚗... ===============")
        df = pd.concat((pd.read_csv(f, low_memory=False) for f in iglob(data_dir, recursive=True)), ignore_index=True)
        print(f"=============== Data_{split} loading has completed ✅ ===============")

        for col in ['Unnamed: 0', 'Timestamp', 'Flow ID']:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)

        for f in df.columns:
            if f == 'Label':
                df[f] = LabelEncoder().fit_transform(df[f])
            if f == 'Dst IP':
                df = pross_dst_src_ip(df, f)
            if f == 'Src IP':
                df = pross_dst_src_ip(df, f)

        df.fillna(0, inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)

        x = df.drop(columns=['Label'])
        y = df['Label']
        self.num_classes = len(np.unique(y))
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

        if split == 'test':
            self.x_data = x_test
            self.y_data = y_test
        elif split == 'train' and fold_indices is not None:
            train_idx, val_idx = fold_indices
            self.x_data = x_train.iloc[train_idx]
            self.y_data = y_train.iloc[train_idx]
        elif split == 'val' and fold_indices is not None:
            train_idx, val_idx = fold_indices
            self.x_data = x_train.iloc[val_idx]
            self.y_data = y_train.iloc[val_idx]
        else:
            self.x_data = x_train
            self.y_data = y_train

        scaler = MinMaxScaler()
        self.x_data = scaler.fit_transform(self.x_data)
        self.x_data = self.x_data[:, :, None]

        self.x_data = torch.tensor(self.x_data, dtype=torch.float32).permute(0, 2, 1).to(device)
        self.y_data = torch.tensor(self.y_data.values, dtype=torch.long).to(device)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return len(self.x_data)


class UNSWNB15(Dataset):
    def __init__(self,
                 csv_path: str = "Data/unsw_nb15/unsw_nb15.csv",
                 split: str = "train",
                 train_ratio: float = 0.85,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.05,
                 random_state: int = 42,
                 device: str = "cpu"):
        super().__init__()
        assert split in {"train", "val", "test"}, "split error!"
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, ""

        df = pd.read_csv(csv_path, low_memory=False)
        print("=============== Data loading 🚗... ===============")
        print(f"=============== Data_{split} loading has completed ✅ ===============")
        for col in ['id', 'proto', 'service', 'state', 'attack_cat']:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)

        if 'label' in df.columns:
            df['label'] = LabelEncoder().fit_transform(df['label'])
        else:
            raise ValueError("label error!")

        df.fillna(0, inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)

        X = df.drop(columns=['label'])
        y = df['label'].values
        self.num_classes = len(np.unique(y))

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_ratio,
            random_state=random_state,
            stratify=y
        )
        val_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_adjusted,
            random_state=random_state,
            stratify=y_temp
        )

        if split == "train":
            X_used, y_used = X_train, y_train
        elif split == "val":
            X_used, y_used = X_val, y_val
        else:
            X_used, y_used = X_test, y_test

        scaler = MinMaxScaler()
        if split == "train":
            X_scaled = scaler.fit_transform(X_used)
        else:
            X_scaled = scaler.fit_transform(X_used)
        X_scaled = X_scaled[:, :, None]
        self.x_data = torch.tensor(X_scaled, dtype=torch.float32).permute(0, 2, 1).to(device)
        self.y_data = torch.tensor(y_used, dtype=torch.long).to(device)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return len(self.x_data)


class Cic_2019_dataset(Dataset):
    def __init__(self,
                 data_dir: str = "Data",
                 data_name: str = "cic2019_LDAP",
                 split: str = "train",
                 train_ratio: float = 0.85,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.05,
                 random_state: int = 42,
                 device: str = "cpu"):
        super().__init__()
        assert split in {"train", "val", "test"}, "split error!"
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "ratio error!"
        prefix, filename = parse_dataset_name(data_name)
        data_path = os.path.join(data_dir, prefix, filename)
        df = pd.read_csv(data_path, encoding='utf-8-sig', low_memory=False)
        df.columns = df.columns.str.strip()
        print("=============== Data loading 🚗... ===============")
        print(f"======= Data_{split} loading has completed ✅ =======")
        df = split_flow_id(df)
        drop_cols = ['Source IP', 'Destination IP', 'Timestamp', "Flow ID", "Flow ID1", "Flow ID2"]
        df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)

        if "Label" in df.columns:
            df["Label"] = LabelEncoder().fit_transform(df["Label"])
        else:
            raise ValueError("Label error!")

        df.fillna(0, inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df = df.map(lambda x: 0 if isinstance(x, str) else x)
        X = df.drop(columns=['Label'])
        y = df['Label'].values
        self.num_classes = len(np.unique(y))

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_ratio,
            random_state=random_state, stratify=y
        )
        val_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_adjusted,
            random_state=random_state, stratify=y_temp
        )
        if split == "train":
            X_used, y_used = X_train, y_train
        elif split == "val":
            X_used, y_used = X_val, y_val
        else:
            X_used, y_used = X_test, y_test

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_used)
        X_scaled = X_scaled[:, :, None]
        self.x_data = torch.tensor(X_scaled, dtype=torch.float32).permute(0, 2, 1).to(device)
        self.y_data = torch.tensor(y_used, dtype=torch.long).to(device)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return len(self.x_data)
