
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from scipy import io
from PIL import Image
from typing import Callable, Optional


class GLDv2(Dataset):
    base_folder = ''
    def __init__(self, root: str, name='gld23k', train: bool = True, transform: Optional[Callable] = None):
        # GLD-160k
        # root = "/data/xuyc/datasets/google-landmark"
        self.root = root 
        self.images_path =  root + '/train'
        self.transform = transform

        if name == 'gld160k':
            self.split_path = root + "/landmarks-user-160k"
            if train:
                self.split_path += "/federated_train.csv"
            else:
                self.split_path += "/test.csv"
        elif name == 'gld23k':
            self.split_path = root + "/landmarks-user-23k"
            if train:
                self.split_path += "/mini_gld_train_split.csv"
            else:
                self.split_path += "/mini_gld_test.csv"
        
        infos = pd.read_csv(self.split_path)
        self.image_ids = infos['image_id'].to_list()
        self.targets = infos['class'].to_list()
        # dict_users
        if train:
            dict_users = infos.groupby(by='user_id').groups
            self.dict_users = {k: v.to_list() for k, v in dict_users.items()}
        
    def __getitem__(self, index):
        image_id, label = self.image_ids[index], self.targets[index]
        path = self.images_path + "/%s/%s/%s/%s.jpg" % (image_id[0], image_id[1], image_id[2], image_id)
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
        
    def __len__(self):
        return len(self.infos)

class Emnist(Dataset):
    base_folder = ''
    def __init__(self, root: str, train: bool = True):
        emnist = io.loadmat(root + "emnist/matlab/emnist-letters.mat")
        # load training dataset
        x_train = emnist["dataset"][0][0][0][0][0][0]
        x_train = x_train.astype(np.float32)

        # load training labels
        y_train = emnist["dataset"][0][0][0][0][0][1] - 1 # make first class 0

        # take first 10 classes of letters
        trn_idx = np.where(y_train < 10)[0]

        y_train = y_train[trn_idx]
        x_train = x_train[trn_idx]

        mean_x = np.mean(x_train)
        std_x = np.std(x_train)

        # load test dataset
        x_test = emnist["dataset"][0][0][1][0][0][0]
        x_test = x_test.astype(np.float32)

        # load test labels
        y_test = emnist["dataset"][0][0][1][0][0][1] - 1 # make first class 0

        tst_idx = np.where(y_test < 10)[0]

        y_test = y_test[tst_idx]
        x_test = x_test[tst_idx]
        
        x_train = x_train.reshape((-1, 1, 28, 28))
        x_test  = x_test.reshape((-1, 1, 28, 28))
        
        # normalise train and test features

        if train:
            self.feature = (x_train - mean_x) / std_x
            self.targets = y_train.squeeze()
        else:
            self.feature = (x_test  - mean_x) / std_x
            self.targets = y_test.squeeze()
            
    def __getitem__(self, index):
        return self.feature[index], self.targets[index]
        
    def __len__(self):
        return len(self.feature)

