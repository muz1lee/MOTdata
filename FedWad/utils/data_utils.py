import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
from utils.constants import *
data_dir=DEFAULT_DATA_DIR

from utils.log_tools import fprint
from utils.global_utils import load_pkl, save_pkl, load_json, set_cuda
from utils.dataset import Emnist
from utils.client_sampling import iid, imitate_sampling, noniid_label, noniid_dir, one_label





trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))]) # channel num = 1
trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                                std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])


def set_data_loader(datasets: dict, args):
    data_loader = {}
    for name, dataset in datasets.items():
        if name == 'train':
            data_loader[name] = DataLoader(dataset, batch_size=args.bs_train, shuffle=True)
        elif name in ['test', 'valid']:
            data_loader[name] = DataLoader(dataset, batch_size=args.bs_test, shuffle=False)
        elif name == 'gtest':
            data_loader[name] = DataLoader(dataset, batch_size=args.bs_test, shuffle=False)
    return data_loader


class Cifar100Wrapper(Dataset):
    def __init__(self, data_dir:str, transform, train:bool=True, download:bool=True, chose_cls:list=None, data_ratio:float=1.0, merged_cls_num:int=100):
        """
            quantity
        """
        self.dataset = datasets.CIFAR100(data_dir, train=train, download=download, transform=transform)
        idxs_dict = {}  # {'label': 'dataset samples' id'}
        targets = self.dataset.targets
        for i in range(len(targets)):
            label = targets[i]
            if label not in idxs_dict.keys():
                idxs_dict[label] = []
            idxs_dict[label].append(i)
        
        # dataset.classes  can get all fine class name.
        self.target_fine  = self.dataset.targets
        info = self.unpickle(os.path.join(data_dir,"cifar-100-python/train"))
        self.targets_coarse = [i for i in info[b'coarse_labels']]
        info_meta = self.unpickle(os.path.join(data_dir,"cifar-100-python/meta"))
        self.fine_class_names = self.dataset.classes
        self.coarse_class_names = [i.decode() for i in info_meta[b'coarse_label_names']]
        del info
        del info_meta

        self.coarse_to_fine_idx = {}
        self.fine_to_coarse_idx = {}

        for i in range(len(self.target_fine)):
            if self.targets_coarse[i] not in self.coarse_to_fine_idx:
                self.coarse_to_fine_idx[self.targets_coarse[i]] = set()
            self.coarse_to_fine_idx[self.targets_coarse[i]].add(self.target_fine[i])
            self.fine_to_coarse_idx[self.target_fine[i]] = self.targets_coarse[i]

        self.coarse_to_fine_idx = {k: list(self.coarse_to_fine_idx[k]) for k in sorted(self.coarse_to_fine_idx)}
        self.coarse_to_fine_name = {}
        for k,v in self.coarse_to_fine_idx.items():
            self.coarse_to_fine_name[self.coarse_class_names[k]] = []
            for i in v:
                self.coarse_to_fine_name[self.coarse_class_names[k]].append(self.fine_class_names[i])


        self.indexs = []
        self.chose_cls = chose_cls
        if chose_cls is None:
            chose_cls = list(idxs_dict.keys())
        for key, value in idxs_dict.items():
            if key not in chose_cls:
                    continue
            else:
                if train is True:
                    num = len(value)
                    chose_num = math.ceil(num * data_ratio)
                    select_ids = np.random.choice(value, chose_num, replace=False)
                else:
                    select_ids = value
                self.indexs.extend(select_ids)

        # if shuffle:
            # random.shuffle(self.indexs)
        

        self.proj_index = {}
        self.cls_num = merged_cls_num
        if self.cls_num == 20:
            self.proj_index = self.fine_to_coarse_idx
        elif self.cls_num == 2:
            creature_idx = ([0,1,7,8] + list(range(11,17)))
            fine_to_2_idx = {i:0 if i in creature_idx else 1 for i in range(20)}
            self.proj_index = {k:fine_to_2_idx[v] for k,v in self.fine_to_coarse_idx.items()}

        if self.chose_cls is not None:
            self.proj_index = {v:k for k,v in enumerate(self.chose_cls)}


        self.targets = [self.proj_index[self.dataset[i][1]] for i in self.indexs]

    def __len__(self):
        return len(self.indexs)

    def __getitem__(self, idx):
        feature, label = self.dataset[self.indexs[idx]]
        if self.cls_num != 100 or self.chose_cls is not None:
            label = self.proj_index[label]
        return (feature, label)

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

def convert_exp_to_param(nid:str="a-1.0"):
    """ convert exploration experiments name into concrete parameters.

    :param str split: "[NAME]-[INDEX]", defaults to "a-1.0"
    """
    
    _ = load_json(os.path.join(data_dir,"cifar-100-python/class_idx_convert.json"))
    coarse_to_fine_idx, fine_to_coarse_idx = _["coarse_to_fine_idx"], _["fine_to_coarse_idx"]
    coarse_to_fine_idx = {int(k):v for k,v in coarse_to_fine_idx.items()}
    fine_to_coarse_idx = {int(k):v for k,v in fine_to_coarse_idx.items()}

    result = {}
    name, index = nid.split("-")
    if name == "a":
        # ind_to_ratio = {1:0.1, 2:0.4, 3:1.0}
        index = float(index)
        assert index <= 1.0, "data_ratio only support <=1.0"
        result = {**result, "data_ratio": index}
    elif name == "b":
        chose_cls = []
        index = int(index)
        assert index <= 20, "chose_cls only support 20 coarse classes"
        for i in range(index):
            chose_cls.extend(coarse_to_fine_idx[i])
        result = {**result, "chose_cls": chose_cls}
    elif name == "c":
        index = int(index)
        assert index in [2, 20, 100], "merged_cls_num only support [2, 20, 100]"
        result = {**result, "merged_cls_num": index}
    # elif name == "d":
    #     indexs = index.split("|")

    #     if index < 6:
    #         chose_cls = 
    #     result = {**result, "chose_cls": chose_cls}

    return result

def get_center_data(dataset, nid=None):
    """center training exploration

    :param _type_ args: _description_
    :return _type_: _description_
    """
    dataset_train, dataset_test = None, None
    if dataset == 'mnist':
        dataset_train = datasets.MNIST(data_dir, train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(data_dir, train=False, download=True, transform=trans_mnist)
    elif dataset == 'cifar10':
        dataset_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10(data_dir, train=False, download=True, transform=trans_cifar10_val)
    elif dataset == 'cifar100':
        split_param = convert_exp_to_param(nid=nid)
        dataset_train = Cifar100Wrapper(data_dir, transform=trans_cifar100_train, train=True, download=True, **split_param)
        dataset_test = Cifar100Wrapper(data_dir, transform=trans_cifar100_val, train=False, download=True, **split_param)
        # dataset_gtest = datasets.CIFAR100(data_dir, transform=trans_cifar100_val, train=False, download=True)
    else:
        exit('Error: unrecognized dataset')

    return {'train': dataset_train, 'test': dataset_test}
    # 'gtest': dataset_gtest}

def get_normal_data(dataset):
    """

    :param args: dataset
    :return:
    """
    dataset_train, dataset_test = None, None
    if dataset == 'mnist':
        dataset_train = datasets.MNIST(data_dir, train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(data_dir, train=False, download=True, transform=trans_mnist)
    elif dataset == 'cifar10':
        dataset_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10(data_dir, train=False, download=True, transform=trans_cifar10_val)
    elif dataset == 'cifar100':
        dataset_train = datasets.CIFAR100(data_dir, train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100(data_dir, train=False, download=True, transform=trans_cifar100_val)
    elif dataset == 'cifar20':
        split_param = convert_exp_to_param(nid='c-20')
        dataset_train = Cifar100Wrapper(data_dir, transform=trans_cifar100_train, train=True, download=True, **split_param)
        dataset_test = Cifar100Wrapper(data_dir, transform=trans_cifar100_val, train=False, download=True, **split_param)
    else:
        exit('Error: unrecognized dataset')

    return {'train': dataset_train, 'test': dataset_test}


def get_fed_data(setting_path, dataset, split, num_users, dir_alpha, clsnum_peruser, imb_alpha, data_frac):
    """

    :param args: dataset/split/num_users/shard_per_user/dir_alpha/
    :return:
    """
    dataset_train, dataset_test = None, None
    dict_save_path = os.path.join(setting_path, 'dict_users.pkl')

    if dataset == 'mnist':
        dataset_train = datasets.MNIST(data_dir, train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(data_dir, train=False, download=True, transform=trans_mnist)
    elif dataset == 'cifar10':
        dataset_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10(data_dir, train=False, download=True, transform=trans_cifar10_val)
    elif dataset == 'cifar100':
        dataset_train = datasets.CIFAR100(data_dir, train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100(data_dir, train=False, download=True, transform=trans_cifar100_val)
    elif dataset == 'cifar20':
        split_param = convert_exp_to_param(nid='c-20')
        dataset_train = Cifar100Wrapper(data_dir, transform=trans_cifar100_train, train=True, download=True, **split_param)
        dataset_test = Cifar100Wrapper(data_dir, transform=trans_cifar100_val, train=False, download=True, **split_param)
    elif dataset == 'emnist':
        dataset_train = Emnist(data_dir, train=True)
        dataset_test = Emnist(data_dir, train=False)
    else:
        exit('Error: unrecognized dataset')

    if os.path.exists(dict_save_path):
        dict_users_train, dict_users_test = load_pkl(dict_save_path)
        fprint('[Load dict_users] from: %s' % dict_save_path)
    else:
        # sample users
        if split == 'iid':
            dict_users_train = iid(dataset_train.targets, num_users, data_frac)
            dict_users_test = iid(dataset_test.targets, num_users)
        elif split == 'niid-label':
            dict_users_train, client_cls_set = noniid_label(dataset_train.targets, num_users, shard_per_user=clsnum_peruser, data_frac=data_frac, is_uniform=False)
            dict_users_test, _ = noniid_label(dataset_test.targets, num_users, shard_per_user=clsnum_peruser, client_cls_set=client_cls_set, is_uniform=False)
        elif split == 'dir':
            dict_users_train, pre_dict_frequency_classes = noniid_dir(dataset_train.targets, num_users, data_frac, alpha=dir_alpha)
            dict_users_test = imitate_sampling(dataset_test.targets, num_users,dict_frequency_classes=pre_dict_frequency_classes)
        elif split == 'one-label':
            dict_users_train, client_cls_set = one_label(dataset_train.targets, num_users, data_frac=data_frac)
            dict_users_test, _ = one_label(dataset_test.targets, num_users, data_frac=data_frac, client_cls_set=client_cls_set)
        # save users
        else:
            raise Exception("\033[0;31m{}\033[0m".format("No matched split, now support iid, niid-label, dir!"))
        
        save_pkl(dict_save_path, (dict_users_train, dict_users_test))
        fprint('[Create dict_users]: %s' % dict_save_path)
    return dataset_train, dataset_test, dict_users_train, dict_users_test


def get_fed_domain_data(dataset, batch_size, data_frac):
    """

    :param args: dataset/split/num_users/shard_per_user/dir_alpha/
    :return:
    """
    # dataset_train, dataset_test = None, None
    # dict_save_path = os.path.join(setting_path, 'dict_users.pkl')

    if dataset == 'digits':
        transform_mnist = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        transform_svhn = transforms.Compose([
                transforms.Resize([28,28]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        transform_usps = transforms.Compose([
                transforms.Resize([28,28]),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        transform_synth = transforms.Compose([
                transforms.Resize([28,28]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        transform_mnistm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        domain_path = os.path.join(data_dir, "domain/digits")
        # MNIST
        mnist_trainset     = DigitsDataset(data_path=f"{domain_path}/MNIST", channels=1, percent=data_frac, train=True,  transform=transform_mnist)
        mnist_testset      = DigitsDataset(data_path=f"{domain_path}/MNIST", channels=1, percent=data_frac, train=False, transform=transform_mnist)

        # SVHN
        svhn_trainset      = DigitsDataset(data_path=f'{domain_path}/SVHN', channels=3, percent=data_frac,  train=True,  transform=transform_svhn)
        svhn_testset       = DigitsDataset(data_path=f'{domain_path}/SVHN', channels=3, percent=data_frac,  train=False, transform=transform_svhn)

        # USPS
        usps_trainset      = DigitsDataset(data_path=f'{domain_path}/USPS', channels=1, percent=data_frac,  train=True,  transform=transform_usps)
        usps_testset       = DigitsDataset(data_path=f'{domain_path}/USPS', channels=1, percent=data_frac,  train=False, transform=transform_usps)

        # Synth Digits
        synth_trainset     = DigitsDataset(data_path=f'{domain_path}/SynthDigits/', channels=3, percent=data_frac,  train=True,  transform=transform_synth)
        synth_testset      = DigitsDataset(data_path=f'{domain_path}/SynthDigits/', channels=3, percent=data_frac,  train=False, transform=transform_synth)

        # MNIST-M
        mnistm_trainset     = DigitsDataset(data_path=f'{domain_path}/MNIST_M/', channels=3, percent=data_frac,  train=True,  transform=transform_mnistm)
        mnistm_testset      = DigitsDataset(data_path=f'{domain_path}/MNIST_M/', channels=3, percent=data_frac,  train=False, transform=transform_mnistm)

        # mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
        # mnist_test_loader  = torch.utils.data.DataLoader(mnist_testset, batch_size=batch_size, shuffle=False)
        # svhn_train_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=batch_size,  shuffle=True)
        # svhn_test_loader = torch.utils.data.DataLoader(svhn_testset, batch_size=batch_size, shuffle=False)
        # usps_train_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=batch_size,  shuffle=True)
        # usps_test_loader = torch.utils.data.DataLoader(usps_testset, batch_size=batch_size, shuffle=False)
        # synth_train_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=batch_size,  shuffle=True)
        # synth_test_loader = torch.utils.data.DataLoader(synth_testset, batch_size=batch_size, shuffle=False)
        # mnistm_train_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=batch_size,  shuffle=True)
        # mnistm_test_loader = torch.utils.data.DataLoader(mnistm_testset, batch_size=batch_size, shuffle=False)


        train_dataset = DatasetMerge([mnist_trainset, svhn_trainset, usps_trainset, synth_trainset, mnistm_trainset])
        test_dataset  = DatasetMerge([mnist_testset, svhn_testset, usps_testset, synth_testset, mnistm_testset])

        return  train_dataset, test_dataset, None, None
 
    # return dataset_train, dataset_test, dict_users_train, dict_users_test


def split_dict_users(targets, dict_users, collection_rounds=1):
    # targets = np.repeat(np.expand_dims(np.repeat(np.arange(10), 1000),0), 10, axis=0).reshape(-1)
    # n_targets = len(targets) // 10
    # dict_users = {i: list(range(n_targets * i, n_targets * (i + 1))) for i in range(10)}
    # dict_users_split=split_collection(targets, dict_users, collection_rounds=10)
    # for k,v in dict_users_split[-1].items():
    #    print(set(dict_users_split[-1][k]) == set(dict_users[k]))
    dict_users = copy.deepcopy(dict_users)
    n_classes = len(set(targets))
    dict_users_split = []
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    elif isinstance(targets, list):
        targets = np.array(targets)
    # new collect number in the current round
    n_per_cls_user = math.floor(len(targets) / (len(dict_users) * n_classes * collection_rounds))

    for rod in range(1, collection_rounds+1):
        dict_users_round = {}
        for user, id_data in dict_users.items():
            dict_users_round[user] = []
            id_data = np.array(id_data)
            for c in range(n_classes):
                ids_cls = id_data[np.where(targets[id_data]==c)[0]]
                ids_select_cls = np.random.choice(ids_cls, n_per_cls_user, replace=False)
                dict_users_round[user] += ids_select_cls.tolist()
                id_data = np.array(list(set(id_data).difference(set(ids_select_cls))))
            dict_users[user] = id_data
            if rod > 1:
                dict_users_round[user] = dict_users_split[-1][user] + dict_users_round[user]
        dict_users_split.append(dict_users_round)
    return dict_users_split


class DigitsDataset(Dataset):
    """
        [reference]: https://github.com/med-air/FedBN
    """
    def __init__(self, data_path, channels, percent=0.1, filename=None, train=True, transform=None):
        if filename is None:
            if train:
                if percent >= 0.1:
                    for part in range(int(percent*10)):
                        if part == 0:
                            self.images, self.targets = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                        else:
                            images, targets = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                            self.images = np.concatenate([self.images,images], axis=0)
                            self.targets = np.concatenate([self.targets,targets], axis=0)
                else:
                    self.images, self.targets = np.load(os.path.join(data_path, 'partitions/train_part0.pkl'), allow_pickle=True)
                    data_len = int(self.images.shape[0] * percent*10)
                    self.images = self.images[:data_len]
                    self.targets = self.targets[:data_len]
            else:
                self.images, self.targets = np.load(os.path.join(data_path, 'test.pkl'), allow_pickle=True)
        else:
            self.images, self.targets = np.load(os.path.join(data_path, filename), allow_pickle=True)

        self.transform = transform
        self.channels = channels
        self.targets = torch.LongTensor(self.targets.astype(np.long).squeeze())

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.targets[idx]
        if self.channels == 1:
            image = Image.fromarray(image, mode='L')
        elif self.channels == 3:
            image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class DatasetMerge(Dataset):
    """
        多个数据集合并为一个测试集，和Dataloader之间的接口。
    """

    def __init__(self, datasets:list):
        self.datasets = datasets
        self.sizes = np.array([len(i) for i in self.datasets])
        self.cum_idxs = np.cumsum(self.sizes) - 1 # the max idxs of each dataset.
        self.targets = np.concatenate([i.targets for i in datasets])

    def __len__(self):
        return self.sizes.sum()

    def __getitem__(self, item):
        idx_dataset = np.searchsorted(self.cum_idxs, item)
        idx = item if idx_dataset == 0 else ( item - self.cum_idxs[idx_dataset - 1] - 1 )
        image, targets = self.datasets[idx_dataset][idx]
        return image, targets
