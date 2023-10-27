
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset

from utils1 import *

from otdd.pytorch.distance import  FeatureCost
from otdd.pytorch.moments import *
from otdd.pytorch.utils import *

import os
from utils.options import get_basic_parser, add_fl_parser
from utils.data_utils import get_fed_domain_data
from utils.log_tools import generate_log_dir
from utils.global_utils import set_random_seed, resolve_args

import sys

from algs.alg_tools import check_client_class_distribution
from utils.global_utils import draw_bars
from utils.data_utils import get_fed_data, get_normal_data
from utils.global_utils import load_pkl
from torchvision.models import resnet18

def bary_difference(diff_list):
    df = pd.DataFrame(diff_list)
    plot = plt.figure(figsize=(10,6))
    plt.plot(df,label='Square Error',linewidth=2, marker='*')
    plt.grid()
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    plt.ylabel('Error (Approx Bary - True Bary) ',size=18)
    plt.xlabel('Iterations',size=18)
    plt.legend(fontsize=18)
    plt.savefig('/Users/muz1lee/Desktop/代码/fedselect/results/gaussian_toy.png')
    plt.show()

def data_process_iid():
    # only MNIST datasets
    parser = add_fl_parser(get_basic_parser())
    args = parser.parse_args(['--exp_name', 'exp0_ot', '--dataset', 'mnist', '--split', 'iid'])

    args, setting_name, method_name = resolve_args(root='.', args=args)
    if not args.save_clients:
        save_dir = os.path.join(args.save_root, args.project_name, args.exp_name, setting_name, method_name)
    else:
        save_dir = os.path.join(args.save_root, args.project_name, args.exp_name, setting_name, '_grad', method_name)
    args.save_dir = generate_log_dir(path=save_dir, is_use_tb=False, has_timestamp=args.timestamp)
    setting_dir = os.path.join(args.save_root, args.project_name, args.exp_name, setting_name)
    # # ---------------------------------------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # set random seed
    set_random_seed(seed=args.seed)

    if args.dataset in ["digits"]:
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_fed_domain_data(args.dataset,
                                                                                             args.bs_train,
                                                                                             args.data_frac)
    else:
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_fed_data(setting_dir, args.dataset,
                                                                                      args.split, args.num_users,
                                                                                      args.dir_alpha,
                                                                                      args.clsnum_peruser,
                                                                                      args.imb_alpha, args.data_frac)


    sys.path.append("/Users/muz1lee/Desktop/代码/fedselect/")

    path = "/Users/muz1lee/Desktop/代码/fedselect/exp0_ot/cifar10_dFr1.0_nUs10_iid_f1.0_e100_lEp5_s1/dict_users.pkl"
    dict_users = load_pkl(path)
    dataset = get_normal_data('mnist')
    dataset_train, dataset_test = dataset['train'], dataset['test']
    results = check_client_class_distribution(dataset_train.targets, dict_users[0])
    results = results.iloc[:, 4:]
    draw_bars(results.to_numpy(), title="mnist")
    df = check_client_class_distribution(dataset_train.targets, dict_users[0])
    print(df)

    # Embed using a pretrained (+frozen) resnet
    embedder = resnet18(pretrained=True).eval()
    embedder.fc = torch.nn.Identity()
    for p in embedder.parameters():
        p.requires_grad = False

    # Here we use same embedder for both datasets

    sys_path = '../fedewasserstein/data/' # you can change your own path to save the augmentated data

    print('extract test data... ')
    path = sys_path + 'central'
    extract_augdata(path, dataset_test)

    for client_idx in range(10):
        print('extract client {}th data...'.format(client_idx + 1))
        path = sys_path + 'c' + str(client_idx + 1)
        local_train = DatasetSplit(dataset_train, dict_users_train[client_idx])
        trainloader = torch.utils.data.DataLoader(local_train, batch_size=32, shuffle=False)
        extract_augdata(path,trainloader)


def extract_augdata(path,data):
    targets1, classes1, idxs1 = extract_data_targets(data)
    vals1, cts1 = torch.unique(targets1, return_counts=True)
    min_labelcount = 2
    V1 = torch.sort(vals1[cts1 >= min_labelcount])[0]
    X1, Y1 = load_full_dataset(data, targets=True,
                               labels_keep=V1,
                               maxsamples=10000,
                               device='cpu',
                               dtype=torch.FloatTensor,
                               reindex=True,
                               reindex_start=0)
    DA = (X1, Y1)

    M1, C1 = compute_label_stats(data, targets1, idxs1, classes1, diagonal_cov=True)
    XA = augmented_dataset(DA, means=M1, covs=C1, maxn=10000)

    np.save(path + 'mean.npy', M1.numpy())
    np.save(path + 'cov.npy', C1.numpy())
    np.save(path + 'xa.npy', XA.numpy())

class DatasetSplit(Dataset):
    """
        数据集和Dataloader之间的接口。
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.targets = torch.LongTensor(self.dataset.targets)[idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
