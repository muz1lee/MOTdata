import argparse
from utils.constants import PROJECT_NAME, DEFAULT_SAVE_ROOT

def get_basic_parser():
    parser = argparse.ArgumentParser()
    # experiment name
    parser.add_argument('--project_name', type=str, default=PROJECT_NAME, help='project name')
    parser.add_argument('--exp_name', type=str, default='exp0_test', help='exp[id]_[exp name]')        
    # parser.add_argument('--alg', type=str, default="fedlocal", help="name of the algorithm")
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--model', type=str, default='mlp', help='model name')

    # basic experiment
    parser.add_argument('--gpu', type=str, default='0', help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    # log related
    parser.add_argument('--save_root', type=str, default=DEFAULT_SAVE_ROOT, help='save root path')
    parser.add_argument('--test_freq', type=int, default=1, help='how often to test on val set')
    parser.add_argument('--save_dir', type=str, default='',
                        help='It will be set automatically after generated the experiment name, it seems like [save_root]/[experiemnt_name]')

    # optimizer related
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--lr', type=float, default=0.05, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--lr_decay', type=float, default=1.0, help="learning rate decay per round")
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--optim', type=str, default="sgd", help="optimizer")

    # data related
    parser.add_argument('--bs_train', type=int, default=10, help="train batch size: B")
    parser.add_argument('--bs_test', type=int, default=200, help="test batch size")
    parser.add_argument('--data_frac', type=float, default=1.0, help='the fraction of dataset used')

    # model related [generality decrease]
    parser.add_argument('--norm', type=str, default='bn', help="bn, gn, in, ln")
    parser.add_argument('--num_groups', type=int, default=32, help="number of group norm")
    parser.add_argument('--std_lin', type=float, default=0.01, help='the std of initializing linear layer.')

    # pretrained model related
    parser.add_argument("--load", type=str, default="", help="the epoch of pretrained model (default: '' not load). If load=Number, it will load the corresponding npy file")
    # model save
    parser.add_argument('--save_model', action='store_true', help='whether save your model during training')
    parser.add_argument('--use_wandb', action='store_true', help='whether use wandb')
    parser.add_argument('--patience_earlystop', type=int, default=-1, help='patience for early stopping')
    parser.add_argument('--timestamp', type=int, default=1, help='add timestamp')

    return parser

def add_fl_parser(parser: argparse.ArgumentParser):
    """
        federated arguments
    :param parser:
    :return:
    """

    # scenario settings
    parser.add_argument('--frac', type=float, default=1.0, help="the fraction of clients: C")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    # local_bs == bs_train

    # log related
    parser.add_argument('--save_clients', action='store_true', 
                        help='whether to save global model and each client models in the every iteration')

    # data related
    parser.add_argument('--split', type=str, default='iid', help="train-test split type, user or sample")
    parser.add_argument('--clsnum_peruser', type=int, default=2, help="the numbers of class belonged to each user.")
    
    parser.add_argument('--dir_alpha', type=float, default=1.0, help="dirichlet sampling ")
    parser.add_argument('--imb_alpha', type=float, default=0.0, help="imbalanced level, recommend=[0.05,0.2,0.5]")

    # train and test
    parser.add_argument('--local_test', action='store_true', help="whether to test locally, it may slow down the training time")
    return parser


if __name__ == '__main__':
    parser = get_basic_parser()
    parser = add_fl_parser(parser)
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(k, v)
