from utils.constants import *
from utils.log_tools import fprint

import pickle
import json
import yaml
import requests
# import gpustat
import subprocess
import itertools
import psutil
from shutil import copy2

import argparse
from itertools import chain, combinations
from matplotlib.ticker import FixedLocator
from scipy.special import comb as comb_op

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_pkl(path: str, obj, protocol=4):
    """_summary_

    :param str path: _description_
    :param _type_ obj: _description_
    :param int protocol: pickle protocal, defaults to 4, bcz python3.8 HIGHT_PROTOCAL is 5 and python 3.6/3.7 is 4.
    """
    
    if '.pkl' not in path:
        path = path + '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)

def load_pkl(path: str):
    
    if '.pkl' not in path:
        path = path + '.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_json(save_path, data):
    assert save_path.split('.')[-1] == 'json'
    with open(save_path, 'w') as file:
        json.dump(data, file)

def load_json(file_path):
    assert file_path.split('.')[-1] == 'json'
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_yaml(file_path: str, data: dict):
    assert file_path.split('.')[-1] == 'yml'
    with open(file_path, 'w', encoding='utf-8') as file:
        yaml.dump(data, file)

def save_yaml_all(file_path: str, data: list):
    assert file_path.split('.')[-1] == 'yml'
    with open(file_path, 'w', encoding='utf-8') as file:
        yaml.dump_all(data, file)

def load_yaml_all(file_path: str):
    assert file_path.split('.')[-1] == 'yml'
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
        data = yaml.load_all(data, Loader=yaml.FullLoader)
    return data

def get_gpu(server_user_name='xuyc'):
    """ 
        get the most empty gpu id.
        If user has other running gpu programs, combine their info to provide gpu.
        If no proper gpu, return -1.

    Args:
        server_user_name (str, optional): name of login user in linux

    Returns:
        int: id of the most empty gpu!
    """
    gpu_stats = gpustat.new_query()
    gpu_memory_usages = []
    gpu_info = np.zeros((len(gpu_stats.gpus),2), dtype=np.int64)
    for gpu in gpu_stats.gpus:
        # 'index';'utilization.gpu';'memory.used';'memory.total'
        totol_memory = gpu['memory.total']
        gpu_info[gpu['index']] = [gpu['utilization.gpu'], gpu['memory.used']]
        
        for proc in gpu['processes']:
            # proc.keys():['username', 'gpu_memory_usage', 'pid', 'full_command']
            # fprint(proc.keys())
            if proc['username'] == server_user_name:
                gpu_memory_usages.append(proc['gpu_memory_usage'])

    most_empty_gpu_id = np.argmin(gpu_info[:,0])

    cpu_usage = psutil.cpu_percent(interval=3)
    if cpu_usage > 50:
        return -1
    if gpu_info[most_empty_gpu_id,0] < 80:
        if len(gpu_memory_usages) != 0:
            max_gpu_memory_usage = np.max(gpu_memory_usages)
            if gpu_info[most_empty_gpu_id,1] + max_gpu_memory_usage < totol_memory:
                return most_empty_gpu_id
            else:
                return -1
        else:
            return most_empty_gpu_id
    return -1

def generate_combined_args(public_str:str, config:dict, alg_config:dict):
    public_str = public_str.strip()
    if len(public_str) == 0:
        public_str = []
    else:
        public_str = public_str.split(" ")

    for values in itertools.product(*config.values()):
        args = ' '.join([f"--{k} {v}" for k, v in zip(config.keys(), values)])

        if len(alg_config) != 0:
            for alg_name, alg_params in alg_config.items():
                if len(alg_params) != 0:
                    keys = alg_params.keys()
                    values = alg_params.values()
                    for value_combination in itertools.product(*values):
                        params = ' '.join([f"--{k} {v}" for k, v in zip(keys, value_combination)])
                        yield public_str + args.split(" ") + [alg_name] + params.split(" ")
                else:
                    yield public_str + args.split(" ") + [alg_name]
        else:
            yield public_str + args.split(" ")

def auto_run_gpu(program_name:str, combined_arg_list: list, isblock=False):
    """auto run a program with multiple combination of args.

    Args:
        program_name (str): py file name.
        combined_arg_list (list[list]): [arg_list, arg_list]
    """
    run_num = len(combined_arg_list)
    for idx, combine_arg in enumerate(combined_arg_list):
        # gpu_id = get_gpu(server_user_name=os.getlogin())
        gpu_id = -1
        try_count = 0
        while gpu_id == -1:
            time.sleep(random.randint(5, 10))
            gpu_id = get_gpu(server_user_name=os.getlogin())
            fprint(f"Try{try_count}")
            try_count+=1

        args = ['python', program_name, '--gpu' , str(gpu_id)] + combine_arg
        fprint(f"{idx+1}/{run_num}", " ".join(args))
        
        with open('/dev/null', 'w') as devnull:
            proc = subprocess.Popen(args
                                    , stdout=devnull
                                    , stderr=subprocess.STDOUT
                                    ) 
            if isblock:
                proc.wait()
        time.sleep(7)

def print_red(x):
    fprint(f"\033[31m{x}\033[0m")

def resolve_args(root: str, args, args_force={}, is_fed=True):
    """load default parameters from _param and exp self.

    :param str root: _description_
    :param _type_ args: _description_
    :param dict args_force: forcely set args except for [alg, dataset], defaults to {}
    :return _type_: _description_
    """

    args_params = vars(args)
    argv_names = obtain_argv_param_names()
    # fix subparsers is not
    if 'alg' not in argv_names:
        argv_names.append('alg')
    # general.yml 
    file_path = os.path.join(root, "/Users/muz1lee/Desktop/代码/fedselect/exps/_params/general.yml")
    datas = list(load_yaml_all(file_path))[0]
    fprint("[general] load exps/_params/general.yml")
    ignore_param_names = datas['ignore']
    default_params = datas['params']

    # expx.x_xx/params.yml
    file_path = os.path.join(root, f"exps/{args.exp_name}/params.yml")
    if os.path.exists(file_path):
        datas = list(load_yaml_all(file_path))[0]
        exp_ignore_param_names = datas['ignore']
        for p in exp_ignore_param_names:
            if p not in ignore_param_names: ignore_param_names.append(p)
        exp_default_params = datas['params']
        for name, value in exp_default_params.items():
            default_params[name] = value
        fprint(f"[{args.exp_name}/params.yml]: Load successfully")
    else:
        print_red(f"Warning: {args.exp_name}/params.yml is not found, we use the params from default ymls")

    special_param_names = []
    # --------------------------------------------------------------------------------------------------
    # according to assigned dataset and alg, acquire default param values for yml file. 
    # to 1) reset args_params and 2) choose special param names.
    for k in args_params.keys():
        if k in default_params:
            if k not in argv_names: # whether params emerged at program start.
                args_params[k] = default_params[k] # case 1.1: arg in default_params && arg not in argv -> set default params.
            else:
                special_param_names.append(k) # case 1.2: arg in default_params && arg in argv -> choose(want to show).
        else:
            if k not in special_param_names: # case 2: arg not in default_params (alg-special) -> choose(want to show).
                special_param_names.append(k)

    # forcely set args except for alg, dataset
    for k,v in args_params.items():
        if k in args_force:
            args_params[k] = v
    
    def abbr(words):
        """_summary_

        :param _type_ s: _description_
        :return _type_: _description_
        """
        split = words.split('_')
        if len(split) == 1:
            name = words[0]
        elif len(split) == 2:
            name = split[0][0] + split[1].capitalize()[:2]
        elif len(split) == 3:
            name = split[0][0] + split[1].capitalize()[:2] + split[2].capitalize()[:2]
        else:
            fprint(words)
            raise Exception('args name should not have > 3 _')
        return name

    # ---------------------------------------------------------------------------------------------------
    # after reset arg params, 
    setting_name = []
    if is_fed:
        setting_param_names = ['dataset', 'data_frac', 'num_users','split', 'dir_alpha','frac' ,'clsnum_peruser','epochs','local_ep', 'seed']
        for k in setting_param_names:
            if args_params['split'] == 'dir' and k in ['dir_alpha','clsnum_peruser']:
                continue
            if args_params['split'] == 'iid' and k in ['dir_alpha','clsnum_peruser']:
                continue
            if args_params['split'] == 'niid-label' and k in ['dir_alpha','clsnum_peruser']:
                continue

            value = args_params[k]
            name = '' if isinstance(args_params[k], str) else abbr(k)
            if value == 'dir':
                value += "%.2f" % args_params['dir_alpha']
            if value == 'niid-label':
                value = 'nlabel' + "%d" % args_params['clsnum_peruser']

            setting_name.append("{}{}".format(name, value))
    else:
        setting_param_names = ['dataset', 'epochs', 'seed']
        for k in setting_param_names:
            value = args_params[k]
            name = '' if isinstance(args_params[k], str) else abbr(k)
            setting_name.append("{}{}".format(name, value))
    setting_name = '_'.join(setting_name)

    args = argparse.Namespace(**args_params)
    method_name = []
    for k in special_param_names:
        value = args_params[k]
        if value == "" or k in ignore_param_names or k in setting_param_names:
            continue
        # if type(arg's value) is 'str', name =''. else we abbreviate it.
        name = '' if isinstance(value, str) else abbr(k)
        method_name.append("{}{}".format(name, value))
    method_name = '_'.join(method_name)

    return args, setting_name, method_name


def append_in_dict(origin_dict:dict, new_dict:dict):
    for k, v in new_dict.items():
        if k not in origin_dict:
            origin_dict[k] = []
        origin_dict[k].append(v)
    return origin_dict

def mean_in_dict(data:dict):
    mean_data = {}
    for k, v in data.items():
        mean_data[k] = sum(v) / len(v)
    return mean_data


def obtain_argv_param_names():
    argvs = sys.argv
    param_names = []
    for i in argvs:
        if '--' in i:
            param_names.append(i.replace('--',''))
    return param_names

def powerset(s:list):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def draw_acc_powersets(test_acc_results, num_users, epochs, name="", dpi=100):

    combine_len = []
    for i,j in enumerate(list(powerset(list(range(num_users))))):
        if len(j) == len(combine_len) + 1:
            combine_len.append(i)

    fig = plt.figure(dpi=dpi)
    ax = fig.gca()

    for k,v in test_acc_results.items():
        plt.plot(v, label=str(k))
    plt.legend()
    xtick = np.arange(0, 1025, 128)
    plt.xticks(xtick)
    plt.xlabel('powerset of users')
    plt.ylabel('test_acc ')

    xminorLocator = FixedLocator(combine_len)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.xaxis.grid(True, which='minor')
    plt.title("test acc of powersets with %d epochs between different collection rounds %s" % (epochs, name))

def draw_sv_users(sv_results, num_users, epochs, dpi=100, name='SV'):
    plt.figure(dpi=dpi)
    total_width, n = 0.8, len(sv_results)
    width = total_width / n
    x = np.arange(num_users)
    x = x - (total_width - width) / 2

    now_width = 0
    for k,v in sv_results.items():
        # plt.plot(v, label=str(k))
        plt.bar(x + now_width,v,width=width, label=str(k))
        now_width += width
    plt.legend(loc="upper right",bbox_to_anchor=(1.2,1))
    plt.title('%s with %d epochs between different collect rounds' % (name, epochs))
    plt.xticks(np.arange(num_users))
    plt.xlabel('User id')
    plt.ylabel('Contribution')

def compute_shapley_value(num_users, test_acc):
    combinations_list = list(powerset(list(range(num_users))))
    combine_2_idx = {v:k for k,v in enumerate(combinations_list)}
    results = np.zeros(num_users)
    for user_id in range(num_users):
        ex_user_combinations = [_ for _ in combinations_list if user_id not in _]
            # ex_user_combinations = [_ for _ in combinations_list if user_id not in _ and len(set(_).difference(set(range(2)))) == 0]
        for ex_user in ex_user_combinations:
            ex_user_add = ex_user + (user_id,)
            ex_user_add = tuple(sorted(list(ex_user_add)))
            # fprint(ex_user_add, ex_user)
            # fprint(test_acc[combine_2_idx[ex_user_add],1], '-', test_acc[combine_2_idx[ex_user],1])
            results[user_id] += (test_acc[combine_2_idx[ex_user_add]] - test_acc[combine_2_idx[ex_user]]) / comb_op(num_users-1, len(ex_user))
    return results

def draw_bars(samples, title="", xlabel="id", ylabel="data num"):
    clients_num = samples.shape[0]
    class_num = samples.shape[1]
    s = samples.copy()
    accum = s[:,0].copy()
    fg = plt.figure(figsize=(9, 7))


    # data
    x = list(range(1,clients_num+1))
    plt.xticks(x)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.bar(x, s[:,0], width=0.3, color=None, label='0')
    for i in range(1, class_num):
        plt.bar(x, s[:,i], bottom=accum, width=0.3, color=None, label=str(i))
        accum += s[:,i]
    plt.legend(loc='upper right', bbox_to_anchor=(0.92, 0.8, 0.2, 0.2))
    ax=plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.show()

def backup_code(src_dir, to_dir):
    ignore_dir_keywords = ['.vscode', '__pycache__', '.git', 'wandb', '.ipynb_checkpoints', 
                           '.pdf', '.csv','.gif','.npy']
    pattern_list = [re.compile(r'{}'.format(keyword)) for keyword in ignore_dir_keywords]
    
    if src_dir[-1] != '/':
        src_dir += '/'

    def is_ignore(path, pattern_list):
        for pattern in pattern_list:
            if pattern.search(path) is not None:
                return True
        return False
    def path_generater(src_dir, pattern_list):
        for root, dir, files in os.walk(src_dir):
            for file in files:
                if not is_ignore(os.path.join(root, file), pattern_list):
                    yield os.path.join(root, file)

    for path in path_generater(src_dir, pattern_list):
        if os.path.getsize(path) / float(1024*1024) >= 1:
            fprint(f"Warning {path_new} > 1 MB")
        path_new = os.path.join(to_dir, path.replace(src_dir, ''))
        dir_new = os.path.dirname(path_new)
        if not os.path.exists(dir_new):
            os.makedirs(dir_new)
        copy2(src=path, dst=path_new)


