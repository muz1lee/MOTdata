from utils.constants import *
import traceback

from torch.utils.tensorboard import SummaryWriter


tb_recorder = None
wandb = None

def init_wandb(project_name, run_name, tags:list, config, mode='online'):
    import wandb as wb
    global wandb
    wandb = wb
    wandb.init(project=project_name, entity="super", tags=tags, config=config, mode=mode)
    wandb.run.name = run_name # wandb.run.id
    wandb.run.save()

def generate_log_dir(path, is_use_tb=True, ind_sim=None, has_timestamp=True):
    """
    initialize wandb enviro
    :param project_name: name
    :param hyper_params: list of params
    :return:
    """
    save_dir = path

    if ind_sim is None:
        # when save_dir has existed, it means that some special parameters did not emerge in the log dir. So we add timestamp to prevent overwrite existing log.
        if os.path.exists(save_dir) and has_timestamp:
            timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
            save_dir = '%s_%s' % (save_dir, timestamp)
    else:
        save_dir = os.path.join(save_dir, ind_sim)
    os.makedirs(save_dir, exist_ok=True)
    
    close_logger()
    set_logger(save_dir)
    if is_use_tb:
        close_tb_recorder()
        init_tb_recorder(path=save_dir)

    return save_dir

## logger save training time information and error.
logger = None
def set_logger(root):
    global logger
    logger = logging.getLogger('autoLog')
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s: %(message)s')

    path = os.path.join(root, 'output.log')
    file_handler = logging.FileHandler(path)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def get_logger():
    global logger
    return logger

def close_logger():
    global logger
    if logger is not None:
        logger.handlers.clear()

# tensorboard recorder.
def init_tb_recorder(path):
    global tb_recorder
    tb_recorder = SummaryWriter(log_dir=path)
    
def add_scalars(epoch, results:dict):
    global tb_recorder
    for metric_name, value in results.items():
        if '/' not in metric_name:
            if 'global' in metric_name:
                tag_name = 'global_metrics/'
            else:
                tag_name = 'metrics/'
        else:
            tag_name = ""
        tb_recorder.add_scalar(tag_name + metric_name, value, global_step=epoch)
    tb_recorder.flush()

def add_best_scalars(params, best_results):
    global tb_recorder
    metric_dict = {"hparam/best_{}".format(key):value for key, value in best_results.items()}
    tb_recorder.add_hparams(params, metric_dict)

def close_tb_recorder():
    global tb_recorder
    if tb_recorder is not None:
        tb_recorder.close()


def fprint(x, *args, level='INFO'):
    """unify logging and print.
    - If logger is not set, use 'print'.
    - If logger is set, use it!

    Args:
        x (_type_): content which wants to print.
        level (str, optional): _description_. Defaults to 'INFO'.
    """
    content = " ".join(list(map(str, [x] + list(args))))
    global logger
    if logger is None:
        print(content)
    else:
        if level == 'INFO':
            logger.info(content)
        elif level == 'DEBUG':
            logger.debug(content)
        elif level == 'ERROR':
            logger.error(content)
        elif level == 'CRITICAL':
            logger.critical(content)

def log_wandb(d):
    global wandb
    if wandb is not None:
        wandb.log(d)

class CatchExcept:
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """When the current run exits abnormally, 
            put the log into the bin directory, and shut down some services.

        Args:
            exc_type: Normal->None, AbNormal-> not None
            exc_value: Normal->None, AbNormal-> not None
            exc_tb (_type_): _description_

        Returns:
            If normally terminate, return 0.
        """
        if exc_type:
            close_tb_recorder() # close tensorboard logger

            if get_logger() is None:
                fprint(traceback.format_exc())
            else:
                fprint(traceback.format_exc(), level="ERROR")
                run_path = os.path.dirname(get_logger().handlers[0].baseFilename)
                close_logger()
                # if run is fail, please mv it to bin/ directory
                exp_path = os.path.dirname(run_path) # expx_xx
                if "KeyboardInterrupt" in traceback.format_exc():
                    mid_dir = "KeyboardInterrupt"
                else:
                    mid_dir = "bin"
                new_path = os.path.join(exp_path, mid_dir, os.path.basename(run_path)) # xx_xx_xx_x
                if 'exp' in exp_path:
                    shutil.move(run_path, new_path)

            if wandb is not None:
                if exc_type:
                    wandb.finish(exit_code=1) # failed
            return 1
        else:
            if wandb is not None: wandb.finish()
            return 0