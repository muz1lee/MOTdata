

#%%
from otdd.pytorch.datasets import load_torchvision_data_shuffle


def load_data_corrupted(corrupt_type='shuffle', dataname=None, data=None, valid_size=0, random_seed=2021, resize=None,
                                        stratified=True, shuffle=False,
                                        training_size=None, test_size=None, currupt_por=0):
    if corrupt_type == 'shuffle':
        loaders, full_dict, shuffle_ind = load_torchvision_data_shuffle(dataname, valid_size=valid_size,
                                                                        random_seed=random_seed,
                                                                        resize=resize, stratified=stratified,
                                                                        shuffle=shuffle, maxsize=training_size,
                                                                        maxsize_test=test_size, shuffle_per=currupt_por)
        return loaders, shuffle_ind
    # elif corrupt_type == 'feature':
    # elif corrupt_type == 'backdoor-blend', 'backdoor-trojan-sq', 'backdoor-trojan-wm'
    else:  # empty or non-implemented == Loading Clean Data
        shuffle_ind = []
        loaders, full_dict = load_torchvision_data_shuffle(dataname, valid_size=valid_size, random_seed=random_seed,
                                                           resize=resize, stratified=stratified, shuffle=shuffle,
                                                           maxsize=training_size, maxsize_test=test_size, shuffle_per=0)
        return loaders, shuffle_ind


