from pathlib import Path
import pickle
from tqdm import trange
import numpy as np


def generate_negative_samples(train, val, test, user_count, item_count, train_negative_sample_size,
                              train_negative_sampling_seed):
    assert train_negative_sampling_seed is not None, 'Specify seed for random sampling'
    np.random.seed(train_negative_sampling_seed)
    items = np.arange(item_count) + 1
    prob = np.ones_like(items)
    prob = prob / prob.sum()

    negative_samples = {}
    print('Sampling negative items')
    for user in trange(user_count):
        if isinstance(train[user][1], tuple):
            seen = set(x[0] for x in train[user])
            seen.update(x[0] for x in val[user])
            seen.update(x[0] for x in test[user])
        else:
            seen = set(train[user])
            seen.update(val[user])
            seen.update(test[user])
        zeros = np.array(list(seen)) - 1  # items start from 1
        p = prob.copy()
        p[zeros] = 0.0
        p = p / p.sum()

        samples = np.random.choice(items, train_negative_sample_size, replace=False, p=p)
        negative_samples[user] = samples.tolist()

    return negative_samples


def get_negative_samples(train, val, test, user_count, item_count, train_negative_sample_size,
                         train_negative_sampling_seed, save_folder):
    savefile_path = _get_save_path(save_folder, train_negative_sample_size,
                                   train_negative_sampling_seed)
    if savefile_path.is_file():
        print('Negatives samples exist. Loading.')
        negative_samples = pickle.load(savefile_path.open('rb'))
        return negative_samples
    print("Negative samples don't exist. Generating.")
    negative_samples = generate_negative_samples(train, val, test, user_count,
                                                 item_count, train_negative_sample_size,
                                                 train_negative_sampling_seed)
    with savefile_path.open('wb') as f:
        pickle.dump(negative_samples, f)
    return negative_samples


def _get_save_path(save_folder, train_negative_sample_size, train_negative_sampling_seed):
    folder = Path(save_folder)
    print('random', train_negative_sample_size, train_negative_sampling_seed)
    filename = '{}-sample_size{}-seed{}.pkl'.format('random', train_negative_sample_size,
                                                    train_negative_sampling_seed)
    return folder.joinpath(filename)
