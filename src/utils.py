import os
import numpy as np
import pandas as pd
import pickle


def save_txt(data_to_save, filepath, mode='a'):
    """
    Save text to a file.
    """
    with open(filepath, mode) as text_file:
        text_file.write(data_to_save + '\n')


def save_outputs(files_to_save: dict,
                 folder_path):
    """
    Save objects as pickle files, in a given folder.
    """
    for name, file in files_to_save.items():
        with open(folder_path + name + '.pkl', 'wb') as f:
            pickle.dump(file, f)


def get_last_checkpoint():
    """
    Fetch path of last checkpoint available in the root folder, based on the date in the filename.
    """
    logdir = '.'
    logfiles = sorted([f for f in os.listdir(logdir) if f.startswith('checkpoint')])
    checkpoint_path = logfiles[-1]
    return checkpoint_path


def read_data(file_path):
    """
    Generic function to read any kind of data. Extensions supported: '.gz', '.csv', '.pkl'
    """
    if file_path.endswith('.gz'):
        obj = pd.read_csv(file_path, compression='gzip',
                     header=0, sep=';', quotechar='"',
                     error_bad_lines=False)
    elif file_path.endswith('.csv'):
        obj = pd.read_csv(file_path)
    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as handle:
            obj = pickle.load(handle)
    else:
        raise KeyError('File extension of {} not recognized.'.format(file_path))
    return obj


def softmax(x):
    """
    (Currently not used.) Compute softmax values for each sets of scores in x.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
