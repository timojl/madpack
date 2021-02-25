from itertools import combinations
import os
import re
import sys
from importlib.util import spec_from_file_location, module_from_spec

import torch

import madpack
from madpack import log
import numpy as np

from os.path import isfile, expanduser, join, dirname, realpath, basename, isdir
from collections import defaultdict
from functools import partial
from inspect import signature, getsource

# Useful Stuff


class StopTrainingInterrupt(Exception):
    pass


class TrainingFailedException(Exception):
    pass


class State(dict):
    """ Wrapper for dict. Intended for better readability. """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def extract_archive(filename, target_folder=None, noarchive_ok=False):
    from subprocess import run, PIPE

    if filename.endswith('.tgz') or filename.endswith('.tar'):
        command = f'tar -xf {filename}'
        command += f' -C {target_folder}' if target_folder is not None else ''
    elif filename.endswith('.tar.gz'):
        command = f'tar -xzf {filename}'
        command += f' -C {target_folder}' if target_folder is not None else ''
    elif filename.endswith('zip'):
        command = f'unzip {filename}'
        command += f' -d {target_folder}' if target_folder is not None else ''
    else:
        if noarchive_ok:
            return
        else:
            raise ValueError(f'unsuppored file ending of {filename}')

    run(command.split(), stdout=PIPE, stderr=PIPE)


def download_file(url, target, exist_ok=False):
    from subprocess import run
    import os

    if not os.path.isfile(target):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        run(f'wget {url} -O {target}'.split())
    else:
        if exist_ok:
            return
        else:
            raise FileExistsError


def collate_element(batch, cuda=False):
    import torch

    if batch is None:
        out = None
    elif type(batch) == list and batch[0] is None:
        out = None
    elif type(batch) == torch.Tensor:
        out = batch
    else:
        if type(batch[0]) == torch.Tensor:
            out = torch.stack(batch)
        elif type(batch[0]) == np.ndarray:
            out = torch.from_numpy(np.array(batch))
        elif type(batch[0]) in {str, np.str_}:
            out = str(batch[0])
        elif type(batch[0]) == float:
            out = torch.FloatTensor(batch)
        elif type(batch[0]) in {int, np.int64}:
            out = torch.LongTensor(batch)
        else:
            raise ValueError('Failed to convert data into tensor. Type: {}, Length: {}, Element types: {}'.format(
                type(batch), len(batch), [type(b) for b in batch]
            ))

    if cuda and hasattr(out, 'cuda'):
        out.pin_memory()
        out = out.cuda()

    return out


def collate(batch, cuda=False):

    variables_x = [collate_element([batch[i][0][j] for i in range(len(batch))], cuda=cuda) for j in range(len(batch[0][0]))]
    variables_y = [collate_element([batch[i][1][j] for i in range(len(batch))], cuda=cuda) for j in range(len(batch[0][1]))]

    return variables_x, variables_y


def count_parameters(model, only_trainable=False):
    """ Count the number of parameters of a torch model. """
    import numpy as np
    return sum([np.prod(p.size()) for p in model.parameters()
                if (only_trainable and p.requires_grad) or not only_trainable])


def split_overlap(dataset_type, splits='train,val,test', show_intersection=False, **kwargs):
    from madpack import datasets

    # dataset_type = getattr(datasets, dataset)
    dataset_args = kwargs

    if type(splits) == str:
        splits = splits.split(',')
        assert 1 <= len(splits) <= 3

    datasets = [(dataset_type(s, **dataset_args), s) for s in splits]

    for dataset, name in datasets:
        print('{}: {} samples'.format(name, len(dataset.sample_ids)))

    for (a, a_name), (b, b_name) in combinations(datasets, 2):
        if a is not None and b is not None:
            intersection = set(a.sample_ids).intersection(b.sample_ids)
            if len(intersection) == 0:
                log.important('{}/{} do not intersect'.format(a_name, b_name))
            else:
                if show_intersection:
                    print(intersection)
                log.warning('{}/{} intersect ({} samples)'.format(a_name, b_name, len(intersection)))


def get_current_git():
    from subprocess import run, PIPE
    root_path = join(dirname(realpath(__file__)), '..', '..',)

    out = run(["git", "rev-parse", "HEAD"], stdout=PIPE, stderr=PIPE, cwd=root_path)

    if out.returncode == 0:
        return out.stdout.decode('utf8')[:-1]
    else:
        return None


def find_in_folders(substr, base_folders):
    matching = []

    if type(base_folders) not in {list, tuple}:
        base_folders = [base_folders]

    for folder in base_folders:
        matching += [join(m_path, m) for m_path, _, files in os.walk(expanduser(folder)) for m in files
                     if m.startswith(substr) and not re.match(r'^.*(\.log-?.*|-args)$', m)]

    return matching


# HASHING

def sha1_hash_object(obj):
    import hashlib
    import base64

    obj_str = str(hash_object_recursive(obj)).encode('utf8')

    hash_str = base64.b64encode(hashlib.sha1(obj_str).digest())
    hash_str = hash_str.decode('utf-8').replace('/', '-').replace('+', '_')[:-1]
    return hash_str


def hash_object_recursive(obj):

    if isinstance(obj, (tuple, list)):
        return tuple(hash_object_recursive(x) for x in obj)

    if isinstance(obj, (dict,)):
        return tuple(sorted((k, hash_object_recursive(obj[k])) for k in obj.keys()))

    if isinstance(obj, (set, frozenset)):
        return tuple(sorted(hash_object_recursive(x) for x in obj))

    if callable(obj):
        if hasattr(obj, '__code__'):
            return hash_object_recursive(getsource(obj)), hash_object_recursive({p.name: p.default for p in signature(obj).parameters.values()})
        elif type(obj) == partial:
            return hash_object_recursive(obj.func), hash_object_recursive(obj.args), hash_object_recursive(obj.keywords)
        else:
            a = [getattr(obj, f) for f in dir(obj) if callable(getattr(obj, f))]
            return tuple(x.__code__.co_code for x in a if hasattr(x, '__code__'))

    return obj


def prepare_object_recursive(obj):
    """ Transforms any object recursively into a primitive type """

    if isinstance(obj, (tuple, list)):
        return '[' + ','.join(str_object_recursive(x) for x in obj) + ']'

    if isinstance(obj, (dict,)):
        return tuple(sorted((k, str_object_recursive(obj[k])) for k in obj.keys()))

    if isinstance(obj, (set, frozenset)):
        return tuple(sorted(str_object_recursive(x) for x in obj))

    if type(obj) == type:
        return obj.__name__

    return str(obj)


def str_object_recursive(obj):
    """ Transforms any object recursively into a string """

    if isinstance(obj, (tuple, list)):
        return '[' + ','.join(str_object_recursive(x) for x in obj) + ']'

    if isinstance(obj, (dict,)):
        return tuple(sorted((k, str_object_recursive(obj[k])) for k in obj.keys()))

    if isinstance(obj, (set, frozenset)):
        return tuple(sorted(str_object_recursive(x) for x in obj))

    if type(obj) == type:
        return obj.__name__

    if callable(obj):
        if hasattr(obj, '__code__'):
            return str_object_recursive(getsource(obj)), str_object_recursive({p.name: p.default for p in signature(obj).parameters.values()})
        elif type(obj) == partial:
            return str_object_recursive(obj.func), str_object_recursive(obj.args), str_object_recursive(obj.keywords)
        else:
            a = [getattr(obj, f) for f in dir(obj) if callable(getattr(obj, f))]
            return tuple(x.__code__.co_code for x in a if hasattr(x, '__code__'))

    return str(obj)


def diff_columns(table):
    """ find columns which have different values """

    all_columns = list(set(k for r in table.values() for k in r.keys()))

    diff_columns, same_columns = [], []
    for c in all_columns:
        all_values = [table[k][c] if c in table[k] else None for k in table.keys()]
        if len(set(all_values)) > 1:
            diff_columns += [c]
        else:
            same_columns += [c]

    return diff_columns, same_columns

