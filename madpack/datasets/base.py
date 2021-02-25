from abc import ABC
import os
import time
from os import makedirs
from os.path import basename, expanduser, isdir, join, isfile, dirname
from shutil import copy, copytree
from yaml import safe_load
from madpack.config import MADPACK_CONFIG

from madpack.utils import collate
from madpack import log
from madpack.utils import extract_archive

import re


class DatasetBase(object):
    """
    Represents a datasets.

    repository_files is a list of
    """
    repository_files = []
    data_types = None

    def __init__(self):
        self.model_config = dict()
        self._n_active_samples = None
        self.sample_ids = []

        # check if dataset available
        def reinstall():

            if len(self.repository_files) == 0:
                log.hint(f'No repository files are provided for this dataset. Try install() method')
                self.install()
            else:
                not_found = [join(self.repository_path(), f)
                             for f in self._repository_file_origins() if not isfile(join(self.repository_path(), f))]
                if len(not_found) == 0:
                    log.hint(f'Found dataset in repository. Copying to local data path')
                    self.install_from_repository()
                else:
                    log.hint(f'Not all files were found in repository {self.repository_path()}.')
                    log.hint(f'Missing {",".join(not_found)}.')
                    self.install()

        self._get_lock()
        try:
            if isdir(self.data_path()):

                integrity_ok = False
                try:
                    integrity_ok = self.check_data_integrity()
                except BaseException as e:
                    log.warning('Integrity check caused this error: ' + str(e))

                if integrity_ok:
                    log.hint('Integrity check: Dataset exists and is valid')
                else:
                    log.hint(f'Integrity check in path {self.data_path()} failed. Reinstall dataset.')
                    reinstall()
                
            else:
                log.hint(f'Path {self.data_path()} does not exist.')
                reinstall()

            self._release_lock()
        except BaseException as e:  # make sure the lock is released
            self._release_lock()
            raise
        
    def _repository_file_origins(self):
        return [f if type(f) == str else f[0] for f in self.repository_files]

    def _repository_file_targets(self):
        return [f if type(f) == str else f[1] for f in self.repository_files]

    def __getitem__(self, index):
        raise NotImplementedError

    def _lock_file(self):
        return join(dirname(self.data_path()), basename(self.data_path()) + '_madpack_lock')

    def _get_lock(self):

        if len(self.repository_files) > 0:

            log.detail('Acquire lock', self._lock_file(), isfile(self._lock_file()))
            while isfile(self._lock_file()):
                time.sleep(1.0)

            with open(self._lock_file(), 'w') as fh:
                fh.write('locked')

    def _release_lock(self):

        if len(self.repository_files) > 0:

            if isfile(self._lock_file()):
                os.remove(self._lock_file())

    def name(self):
        return self.__class__.__name__

    def resize(self, n_elements):

        if n_elements is not None and n_elements > len(self.sample_ids):
            raise ValueError('n_element must be smaller or equal the number of samples in the dataset.')

        self._n_active_samples = n_elements

    def __len__(self):
        if self._n_active_samples is not None:
            return self._n_active_samples
        else:
            return len(self.sample_ids)

    def get_batch(self, start, end, cuda=False, shuffle=False):
        samples = []

        if not shuffle:
            indices = range(start, end)
        else:
            import random
            indices = list(range(self.__len__()))
            random.shuffle(indices)
            indices = indices[start: end]

        for i in indices:
            samples += [self.__getitem__(i)]

        assert len(set(([(len(sx), len(sy)) for sx, sy in samples]))) == 1

        batch_x, batch_y = collate(samples, cuda=cuda)
        return batch_x, batch_y

    def data_loader(self, shuffle, batch_size, threads=4, drop_last=False):
        from torch.utils.data import DataLoader
        import signal

        def worker_init(x): 
            signal.signal(signal.SIGINT, signal.SIG_IGN)  # hack to avoid unnecessary messages
            
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                          num_workers=threads, drop_last=drop_last, pin_memory=True, worker_init_fn=worker_init)

    def check_data_integrity(self):
        """ [Optional] Validates that all files are available and correct """
        return True

    def data_path_name(self):
        return None

    def data_path(self, other=None):
        own_name = self.data_path_name() if self.data_path_name() is not None else self.__class__.__name__
        return join(expanduser(MADPACK_CONFIG['DATASETS_PATH']), own_name if other is None else other)

    def repository_path(self):
        return MADPACK_CONFIG['REPOSITORY_PATH']

    def install_from_repository(self):
        """ Default installation from repository: Copy and extract all tars in self.repository_files """

        for i, filename in enumerate(self.repository_files):

            origin, target = (filename, filename) if type(filename) == str else filename

            log.hint(f'copy: {join(self.repository_path(), origin)} to {join(self.data_path(), target)}')

            # make sure the path exists
            if not isdir(dirname(join(self.data_path(), target))):
                makedirs(dirname(join(self.data_path(), target)), exist_ok=True)

            copy(join(self.repository_path(), origin), join(self.data_path(), target))
            extract_archive(join(self.data_path(), target), self.data_path(), noarchive_ok=True)

    def install(self):
        """ Download and install the dataset in self.data_path() """
        pass

    def parse_data_types(self):
        """ Datasets can optionally specify the data type"""

        if self.data_types is None:
            raise AttributeError('Dataset does specify explicit data types')
        else:
            inp, out = self.data_types.split('->')
            inp = inp.split(',')
            out = out.split(',')

            regexp = r'^(\w*)(\((\w*)\))?$'

            try:
                inp = [re.match(regexp, s).groups() for s in inp]
                inp = [(s[0], s[2] if s[2] is not None else s[0]) for s in inp]
                # inp = [(s[1], s.group(2) if s is not None else None) for s in inp]

                out = [re.match(regexp, s).groups() for s in out]
                out = [(s[0], s[2] if s[2] is not None else s[0]) for s in out]
                # out = [(s.group(1), s.group(2) if s is not None else None) for s in out]

                return inp, out
            except AttributeError as e:
                log.warning('Failed to parse datatype string: {}'.format(self.data_types))
