# -*- coding: utf-8 -*-
import commandr
import importlib
from nlp.process import naive_tokenize
from dataset.common.const import *


def iterate_file(filename):
    with open(filename, 'r') as file_obj:
        for line in file_obj:
            line = line.strip()
            if line == '':
                continue
            yield line


@commandr.command('naive')
def to_naive(dataset_key):
    data_config = getattr(importlib.import_module('dataset.{}.config'.format(dataset_key)), 'config')
    for mode in [TRAIN, TEST]:
        text_path = data_config.path(mode, TEXT)
        out_path = data_config.path(mode, TEXT, 'naive')
        with open(out_path, 'w') as out_obj:
            for line in iterate_file(text_path):
                tokens = naive_tokenize(line)
                out_obj.write(' '.join(tokens).encode('utf8') + '\n')


if __name__ == '__main__':
    commandr.Run()
