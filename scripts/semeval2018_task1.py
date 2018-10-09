# -*- coding: utf-8 -*-
from collections import defaultdict

import commandr
from dataset.semeval2018.task1.config import config

from dataset.common.const import *
from dataset.semeval2018_task1.process import Processor
from nlp.process import naive_tokenize


@commandr.command
def build_vocab(out_filename):
    """
    结合train, test生成字典
        <token>(tab)<count>
    """
    func_load = [Processor.load_train, Processor.load_test]

    token_count = defaultdict(lambda: 0)
    for func in func_load:
        dataset = func()
        for label, text in dataset:
            tokens = naive_tokenize(text)
            for token in tokens:
                token_count[token] += 1

    token_count = sorted(token_count.items(), key=lambda item: -item[1])
    with open(out_filename, 'w') as file_obj:
        for token, count in token_count:
            file_obj.write(u'{}\t{}\n'.format(token, count).encode('utf8'))


@commandr.command
def build_text_label():
    path_list = {
        TRAIN: config.path_E_c_train,
        TEST: config.path_E_c_dev
    }
    for key, path in path_list.items():
        text_path = config.path(key, TEXT)
        label_path = config.path(key, LABEL)
        with open(text_path, 'w') as text_obj, open(label_path, 'w') as label_obj:
            for label, text in Processor.load_origin(path):
                text_obj.write(text + '\n')
                label_obj.write(str(label) + '\n')


@commandr.command
def test_process():
    print '[TRAIN]'
    Processor.load_train()

    print '\n[TEST]'
    Processor.load_test()


if __name__ == '__main__':
    commandr.Run()
