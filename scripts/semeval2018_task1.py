# -*- coding: utf-8 -*-
import commandr
from collections import defaultdict
from nlp.process import naive_tokenize
from dataset.semeval2018.task1.process import Processor, config


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
def build_text():
    path_list = [config.path_E_c_train, config.path_E_c_dev]
    for path in path_list:
        output_path = '{}.text'.format(path)
        with open(output_path, 'w') as file_obj:
            for label, text in Processor.load_dataset(path):
                file_obj.write(text + '\n')


@commandr.command
def test_process():
    print '[TRAIN]'
    Processor.load_train()

    print '\n[TEST]'
    Processor.load_test()


if __name__ == '__main__':
    commandr.Run()
