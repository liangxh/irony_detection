# -*- coding: utf-8 -*-
import os
import commandr
from collections import defaultdict
from nlp.process import naive_tokenize
from dataset.semeval2018.task3.process import Processor, config


@commandr.command
def build_vocab(out_filename):
    """
    结合subtask A 的train, test生成字典
        <token>(tab)<count>
    """
    subtask = 'A'
    func_load = [Processor.load_train, Processor.load_test]

    token_count = defaultdict(lambda: 0)
    for func in func_load:
        dataset = func(subtask)
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
    for mode, func in {'train': Processor.load_train, 'test': Processor.load_test}.items():
        output_path = os.path.join(
            os.environ['HOME'],
            'lab/irony_detection/data/semeval2018_task3/{mode}.text'.format(mode=mode)
        )
        with open(output_path, 'w') as file_obj:
            for _, text in func('A'):
                file_obj.write(text + '\n')


if __name__ == '__main__':
    commandr.Run()
