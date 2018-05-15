# -*- coding: utf-8 -*-
import commandr
from collections import defaultdict
from dataset.semeval2018.task3.process import Processor
from nlp.process import process_tweet


@commandr.command
def build_vocab(out_filename):
    subtask = 'A'
    func_load = [Processor.load_train, Processor.load_test]

    token_count = defaultdict(lambda :0)
    for func in func_load:
        dataset = func(subtask)
        for text, label in dataset:
            tokens = process_tweet(text)
            for token in tokens:
                token_count[token] += 1

    token_count = sorted(token_count.items(), key=lambda item: -item[1])
    with open(out_filename, 'w') as file_obj:
        for token, count in token_count:
            file_obj.write(u'{}\t{}\n'.format(token, count).encode('utf8'))


if __name__ == '__main__':
    commandr.Run()
