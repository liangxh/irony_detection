# -*- coding: utf-8 -*-
from __future__ import print_function
import re
import commandr
import importlib
from textblob import TextBlob
from collections import defaultdict
from dataset.common.const import *
from dataset.common.load import *


@commandr.command
def build_vocab(dataset_key, text_version):
    data_config = getattr(importlib.import_module('dataset.{}.config'.format(dataset_key)), 'config')

    keys =  [TRAIN, TEST]
    if dataset_key == 'semeval2019_task3_dev':
        keys += [FINAL]

    all_token = set()
    for key in keys:
        text_path = data_config.path(key, TEXT, text_version)
        output_path = data_config.path(key, VOCAB, text_version)
        print(output_path)

        tf = defaultdict(lambda: 0)
        df = defaultdict(lambda: 0)

        tokenized_list = load_tokenized_list(text_path)
        for tokens in tokenized_list:
            for token in tokens:
                tf[token] += 1
            for token in set(tokens):
                df[token] += 1

        token_list = tf.keys()
        token_list = sorted(token_list, key=lambda _t: (-df[_t], -tf[_t]))
        with open(output_path, 'w') as file_obj:
            for token in token_list:
                data = {'t': token, 'tf': tf[token], 'df': df[token]}
                file_obj.write('{}\n'.format(json.dumps(data)))

        all_token |= set(token_list)

    output_path = data_config.path(ALL, VOCAB, text_version)
    with open(output_path, 'w') as file_obj:
        for token in all_token:
            file_obj.write('{}\n'.format(token.encode('utf8')))
    print(output_path)


@commandr.command
def pos(dataset_key):
    data_config = getattr(importlib.import_module('dataset.{}.config'.format(dataset_key)), 'config')
    for mode in [TRAIN, TEST]:
        out_path = data_config.path(mode, POS)
        text_path = data_config.path(mode, TEXT, EK)
        with open(text_path, 'r') as text_obj, open(out_path, 'w') as out_obj:
            for line in text_obj:
                text = line.strip()
                if text == '':
                    continue
                text = re.sub('\<[^\>]+\>', '', text)
                text = re.sub('\s+', ' ', text)
                blob = TextBlob(text.decode('utf8'))
                try:
                    tags = blob.tags
                except UnicodeDecodeError:
                    print(tags)
                    print(out_path)
                out_obj.write(json.dumps(tags) + '\n')


if __name__ == '__main__':
    commandr.Run()
