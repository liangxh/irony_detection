# -*- coding: utf-8 -*-
from __future__ import print_function
import re
import commandr
import json
from collections import defaultdict
from dataset.common.const import *
from dataset.semeval2014_task9.process import Processor
from dataset.semeval2014_task9.config import config
from nlp.process import naive_tokenize


@commandr.command
def build_origin(infile, outfile):
    """
    将SemEval2014 Task9的原始文件转换成
        <label_id>(tab)<text>
    """
    label_map = {
        'negative': 0,
        'neutral': 1,
        'positive': 2,
        'objective': 1,  # 根据官方README, objective在taskB中视为neutral
    }

    texts = set()
    lines = set()

    with open(infile, 'r') as in_obj, open(outfile, 'w') as out_obj:
        for line in in_obj:
            line = line.strip()
            if line == '':
                continue
            parts = line.split('\t', 3)
            label = parts[-2]
            if label in label_map:
                label_id = label_map[label]
                text = parts[-1].replace('\t', ' ').replace('', ' ').strip()
                text = re.sub('Not Available$', '', text)
                text = re.sub('\s+', ' ', text)

                text = text.strip().decode('utf8')
                if text != '' and text not in texts:
                    line = u'{}\t{}'.format(label_id, text).encode('utf8')
                    if line in lines:
                        print(line)
                    out_obj.write(line + '\n')
                    texts.add(text)
            elif label.find('-OR-') < 0:
                raise Exception('unknown label: {}'.format(label))


@commandr.command
def merge_train_dev():
    exclude_texts = set()
    for label, text in Processor.load_origin(config.path(TEST, RAW)):
        exclude_texts.add(text)

    with open(config.path(TRAIN, TXT), 'w') as fobj:
        for path in [config.path(TRAIN, RAW), config.path(DEV, RAW)]:
            for label, text in Processor.load_origin(path):
                if text not in exclude_texts:
                    fobj.write('{}\t{}\n'.format(label, text))
                    exclude_texts.add(text)


@commandr.command
def build_text_label():
    path_list = {
        TRAIN: config.path(TRAIN, TXT),
        TEST: config.path(TEST, TXT)
    }
    for key, path in path_list.items():
        text_path = config.path(key, TEXT)
        label_path = config.path(key, LABEL)
        with open(text_path, 'w') as text_obj, open(label_path, 'w') as label_obj:
            for label, text in Processor.load_origin(path):
                text_obj.write(text + '\n')
                label_obj.write(str(label) + '\n')


@commandr.command
def build_vocab(version='v0'):
    all_token = set()

    for key in [TRAIN, TEST]:
        text_path = config.path(key, TEXT)
        output_path = config.path(key, VOCAB, version)
        print(output_path)

        tf = defaultdict(lambda: 0)
        df = defaultdict(lambda: 0)

        with open(text_path, 'r') as file_obj:
            for line in file_obj:
                line = line.strip()
                if line == '':
                    continue
                tokens = naive_tokenize(line)
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

    output_path = config.path(ALL, VOCAB, version)
    with open(output_path, 'w') as file_obj:
        for token in all_token:
            file_obj.write('{}\n'.format(token.encode('utf8')))
    print(output_path)


if __name__ == '__main__':
    commandr.Run()
