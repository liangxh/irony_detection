# -*- coding: utf-8 -*-
import commandr
from collections import defaultdict
from nlp.process import naive_tokenize
from dataset.semeval2014.task9.process import Processor, config


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

    with open(infile, 'r') as in_obj, open(outfile, 'w') as out_obj:
        for line in in_obj:
            line = line.strip()
            if line == '':
                continue
            parts = line.split('\t', 3)
            label = parts[-2]

            if label in label_map:
                label_id = label_map[label]
                text = parts[-1].replace('\t', ' ')
                out_obj.write('{}\t{}\n'.format(label_id, text))
            elif label.find('-OR-') < 0:
                raise Exception('unknown label: {}'.format(label))


@commandr.command
def build_text():
    path_list = [config.path_train, config.path_dev, config.path_test]
    for path in path_list:
        output_path = '{}.text'.format(path)
        with open(output_path, 'w') as file_obj:
            for label, text in Processor.load_dataset(path):
                file_obj.write(text + '\n')


@commandr.command
def build_vocab(out_filename):
    """
    结合train, dev, test生成字典
        <token>(tab)<count>
    """
    func_load = [Processor.load_train, Processor.load_dev, Processor.load_test]

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


if __name__ == '__main__':
    commandr.Run()
