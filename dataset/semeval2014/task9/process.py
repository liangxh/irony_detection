# -*- coding: utf-8 -*-
from dataset.semeval2014.task9 import config


class Processor(object):
    @classmethod
    def load_origin(cls, filename):
        """
        加载由scripts.semeval2014_task9.build_origin生成的文件

        :param filename:
        :return: list of pair<LABEL, TEXT>
            LABEL: int, 取值为 0(negative) / 1(neural) / 2(positive)
            TEXT: string
        """
        samples = list()
        with open(filename, 'r') as file_obj:
            for line in file_obj:
                line = line.strip()
                if line == '':
                    continue
                parts = line.split('\t')
                label = int(parts[0])
                text = parts[1]
                sample = (label, text)
                samples.append(sample)
        return samples

    @classmethod
    def build_vocab(cls):
        fro