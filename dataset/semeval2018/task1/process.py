# -*- coding: utf-8 -*-
from dataset.semeval2018.task1 import config


class Processor(object):
    @classmethod
    def load_origin(cls, filename):
        """
        :param filename: string
        :return: list of pair<LABLE, TEXT>
            LABEL: list of float, 共11维, 順序如下, 取值为0/1
                0) anger
                1) anticipation
                2) disgust
                3) fear
                4) joy
                5) love
                6) optimism
                7) pessimism
                8) sadness
                9) surprise
                10) trust
            TEXT: string
        """
        samples = list()
        with open(filename, 'r') as file_obj:
            file_obj.readline()

            for line in file_obj:
                line = line.strip()
                if line == '':
                    continue
                values = line.split('\t')
                text = values[1]
                label = map(int, values[2:])
                sample = (label, text)
                samples.append(sample)
        return samples

    @classmethod
    def load_train(cls):
        return cls.load_origin(config.path_E_c_train)

    @classmethod
    def load_test(cls):
        return cls.load_origin(config.path_E_c_dev)
