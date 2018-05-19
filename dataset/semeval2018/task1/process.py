# -*- coding: utf-8 -*-
from dataset.semeval2018.task1 import config


class Processor(object):
    @classmethod
    def load_file(cls, filename):
        samples = list()
        with open(filename, 'r') as file_obj:
            file_obj.readline()

            for line in file_obj:
                line = line.strip()
                if line == '':
                    continue
                values = line.split('\t')
                text = values[1]
                label = map(float, values[2:])
                sample = (label, text)
                samples.append(sample)
        return samples

    @classmethod
    def load_train(cls):
        return cls.load_file(config.path_E_c_train)

    @classmethod
    def load_test(cls):
        return cls.load_file(config.path_E_c_dev)
