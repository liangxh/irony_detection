# -*- coding: utf-8 -*-
from __future__ import print_function
from dataset.semeval2019_task3_dev.config import config

label_str = ['others', 'happy', 'sad', 'angry']
label_to_idx = {v: i for i, v in enumerate(label_str)}


class Processor(object):
    @classmethod
    def load_origin(cls, path):
        dataset = list()
        with open(path, 'r') as file_obj:
            file_obj.readline()
            for line in file_obj:
                line = line.strip()
                if line == '':
                    continue
                parts = line.split('\t')
                assert len(parts) == 5
                turn_1 = parts[1].strip()
                turn_2 = parts[2].strip()
                turn_3 = parts[3].strip()
                label_idx = label_to_idx[parts[4]]

                sample = (turn_1, turn_2, turn_3, label_idx)
                dataset.append(sample)
        return dataset

    @classmethod
    def load_origin_train(cls):
        return cls.load_origin(config.path_train)

    @classmethod
    def load_origin_dev(cls):
        return cls.load_origin(config.path_dev)

    @classmethod
    def load_origin_dev_no_labels(cls):
        path = config.path_dev_no_labels
        dataset = list()
        with open(path, 'r') as file_obj:
            file_obj.readline()
            for line in file_obj:
                line = line.strip()
                if line == '':
                    continue
                parts = line.split('\t')
                if not len(parts) == 4:
                    print(parts)
                    raise Exception
                turn_1 = parts[1].strip()
                turn_2 = parts[2].strip()
                turn_3 = parts[3].strip()

                sample = (turn_1, turn_2, turn_3)
                dataset.append(sample)
        return dataset
