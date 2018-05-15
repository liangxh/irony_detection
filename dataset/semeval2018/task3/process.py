# -*- coding: utf-8 -*-
from dataset.semeval2018.task3 import config


class Processor(object):
    @classmethod
    def load_train(cls, subtask):
        path = config.path_train_emoji.format(subtask=subtask)
        dataset = list()
        with open(path, 'r') as file_obj:
            file_obj.readline()
            for line in file_obj:
                line = line.strip()
                if line == '':
                    continue
                parts = line.split('\t')
                text = parts[2]
                label = int(parts[1])
                sample = (text, label)
                dataset.append(sample)
        return dataset

    @classmethod
    def load_test(cls, subtask):
        labels = list()
        texts = list()

        path = config.path_goldtest.format(subtask=subtask)
        with open(path, 'r') as file_obj:
            file_obj.readline()
            for line in file_obj:
                line = line.strip()
                if line == '':
                    continue
                parts = line.split('\t')
                label = int(parts[1])
                labels.append(label)

        path = config.path_test_input_emoji.format(subtask=subtask)
        with open(path, 'r') as file_obj:
            file_obj.readline()
            for line in file_obj:
                line = line.strip()
                if line == '':
                    continue
                parts = line.split('\t')
                text = parts[-1]
                texts.append(text)

        return list(zip(texts, labels))
