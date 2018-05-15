# -*- coding: utf-8 -*-
import commandr
from dataset.semeval2018.task3.process import Processor


@commandr.command
def load_train():
    loader = Processor()
    for subtask in ['A', 'B']:
        dataset = loader.load_train(subtask)
        print 'subtask {}: {}'.format(subtask, len(dataset))


@commandr.command
def load_test():
    loader = Processor()
    for subtask in ['A', 'B']:
        dataset = loader.load_test(subtask)
        print 'subtask {}: {}'.format(subtask, len(dataset))


if __name__ == '__main__':
    commandr.Run()
