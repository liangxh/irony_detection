# -*- coding: utf-8 -*-
import re
import commandr
from dataset.semeval2018_task3.process import Processor
from dataset.common.const import *
from dataset.semeval2018_task3.config import config


@commandr.command
def check_data():
    for sA, sB in zip(Processor.load_origin_train('A'), Processor.load_origin_train('B')):
        if sA[1] != sB[1]:
            print sA[1]
            print sB[1]
        if sA[0] != (0 if sB[0] == 0 else 1):
            print sA[0], sB[0]

    for sA, sB in zip(Processor.load_origin_test('A'), Processor.load_origin_test('B')):
        if sA[1] != sB[1]:
            print sA[1]
            print sB[1]
        if sA[0] != (0 if sB[0] == 0 else 1):
            print sA[0], sB[0]


@commandr.command
def build_text_label():
    for key, func in {TRAIN: Processor.load_origin_train, TEST: Processor.load_origin_test}.items():
        text_path = config.path(key, TEXT)
        label_A_path = config.path(key, LABEL, 'A')
        label_B_path = config.path(key, LABEL, 'B')

        labels_A = list()
        with open(text_path, 'w') as text_obj, open(label_A_path, 'w') as label_A_obj, open(label_B_path, 'w') as label_B_obj:
            for label, text in func('B'):
                text = re.sub('\s+', ' ', text)
                text_obj.write(text + '\n')
                label_B_obj.write(str(label) + '\n')
                label_A_obj.write(str(0 if label == 0 else 1) + '\n')
                labels_A.append(0 if label == 0 else 1)

        mismatch = 0
        for i, res in enumerate(func('A')):
            if not res[0] == labels_A[i]:
                mismatch += 1
        print key, mismatch


if __name__ == '__main__':
    commandr.Run()
