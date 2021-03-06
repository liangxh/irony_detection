# -*- coding: utf-8 -*-
from __future__ import print_function
import commandr
from dataset.common.const import *
from dataset.semeval2018_task1.config import config
from dataset.semeval2018_task1.process import Processor, label_names


@commandr.command
def build_text_label():
    path_list = {
        TRAIN: config.path_E_c_train,
        TEST: config.path_E_c_dev
    }
    for key, path in path_list.items():
        text_path = config.path(key, TEXT)

        label_objs = {
            _label_name: open(config.path(key, LABEL, _label_name), 'w')
            for _label_name in label_names
        }

        with open(text_path, 'w') as text_obj:
            for label, text in Processor.load_origin(path):
                text_obj.write(text + '\n')
                for _label_name, _label in label.items():
                    label_objs[_label_name].write('{}\n'.format(_label))
        list(map(lambda _obj: _obj.close(), label_objs.values()))


if __name__ == '__main__':
    commandr.Run()
