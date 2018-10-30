# -*- coding: utf-8 -*-
import commandr
import importlib
from dataset.common.const import *
from dataset.common.load import *


@commandr.command
def tf_idf(dataset_key, text_version):
    data_config = getattr(importlib.import_module('dataset.{}.config'.format(dataset_key)), 'config')
    for mode in [TRAIN, TEST]:
        text_path = data_config.path(mode, text_version)
        tokenized_list = load_tokenized_list(text_path)
