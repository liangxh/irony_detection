# -*- coding: utf-8 -*-
import os


path_root = os.path.join(os.environ['HOME'], 'lab', 'SemEval2018-Task3')
path_datasets = os.path.join(path_root, 'datasets')

path_train_root = os.path.join(path_datasets, 'train')

path_train = os.path.join(path_train_root, 'SemEval2018-T3-train-task{subtask}.txt')
path_train_emoji = os.path.join(path_train_root, 'SemEval2018-T3-train-task{subtask}_emoji.txt')
path_train_emoji_tag = os.path.join(path_train_root, 'SemEval2018-T3-train-task{subtask}_emoji_ironyHashtags.txt')

path_test = os.path.join(path_datasets, 'test_Task{subtask}')
path_test_input = os.path.join(path_test, 'SemEval2018-T3_input_test_Task{subtask}.txt')
path_test_input_emoji = os.path.join(path_test, 'SemEval2018-T3_input_test_Task{subtask}_emoji.txt')

path_goldtest = os.path.join(path_datasets, 'goldtest_Task{subtask}', 'SemEval2018-T3_gold_test_task{subtask}_emoji.txt')
