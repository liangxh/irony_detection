# -*- coding: utf-8 -*-
import json


def load_tokenized_list(path):
    text_list = load_text_list(path)
    if hasattr(str, 'decode'):
        return list(map(lambda line: line.decode('utf8').split(' '), text_list))
    else:
        return list(map(lambda line: line.split(' '), text_list))


def load_text_list(path):
    """
    :param path: string
    :return: list of string
    """
    text_list = list()
    with open(path, 'r') as file_obj:
        for line in file_obj:
            line = line.strip()
            if line == '':
                continue
            text = line
            text_list.append(text)
    return text_list


def load_label_list(path):
    """
    :param path: string
    :return: list of int
    """
    label_list = list()
    with open(path, 'r') as file_obj:
        for line in file_obj:
            line = line.strip()
            if line == '':
                continue
            label = int(line)
            label_list.append(label)
    return label_list


def load_vocab_list(path):
    """
    :param path: string
    :return: list of dict, {"t": "xxx", "tf": int, "df": int}
    """
    vocab_meta_list = list()
    with open(path, 'r') as file_obj:
        for line in file_obj:
            line = line.strip()
            if line == '':
                continue
            data = json.loads(line)
            vocab_meta_list.append(data)
    return vocab_meta_list


def load_feat(path, separator='\t'):
    """
    :param path: string
    :param separator: string
    :return:
        vec_list, list of list of float
        dim, float
    """
    vec_list = list()
    dim = None
    with open(path) as file_obj:
        for i, line in enumerate(file_obj):
            line = line.strip()
            if line == '':
                continue
            vec = list(map(float, line.split(separator)))
            if dim is None:
                dim = len(vec)
            elif len(vec) != dim:
                raise Exception('expected dim={}, got dim={} at line {}'.format(dim, len(vec), i + 1))
            vec_list.append(vec)
    return vec_list, dim


def seq_to_len_list(seq_list):
    return list(map(len, seq_list))


def zero_pad_seq_list(seq_list, seq_len):
    return list(map(lambda _seq: _seq + [0] * (seq_len - len(_seq)), seq_list))


def trim_tid_list(tid_list, max_len):
    return list(map(lambda _seq: _seq[:max_len], tid_list))
