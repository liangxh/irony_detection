# -*- coding: utf-8 -*-
import numpy as np
import json
from nlp.model.config import config


class Word2VecBin(object):
    """
    由于word2ved的bin文件一般比较大, 需要首先为字典建立索引
    """
    def __init__(self, filename_model, filename_index):
        self.file_model = open(filename_model, 'r')
        header = self.file_model.readline()
        vocab_size, dim = map(int, header.split())
        self.dim = dim
        self.index = json.load(open(filename_index, 'r'))
        self._binary_len = np.dtype(np.float32).itemsize * dim

    def get(self, vocab):
        if isinstance(vocab, str):
            vocab = vocab.decode('utf8')
        offset = self.index.get(vocab)
        if offset is None:
            return None
        else:
            self.file_model.seek(offset)
            bytes_vec = self.file_model.read(self._binary_len)
            vec = np.fromstring(bytes_vec, dtype=np.float32)
            return vec

    @classmethod
    def load_vocab(cls, filename):
        vocab_list = list()
        with open(filename, 'r') as file_obj:
            for line in file_obj:
                line = line.strip()
                if line == '':
                    continue
                parts = line.strip('\t')
                vocab = parts[0]
                vocab_list.append(vocab)
        return vocab_list

    @classmethod
    def save_index(cls, filename, index):
        with open(filename, 'w') as file_obj:
            for vocab, offset in index.items():
                file_obj.write('{}\t{}\n').format(vocab, offset)

    @classmethod
    def load_index(cls, filename):
        index = dict()
        with open(filename, 'r') as file_obj:
            for line in file_obj:
                line = line.strip()
                if line == '':
                    continue
                parts = line.split('\t')
                vocab = parts[0]
                offset = int(parts[1])
                index[vocab] = offset
        return index

    @classmethod
    def build_index(cls, filename_model, filename_vocab, filename_index):
        vocab_list = cls.load_vocab(filename_vocab)
        vocab_set = set(vocab_list)
        index = dict()

        with open(filename_model, 'r') as file_obj:
            header = file_obj.readline().decode('utf8')
            vocab_size, dim = map(int, header.split())  # throws for invalid file format
            binary_len = np.dtype(np.float32).itemsize * dim
            for _ in xrange(vocab_size):
                ch_list = []
                while True:
                    ch = file_obj.read(1)
                    if ch == b' ':
                        break
                    if ch != b'\n':  # ignore newlines in front of words (some binary files have newline, some don't)
                        ch_list.append(ch)
                token = unicode(b''.join(ch_list), encoding='latin-1')
                offset = file_obj.tell()

                if token in vocab_set:
                    index[token] = offset
                    if len(index) == len(vocab_set):
                        break
                file_obj.read(binary_len)

        cls.save_index(filename_index, index)
