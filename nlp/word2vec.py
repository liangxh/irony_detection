# -*- coding: utf-8 -*-
import numpy as np
import json


class PlainModelWriter(object):
    def __init__(self, filename):
        self.file_obj = open(filename, 'w')

    def add(self, token, vec):
        self.file_obj.write('{}\t{}\n'.format(
            token, '\t'.join(map(str, vec))
        ))

    def close(self):
        self.file_obj.close()


class PlainModel(object):
    """

    """
    def __init__(self, filename_model):
        self.file_model = open(filename_model, 'r')
        self.index = self.build_index(filename_model)
        self.dim = self.get_dim(filename_model)

    @classmethod
    def build_index(cls, filename_model):
        index = dict()
        with open(filename_model, 'r') as file_obj:
            offset = 0
            for line in file_obj:
                parts = line.split('\t', 1)
                token = parts[0]
                index[token] = offset
                offset = file_obj.tell()
        return index

    @classmethod
    def get_dim(cls, filename_model):
        with open(filename_model, 'r') as file_obj:
            line = file_obj.readline()
            parts = line.split('\t')
            dim = len(parts) - 1
        return dim

    def get(self, vocab):
        if isinstance(vocab, str):
            vocab = vocab.decode('utf8')

        offset = self.index.get(vocab)
        if offset is None:
            return None
        else:
            self.file_model.seek(offset)
            line = self.file_model.readline()
            line = line.strip()
            parts = line.split('\t')
            vec = map(float, parts[1:])
            return vec


class BinModel(object):
    """
    由于word2ved的bin文件一般比较大, 需要首先为字典建立索引
    """
    def __init__(self, filename_model, filename_index):
        self.file_model = open(filename_model, 'r')
        header = self.file_model.readline()
        vocab_size, dim = map(int, header.split())
        self.dim = dim
        self.index = self.load_index(filename_index)
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
                parts = line.split('\t')
                vocab = parts[0]
                count = int(parts[1])
                if count > 1:
                    vocab_list.append(vocab)
        return vocab_list

    @classmethod
    def save_index(cls, filename, index):
        with open(filename, 'w') as file_obj:
            for vocab, offset in index.items():
                file_obj.write('{}\t{}\n'.format(vocab, offset))

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
    def create(cls, filename_model, filename_vocab, filename_index):
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
        return cls(filename_model, filename_index)

    def to_plain(self, filename_output):
        writer = Writer(filename_output)
        for token in self.index.keys():
            vec = self.get(token)
            writer.add(token, vec)
        writer.close()
