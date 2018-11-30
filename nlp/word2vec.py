# -*- coding: utf-8 -*-
from __future__ import print_function
import commandr
import os
import numpy as np


class PlainModelWriter(object):
    def __init__(self, filename):
        self.file_obj = open(filename, 'w')

    def add(self, token, vec):
        self.file_obj.write('{}\t{}\n'.format(
            token.encode('utf8'), '\t'.join(map(str, vec))
        ))

    def close(self):
        self.file_obj.close()


class PlainModel(object):
    def __init__(self, filename_model, separator='\t'):
        self.separator = separator
        self.file_model = open(filename_model, 'r')
        self.index = self.build_index(filename_model)
        self.dim = self.get_dim(filename_model)

    def build_index(self, filename_model):
        index = dict()

        with open(filename_model, 'r') as file_obj, open(filename_model, 'r') as _file_obj:
            while True:
                offset = file_obj.tell()
                line = file_obj.readline()
                if line == '':
                    break

                parts = line.strip().split(self.separator)
                if len(parts) == 0:
                    # 部分模型文件头会带 n_vocab, n_dim, 直接跳过
                    offset += len(line)
                    continue

                token = parts[0]
                if hasattr(str, 'decode'):
                    token = token.decode('utf8')
                index[token] = offset

                # 检查index创建是否正确
                _file_obj.seek(offset)
                _line = _file_obj.readline()
                assert _line.strip() == line.strip()

                offset += len(line)
        return index

    def get_dim(self, filename_model):
        with open(filename_model, 'r') as file_obj:
            line = file_obj.readline()
            parts = line.split(self.separator)
            dim = len(parts) - 1
        return dim

    def get(self, vocab):
        if isinstance(vocab, str) and hasattr(str, 'decode'):
            vocab = vocab.decode('utf8')

        offset = self.index.get(vocab)
        if offset is None:
            return None
        else:
            self.file_model.seek(offset)
            line = self.file_model.readline()
            line = line.strip()
            parts = line.split(self.separator)
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
        """
        :param vocab: string
        :return: 若vocab在索引当中则返回词向量(list of float)，否则返回None
        """
        if isinstance(vocab, str) and hasattr(str, 'decode'):
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
        """
        :param filename: string, 字典文件名, 文件格式形如 <vocab>(tab)xxxxxxxx
        :return: list of string
        """
        vocab_list = set()
        filenames = filename.split(',')
        print(filenames)

        for filename in filenames:
            with open(filename, 'r') as file_obj:
                for line in file_obj:
                    line = line.strip()
                    if line == '':
                        continue
                    parts = line.split('\t')
                    vocab = parts[0]
                    if hasattr(str, 'decode'):
                        vocab = vocab.decode('utf8')
                    vocab_list.add(vocab)
            print(len(vocab_list))
        return vocab_list

    @classmethod
    def save_index(cls, filename, index):
        """
        :param filename: string
        :param index: dict<string, int>, vocab以及其对应文件中的offset
        :return:
        """
        with open(filename, 'w') as file_obj:
            for vocab, offset in index.items():
                file_obj.write('{}\t{}\n'.format(vocab, offset))

    @classmethod
    def load_index(cls, filename):
        """
        :param filename: string
        :return: dict<string, int>, vocab以及其对应文件中的offset
        """
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
    def init(cls, filename_model, filename_index, filename_vocab=None):
        """
        实例初始化
        若filename_index未生成 则要求filename_vocab不是空，利用之生成index文件
        否则直接加载

        :param filename_model: string, 待加载的word2vec的二进制模型文件
        :param filename_index: string, 生成的索引存储的文件路径
        :param filename_vocab: string
        :return: BinModel实例
        """
        if os.path.exists(filename_index):
            return cls(filename_model, filename_index)

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
        """
        生成PlainModel的文件

        :param filename_output: 输出文件的路径
        :return:
        """
        writer = PlainModelWriter(filename_output)
        for token in self.index.keys():
            vec = self.get(token)
            writer.add(token, vec)
        writer.close()


@commandr.command('w2v')
def build_plain(input_filename, vocab_filename, output_filename):
    """
    # naive

    python nlp/word2vec.py w2v \
        ~/Downloads/GoogleNews-vectors-negative300.bin \
        ../irony_detection_data/semeval2018_task3/all.vocab.naive \
        ../irony_detection_data/semeval2018_task3/all.w2v.google_naive

    python nlp/word2vec.py w2v \
        ~/Downloads/GoogleNews-vectors-negative300.bin \
        ../irony_detection_data/semeval2018_task3/all.vocab.naive,../irony_detection_data/semeval2018_task1/all.vocab.naive \
        ../irony_detection_data/semeval2018_task1/all.w2v.google_naive

    python nlp/word2vec.py w2v \
        ~/Downloads/GoogleNews-vectors-negative300.bin \
        ../irony_detection_data/semeval2018_task3/all.vocab.naive,../irony_detection_data/semeval2014_task9/all.vocab.naive \
        ../irony_detection_data/semeval2014_task9/all.w2v.google_naive

    python nlp/word2vec.py index \
        ~/Downloads/GoogleNews-vectors-negative300.bin \
        ../irony_detection_data/semeval2018_task3/all.vocab.naive,../irony_detection_data/semeval2015_task9/all.vocab.naive \
        ../irony_detection_data/semeval2015_task11/all.w2v.google_naive

    # ek

    python nlp/word2vec.py w2v \
        ~/Downloads/GoogleNews-vectors-negative300.bin \
        ../irony_detection_data/semeval2018_task3/all.vocab.ek \
        ../irony_detection_data/semeval2018_task3/all.w2v.google_ek

    python nlp/word2vec.py w2v \
        ~/Downloads/GoogleNews-vectors-negative300.bin \
        ../irony_detection_data/semeval2018_task3/all.vocab.ek,../irony_detection_data/semeval2018_task1/all.vocab.ek \
        ../irony_detection_data/semeval2018_task1/all.w2v.google_ek

    python nlp/word2vec.py w2v \
        ~/Downloads/GoogleNews-vectors-negative300.bin \
        ../irony_detection_data/semeval2018_task3/all.vocab.ek,../irony_detection_data/semeval2014_task9/all.vocab.ek \
        ../irony_detection_data/semeval2014_task9/all.w2v.google_ek

    python nlp/word2vec.py w2v \
        ~/Downloads/GoogleNews-vectors-negative300.bin \
        ../irony_detection_data/semeval2018_task3/all.vocab.ek,../irony_detection_data/semeval2015_task11/all.vocab.ek \
        ../irony_detection_data/semeval2015_task11/all.w2v.google_ek

    """
    BinModel.init(input_filename, '_index.tmp', vocab_filename).to_plain(output_filename)
    os.remove('_index.tmp')


@commandr.command('glove')
def build_glove(input_filename, vocab_filename, output_filename):
    """
    dim=25
    python nlp/word2vec.py glove \
        ~/lab/glove/twitter.${dim}d.txt \
        ../irony_detection_data/semeval2018_task3/all.vocab.ek \
        ../irony_detection_data/semeval2018_task3/all.w2v.glove_${dim}_ek

    python nlp/word2vec.py glove \
        ~/Downloads/ntua_twitter_300.txt \
        ../irony_detection_data/semeval2018_task3/all.vocab.ek \
        ../irony_detection_data/semeval2018_task3/all.w2v.ntua_ek

    python nlp/word2vec.py glove \
        ~/Downloads/ntua_twitter_300.txt \
        ../irony_detection_data/semeval2019_task3_dev/all.vocab.ek \
        ../irony_detection_data/semeval2019_task3_dev/all.w2v.ntua_ek
    """
    original_model = PlainModel(input_filename, separator=' ')
    vocabs = BinModel.load_vocab(vocab_filename)

    writer = PlainModelWriter(output_filename)
    for vocab in vocabs:
        vec = original_model.get(vocab)
        if vec is not None:
            writer.add(vocab, vec)
    writer.close()


@commandr.command('test_glove')
def test_glove(filename):
    PlainModel(filename)


if __name__ == '__main__':
    commandr.Run()
