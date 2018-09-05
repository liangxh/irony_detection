# -*- coding: utf-8 -*-
import commandr
from collections import defaultdict


@commandr.command
def merge(input_filename_list, output_filename):
    """
    结合多个形如<token>(tab)<count>的字典文件生成单个字典文件

    :param input_filename_list: string, "FILENAME1,FILENAME2,FILENAME1,"
    :param output_filename:
    :return:
    """
    filename_list = input_filename_list.split(',')

    token_count = defaultdict(lambda: 0)
    for filename in filename_list:
        with open(filename, 'r') as file_obj:
            for line in file_obj:
                line = line.strip()
                if line == '':
                    continue
                parts = line.split('\t')
                token = parts[0]
                count = int(parts[1])
                token_count[token] += count

    token_count = sorted(token_count.items(), key=lambda item: -item[1])
    with open(output_filename, 'w') as file_obj:
        for token, count in token_count:
            file_obj.write('{}\t{}\n'.format(token, count))


if __name__ == '__main__':
    commandr.Run()
