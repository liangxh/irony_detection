# -*- coding: utf-8 -*-


def iterate_file(filename):
    with open(filename, 'r') as file_obj:
        for line in file_obj:
            line = line.strip()
            if line == '':
                continue
            yield line
