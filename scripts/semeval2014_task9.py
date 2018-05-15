# -*- coding: utf-8 -*-
import commandr


@commandr.command
def build_origin(infile, outfile):
    label_map = {
        'negative': 0,
        'neutral': 1,
        'positive': 2,
        'objective': 1,  # 根據官方README, objective在taskB中视为neutral
    }

    with open(infile, 'r') as in_obj, open(outfile, 'w') as out_obj:
        for line in in_obj:
            line = line.strip()
            if line == '':
                continue
            parts = line.split('\t', 3)
            label = parts[-2]

            if label in label_map:
                label_id = label_map[label]
                text = parts[-1].replace('\t', ' ')
                out_obj.write('{}\t{}\n'.format(label_id, text))
            elif label.find('-OR-') < 0:
                raise Exception('unknown label: {}'.format(label))


if __name__ == '__main__':
    commandr.Run()
