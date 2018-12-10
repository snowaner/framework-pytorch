#! /usr/bin/env python
#################################################################################
#     File Name           :     tools.py
#     Created By          :     WanCQ
#     Creation Date       :     [2018-12-04 03:32]
#     Last Modified       :     [2018-12-04 03:42]
#     Description         :     some tools for the framework
#################################################################################

import errno

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def readtxt(path, label_info):
    lines = []
    with open(path, 'r') as f:
        data = f.readlines()
    for d in data:
        d.replace(',', ' ')
        d.replace(';', ' ')
        ret = d.split()
        for i, info in enumerate(label_info):
            if info == 'str':
                continue
            elif info == 'int':
                ret[i] = int(ret[i])
            elif info == 'float':
                ret[i] = float(ret[i])
            else:
                raise RuntimeError('Invalid type of label {}'
                                   .format(info))
        lines.append(ret)
    return lines
