from collections import defaultdict
import os
import sys
import numpy as np


def read_bvec(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        p = [list(map(float, l.split())) for l in lines if not l.isspace()]
    return np.array(p)


def read_bvec_bval(bvec_file, bval_file, start=0):
    with open(bvec_file, "r") as f:
        lines = f.readlines()
        bvec = [list(map(float, l.split())) for l in lines[start:]]
    with open(bval_file, "r") as f:
        lines = f.readlines()
        bval = [int(l) for l in lines[start:]]
    d = defaultdict(list)
    for val, vec in zip(bval, bvec):
        d[val].append(vec)
    ks, vs = [], []
    for k, v in d.items():
        ks.append(k)
        vs.append(np.array(v))
    return ks, vs


def write_bvec(filename, bvec):
    with open(filename, "w") as f:
        f.writelines(map(lambda x: f"{' '.join(map(str, x))}\n", bvec))


def write_bval(filename, bval):
    with open(filename, "w") as f:
        f.writelines(map(lambda x: f"{x}\n", bval))


def do_func(flag, f, *args, **kwargs):
    if flag:
        ret = f(*args, **kwargs)
    else:
        try:
            with open(os.devnull, "w", encoding="utf-8") as target:
                sys.stdout = target
                ret = f(*args, **kwargs)
        finally:
            sys.stdout = sys.__stdout__
    return ret
