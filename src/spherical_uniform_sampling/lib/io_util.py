import argparse
import os
import sys
from collections import defaultdict
from typing import List

import numpy as np


def read_bvec(filename: str, fsl_flag=False):
    """Read from bvec file

    Args:
        filename (str): filename
        fsl_flag (bool, optional): Whether the bvec file is fsl format. Defaults to False.

    Returns:
        np.ndarray shape (N * 3) : b-vectors
    """
    with open(filename, "r") as f:
        lines = f.readlines()
        p = np.array([list(map(float, l.split())) for l in lines if not l.isspace()])
    if fsl_flag:
        p = p.T
    return p


def read_bvec_bval(bvec_file: str, bval_file: str, fsl_flag=False):
    """Read from corresponding bvec and bval files

    Args:
        bvec_file (str): bvec filename
        bval_file (str): bval filename
        fsl_flag (bool, optional): Whether the input file is in fsl format. Defaults to False.

    Returns:
        (list[int], list[np.ndarray]): distinct b-values and their corresponding v-vectors
    """
    with open(bvec_file, "r") as f:
        lines = f.readlines()
        bvec = [list(map(float, l.split())) for l in lines]
    if fsl_flag:
        bvec = list(zip(*bvec))
    with open(bval_file, "r") as f:
        lines = f.readlines()
        if fsl_flag:
            bval = list(map(int, lines[0].split()))
        else:
            bval = [float(l) for l in lines]
    d = defaultdict(list)
    for val, vec in zip(bval, bvec):
        d[val].append(vec)
    ks, vs = [], []
    for k, v in d.items():
        ks.append(k)
        vs.append(np.array(v))
    return ks, vs


def write_bvec(
    filename: str, bvec: np.ndarray, fsl_flag=False, verbose_flag=1, message=""
):
    """write bvec to file

    Args:
        filename (str): bvec filename
        bvec (np.ndarray): b-vectors
        fsl_flag (bool, optional): Whether to write in fsl format. Defaults to False.
        verbose_flag (int, optional): Whether to output help message
        message (str, optinal): message to write
    """
    with open(filename, "w") as f:
        if fsl_flag:
            bvec = list(zip(*bvec))
        f.writelines(map(lambda x: f"{' '.join(map(str, x))}\n", bvec))
    if verbose_flag:
        print(f"Save {message} to {filename}")


def write_bval(
    filename: str, bval: List[int], fsl_flag=False, verbose_flag=1, message=""
):
    """write bval to file

    Args:
        filename (str): bval filename
        bval (List[int]): b-values
        fsl_flag (bool, optional): Whether to write in fsl format. Defaults to False.
        verbose_flag (int, optional): Whether to output help message
        message (str, optinal): message to write
    """
    with open(filename, "w") as f:
        if fsl_flag:
            f.write(f"{' '.join(map(str, bval))}")
        else:
            f.writelines(map(lambda x: f"{x}\n", bval))
    if verbose_flag:
        print(f"Save {message} to {filename}")


def combine_bvec_bval(bvec: List[np.ndarray], bval: List[float]):
    """Combine list of bvec and bval into a single list

    Args:
        bvec (List[np.ndarray]): list of b-vectors
        bval (List[float]): list of b-values

    Returns:
        (np.ndarray, list[float]): A single list of bvecs and bvals
    """
    N_s = [len(l) for l in bvec]
    bvalList = sum(([b for _ in range(n_s)] for n_s, b in zip(N_s, bval)), [])
    return np.concatenate(bvec), bvalList


def arg_values(value, typefunc, numberOfValues=-1, is_single=False):
    """split comma seperated value and convert them using typefunc

    Args:
        value (Any): value
        typefunc : conversion function
        numberOfValues (int) : number of values,f numberOfValues < 0, it supports arbitrary number of inputs
        is_single (bool) : whether to choose value as a single item rather than list

    Returns:
        result
    """
    if not value:
        return None
    value = value.strip()
    if value[0] == "(" and value[-1] == ")":
        value = value[1:-1]
    values = value.split(",")
    if numberOfValues > 0 and len(values) != numberOfValues:
        raise argparse.ArgumentError
    if is_single:
        return typefunc(values[0])
    return list(map(typefunc, values))


def arg_bool(value, typefunc):
    """change value into typefunc bool

    Args:
        value (Any): value
        typefunc : conversion function

    Returns:
        result
    """
    if value:
        return typefunc(1)
    else:
        return typefunc(0)


def do_func(flag, f, f_name, *args, **kwargs):
    """execute and return f(*args, **kwargs), and optionally redirect output to devnull

    Args:
        flag (int): verbose level
        f (function): function to execute
        f_name (str): funtion name
    """
    if flag >= 1:
        print(f"Running {f_name} optimization ...")
    if flag == 2:
        ret = f(*args, **kwargs)
    else:
        try:
            with open(os.devnull, "w", encoding="utf-8") as target:
                sys.stdout = target
                ret = f(*args, **kwargs)
        finally:
            sys.stdout = sys.__stdout__
    if flag >= 1:
        print(f"Done")
    return ret
