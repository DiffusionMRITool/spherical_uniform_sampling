#!/usr/bin/env python
"""
Usage:
    direction_order.py BVEC BVAL [-v | -q] --output=OUTPUT [-t TIME] [-n NUM] [-w WEIGHT]
    direction_order.py BVEC [-v | -q] --output=OUTPUT [-t TIME] [-n NUM]

Options:
    -o OUTPUT, --output OUTPUT  output file 
    -v, --verbose               output gurobi message
    -q, --quiet                 do not output gurobi message
    -n NUM, --number NUM        number of points per split. [default: 3]
    -w WEIGHT, --weight WEIGHT  Weight for single shell term, 1-weight for mutiple shell term. [default: 0.5]
    -t TIME, --time_limit TIME  Maximum time to run milp algorithm    [default: 600]
"""
import os
from docopt import docopt
from sampling import optimalSplit, optimalSplitMultiShell
from io_util import read_bvec, read_bvec_bval, write_bvec, write_bval, do_func


def gen_split(num_per_split: int, n: int):
    l = [num_per_split]
    s = num_per_split
    while s + num_per_split <= n:
        l.append(num_per_split)
        s += num_per_split
    if s < n:
        l.append(n - s)
    return l


if __name__ == "__main__":
    arguments = docopt(__doc__)

    inputBVecFile = arguments["BVEC"]

    time = float(arguments["--time_limit"])

    output_flag = 1 if arguments["--verbose"] else 0

    num = int(arguments["--number"])

    weight = float(arguments["--weight"])

    outputFile = arguments["--output"]
    root, ext = os.path.splitext(outputFile)

    if arguments["BVAL"]:
        inputBValFile = arguments["BVAL"]
        bvalues, bvecs = read_bvec_bval(inputBVecFile, inputBValFile)

        bvec, bval = do_func(
            output_flag,
            optimalSplitMultiShell,
            bvecs,
            bvalues,
            gen_split(num, sum(len(l) for l in bvecs)),
            w=weight,
            time_limit=time,
            output_flag=output_flag,
        )
        write_bvec(f"{root}_bvec{ext}", bvec)
        write_bval(f"{root}_bval{ext}", bval)
    else:
        bvec = read_bvec(inputBVecFile)
        output = do_func(
            output_flag,
            optimalSplit,
            bvec,
            gen_split(num, len(bvec)),
            time,
            output_flag,
        )
        write_bvec(f"{root}{ext}", output)
