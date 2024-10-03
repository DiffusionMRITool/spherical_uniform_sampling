#!/usr/bin/env python
"""
Description:
    Order a given sampling scheme.

Usage:
    direction_order.py BVEC BVAL [-v] --output=OUTPUT [-t TIME] [-n NUM] [-w WEIGHT] [--fslgrad]
    direction_order.py BVEC [-v] --output=OUTPUT [-t TIME] [-n NUM] [--fslgrad]

Options:
    -o OUTPUT, --output OUTPUT  Output file 
    -v, --verbose               Output gurobi message
    -n NUM, --number NUM        Number of points per split. [default: 3]
    -w WEIGHT, --weight WEIGHT  Weight for single shell term, 1-weight for mutiple shell term. [default: 0.5]
    -t TIME, --time_limit TIME  Maximum time to run milp algorithm    [default: 600]
    --fslgrad                   If set, program will read and write in fslgrad format

Examples:
    python -m qspace_direction.direction_order scheme.txt --output ordered.txt
    python -m qspace_direction.direction_order bvec.txt bval.txt --output ordered.txt
"""
import os

from docopt import docopt

from .lib import do_func, read_bvec, read_bvec_bval, write_bval, write_bvec
from .sampling import incremental_sorting_multi_shell, incremental_sorting_single_shell


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

    fsl_flag = True if arguments["--fslgrad"] else False
    inputBVecFile = arguments["BVEC"]

    time = float(arguments["--time_limit"])

    output_flag = 1 if arguments["--verbose"] else 0

    num = int(arguments["--number"])

    weight = float(arguments["--weight"])

    outputFile = arguments["--output"]
    root, ext = os.path.splitext(outputFile)

    if arguments["BVAL"]:
        inputBValFile = arguments["BVAL"]
        bvalues, bvecs = read_bvec_bval(inputBVecFile, inputBValFile, fsl_flag)

        if len(bvalues) == 1:
            bvec = do_func(
                output_flag,
                incremental_sorting_single_shell,
                bvecs,
                gen_split(num, len(bvecs)),
                time,
                output_flag,
            )
            bval = [bvalues[0] for _ in len(bvecs)]
        else:
            bvec, bval = do_func(
                output_flag,
                incremental_sorting_multi_shell,
                bvecs,
                bvalues,
                gen_split(num, sum(len(l) for l in bvecs)),
                w=weight,
                time_limit=time,
                output_flag=output_flag,
            )
        write_bvec(f"{root}_bvec{ext}", bvec, fsl_flag)
        write_bval(f"{root}_bval{ext}", bval, fsl_flag)
    else:
        bvec = read_bvec(inputBVecFile, fsl_flag)
        output = do_func(
            output_flag,
            incremental_sorting_single_shell,
            bvec,
            gen_split(num, len(bvec)),
            time,
            output_flag,
        )
        write_bvec(f"{root}{ext}", output, fsl_flag)
