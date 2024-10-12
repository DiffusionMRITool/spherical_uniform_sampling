#!/usr/bin/env python
"""
Description:
    Optimize the order of a given sampling scheme.

Usage:
    direction_order.py BVEC BVAL [-v | -q] --output=OUTPUT [-t TIME] [-s SPLIT] [-w WEIGHT] [--fslgrad]
    direction_order.py BVEC [-v | -q] --output=OUTPUT [-t TIME] [-s SPLIT] [--fslgrad]

Options:
    -o OUTPUT, --output OUTPUT  Output file 
    -v, --verbose               Output gurobi message
    -q, --quiet                 Don't output any message
    -s SPLIT, --split SPLIT     Number of points per split for order optimization. [default: 3]
    -w WEIGHT, --weight WEIGHT  Weight for single shell term, 1-weight for mutiple shell term. [default: 0.5]
    -t TIME, --time_limit TIME  Maximum time to run milp algorithm    [default: 600]
    --fslgrad                   If set, program will read and write in fslgrad format

Examples:
    # Optimize the order of a single shell sampling scheme with bvec.txt
    direction_order.py bvec.txt --output ordered.txt
    # Optimize the order of a sampling scheme with bvec and bval. This works for both single and multiple shell case.
    direction_order.py bvec.txt bval.txt --output ordered.txt
"""
import os

from docopt import docopt

from spherical_uniform_sampling.lib.io_util import (
    arg_bool,
    arg_values,
    do_func,
    read_bvec,
    read_bvec_bval,
    write_bval,
    write_bvec,
)
from spherical_uniform_sampling.sampling.packing_density import (
    incremental_sorting_multi_shell,
    incremental_sorting_single_shell,
)


def gen_split(num_per_split: int, n: int):
    l = [num_per_split]
    s = num_per_split
    while s + num_per_split <= n:
        l.append(num_per_split)
        s += num_per_split
    if s < n:
        l.append(n - s)
    return l


def main(arguments):
    fsl_flag = arg_bool(arguments["--fslgrad"], bool)
    inputBVecFile = arguments["BVEC"]

    time = arg_values(arguments["--time_limit"], float, 1, is_single=True)

    output_flag = 1
    if arguments["--verbose"]:
        output_flag = 2
    if arguments["--quiet"]:
        output_flag = 0

    num = arg_values(arguments["--split"], int, 1, is_single=True)

    weight = arg_values(arguments["--weight"], float, 1, is_single=True)

    outputFile = arguments["--output"]
    root, ext = os.path.splitext(outputFile)

    if arguments["BVAL"]:
        inputBValFile = arguments["BVAL"]
        bvalues, bvecs = read_bvec_bval(inputBVecFile, inputBValFile, fsl_flag)

        if len(bvalues) == 1:
            bvec = do_func(
                output_flag,
                incremental_sorting_single_shell,
                "order",
                bvecs[0],
                gen_split(num, len(bvecs[0])),
                time,
            )
            bval = [bvalues[0] for _ in range(len(bvecs[0]))]
        else:
            bvec, bval = do_func(
                output_flag,
                incremental_sorting_multi_shell,
                "order",
                bvecs,
                bvalues,
                gen_split(num, sum(len(l) for l in bvecs)),
                w=weight,
                time_limit=time,
            )
        write_bvec(
            f"{root}_bvec{ext}",
            bvec,
            fsl_flag,
            output_flag,
            "order optimized b-vectors",
        )
        write_bval(
            f"{root}_bval{ext}", bval, fsl_flag, output_flag, "order optimized b-values"
        )
    else:
        bvec = read_bvec(inputBVecFile, fsl_flag)
        output = do_func(
            output_flag,
            incremental_sorting_single_shell,
            "order",
            bvec,
            gen_split(num, len(bvec)),
            time,
            output_flag,
        )
        write_bvec(
            f"{root}{ext}", output, fsl_flag, output_flag, "order optimized b-vectors"
        )


if __name__ == "__main__":
    arguments = docopt(__doc__)

    main(arguments)
