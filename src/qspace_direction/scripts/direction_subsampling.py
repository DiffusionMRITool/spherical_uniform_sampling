#!/usr/bin/env python
"""
Description:
    Subsample a single/multiple set of points from a given single/multiple set of points.

Usage:
    direction_subsampling.py [-v] [-a] --input=INPUT --number=NUMBER --output=OUTPUT [--lower_bound LB] [-w WEIGHT] [-t TIME] [--fslgrad]

Options:
    -o OUTPUT, --output OUTPUT  Output file 
    -i INPUT, --input INPUT     Input bvec files
    -n NUMBER, --number NUMBER  Number chosen from each shell(in the same order of input files)
    --lower_bound LB            Lower bound for each shell and all shell conbined. This helps milp to optimize better
    -w WEIGHT, --weight WEIGHT  Weight for single shell term, 1-weight for mutiple shell term. [default: 0.5]
    -v, --verbose               Output gurobi message
    -a, --antipodal             Treat antipolar points as same
    -t TIME, --time_limit TIME  Maximum time to run milp algorithm    [default: 600]
    --fslgrad                   If set, program will read and write in fslgrad format

Examples:
    # Subsample 30 points from a single bvec scheme.
    python -m qspace_direction.direction_subsampling -i bvec.txt -n 30 -o subsample.txt 
    # Subsample 3 shells with each 10 points from a single bvec scheme.
    python -m qspace_direction.direction_subsampling -i bvec.txt -n 10,10,10 -o subsample.txt
    # Subsample from three shells, with each shell choosing 30 points
    python -m qspace_direction.direction_subsampling -i bvec_shell0.txt,bvec_shell1.txt,bvec_shell2.txt -n 30,30,30 -o subsample.txt

Reference:
    1. Jian Cheng, Dinggang Shen, Pew-Thian Yap and Peter J. Basser, "Single- and Multiple-Shell Uniform Sampling Schemes for Diffusion MRI Using Spherical Codes," in IEEE Transactions on Medical Imaging, vol. 37, no. 1, pp. 185-199
"""
import os

import numpy as np
from docopt import docopt

from qspace_direction.lib.io_util import (
    arg_bool,
    arg_values,
    do_func,
    read_bvec,
    write_bvec,
)
from qspace_direction.sampling.subsample import (
    multiple_subset_from_multiple_set,
    multiple_subset_from_single_set,
    single_subset_from_single_set,
)


def main(arguments):
    fsl_flag = arg_bool(arguments["--fslgrad"], bool)
    numbers = arg_values(arguments["--number"], int)
    time = arg_values(arguments["--time_limit"], float, is_single=True)
    output_flag = arg_bool(arguments["--verbose"], int)

    outputFile = arguments["--output"]
    root, ext = os.path.splitext(outputFile)
    antipodal = not arg_bool(arguments["--asym"], bool)
    lb = (
        None
        if arguments["--lower_bound"] is None
        else list(map(float, arguments["--lower_bound"].split(",")))
    )
    if lb:
        assert (
            len(lb) == len(numbers) + 1
        ), "number of lower bounds and number of shells mismatch "
    weight = arg_values(arguments["--weight"], float, is_single=True)
    inputBvec = arg_values(arguments["--input"], lambda f: read_bvec(f, fsl_flag))

    if len(inputBvec) == 1:
        if len(numbers) == 1:
            output = [
                do_func(
                    output_flag,
                    single_subset_from_single_set,
                    inputBvec[0],
                    numbers[0],
                    lb=lb,
                    time_limit=time,
                    antipodal=antipodal,
                ),
            ]
        else:
            output = do_func(
                output_flag,
                multiple_subset_from_single_set,
                inputBvec[0],
                np.array(numbers),
                w=weight,
                lb=lb,
                time_limit=time,
                antipodal=antipodal,
            )
    else:
        len(inputBvec) == len(numbers)
        output = do_func(
            output_flag,
            multiple_subset_from_multiple_set,
            inputBvec,
            np.array(numbers),
            w=weight,
            lb=lb,
            time_limit=time,
            antipodal=antipodal,
        )

    if len(inputBvec) == 1 and len(numbers) == 1:
        realPath = f"{root}{ext}"
        write_bvec(realPath, output[0], fsl_flag)
    else:
        for i, points in enumerate(output):
            realPath = f"{root}_shell{i}{ext}"
            write_bvec(realPath, points, fsl_flag)


if __name__ == "__main__":
    arguments = docopt(__doc__)

    main(arguments)