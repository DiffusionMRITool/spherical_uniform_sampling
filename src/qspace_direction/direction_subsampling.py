#!/usr/bin/env python
"""
Usage:
    direction_subsampling.py [-v | -q] [-a] --input=INPUT --number=NUMBER --output=OUTPUT [--lower_bound LB] [-t TIME] [--fslgrad]

Options:
    -o OUTPUT, --output OUTPUT  output file 
    -i INPUT, --input INPUT     input bvec files
    -n NUMBER, --number NUMBER  number chosen from each shell(in the same order of input files)
    --lower_bound LB            lower bound for each shell and all shell conbined. This helps milp to optimize better
    -v, --verbose               output gurobi message
    -q, --quiet                 do not output gurobi message
    -a, --antipodal             treat antipolar points as same
    -t TIME, --time_limit TIME  maximum time to run milp algorithm    [default: 600]
    --fslgrad,                  if set, program will read and write in fslgrad format

Reference:
    1. Jian Cheng, Dinggang Shen, Pew-Thian Yap and Peter J. Basser, "Single- and Multiple-Shell Uniform Sampling Schemes for Diffusion MRI Using Spherical Codes," in IEEE Transactions on Medical Imaging, vol. 37, no. 1, pp. 185-199
"""
import os

import numpy as np
from docopt import docopt
from sampling import (
    multiple_subset_from_multiple_set,
    multiple_subset_from_single_set,
    single_subset_from_single_set,
)
from io_util import read_bvec, write_bvec, do_func


if __name__ == "__main__":
    arguments = docopt(__doc__)

    fsl_flag = True if arguments["--fslgrad"] else False
    inputFiles = arguments["--input"].split(",")
    numbers = list(map(int, arguments["--number"].split(",")))
    time = float(arguments["--time_limit"])
    output_flag = 1 if arguments["--verbose"] else 0
    outputFile = arguments["--output"]
    root, ext = os.path.splitext(outputFile)
    antipodal = arguments["--antipodal"]
    lb = (
        None
        if arguments["--lower_bound"] is None
        else list(map(float, arguments["--lower_bound"].split(",")))
    )
    if lb:
        assert (
            len(lb) == len(numbers) + 1
        ), "number of lower bounds and number of shells mismatch "
    inputBvec = [read_bvec(f, fsl_flag) for f in inputFiles]

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
