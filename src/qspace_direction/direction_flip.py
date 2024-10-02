#!/usr/bin/env python
"""
Usage:
    direction_flip.py [-v | -q] --input=INPUT --output=OUTPUT [-t TIME] [-c CRITERIA] [--fslgrad]

Options:
    -o OUTPUT, --output OUTPUT        output file
    -i INPUT, --input INPUT           input bvec files  
    -v, --verbose                     output gurobi message
    -q, --quiet                       do not output gurobi message
    -c CRITERIA, --criteria CRITERIA  criteria Type(DISTANCE or ELECTROSTATIC). [default: ELECTROSTATIC]
    -t TIME, --time_limit TIME        maximum time to run milp algorithm    [default: 600]
    --fslgrad,                        if set, program will read and write in fslgrad format
"""
import os

from docopt import docopt
from sampling import (
    milpflip_SC,
    milpflip_EEM,
    milp_multi_shell_SC,
    milpflip_multi_shell_EEM,
)
from io_util import do_func, read_bvec, write_bvec


if __name__ == "__main__":
    arguments = docopt(__doc__)

    fsl_flag = True if arguments["--fslgrad"] else False
    inputFiles = arguments["--input"].split(",")
    inputBvec = [read_bvec(f, fsl_flag) for f in inputFiles]

    time = float(arguments["--time_limit"])

    output_flag = 1 if arguments["--verbose"] else 0

    outputFile = arguments["--output"]
    root, ext = os.path.splitext(outputFile)

    criteria = arguments["--criteria"]

    if len(inputBvec) == 1:
        method = milpflip_EEM if criteria == "ELECTROSTATIC" else milpflip_SC
        output = do_func(output_flag, method, inputBvec[0], time_limit=time)
    else:
        method = (
            milpflip_multi_shell_EEM
            if criteria == "ELECTROSTATIC"
            else milp_multi_shell_SC
        )
        output = do_func(output_flag, method, inputBvec, time_limit=time)

    if len(inputBvec) == 1:
        realPath = f"{root}{ext}"
        write_bvec(realPath, output, fsl_flag)
    else:
        for i, points in enumerate(output):
            realPath = f"{root}_shell{i}{ext}"
            write_bvec(realPath, points, fsl_flag)
