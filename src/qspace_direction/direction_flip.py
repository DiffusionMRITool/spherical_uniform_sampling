#!/usr/bin/env python
"""
Description:
    Flip a given sampling scheme.

Usage:
    direction_flip.py [-v] --input=INPUT --output=OUTPUT [-t TIME] [-c CRITERIA] [--fslgrad]

Options:
    -o OUTPUT, --output OUTPUT        Output file
    -i INPUT, --input INPUT           Input bvec files  
    -v, --verbose                     Output gurobi message
    -c CRITERIA, --criteria CRITERIA  Criteria type (DISTANCE or ELECTROSTATIC). [default: ELECTROSTATIC]
    -t TIME, --time_limit TIME        Maximum time to run milp algorithm    [default: 600]
    --fslgrad                         If set, program will read and write in fslgrad format

Examples: 
    python -m qspace_direction.direction_flip --input scheme.txt --output flipped.txt
    python -m qspace_direction.direction_flip --input scheme_shell0.txt,scheme_shell1.txt,scheme_shell2.txt --output flipped.txt 
"""
import os

from docopt import docopt
from .sampling import (
    milpflip_SC,
    milpflip_EEM,
    milp_multi_shell_SC,
    milpflip_multi_shell_EEM,
)
from .lib import do_func, read_bvec, write_bvec


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
