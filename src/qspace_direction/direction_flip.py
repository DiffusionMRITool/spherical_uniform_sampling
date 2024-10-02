#!/usr/bin/env python
"""
Usage:
    direction_flip.py [-v | -q] --input=INPUT --output=OUTPUT [-t TIME] [-c CRITERIA]

Options:
    -o OUTPUT, --output OUTPUT  output file [default: ./output.txt]
    -i INPUT, --input INPUT     input bvec files  
    -v, --verbose               output gurobi message
    -q, --quiet                 do not output gurobi message
    -c CRITERIA, --criteria CRITERIA             Criteria Type(DISTANCE or ELECTROSTATIC). [default: ELECTROSTATIC]
    -t TIME, --time_limit TIME  Maximum time to run milp algorithm    [default: 600]
"""
import os

from docopt import docopt
from sampling import (
    dirflip_SC,
    dirflip_EEM,
    dirflip_multi_shell_SC,
    dirflip_multi_shell_EEM,
)
from io_util import do_func, read_bvec, write_bvec


if __name__ == "__main__":
    arguments = docopt(__doc__)
    inputFiles = arguments["--input"].split(",")
    inputBvec = [read_bvec(f) for f in inputFiles]

    time = float(arguments["--time_limit"])

    output_flag = 1 if arguments["--verbose"] else 0

    outputFile = arguments["--output"]
    root, ext = os.path.splitext(outputFile)

    criteria = arguments["--criteria"]

    if len(inputBvec) == 1:
        method = dirflip_EEM if criteria == "ELECTROSTATIC" else dirflip_SC
        output = do_func(output_flag, method, inputBvec[0], time_limit=time)
    else:
        method = (
            dirflip_multi_shell_EEM
            if criteria == "ELECTROSTATIC"
            else dirflip_multi_shell_SC
        )
        output = do_func(output_flag, method, inputBvec, time_limit=time)

    if len(inputBvec) == 1:
        realPath = f"{root}{ext}"
        write_bvec(realPath, output)
    else:
        for i, points in enumerate(output):
            realPath = f"{root}_shell{i}{ext}"
            write_bvec(realPath, points)
