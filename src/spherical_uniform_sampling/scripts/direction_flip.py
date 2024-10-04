#!/usr/bin/env python
"""
Description:
    Optimize the polarity of a given sampling scheme by flipping certain points within it.

Usage:
    direction_flip.py [-v] --input=INPUT --output=OUTPUT [-w WEIGHT] [-t TIME] [-c CRITERIA] [--fslgrad]

Options:
    -o OUTPUT, --output OUTPUT        Output file
    -i INPUT, --input INPUT           Input bvec files  
    -v, --verbose                     Output gurobi message
    -w WEIGHT, --weight WEIGHT        Weight for single shell term, 1-weight for mutiple shell term. [default: 0.5]
    -c CRITERIA, --criteria CRITERIA  Criteria type (DISTANCE or ELECTROSTATIC). [default: ELECTROSTATIC]
    -t TIME, --time_limit TIME        Maximum time to run milp algorithm    [default: 600]
    --fslgrad                         If set, program will read and write in fslgrad format

Examples: 
    # Optimize the polarity of a single shell scheme
    direction_flip.py --input bvec.txt --output flipped.txt
    # Optimize the polarity of a multiple shell scheme
    direction_flip.py --input bvec_shell0.txt,bvec_shell1.txt,bvec_shell2.txt --output flipped.txt 
"""
import os

from docopt import docopt

from spherical_uniform_sampling.lib.io_util import (
    arg_bool,
    arg_values,
    do_func,
    read_bvec,
    write_bvec,
)
from spherical_uniform_sampling.sampling.flip import (
    milp_multi_shell_SC,
    milpflip_EEM,
    milpflip_multi_shell_EEM,
    milpflip_SC,
)


def main(arguments):
    fsl_flag = arg_bool(arguments["--fslgrad"], bool)
    inputBvec = arg_values(arguments["--input"], lambda f: read_bvec(f, fsl_flag))

    time = arg_values(arguments["--time_limit"], float, is_single=True)

    output_flag = arg_bool(arguments["--verbose"], int)

    outputFile = arguments["--output"]
    root, ext = os.path.splitext(outputFile)

    criteria = arguments["--criteria"]
    weight = arg_values(arguments["--weight"], float, is_single=True)

    if len(inputBvec) == 1:
        method = milpflip_EEM if criteria == "ELECTROSTATIC" else milpflip_SC
        output = do_func(output_flag, method, inputBvec[0], time_limit=time)
    else:
        method = (
            milpflip_multi_shell_EEM
            if criteria == "ELECTROSTATIC"
            else milp_multi_shell_SC
        )
        output = do_func(output_flag, method, inputBvec, w=weight, time_limit=time)

    if len(inputBvec) == 1:
        realPath = f"{root}{ext}"
        write_bvec(realPath, output, fsl_flag)
    else:
        for i, points in enumerate(output):
            realPath = f"{root}_shell{i}{ext}"
            write_bvec(realPath, points, fsl_flag)


if __name__ == "__main__":
    arguments = docopt(__doc__)

    main(arguments)
