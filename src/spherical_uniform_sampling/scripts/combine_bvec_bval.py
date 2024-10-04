#!/usr/bin/env python
"""
Description:
    Combine a bvec list and a bval list

Usage:
    combine_bvec_bval.py BVEC BVAL --output=OUTPUT [--fslgrad]

Options:
    -o OUTPUT, --output OUTPUT  Output file 
    --fslgrad                   If set, program will read and write in fslgrad format

Examples:
    # different bvec and bval must be seperated with a comma and without any space
    combine_bvec_bval.py bvec_shell0.txt,bvec_shell0.txt,bvec_shell0.txt 1000,2000,3000 --output ordered.txt
"""
import os

from docopt import docopt

from spherical_uniform_sampling.lib.io_util import (
    arg_bool,
    arg_values,
    combine_bvec_bval,
    read_bvec,
    write_bval,
    write_bvec,
)


def main(arguments):
    fsl_flag = arg_bool(arguments["--fslgrad"], bool)
    bvecs = arg_values(arguments["BVEC"], lambda f: read_bvec(f, fsl_flag))
    bvals = arg_values(arguments["BVAL"], float)

    assert len(bvecs) == len(
        bvals
    ), "Number of bvec shell and number of bvals don't match!"

    outputFile = arguments["--output"]
    root, ext = os.path.splitext(outputFile)

    bvecList, bvalList = combine_bvec_bval(bvecs, bvals)
    write_bvec(f"{root}_bvec{ext}", bvecList, fsl_flag)
    write_bval(f"{root}_bval{ext}", bvalList, fsl_flag)


if __name__ == "__main__":
    arguments = docopt(__doc__)

    main(arguments)
