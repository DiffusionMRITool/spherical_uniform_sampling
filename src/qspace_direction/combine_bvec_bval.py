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
    python -m qspace_direction.combine_bvec_bval bvec_shell0.txt,bvec_shell0.txt,bvec_shell0.txt 1000,2000,3000 --output ordered.txt
"""
import os
from docopt import docopt
from .lib import read_bvec, combine_bvec_bval, write_bvec, write_bval

if __name__ == "__main__":
    arguments = docopt(__doc__)

    fsl_flag = True if arguments["--fslgrad"] else False
    inputBVecFiles = arguments["BVEC"]
    bvecs = [read_bvec(f, fsl_flag) for f in inputBVecFiles.split(',')]
    bvals = list(map(float, arguments["BVAL"].split(',')))

    assert len(bvecs) == len(bvals), "Number of bvec shell and number of bvals don't match!"

    outputFile = arguments["--output"]
    root, ext = os.path.splitext(outputFile)

    bvecList, bvalList = combine_bvec_bval(bvecs, bvals)
    write_bvec(f"{root}_bvec{ext}", bvecList, fsl_flag)
    write_bval(f"{root}_bval{ext}", bvalList, fsl_flag)
