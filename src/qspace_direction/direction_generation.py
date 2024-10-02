#!/usr/bin/env python
"""
Usage:
    direction_generation.py [-v | -q] --number=NUMBER --output=OUTPUT [--asym] [--max_iter ITER] [--initialization INIT] [--fslgrad]

Options:
    -o OUTPUT, --output OUTPUT      output file 
    -n NUMBER, --number NUMBER      number chosen from each shell
    -v, --verbose                   output message
    -q, --quiet                     do not output message
    -i INIT, --initialization INIT  if set, use this file as initialization for CNLO algorithm, else use GEEM as initialization by default
    -a, --asym                      If set, the orientation is not antipodal symmetric 
    --max_iter ITER  Maximum iteration rounds for optimization    [default: 1000]
    --fslgrad,                      if set, program will read and write in fslgrad format

Reference:
    1. Jian Cheng, Dinggang Shen, Pew-Thian Yap and Peter J. Basser, "Single- and Multiple-Shell Uniform Sampling Schemes for Diffusion MRI Using Spherical Codes," in IEEE Transactions on Medical Imaging, vol. 37, no. 1, pp. 185-199
    1. Caruyer, Emmanuel, Christophe Lenglet, Guillermo Sapiro, and Rachid Deriche. "Design of multishell sampling schemes with uniform coverage in diffusion MRI." Magnetic Resonance in Medicine 69, no. 6 (2013): 1534-1540.
    2. Caruyer, Emmanuel, and Rachid Deriche. "Diffusion MRI signal reconstruction with continuity constraint and optimal regularization." Medical Image Analysis 16, no. 6 (2012): 1113-1120.
"""
import os

import numpy as np
from docopt import docopt
from sampling import cnlo_optimize
from io_util import do_func, read_bvec, write_bvec


if __name__ == "__main__":
    arguments = docopt(__doc__)

    fsl_flag = True if arguments["--fslgrad"] else False
    initVecs = None
    if arguments["--initialization"]:
        fileList = arguments["--initialization"].split(",")
        initVecs = np.concatenate([read_bvec(name, fsl_flag) for name in fileList])

    numbers = list(map(int, arguments["--number"].split(",")))

    num_iter = int(arguments["--max_iter"])

    output_flag = 1 if arguments["--verbose"] else 0

    antipodal = False if arguments["--asym"] else True

    outputFile = arguments["--output"]
    root, ext = os.path.splitext(outputFile)

    vects = do_func(
        output_flag,
        cnlo_optimize,
        numbers,
        initVecs,
        antipodal=antipodal,
        max_iter=num_iter,
    )
    splitPoint = np.cumsum(numbers).tolist()
    splitPoint.insert(0, 0)

    if len(numbers) == 1:
        realPath = f"{root}{ext}"
        write_bvec(realPath, vects, fsl_flag)
    else:
        for i in range(len(numbers)):
            realPath = f"{root}_shell{i}{ext}"
            write_bvec(realPath, vects[splitPoint[i] : splitPoint[i + 1]], fsl_flag)
