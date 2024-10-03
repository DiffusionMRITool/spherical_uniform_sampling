#!/usr/bin/env python
"""
Description:
    Generate a set of uniform sampling schemes given the number on each shell using GEEM algorithm.

Usage:
    generation_geem.py [-v] --number=NUMBER --output=OUTPUT [--asym] [--max_iter ITER] [--initialization INIT] [--fslgrad]

Options:
    -o OUTPUT, --output OUTPUT      Output file 
    -n NUMBER, --number NUMBER      Number chosen from each shell
    -v, --verbose                   Output message
    -i INIT, --initialization INIT  Optimal initialization bvec files
    -a, --asym                      If set, the orientation is not antipodal symmetric 
    --max_iter ITER                 Maximum iteration rounds for optimization    [default: 1000]
    --fslgrad                       If set, program will read and write in fslgrad format

Example:
    # Generate a 30 points scheme
    python -m qspace_direction.generation_geem --output bvec.txt -n 30
    # Generate a 90x3 points scheme
    python -m qspace_direction.generation_geem --output bvec.txt -n 90,90,90    

Reference:
    1. Emmanuel Caruyer, Christophe Lenglet, Guillermo Sapiro, and Rachid Deriche. "Design of multishell sampling schemes with uniform coverage in diffusion MRI." Magnetic Resonance in Medicine 69, no. 6 (2013): 1534-1540.
    2. Emmanuel Caruyer, Jian Cheng, Christophe Lenglet, Guillermo Sapiro, Tianzi Jiang, and Rachid Deriche,"Optimal Design of Multiple Q-shells experiments for Diffusion MRI",MICCAI Workshop on Computational Diffusion MRI (CDMRI'11), pp. 45–53, 2011.
"""
import os

import numpy as np
from docopt import docopt

from qspace_direction.lib.io_util import (arg_bool, arg_values, do_func,
                                          read_bvec, write_bvec)
from qspace_direction.sampling.geem import compute_weights
from qspace_direction.sampling.geem import optimize as geem_optimize


def main(arguments):
    fsl_flag = arg_bool(arguments["--fslgrad"], bool)
    initVecs = None
    if arguments["--initialization"]:
        initVecs = np.concatenate(
            arg_values(arguments["--initialization"], lambda f: read_bvec(f, fsl_flag))
        )

    numbers = arg_values(arguments["--number"], int)

    num_iter = arg_values(arguments["--max_iter"], int)

    output_flag = arg_bool(arguments["--verbose"], int)

    antipodal = not arg_bool(arguments["--asym"], bool)

    outputFile = arguments["--output"]
    root, ext = os.path.splitext(outputFile)

    nb_shells = len(numbers)
    shell_groups = [[i] for i in range(nb_shells)]
    shell_groups.append(range(nb_shells))
    alphas = np.ones(len(shell_groups))
    weights = compute_weights(nb_shells, numbers, shell_groups, alphas)

    vects = do_func(
        output_flag,
        geem_optimize,
        nb_shells,
        numbers,
        weights,
        antipodal=antipodal,
        max_iter=num_iter,
        init_points=initVecs,
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


if __name__ == "__main__":
    arguments = docopt(__doc__)

    main(arguments)
