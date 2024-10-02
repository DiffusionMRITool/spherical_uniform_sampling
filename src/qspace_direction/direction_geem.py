#!/usr/bin/env python
"""
Usage:
    generation_geem.py [-v | -q] --number=NUMBER --output=OUTPUT [--asym] [--max_iter ITER] [--initialization INIT] 

Options:
    -o OUTPUT, --output OUTPUT  output file 
    -n NUMBER, --number NUMBER  number chosen from each shell
    -v, --verbose               output gurobi message
    -q, --quiet                 do not output gurobi message
    -i INIT, --initialization INIT     optimal initialization bvec files
    -a, --asym                  If set, the orientation is not antipodal symmetric 
    --max_iter ITER  Maximum iteration rounds for optimization    [default: 1000]

Reference:
    1. Caruyer, Emmanuel, Christophe Lenglet, Guillermo Sapiro, and Rachid Deriche. "Design of multishell sampling schemes with uniform coverage in diffusion MRI." Magnetic Resonance in Medicine 69, no. 6 (2013): 1534-1540.
    2. Caruyer, Emmanuel, and Rachid Deriche. "Diffusion MRI signal reconstruction with continuity constraint and optimal regularization." Medical Image Analysis 16, no. 6 (2012): 1113-1120.
"""
import os

import numpy as np
from docopt import docopt
from sampling.geem import optimize as geem_optimize, compute_weights
from io_util import do_func, read_bvec, write_bvec


if __name__ == "__main__":
    arguments = docopt(__doc__)

    initVecs = None
    if arguments["--initialization"]:
        fileList = arguments["--initialization"].split(",")
        initVecs = np.concatenate([read_bvec(name) for name in fileList])

    numbers = list(map(int, arguments["--number"].split(",")))

    num_iter = int(arguments["--max_iter"])

    output_flag = 1 if arguments["--verbose"] else 0

    antipodal = False if arguments["--asym"] else True

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
        write_bvec(realPath, vects)
    else:
        for i in range(len(numbers)):
            realPath = f"{root}_shell{i}{ext}"
            write_bvec(realPath, vects[splitPoint[i] : splitPoint[i + 1]])
