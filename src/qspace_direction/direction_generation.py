#!/usr/bin/env python
"""
Description:
    Perform a pipeline in dMRI sampling, which contains generating continous sampling scheme using CNLO, applying polarity optimization and applying order otimization.
     
Usage:
    direction_generation.py [-v] --number=NUMBER --output=OUTPUT [--bval BVAL] [--initialization INIT] [-w WEIGHT] [-c CRITERIA] [-s SPLIT] [--max_iter ITER] [-t TIME] [--fslgrad]

Options:
    -o OUTPUT, --output OUTPUT        Output file 
    -n NUMBER, --number NUMBER        Number chosen from each shell
    -v, --verbose                     Output message
    --bval BVAL                       Set bval for each shell
    -i INIT, --initialization INIT    If set, use this file as initialization for CNLO algorithm, else use GEEM as initialization by default
    -w WEIGHT, --weight WEIGHT        Weight for single shell term, 1-weight for mutiple shell term. [default: 0.5]
    -c CRITERIA, --criteria CRITERIA  Criteria type for polarity optimization (DISTANCE or ELECTROSTATIC). [default: ELECTROSTATIC]
    -s SPLIT, --split SPLIT           Number of points per split for order optimization. [default: 3]
    --max_iter ITER                   Maximum iteration rounds for continous optimization    [default: 1000]
    -t TIME, --time_limit TIME        Maximum time to run milp algorithm    [default: 600]
    --fslgrad                         If set, program will read and write in fslgrad format

Example:
    python -m qspace_direction.direction_generation --output scheme.txt -n 30
    python -m qspace_direction.direction_generation --output scheme.txt -n 90,90,90 --bval 1000,2000,3000   

Reference:
    1. Jian Cheng, Dinggang Shen, Pew-Thian Yap and Peter J. Basser, "Single- and Multiple-Shell Uniform Sampling Schemes for Diffusion MRI Using Spherical Codes," in IEEE Transactions on Medical Imaging, vol. 37, no. 1, pp. 185-199
    2. Emmanuel Caruyer, Christophe Lenglet, Guillermo Sapiro, and Rachid Deriche. "Design of multishell sampling schemes with uniform coverage in diffusion MRI." Magnetic Resonance in Medicine 69, no. 6 (2013): 1534-1540.
    3. Emmanuel Caruyer, Jian Cheng, Christophe Lenglet, Guillermo Sapiro, Tianzi Jiang, and Rachid Deriche,"Optimal Design of Multiple Q-shells experiments for Diffusion MRI",MICCAI Workshop on Computational Diffusion MRI (CDMRI'11), pp. 45–53, 2011.
"""

import os

import numpy as np
from docopt import docopt

from .lib import do_func, read_bvec, write_bvec
from .sampling import cnlo_optimize

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
