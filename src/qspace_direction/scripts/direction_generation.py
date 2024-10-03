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
    # Generate and optimze a single shell sampling scheme with 30 points without b-value
    direction_generation.py --output scheme.txt -n 30
    # Generate and optimze a single shell points sampling with a single b-value 1000
    direction_generation.py --output scheme.txt -n 30 --bval 1000
    # Generate and optimze a multiple shell sampling scheme with 90x3, and each shell having b-value 1000, 2000 and 3000
    direction_generation.py --output scheme.txt -n 90,90,90 --bval 1000,2000,3000   

Reference:
    1. Jian Cheng, Dinggang Shen, Pew-Thian Yap and Peter J. Basser, "Single- and Multiple-Shell Uniform Sampling Schemes for Diffusion MRI Using Spherical Codes," in IEEE Transactions on Medical Imaging, vol. 37, no. 1, pp. 185-199
    2. Emmanuel Caruyer, Christophe Lenglet, Guillermo Sapiro, and Rachid Deriche. "Design of multishell sampling schemes with uniform coverage in diffusion MRI." Magnetic Resonance in Medicine 69, no. 6 (2013): 1534-1540.
    3. Emmanuel Caruyer, Jian Cheng, Christophe Lenglet, Guillermo Sapiro, Tianzi Jiang, and Rachid Deriche,"Optimal Design of Multiple Q-shells experiments for Diffusion MRI",MICCAI Workshop on Computational Diffusion MRI (CDMRI'11), pp. 45–53, 2011.
"""

import os
import shutil
import uuid

import numpy as np
from combine_bvec_bval import main as combine_main
from direction_continous_optimization import main as continous_main
from direction_flip import main as flip_main
from direction_order import main as order_main
from docopt import docopt

from qspace_direction.lib.io_util import write_bval


def main(arguments: dict):
    # create a temporary directory and work in it
    rd_path = "tmp" + str(uuid.uuid4())
    os.mkdir(rd_path)
    l = len(arguments["--number"].split(","))

    output = arguments["--output"]
    arguments["--output"] = os.path.join(rd_path, "scheme.txt")
    arguments["--asym"] = None
    continous_main(arguments)
    if l == 1:
        arguments["--input"] = os.path.join(rd_path, "scheme.txt")
        arguments["--output"] = os.path.join(rd_path, "flipped.txt")
        flip_main(arguments)

        if arguments["--bval"]:
            bval = arguments["--bval"]
            write_bval(
                os.path.join(rd_path, "bval.txt"),
                [bval] * int(arguments["--number"]),
                arguments["--fslgrad"],
            )
            arguments["BVAL"] = os.path.join(rd_path, "bval.txt")
        else:
            arguments["BVAL"] = None
        arguments["BVEC"] = os.path.join(rd_path, "flipped.txt")
        arguments["--output"] = output
        order_main(arguments)
    else:
        arguments["--input"] = ",".join(
            os.path.join(rd_path, f"scheme_shell{i}.txt") for i in range(l)
        )
        arguments["--output"] = os.path.join(rd_path, "flipped.txt")
        flip_main(arguments)

        assert arguments["--bval"], "Must set --bval for each shell!"
        arguments["BVEC"] = ",".join(
            os.path.join(rd_path, f"flipped_shell{i}.txt") for i in range(l)
        )
        arguments["BVAL"] = arguments["--bval"]
        arguments["--output"] = os.path.join(rd_path, "combine.txt")
        combine_main(arguments)

        arguments["BVEC"] = os.path.join(rd_path, "combine_bvec.txt")
        arguments["BVAL"] = os.path.join(rd_path, "combine_bval.txt")
        arguments["--output"] = output
        order_main(arguments)
    shutil.rmtree(rd_path)


if __name__ == "__main__":

    arguments = docopt(__doc__)

    main(arguments)
