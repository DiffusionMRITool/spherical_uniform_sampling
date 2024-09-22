"""
    Usage:
        direction_generation.py [-v | -q] --number=NUMBER --output=OUTPUT [--asym] [--max_iter ITER] [--initialization INIT] 

    Options:
        -o OUTPUT, --output OUTPUT  output file 
        -n NUMBER, --number NUMBER  number chosen from each shell
        -v, --verbose               output gurobi message
        -q, --quiet                 do not output gurobi message
        -i INIT, --initialization INIT     optimal initialization bvec files
        -a, --asym                  If set, the orientation is not antipodal symmetric 
        --max_iter ITER  Maximum iteration rounds for optimization    [default: 1000]
"""
import os

import numpy as np
from docopt import docopt
from sampling import optimize
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

    vects = do_func(
        output_flag,
        optimize,
        numbers,
        initVecs,
        antipodal=antipodal,
        max_iter=num_iter,
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
