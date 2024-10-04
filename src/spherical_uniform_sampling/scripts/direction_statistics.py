#!/usr/bin/env python
"""
Description:
    Get statistics information from a given sampling scheme. Show covering radius, electrastatic energy and norm of mean direction vector of the scheme.

Usage:
    direction_statistics.py BVEC [-w WEIGHT] [-o ORDER] [-a] [-c] [--fslgrad]
    direction_statistics.py BVEC BVAL [-w WEIGHT] [-o ORDER] [-a] [-c] [--fslgrad]

Options:
    -w WEIGHT, --weight WEIGHT  Weight for single shell term, 1-weight for mutiple shell term. [default: 0.5]
    -o ORDER, --order ORDER     Order for the calculation electrostatic energy [default: 2]
    -a, --asym                  If set, the orientation is not antipodal symmetric 
    -c, --combine               If set, also show the statistics for the orientation combining all input orientations
    --fslgrad                   If set, program will read and write in fslgrad format

Examples:
    # Show statistics for a single shell
    direction_statistics.py bvec.txt
    # Show statistics for multiple shells (bvec files must be seperated by a comma and without any space), use -c option to show combined shell
    direction_statistics.py bvec_shell0.txt,bvec_shell1.txt,bvec_shell2.txt -c
    # Show statistics for a bvec and corresponding bval file. Show combined shell option is automatically open if there is more than one b-values.
    direction_statistics.py bvec.txt bval.txt
"""
import numpy as np
from docopt import docopt

from spherical_uniform_sampling.lib.io_util import (
    arg_bool,
    arg_values,
    read_bvec,
    read_bvec_bval,
)
from spherical_uniform_sampling.sampling.loss import (
    covering_radius,
    electrostatic_energy,
    weighted_cost_multi_shell,
    norm_of_mean,
)


def display_bvec_stat(name, bvec, order, antipodal):
    print(f"{name} ({len(bvec)} points)")
    cr = covering_radius(bvec, antipodal)
    print(f"Covering radius = {cr * 180 / np.pi:.4f}°, radian = {cr:.6f}")
    print(
        f"Electrastatics energy (order={order}) = {electrostatic_energy(bvec, order, antipodal):.6f}"
    )
    print(f"Norm of mean direction vector = {norm_of_mean(bvec):.6f}\n")


def display_bvec_stat_combined(bvec, order, antipodal, weight):
    print(f"Combined {len(bvec)} shell ({sum(len(l) for l in bvec)} points)")
    cr = covering_radius(np.concatenate(bvec), antipodal)
    print(f"Covering radius = {cr * 180 / np.pi:.4f}°, radian = {cr:.6f}")
    print(
        f"Weighted covering radius = {weighted_cost_multi_shell(bvec, covering_radius, weight, antipodal):.6f}"
    )
    print(
        f"Electrastatics energy (order={order}) = {electrostatic_energy(np.concatenate(bvec), order, antipodal):.6f}"
    )
    print(
        f"Weighted electrastatics energy (order={order}) = {weighted_cost_multi_shell(bvec, electrostatic_energy, weight, order, antipodal):.6f}"
    )
    print(f"Norm of mean direction vector = {norm_of_mean(np.concatenate(bvec)):.6f}\n")


def main(arguments):
    fsl_flag = arg_bool(arguments["--fslgrad"], bool)

    weight = arg_values(arguments["--weight"], float, is_single=True)
    show_combine = arg_bool(arguments["--combine"], bool)
    antipodal = not arg_bool(arguments["--asym"], bool)
    order = arg_values(arguments["--order"], float, is_single=True)

    if arguments["BVAL"]:
        bvals, bvecs = read_bvec_bval(arguments["BVEC"], arguments["BVAL"], fsl_flag)
        for bval, bvec in zip(bvals, bvecs):
            display_bvec_stat(f"b-value {bval}", bvec, order, antipodal)
        if len(bvals) > 1:
            display_bvec_stat_combined(bvecs, order, antipodal, weight)
    else:
        bvecFileList = arg_values(arguments["BVEC"], str)
        bvecList = list(map(lambda f: read_bvec(f, fsl_flag), bvecFileList))
        for filename, bvec in zip(bvecFileList, bvecList):
            display_bvec_stat(filename, bvec, order, antipodal)
        if show_combine:
            display_bvec_stat_combined(bvecList, order, antipodal, weight)


if __name__ == "__main__":
    arguments = docopt(__doc__)

    main(arguments)
