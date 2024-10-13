#!/usr/bin/env python
"""
Description:
    View the orientation

Usage:
    direction_view.py BVEC ... [--asym] [--combine] [--save SAVE]

Options:
    -a, --asym     If set, the orientation is not antipodal symmetric 
    -c, --combine  If set, only show points on combined shell
    -s SAVE, --save SAVE    If set, save the orientation view. The output format are deduced by the extension to filename.

Examples:
    # View single shell
    direction_view.py bvec.txt
    # View every shell in a multiple shell scheme
    direction_view.py bvec1.txt bvec2.txt
    # View multiple shell scheme by projecting onto a single sphere
    direction_view.py bvec1.txt bvec2.txt -c
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from docopt import docopt

from spherical_uniform_sampling.lib.io_util import read_bvec, arg_bool, arg_values

def simplices2edge(simplices):
    start, end = [], []

    for p in simplices:
        start.extend([p[0], p[1], p[2]])
        end.extend([p[1], p[2], p[0]])

    return np.array(start), np.array(end)

def get_colors(num):
    if num == 1 or num > 9:
        return [(1, 1, 1)] * num
    colors = [None for _ in range(num)]
    colors[0] = (1, 0, 0)
    colors[1] = (0, 1, 0)
    if num > 2:
        colors[2] = (0, 0, 1)
    if num > 3:
        colors[3] = (0.5, 0, 0)
    if num > 4:
        colors[4] = (0, 0.5, 0)
    if num > 5:
        colors[5] = (0, 0, 0.5)
    if num > 6:
        colors[6] = (0.5, 0.5, 0)
    if num > 7:
        colors[7] = (0, 0.5, 0.5)
    if num > 8:
        colors[8] = (0.5, 0, 0.5)

    return colors

def get_opacity(num):
    if num == 1:
        return 1
    rg = np.arange(num)
    return 1 - 0.7 / (num - 1) * rg

def draw_mesh(ax, bvecs, radius=1, opacity=1):
    bvecs = bvecs * radius
    tri = ConvexHull(bvecs)
    x = bvecs.T[0]
    y = bvecs.T[1]
    z = bvecs.T[2]

    ax.plot_trisurf(x, y, z, triangles=tri.simplices, shade=True, color=(0.5, 0.5, 0.5), alpha=opacity)


def draw_point(ax, bvecs, radius=1, color=(1, 1, 1)):
    bvecs = bvecs * radius * 1.01
    x = bvecs.T[0]
    y = bvecs.T[1]
    z = bvecs.T[2]

    ax.scatter(x, y, z, color=color)

def main(arguments):
    antipodal = not arg_bool(arguments["--asym"], bool)
    only_combined = arg_bool(arguments["--combine"], bool)
    save_flle = arg_values(arguments["--save"], str, 1, True)

    bvecs = list(map(lambda path: read_bvec(path), arguments["BVEC"]))
    if antipodal:
        bvecs = list(map(lambda v: np.concatenate([v, -v]), bvecs))

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    colors = get_colors(len(bvecs))
    if only_combined:
        combined = np.concatenate(bvecs)
        draw_mesh(ax, combined)
        for i, vec in enumerate(bvecs):
            draw_point(ax, vec, 1, colors[i])
    else:
        opacity = get_opacity(len(bvecs))
        for i, vec in enumerate(bvecs):
            draw_mesh(ax, vec, i + 1, opacity[i])
            draw_point(ax, vec, i + 1, colors[i])

    ax.set_box_aspect((1, 1, 1))
    plt.axis("off")

    if save_flle:
        plt.savefig(save_flle)
    else:
        plt.show()


if __name__ == "__main__":
    arguments = docopt(__doc__)

    main(arguments)
