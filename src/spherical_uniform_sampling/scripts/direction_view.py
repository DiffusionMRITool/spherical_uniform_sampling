#!/usr/bin/env python
"""
Description:
    View the orientation

Usage:
    direction_view.py BVEC ... [--asym] [--combine] [--save SAVE]

Options:
    -a, --asym     If set, the orientation is not antipodal symmetric 
    -c, --combine  If set, only show points on combined shell
    -s SAVE, --save SAVE    If set, save the orientation view. The output will be in png format regradless of the extension to filename.

Examples:
    # View single shell
    direction_view.py bvec.txt
    # View every shell in a multiple shell scheme
    direction_view.py bvec1.txt bvec2.txt
    # View multiple shell scheme by projecting onto a single sphere
    direction_view.py bvec1.txt bvec2.txt -c
"""
import numpy as np
import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2
from docopt import docopt
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkFiltersCore import vtkDelaunay3D, vtkGlyph3D
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkRenderingCore import (
    vtkDataSetMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkWindowToImageFilter,
)
from vtkmodules.vtkRenderingLOD import vtkLODActor

from spherical_uniform_sampling.lib.io_util import arg_bool, arg_values, read_bvec


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


def draw_mesh(renderer: vtkRenderer, bvecs, radius=1, opacity=1):
    points = vtkPoints()
    for vec in bvecs:
        points.InsertNextPoint(radius * vec)
    polydata = vtkPolyData()
    polydata.SetPoints(points)
    delaunay3D = vtkDelaunay3D()
    delaunay3D.SetInputData(polydata)
    delaunay3D.SetTolerance(1e-8)
    delaunay3D.SetOffset(10)
    delaunay3D.Update()
    delaunayMapper = vtkDataSetMapper()
    delaunayMapper.SetInputConnection(delaunay3D.GetOutputPort())
    actor = vtkLODActor()
    actor.SetMapper(delaunayMapper)
    actor.GetProperty().SetColor(1, 1, 1)
    actor.GetProperty().SetOpacity(opacity)
    renderer.AddActor(actor)


def draw(renderer, bvecs, radius=2):
    points = vtkPoints()
    for vec in bvecs:
        points.InsertNextPoint(radius * vec)
    polydata = vtkPolyData()
    polydata.SetPoints(points)
    delaunay3D = vtkDelaunay3D()
    delaunay3D.SetInputData(polydata)
    delaunay3D.SetTolerance(1e-8)
    delaunay3D.SetOffset(10)
    delaunay3D.Update()
    delaunayMapper = vtkDataSetMapper()
    delaunayMapper.SetInputConnection(delaunay3D.GetOutputPort())
    actor = vtkLODActor()
    actor.SetMapper(delaunayMapper)
    actor.GetProperty().SetColor(1, 1, 1)
    actor.GetProperty().SetOpacity(1)
    renderer.AddActor(actor)


def draw_point(
    renderer: vtkRenderer,
    sphereSource: vtkSphereSource,
    bvecs,
    radius=1,
    color=(1, 1, 1),
):
    points = vtkPoints()
    for vec in bvecs:
        points.InsertNextPoint(radius * vec)
    polydata = vtkPolyData()
    polydata.SetPoints(points)
    glyph3D = vtkGlyph3D()
    glyph3D.SetSourceConnection(sphereSource.GetOutputPort())
    glyph3D.SetInputData(polydata)
    glyph3D.Update()
    pointMapper = vtkDataSetMapper()
    pointMapper.SetInputData(glyph3D.GetOutput())

    pointsActor = vtkLODActor()
    pointsActor.SetMapper(pointMapper)
    pointsActor.GetProperty().SetColor(color)
    renderer.AddActor(pointsActor)


def main(arguments):
    antipodal = not arg_bool(arguments["--asym"], bool)
    only_combined = arg_bool(arguments["--combine"], bool)
    save_flle = arg_values(arguments["--save"], str, 1, True)

    bvecs = list(map(lambda path: read_bvec(path), arguments["BVEC"]))
    if antipodal:
        bvecs = list(map(lambda v: np.concatenate([v, -v]), bvecs))

    renderer = vtkRenderer()
    renderer.SetBackground(0, 0, 0)
    renderWindow = vtkRenderWindow()
    sphereSource = vtkSphereSource()
    sphereSource.SetCenter(0, 0, 0)
    sphereSource.SetRadius(0.04)
    sphereSource.SetThetaResolution(20)
    sphereSource.SetPhiResolution(20)

    colors = get_colors(len(bvecs))
    if only_combined:
        combined = np.concatenate(bvecs)
        draw_mesh(renderer, combined)
        for i, vec in enumerate(bvecs):
            draw_point(renderer, sphereSource, vec, 1, colors[i])
    else:
        opacity = get_opacity(len(bvecs))
        for i, vec in enumerate(bvecs):
            draw_mesh(renderer, vec, i + 1, opacity[i])
            draw_point(renderer, sphereSource, vec, i + 1, colors[i])

    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(600, 600)
    renderWindowInteractor = vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindowInteractor.SetDesiredUpdateRate(25)

    renderWindow.Render()
    renderer.GetActiveCamera().Roll(0)
    renderer.GetActiveCamera().Elevation(0)
    renderWindow.Render()

    if save_flle:
        windowToImage = vtkWindowToImageFilter()
        windowToImage.SetInput(renderWindow)
        pngWriter = vtkPNGWriter()
        pngWriter.SetInputConnection(windowToImage.GetOutputPort())
        pngWriter.SetFileName(save_flle)
        pngWriter.Write()
    else:
        renderWindowInteractor.Start()


if __name__ == "__main__":
    arguments = docopt(__doc__)

    main(arguments)
