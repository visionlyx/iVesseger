import numpy as np
import vtk
from vtk.util import *
from libtiff import TIFF
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkPolyData
)
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
)


def libTIFFRead(src):

    tif = TIFF.open(src, mode="r")
    im_stack = list()
    for im in list(tif.iter_images()):
        im_stack.append(im)
    tif.close()
    im_stack = np.array(im_stack)
    if(im_stack.shape[0] == 1):
        im_stack = im_stack[0]
    return im_stack


def libTIFFWrite(path, img):

    tif = TIFF.open(path, mode='w')
    if (img.ndim == 2):
        tif.write_image(img, compression=None)
    if(img.ndim==3):
        for i in range(0, img.shape[0]):
            im = img[i]
            tif.write_image(im, compression=None)
    tif.close()


def creat_volume_actor(volume_data, volume_brightness):

    # Create a VTK volume
    volume = vtk.vtkImageData()
    volume.SetDimensions(volume_data.shape)
    volume.SetOrigin(0, 0, 0)
    volume.SetSpacing(1, 1, 1)

    if volume_data.dtype == 'uint8':
        volume.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        # Copy the volume data to the VTK volume
        volume_array = vtk.util.numpy_support.numpy_to_vtk(volume_data.ravel(), deep=True,
                                                           array_type=vtk.VTK_UNSIGNED_CHAR)
    if volume_data.dtype == 'uint16':
        volume.AllocateScalars(vtk.VTK_UNSIGNED_SHORT, 1)
        # Copy the volume data to the VTK volume
        volume_array = vtk.util.numpy_support.numpy_to_vtk(volume_data.ravel(), deep=True,
                                                           array_type=vtk.VTK_UNSIGNED_SHORT)

    volume.GetPointData().SetScalars(volume_array)

    # Create a VTK volume mapper
    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetBlendModeToMaximumIntensity()
    volume_mapper.SetInputData(volume)

    colorTransfer = vtk.vtkColorTransferFunction()
    colorTransfer.AddRGBPoint(0, 0, 0, 0)
    colorTransfer.AddRGBPoint(volume_brightness, 1, 1, 1)

    opacFunction = vtk.vtkPiecewiseFunction()
    opacFunction.AddPoint(0, 0.1)
    opacFunction.AddPoint(volume_brightness, 1)
    opacFunction.ClampingOn()

    # Create a VTK volume property
    volume_property = vtk.vtkVolumeProperty()

    # volume_property.ShadeOn()
    volume_property.SetColor(colorTransfer)
    volume_property.SetScalarOpacity(opacFunction)
    volume_property.SetScalarOpacityUnitDistance(0.1)
    volume_property.SetInterpolationType(2)

    # Create a VTK volume actor
    volume_actor = vtk.vtkVolume()
    volume_actor.SetMapper(volume_mapper)
    volume_actor.SetProperty(volume_property)

    return volume_actor


def creat_points_actor(img):

    res = np.where(img != 0)

    points = vtkPoints()
    vertices = vtkCellArray()

    for i in range(0, len(res[0])):
        point_id = points.InsertNextPoint(res[2][i], res[1][i], res[0][i])
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(point_id)

    # Create a polydata object
    point = vtkPolyData()

    # Set the points and vertices we created as the geometry and topology of the polydata
    point.SetPoints(points)
    point.SetVerts(vertices)

    # Visualize
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(point)
    mapper.SetScalarModeToUsePointData()

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0, 255, 255)
    actor.GetProperty().SetPointSize(2)

    return actor
