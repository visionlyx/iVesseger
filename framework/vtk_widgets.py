import sys
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import vtk
from PyQt5 import QtCore, QtGui, QtWidgets
from get_points import *
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from creat_actors import *


class MyVTK(QVTKRenderWindowInteractor):
    def __init__(self, p):
        super(MyVTK, self).__init__(p)
        self.interactor = QVTKRenderWindowInteractor()
        self.ren = vtk.vtkRenderer()
        self.ren.GetActiveCamera().ParallelProjectionOn()
        self.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.GetRenderWindow().GetInteractor()
        self.model = "Single"
        self.iren.SetInteractorStyle(MyInteractorStyle(self.iren, self.ren, self, self.model))

        self.img = 0
        self.label = 0
        self.max_x = 0
        self.max_y = 0
        self.max_z = 0
        self.image_actor = 0
        self.label_actor = 0
        self.current_min_brightness = 0
        self.current_max_brightness = 0

        self.actors = list()
        self.lines = list()

        self.fp_volumes = list()
        self.fn_volumes = list()
        self.fp_change = False
        self.fn_change = False

    def start(self):
        self.ren.ResetCamera()
        self.iren.Initialize()

    def set_img_label(self, img, label):
        self.img = img
        self.label = label
        self.max_x = img.shape[2]
        self.max_y = img.shape[1]
        self.max_z = img.shape[0]

    def smooth(self, y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def point_to_line_distance(self, point, line):
        point = np.array(point)
        point_A = np.array(line[0])
        point_B = np.array(line[-1])
        vector = np.array(point_B) - np.array(point_A)
        vector = vector / np.linalg.norm(vector)

        distance = np.linalg.norm(point - line - np.dot(vector, (point - point_A)) / np.dot(vector, vector) * vector)
        distance = distance / len(line)
        return distance

    def brightness_min_change(self, new_value):
        volProperty = self.image_actor.GetProperty()

        colorTransfer = vtk.vtkColorTransferFunction()
        colorTransfer.AddRGBPoint(self.current_max_brightness, 1, 1, 1)
        colorTransfer.AddRGBPoint(new_value, 0, 0, 0)

        opacFunction = vtk.vtkPiecewiseFunction()
        opacFunction.AddPoint(new_value, 0.1)
        opacFunction.AddPoint(self.current_max_brightness, 1)
        opacFunction.ClampingOn()

        volProperty.SetColor(colorTransfer)
        volProperty.SetScalarOpacity(opacFunction)

        self.GetRenderWindow().Render()

    def brightness_max_change(self, new_value):
        volProperty = self.image_actor.GetProperty()

        colorTransfer = vtk.vtkColorTransferFunction()
        colorTransfer.AddRGBPoint(self.current_min_brightness, 0, 0, 0)
        colorTransfer.AddRGBPoint(new_value, 1, 1, 1)

        opacFunction = vtk.vtkPiecewiseFunction()
        opacFunction.AddPoint(self.current_min_brightness, 0.1)
        opacFunction.AddPoint(new_value, 1)
        opacFunction.ClampingOn()

        volProperty.SetColor(colorTransfer)
        volProperty.SetScalarOpacity(opacFunction)

        self.GetRenderWindow().Render()

    def single_model(self):
        self.model = "Single"
        self.iren.SetInteractorStyle(MyInteractorStyle(self.iren, self.ren, self, self.model))

    def double_model(self):
        self.model = "Double"
        self.iren.SetInteractorStyle(MyInteractorStyle(self.iren, self.ren, self, self.model))

    def add_line(self, x, y, z):
        a = list()
        a.append([x, y, z])
        c = self.ren.GetActiveCamera().GetDirectionOfProjection()

        temp_x = x
        temp_y = y
        temp_z = z
        while True:
            temp_x = temp_x + c[0]
            temp_y = temp_y + c[1]
            temp_z = temp_z + c[2]
            if temp_x < 0 or temp_x > self.max_x - 1 or temp_y < 0 or temp_y > self.max_y - 1 or temp_z < 0 or temp_z > self.max_z - 1:
                break
            a.append([temp_x, temp_y, temp_z])

        temp_x = x
        temp_y = y
        temp_z = z
        while True:
            temp_x = temp_x - c[0]
            temp_y = temp_y - c[1]
            temp_z = temp_z - c[2]
            if temp_x < 0 or temp_x > self.max_x - 1 or temp_y < 0 or temp_y > self.max_y - 1 or temp_z < 0 or temp_z > self.max_z - 1:
                break
            a.append([temp_x, temp_y, temp_z])

        a_list = np.array(a)
        a_list = np.around(a_list)

        unique_rows = np.unique(a_list, axis=0)

        points = vtkPoints()
        vertices = vtkCellArray()
        for i in range(0, len(unique_rows)):
            point_id = points.InsertNextPoint(unique_rows[i][0], unique_rows[i][1], unique_rows[i][2])
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(point_id)

        point = vtkPolyData()
        point.SetPoints(points)
        point.SetVerts(vertices)

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(point)
        mapper.SetScalarModeToUsePointData()

        actor = vtkActor()
        self.lines.append(actor)
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(255, 255, 0)
        actor.GetProperty().SetPointSize(8)

        self.ren.AddActor(actor)
        self.GetRenderWindow().Render()
        return a

    def add_single_point(self, x, y, z):
        a = list()
        a.append([x, y, z])
        c = self.ren.GetActiveCamera().GetDirectionOfProjection()

        temp_x = x
        temp_y = y
        temp_z = z
        while True:
            temp_x = temp_x + c[0]
            temp_y = temp_y + c[1]
            temp_z = temp_z + c[2]
            if temp_x < 0 or temp_x > self.max_x - 1 or temp_y < 0 or temp_y > self.max_y - 1 or temp_z < 0 or temp_z > self.max_z - 1:
                break
            a.append([temp_x, temp_y, temp_z])

        temp_x = x
        temp_y = y
        temp_z = z
        while True:
            temp_x = temp_x - c[0]
            temp_y = temp_y - c[1]
            temp_z = temp_z - c[2]
            if temp_x < 0 or temp_x > self.max_x - 1 or temp_y < 0 or temp_y > self.max_y - 1 or temp_z < 0 or temp_z > self.max_z - 1:
                break
            a.append([temp_x, temp_y, temp_z])

        a_list = np.array(a)
        a_list = np.around(a_list)

        unique_rows = np.unique(a_list, axis=0)
        unique_rows = unique_rows.astype(int)

        img_pixel_values = list()
        label_pixel_values = list()
        for i in range(len(unique_rows)):
            c_img = self.img[unique_rows[i][2]][unique_rows[i][1]][unique_rows[i][0]]
            c_label = self.label[unique_rows[i][2]][unique_rows[i][1]][unique_rows[i][0]]

            img_pixel_values.append(c_img)
            label_pixel_values.append(c_label)

        img_pixel_values = np.array(img_pixel_values)
        label_pixel_values = np.array(label_pixel_values)

        self.fp_change = False
        self.fn_change = False

        if(max(label_pixel_values)==255):  # fp

            pixel_values = self.smooth(label_pixel_values, 3)
            maxIndex = np.argmax(pixel_values, axis=0)
            self.fp_volumes.append([unique_rows[maxIndex][0], unique_rows[maxIndex][1], unique_rows[maxIndex][2]])
            self.fp_change = True

        else:  # fn
            pixel_values = self.smooth(img_pixel_values, 3)
            maxIndex = np.argmax(pixel_values, axis=0)
            self.fn_volumes.append([unique_rows[maxIndex][0], unique_rows[maxIndex][1], unique_rows[maxIndex][2]])
            self.fn_change = True

        sphereSource = vtkSphereSource()
        sphereSource.SetCenter(unique_rows[maxIndex][0], unique_rows[maxIndex][1], unique_rows[maxIndex][2])
        sphereSource.SetRadius(2)

        sphereMapper = vtkPolyDataMapper()
        sphereMapper.SetInputConnection(sphereSource.GetOutputPort())

        sphereActor = vtkActor()
        self.actors.append(sphereActor)
        sphereActor.SetMapper(sphereMapper)

        if (max(label_pixel_values) == 255):
            sphereActor.GetProperty().SetColor(255, 0, 0)
        else:
            sphereActor.GetProperty().SetColor(0, 255, 0)

        self.ren.AddActor(sphereActor)
        self.GetRenderWindow().Render()

    def add_double_point(self, first_list, second_list):
        first_list = np.array(first_list)
        first_list = np.around(first_list)

        second_list = np.array(second_list)
        second_list = np.around(second_list)

        first_list = first_list.tolist()
        second_list = second_list.tolist()

        min_distance_first = len(second_list)
        min_distance_second = len(first_list)
        min_point_first = []
        min_point_second = []

        for point in first_list:
            distance = self.point_to_line_distance(point, second_list)
            if (distance < min_distance_first):
                min_distance_first = distance
                min_point_first = point

        for point in second_list:
            distance = self.point_to_line_distance(point, first_list)
            if (distance < min_distance_second):
                min_distance_second = distance
                min_point_second = point

        point_x = int((min_point_first[0] + min_point_second[0]) / 2)
        point_y = int((min_point_first[1] + min_point_second[1]) / 2)
        point_z = int((min_point_first[2] + min_point_second[2]) / 2)

        sphereSource = vtkSphereSource()
        sphereSource.SetCenter(point_x, point_y, point_z)
        sphereSource.SetRadius(2)

        sphereMapper = vtkPolyDataMapper()
        sphereMapper.SetInputConnection(sphereSource.GetOutputPort())

        sphereActor = vtkActor()
        self.actors.append(sphereActor)
        sphereActor.SetMapper(sphereMapper)

        if self.label[point_z][point_y][point_x] == 255:
            sphereActor.GetProperty().SetColor(200, 0, 0)
            self.fp_volumes.append([point_x, point_y, point_z])
        else:
            sphereActor.GetProperty().SetColor(0, 200, 0)
            self.fn_volumes.append([point_x, point_y, point_z])

        self.ren.AddActor(sphereActor)
        self.GetRenderWindow().Render()

    def remove_last_point(self):
        if(len(self.actors)!=0):

            if self.fp_change == True:
                self.fp_volumes = self.fp_volumes[:-1]
                self.fp_change = False

            if self.fn_change == True:
                self.fn_volumes = self.fn_volumes[:-1]
                self.fn_change = False

            temp_actor = self.actors[len(self.actors) - 1]
            self.ren.RemoveActor(temp_actor)
            self.actors.pop()
            self.GetRenderWindow().Render()

    def remove_last_line(self):
        if(len(self.lines) != 0):
            temp_line = self.lines[len(self.lines) - 1]
            self.ren.RemoveActor(temp_line)
            self.lines.pop()
            self.GetRenderWindow().Render()

    def remove_all_actors(self):
        self.fp_volumes = []
        self.fn_volumes = []

        if self.image_actor != 0:
            self.ren.RemoveActor(self.image_actor)
            self.image_actor = 0
        if self.label_actor != 0:
            self.ren.RemoveActor(self.label_actor)
            self.label_actor = 0

        if (len(self.actors) != 0):

            for i in range(len(self.actors)):
                temp_actor = self.actors[i]
                self.ren.RemoveActor(temp_actor)
            self.actors = list()

        if (len(self.lines) != 0):

            for i in range(len(self.lines)):
                temp_actor = self.lines[i]
                self.ren.RemoveActor(temp_actor)
            self.lines = list()

        self.GetRenderWindow().Render()

    def remove_seg_actors(self):
        self.fp_volumes = []
        self.fn_volumes = []

        if self.label_actor != 0:
            self.ren.RemoveActor(self.label_actor)
            self.label_actor = 0

        if (len(self.actors) != 0):

            for i in range(len(self.actors)):
                temp_actor = self.actors[i]
                self.ren.RemoveActor(temp_actor)
            self.actors = list()

        if (len(self.lines) != 0):

            for i in range(len(self.lines)):
                temp_actor = self.lines[i]
                self.ren.RemoveActor(temp_actor)
            self.lines = list()

        self.GetRenderWindow().Render()