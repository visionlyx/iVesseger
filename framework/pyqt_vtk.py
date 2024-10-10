from vtk_widgets import *
from thick_detect import *
from refine_detect import *


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

        self.thick_seg_thread = Thick_Seg()
        self.thick_seg_thread.seg_output.connect(self.thick_seg_output)
        self.refine_seg_thread = Refine_Seg()
        self.refine_seg_thread.seg_output.connect(self.refine_seg_output)

        self.image_size = 0
        self.temp_min_brightness = 0
        self.temp_max_brightness = 0
        self.temp_label = np.zeros([self.image_size, self.image_size, self.image_size], dtype=np.uint8)
        self.temp_image = np.zeros([self.image_size, self.image_size, self.image_size], dtype=np.uint8)

    def init_ui(self):
        self.ui = uic.loadUi("UI/UI.ui")

        self.layout = self.ui.layout
        self.min_slider = self.ui.min_slider
        self.max_slider = self.ui.max_slider

        self.min_label = self.ui.min_label
        self.max_label = self.ui.max_label

        self.vtkWidget = MyVTK(self)
        self.layout.addWidget(self.vtkWidget)

        self.ui.load_button.clicked.connect(self.open_image)
        self.ui.pre_seg_button.clicked.connect(self.thick_seg)
        self.ui.seg_button.clicked.connect(self.refine_seg)
        self.ui.save_button.clicked.connect(self.save_label)

        self.ui.min_slider.valueChanged.connect(self.min_slider_change)
        self.ui.max_slider.valueChanged.connect(self.max_slider_change)

        self.ui.min_slider.valueChanged.connect(self.vtkWidget.brightness_min_change)
        self.ui.max_slider.valueChanged.connect(self.vtkWidget.brightness_max_change)

        self.ui.single_line.clicked.connect(self.vtkWidget.single_model)
        self.ui.double_line.clicked.connect(self.vtkWidget.double_model)

        self.vtkWidget.ren.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

        self.vtkWidget.iren.Initialize()

    def drawBox(self, pointA, pointB):
        minX, minY, minZ = pointA
        maxX, maxY, maxZ = pointB

        boxGridPoints = vtk.vtkPoints()
        boxGridPoints.SetNumberOfPoints(8)
        boxGridPoints.SetPoint(0, minX, maxY, minZ)
        boxGridPoints.SetPoint(1, maxX, maxY, minZ)
        boxGridPoints.SetPoint(2, maxX, minY, minZ)
        boxGridPoints.SetPoint(3, minX, minY, minZ)

        boxGridPoints.SetPoint(4, minX, maxY, maxZ)
        boxGridPoints.SetPoint(5, maxX, maxY, maxZ)
        boxGridPoints.SetPoint(6, maxX, minY, maxZ)
        boxGridPoints.SetPoint(7, minX, minY, maxZ)

        boxGridCellArray = vtk.vtkCellArray()
        for i in range(12):
            boxGridCell = vtk.vtkLine()
            if i < 4:
                temp_data = (i + 1) if (i + 1) % 4 != 0 else 0
                boxGridCell.GetPointIds().SetId(0, i)
                boxGridCell.GetPointIds().SetId(1, temp_data)
            elif i < 8:
                temp_data = (i + 1) if (i + 1) % 8 != 0 else 4
                boxGridCell.GetPointIds().SetId(0, i)
                boxGridCell.GetPointIds().SetId(1, temp_data)
            else:
                boxGridCell.GetPointIds().SetId(0, i % 4)
                boxGridCell.GetPointIds().SetId(1, i % 4 + 4)
            boxGridCellArray.InsertNextCell(boxGridCell)

        boxGridData = vtk.vtkPolyData()
        boxGridData.SetPoints(boxGridPoints)
        boxGridData.SetLines(boxGridCellArray)
        boxGridMapper = vtk.vtkPolyDataMapper()
        boxGridMapper.SetInputData(boxGridData)
        return boxGridMapper

    def drawAxes(self, axesActor):
        axesActor.SetTotalLength(20, 20, 20)
        axesActor.SetShaftTypeToCylinder()
        axesActor.SetCylinderRadius(.05)

        axesActor.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axesActor.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(20)

        axesActor.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axesActor.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(20)

        axesActor.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axesActor.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(20)

        transform = vtk.vtkTransform()
        transform.Translate(-5.0, -5.0, -5.0)
        axesActor.SetUserTransform(transform)

        return axesActor

    def open_image(self):
        self.image_path, self.image_type = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image", "", "*.tif")

        if self.image_path == "":
            return

        self.vtkWidget.remove_all_actors()
        img = libTIFFRead(self.image_path)

        self.temp_image = img
        self.vtkWidget.img = img
        self.image_size = img.shape[0]

        self.vtkWidget.max_x = img.shape[2]
        self.vtkWidget.max_y = img.shape[1]
        self.vtkWidget.max_z = img.shape[0]

        volume_img = creat_volume_actor(img, img.max())

        self.vtkWidget.ren.AddActor(volume_img)
        self.vtkWidget.image_actor = volume_img

        self.refine_seg_thread.fp_point_volume = list()
        self.refine_seg_thread.fn_point_volume = list()

        self.vtkWidget.ren.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

        self.min_slider.setMaximum(img.max())
        self.max_slider.setMaximum(img.max())
        self.min_slider.setValue(img.min())
        self.max_slider.setValue(img.max())
        self.min_label.setText(str(img.min()))
        self.max_label.setText(str(img.max()))
        self.vtkWidget.current_min_brightness = img.min()
        self.vtkWidget.current_max_brightness = img.max()

        # Draw bounding box
        pointA = [0, 0, 0]
        pointB = [img.shape[0], img.shape[1], img.shape[2]]
        boxGridMapper = self.drawBox(pointA, pointB)
        bounding_actor = vtk.vtkActor()
        bounding_actor.SetMapper(boxGridMapper)
        self.vtkWidget.ren.AddActor(bounding_actor)

        # Draw axes
        axesActor = vtk.vtkAxesActor()
        axesActor = self.drawAxes(axesActor)
        self.vtkWidget.ren.AddActor(axesActor)

    def min_slider_change(self):
        if self.min_slider.value() < self.max_slider.value():
            self.min_label.setText(str(self.min_slider.value()))
            self.vtkWidget.current_min_brightness = self.min_slider.value()
            self.temp_min_brightness = self.min_slider.value()
        else:
            self.min_slider.setValue(self.max_slider.value()-1)
            self.vtkWidget.current_min_brightness = self.min_slider.value()
            self.temp_min_brightness = self.min_slider.value()

    def max_slider_change(self):
        if self.max_slider.value() > self.min_slider.value():
            self.max_label.setText(str(self.max_slider.value()))
            self.vtkWidget.current_max_brightness = self.max_slider.value()
            self.temp_max_brightness = self.max_slider.value()
        else:
            self.max_slider.setValue(self.min_slider.value() + 1)
            self.vtkWidget.current_max_brightness = self.max_slider.value()
            self.temp_max_brightness = self.max_slider.value()

    def thick_seg(self):
        self.thick_seg_thread.image_size = self.image_size
        self.thick_seg_thread.temp_image = self.temp_image
        self.thick_seg_thread.start()

    def thick_seg_output(self, label):
        self.vtkWidget.remove_seg_actors()
        self.temp_label = label
        self.vtkWidget.label = label

        lab = creat_points_actor(label)
        self.vtkWidget.label_actor = lab

        self.vtkWidget.ren.AddActor(lab)
        self.vtkWidget.GetRenderWindow().Render()

    def refine_seg(self):
        self.refine_seg_thread.image_size = self.image_size
        self.refine_seg_thread.temp_image = self.temp_image
        self.refine_seg_thread.temp_label = self.temp_label
        self.refine_seg_thread.fp_point_volume = self.vtkWidget.fp_volumes
        self.refine_seg_thread.fn_point_volume = self.vtkWidget.fn_volumes

        self.refine_seg_thread.start()

    def refine_seg_output(self, out):
        self.vtkWidget.remove_seg_actors()
        label = np.array(out[1], dtype=np.uint8)

        self.vtkWidget.label = label
        self.temp_label = label
        self.temp_image = out[0]

        lab = creat_points_actor(label)
        self.vtkWidget.label_actor = lab

        self.vtkWidget.ren.AddActor(lab)
        self.vtkWidget.GetRenderWindow().Render()

    def save_label(self):
        label_path = QFileDialog.getSaveFileName(self, "Save Image", ".tif", "*.tif")

        if label_path == "":
            return

        save_path = label_path[0]
        self.temp_label = np.array(self.temp_label, dtype=np.uint8)
        libTIFFWrite(save_path, self.temp_label)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWindow()
    w.ui.show()
    app.exec_()

