from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera


class MyInteractorStyle(vtkInteractorStyleTrackballCamera):
    def __init__(self, interactor, renderer, myvtkwidget, model):
        self.myvtkwidget = myvtkwidget
        self.interactor = interactor
        self.renderer = renderer

        self.model = model
        self.check = 1

        self.first_index = []
        self.second_index = []
        self.first_list = list()
        self.second_list = list()

        self.AddObserver("RightButtonReleaseEvent", self.right_button_release_event)
        self.AddObserver("LeftButtonReleaseEvent", self.left_button_release_event)
        self.AddObserver("MouseMoveEvent", self.Mouse_Move_Event)
        self.number_of_click = 0

    def left_button_release_event(self, obj, event):
        if self.model == "Single":
            self.number_of_click = self.number_of_click + 1
            if self.number_of_click == 2:
                screenX, screenY = self.interactor.GetEventPosition()

                self.interactor.GetPicker().Pick(screenX, screenY, 0, self.renderer)
                picked = self.interactor.GetPicker()
                x, y, z = picked.GetPickPosition()

                self.number_of_click = 0
                self.myvtkwidget.add_single_point(x, y, z)

            self.OnLeftButtonUp()
            return

        if self.model == "Double":
            self.number_of_click = self.number_of_click + 1
            if self.number_of_click == 2:
                screenX, screenY = self.interactor.GetEventPosition()

                self.interactor.GetPicker().Pick(screenX, screenY, 0, self.renderer)
                picked = self.interactor.GetPicker()
                x, y, z = picked.GetPickPosition()

                if self.check == 1:
                    self.number_of_click = 0
                    self.first_index = [x, y, z]
                    self.first_list = self.myvtkwidget.add_line(x, y, z)
                    self.check = 2

                elif self.check == 2:
                    self.number_of_click = 0
                    self.second_index = [x, y, z]
                    self.second_list = self.myvtkwidget.add_line(x, y, z)
                    self.myvtkwidget.add_double_point(self.first_list, self.second_list)
                    self.myvtkwidget.remove_last_line()
                    self.myvtkwidget.remove_last_line()
                    self.check = 1

            self.OnLeftButtonUp()
            return

    def Mouse_Move_Event(self, obj, event):
        self.number_of_click = 0
        self.OnMouseMove()
        return

    def right_button_release_event(self, obj, event):
        if self.model == "Single":
            self.myvtkwidget.remove_last_point()

        if self.model == "Double":
            if self.check == 2:
                self.myvtkwidget.remove_last_line()
                self.check = 1
            elif self.check == 1:
                self.myvtkwidget.remove_last_point()

        self.OnRightButtonUp()
        return
