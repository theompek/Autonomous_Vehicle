"""
This module represent all possible dynamic objects around the car. Dynamic obstacles are vehicles, pedestrians,
traffic signs, traffic lights.
Module functions:
1) Objects around the car: Instances of the objects like other cars and pedestrians around the car
2) The coordinates of the obstacles: The x-y coordinates


"""

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
from time import sleep
import math
import numpy as np

# ==============================================================================
# -- Implementation ------------------------------------------------------------
# ==============================================================================


def vehicle_boundaries(vehicle, l_offset=0.3, w_offset=0.2):
    length = vehicle.bounding_box.extent.x * 2
    width = vehicle.bounding_box.extent.y * 2
    a = length + l_offset
    b = width + w_offset
    r = [a*b/np.hypot(a*math.sin(math.radians(fi)), b*math.cos(math.radians(fi))) for fi in range(360)]
    q1, q2, q3, q4 = 30, 150, 210, 330

    d1 = r[q1]*math.sin(math.radians(q1))
    d2 = -r[q3]*math.sin(math.radians(q3))
    for i in range(q1, q2):
        r[i] = d1/math.cos(math.radians(i-90))
    for i in range(q3, q4):
        r[i] = d2/math.cos(math.radians(i-270))

    return r


class Objects:
    """
        The class represents the dynamic objects of the world like vehicle and pedestrians alongside with dynamic
        actors like traffic lights and signs that can been used like objects as they change the vehicle behavior.
    """
    def __init__(self, obj_type="vehicles", obj_list=None):
        """
        :param obj_type: A string with the name/type of the obstacle/object
        :param obj_list: A list of the objects
        """
        if obj_type is None:
            self.object_type = "None"
            self.objects_list = []
            self.object_id = 0
            self.acceleration = 0
            self.x = 0
            self.y = 0
            self.yaw = 0
            self.vel_x = 0
            self.vel_y = 0
            self.speed = 0
            self.dimensions = {"length": 0, "width": 0}
        elif type(obj_list) is not list:
            self.object_type = obj_type
            self.objects_list = [obj_list]
            self.object_id = self.get_id()[0]
            self.acceleration = self.get_total_acceleration()[0]
            self.x = self.get_x_coordinates()[0]
            self.y = self.get_y_coordinates()[0]
            self.yaw = self.get_yaw()[0]
            self.vel_x = self.get_velocity_x()[0]
            self.vel_y = self.get_velocity_y()[0]
            self.speed = self.get_total_speed()[0]
            self.dimensions = self.get_object_dimensions()[0]
        else:
            self.object_type = obj_type
            self.objects_list = obj_list
            self.object_id = self.get_id()
            self.acceleration = self.get_total_acceleration()
            self.x = self.get_x_coordinates()
            self.y = self.get_y_coordinates()
            self.yaw = self.get_yaw()
            self.vel_x = self.get_velocity_x()
            self.vel_y = self.get_velocity_y()
            self.speed = self.get_total_speed()
            self.dimensions = self.get_object_dimensions()

    def get_xy_coordinates(self):
        """
            The method returns the x-y coordinates of the objects
        """
        return [[object_i.get_location().x, object_i.get_location().y] for object_i in self.objects_list if object_i is not None]

    def get_x_coordinates(self):
        """
            The method returns the x coordinate of the objects
        """
        return [object_i.get_location().x for object_i in self.objects_list if object_i is not None]

    def get_y_coordinates(self):
        """
            The method returns the y coordinate of the objects
        """
        return [object_i.get_location().y for object_i in self.objects_list if object_i is not None]

    def get_yaw(self):
        """
            The method returns the yaw value of the objects
        """
        return [object_i.get_transform().rotation.yaw for object_i in self.objects_list if object_i is not None]

    def get_velocity_x(self):
        """
            The method returns the objects' velocity of the x axis
        """
        return [object_i.get_velocity().x for object_i in self.objects_list if object_i is not None]

    def get_velocity_y(self):
        """
            The method returns the objects' velocity of the y axis
        """
        return [object_i.get_velocity().y for object_i in self.objects_list if object_i is not None]

    def get_total_acceleration(self):
        """
            The method returns the objects' speed in m/s
        """
        return [math.hypot(object_i.get_acceleration().y, object_i.get_acceleration().x) for object_i in self.objects_list if object_i is not None]

    def get_total_speed(self):
        """
            The method returns the objects' speed in m/s
        """
        return [math.hypot(object_i.get_velocity().y, object_i.get_velocity().x) for object_i in self.objects_list if object_i is not None]

    def get_object_dimensions(self):
        """
            The method returns a dictionary with the objects' dimensions length and width
        """
        if self.object_type not in ["traffic_lights", "traffic_signs"]:
            return [{"length": object_i.bounding_box.extent.x, "width": object_i.bounding_box.extent.y}
                    for object_i in self.objects_list if object_i is not None]
        else:
            return [{"length": None, "width": None}]

    def get_id(self):
        """
            The method returns a unique id
        """
        return [object_i.id for object_i in self.objects_list if object_i is not None]

