#!/usr/bin/env python

"""

"""
# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../EggFiles/carla-0.9.7-py2.7-linux-x86_64.egg')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import carla
import time
from time import sleep
import math
import numpy as np
from scipy import interpolate
import random
import copy

# ==============================================================================
# -- Implementation ------------------------------------------------------------
# ==============================================================================
SEARCH_STEP = 2


class Node:
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None, yaw=0):
        self.parent = parent
        self.position = position
        self.map_yaw = yaw
        self.g = 0.0
        self.h = 1e10
        self.f = 1e10

    def __eq__(self, other):
        return math.hypot(self.position.x - other.position.x, self.position.y - other.position.y) <= math.sqrt(SEARCH_STEP)

    def calc_g(self):
        # Mporw na valw kai thn metabolh ths kateu8hnshs san parametro a3iologhshs
        if self.parent is not None:
            self.g = self.parent.g + 1.0 + self.position.yaw**10
        else:
            self.g = 0.0

    def calc_h(self, goal_position):
        # Euclidean distance from the goal
        self.h = math.hypot(goal_position.y - self.position.y, goal_position.x - self.position.x)

    def calc_f(self):
        self.f = self.g + self.h


class Position:
    """The position of a node and the function to check if can be permitted"""
    def __init__(self, x, y, yaw=0.0, lane_id=False, junction=False, lane_type="Driving"):
        self.x = x
        self.y = y
        # change of yaw related to parent node
        self.yaw = yaw
        # Lane id and road inforation
        self.same_lane_id = lane_id
        self.is_junction = junction
        self.lane_type = lane_type

    def position_permitted(self):
        if (not self.same_lane_id and not self.is_junction) or abs(self.yaw) > math.pi/2 or self.lane_type != "Driving":
            return False
        return True


class AStar:
    def __init__(self, start_position, end_position, the_map, world):
        # Create start and end node
        self.the_map = the_map
        self.world = world
        self.start_node = Node()
        self.end_node = Node()
        self.initialization_star_end_positions(start_position, end_position, the_map)
        self.travel_step = SEARCH_STEP
        # Initialize both open and closed list
        self.open_list = []
        self.closed_list = []
        # Add the start node
        self.open_list.append(self.start_node)
        self.initial_path = []
        self.smoothed_path = []
        self.world.debug.draw_string(carla.Location(x=self.start_node.position.x, y=self.start_node.position.y),
                                     "<---- Start", draw_shadow=False, color=carla.Color(r=255, g=200, b=0),
                                     life_time=40, persistent_lines=True)
        self.world.debug.draw_string(carla.Location(x=self.end_node.position.x, y=self.end_node.position.y),
                                     "<---- End", draw_shadow=False,
                                     color=carla.Color(r=255, g=200, b=0), life_time=40, persistent_lines=True)

    def initialization_star_end_positions(self, start_position, end_position, the_map):
        x = start_position[0]
        y = start_position[1]
        w_start = the_map.get_waypoint(carla.Location(x=x, y=y), lane_type=carla.LaneType.Driving)
        yaw_start = math.radians(w_start.transform.rotation.yaw)
        w_start = w_start.transform.location
        x = end_position[0]
        y = end_position[1]
        w_end = the_map.get_waypoint(carla.Location(x=x, y=y), lane_type=carla.LaneType.Driving)
        yaw_end = math.radians(w_end.transform.rotation.yaw)
        w_end = w_end.transform.location
        self.start_node = Node(None, Position(w_start.x, w_start.y, yaw_start))
        self.end_node = Node(None, Position(w_end.x, w_end.y, yaw_end))

    def get_adjacent_positions(self, node):
        waypoint = self.the_map.get_waypoint(carla.Location(x=node.position.x, y=node.position.y), lane_type=carla.LaneType.Any)
        node_lane_id = str(waypoint.road_id) + str(waypoint.section_id) + str(waypoint.lane_id)
        #waypoint = node.position
        positions_list = []

        for point in waypoint.next(2*self.travel_step):
            x = point.transform.location.x
            y = point.transform.location.y
            yaw = point.transform.rotation.yaw
            point_lane_id = str(point.road_id) + str(point.section_id) + str(point.lane_id)
            is_junction = point.is_junction or node.position.is_junction
            yaw_change = math.acos(math.cos(yaw) * math.cos(node.map_yaw) + math.sin(yaw) * math.sin(node.map_yaw)) % math.pi
            positions_list.append(Position(x, y, yaw_change, point_lane_id == node_lane_id, is_junction, str(point.lane_type)))
        waypoint = waypoint.transform.location
        yaw_list = [-90.0, 90.0, 180.0, 0.0, -135.0, 135.0,  -45.0, 45.0]
        for i, point in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]):
            x = waypoint.x + self.travel_step * point[0]
            y = waypoint.y + self.travel_step * point[1]
            yaw = math.radians(yaw_list[i])
            yaw_change = math.acos(math.cos(yaw)*math.cos(node.map_yaw) + math.sin(yaw)*math.sin(node.map_yaw)) % math.pi
            w_nearest = self.the_map.get_waypoint(carla.Location(x=x, y=y), lane_type=carla.LaneType.Any)
            point_lane_id = str(w_nearest.road_id) + str(w_nearest.section_id) + str(w_nearest.lane_id)
            is_junction = w_nearest.is_junction or node.position.is_junction
            positions_list.append(Position(x, y, yaw_change, point_lane_id == node_lane_id, is_junction, str(w_nearest.lane_type)))
            self.world.debug.draw_string(carla.Location(x=x, y=y), "0", draw_shadow=False, color=carla.Color(r=255, g=20, b=0),
                                         life_time=0.5, persistent_lines=True)
            if not any([pos.same_lane_id for pos in positions_list]):
                for pos in positions_list:
                    pos.is_junction = True
        return positions_list

    def find_path(self):
        """Returns a list of tuples as a path from the given start to the given end in the given maze"""
        # Loop until you find the end
        while len(self.open_list) > 0:
            # Get the current node, the node with the lower f value
            f_values = [node.f for node in self.open_list]
            current_index = f_values.index(min(f_values))
            current_node = self.open_list[current_index]
            # Pop current off open list, add to closed list
            self.open_list.pop(current_index)
            self.closed_list.append(current_node)

            if current_node == self.end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                self.initial_path = path[::-1]
                self.generate_target_course()
                return self.smoothed_path  # Return reversed path

            # Generate children
            children = []
            for new_position in self.get_adjacent_positions(current_node):  # Adjacent squares
                # Make sure is in a driving path
                if not new_position.position_permitted():
                    continue
                self.world.debug.draw_string(carla.Location(x=new_position.x, y=new_position.y), "X", draw_shadow=False,
                                             color=carla.Color(r=2, g=200, b=0),
                                             life_time=0.5, persistent_lines=True)
                # Create new node
                waypoint = self.the_map.get_waypoint(carla.Location(x=new_position.x, y=new_position.y),
                                                     lane_type=carla.LaneType.Driving)
                map_yaw = math.radians(waypoint.transform.rotation.yaw)
                new_node = Node(current_node, new_position, map_yaw)
                # Append
                children.append(new_node)
            # Loop through children
            for child in children:
                # Child is on the closed list
                in_closed_list = False
                for closed_child in self.closed_list:
                    if child == closed_child:
                        in_closed_list = True
                        break
                if in_closed_list:
                    continue

                # Create the f, g, and h values
                child.calc_g()
                child.calc_h(self.end_node.position)
                child.calc_f()

                # Child is already in the open list
                in_open_list = False
                for i, open_node in enumerate(self.open_list):
                    if child == open_node:
                        if child.g < open_node.g:
                            self.open_list[i] = child
                        in_open_list = True
                        break
                if in_open_list:
                    continue

                # Add the child to the open list
                self.open_list.append(child)
        # If there is no path directly to the points take the closer one
        f_values = [node.f for node in self.closed_list]
        current_index = f_values.index(min(f_values))
        current_node = self.closed_list[current_index]
        path = []
        current = current_node
        while current is not None:
            path.append(current.position)
            current = current.parent
        self.initial_path = path[::-1]
        self.generate_target_course()
        return self.smoothed_path  # Return reversed path

    def generate_target_course(self):
        x = [pos.x for pos in self.initial_path]
        y = [pos.y for pos in self.initial_path]
        xn = np.r_[x, x[0]]
        yn = np.r_[y, y[0]]
        # Path smoothness
        m = len(xn)
        smoothness = m + math.sqrt(2*m)  # recommended
        print(smoothness)
        # create spline function
        f, u = interpolate.splprep([xn, yn], s=smoothness, per=True)
        # create interpolated lists of points
        xint, yint = interpolate.splev(np.linspace(0, 1, 400), f)
        dist_to_end = []
        for xi, yi in zip(xint, yint):
            dist_to_end.append(math.hypot(self.end_node.position.x - xi, self.end_node.position.y - yi))
            self.smoothed_path.append(Position(xi, yi))
        self.smoothed_path = self.smoothed_path[0:dist_to_end.index(min(dist_to_end))]
        if self.smoothed_path is not None:
            for p in self.smoothed_path:
                sleep(0.001)
                self.world.debug.draw_string(carla.Location(x=p.x, y=p.y), "++", draw_shadow=False,
                                        color=carla.Color(r=20, g=20, b=220),
                                        life_time=500, persistent_lines=True)


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)  # seconds
    world = client.get_world()
    the_map = world.get_map()
    start = (10, -200)
    end = (-10, 130)

    start = (170, -200)
    end = (random.randint(-300, 300), random.randint(30, 300))

    a_star_algorithm = AStar(start, end, the_map, world)
    smoothed_path = a_star_algorithm.find_path()
    if smoothed_path is not None:
        for p in smoothed_path:
            sleep(0.001)
            world.debug.draw_string(carla.Location(x=p.x, y=p.y), "x", draw_shadow=False,
                                    color=carla.Color(r=20, g=20, b=220),
                                    life_time=5, persistent_lines=True)
    else:
        print("Path not found")
        world.debug.draw_string(carla.Location(x=start[1], y=start[1]), "Path not found", draw_shadow=False,
                                color=carla.Color(r=250, g=20, b=20),
                                life_time=5, persistent_lines=True)


if __name__ == '__main__':
    main()