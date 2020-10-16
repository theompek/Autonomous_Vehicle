#!/usr/bin/env python

# ==============================================================================
# --General imports ------------------------------------------------------------
# ==============================================================================
import time
import math
import rospy
import os
import sys
from time import sleep
import local_map
try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

# ==============================================================================
# -- ROS imports ---------------------------------------------------------------
# ==============================================================================
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from perception.msg import Object
from perception.msg import TrafficSigns
from perception.msg import WaypointsList
from perception.msg import Waypoint
from perception.msg import ObjectsList
from perception.msg import Lane
from perception.msg import RoadLanes
from perception.msg import TrajectoriesList
from perception.msg import Float64List
from perception.msg import StringList
from control.msg import VehicleCmd
from local_path_planner.msg import LocalOptimalPath
from std_msgs.msg import Float64
from std_msgs.msg import Bool

from perception.srv import EgoVehicleGeometry, EgoVehicleGeometryRequest, EgoVehicleGeometryResponse
from perception.srv import VehicleByID, VehicleByIDRequest, VehicleByIDResponse
from perception.srv import DrivingPaths, DrivingPathsRequest, DrivingPathsResponse
from perception.srv import EgoVehicleName, EgoVehicleNameRequest, EgoVehicleNameResponse
from route_planner.srv import RoutePath, RoutePathRequest, RoutePathResponse
from route_planner.msg import RouteLocation
from maneuver_generator.msg import ManeuverDataROS

# ==============================================================================
# -- Implementation ------------------------------------------------------------
# ==============================================================================
DRAW_TIME = 0.1


class PygameDraw:
    """
    Class to open a pygame window and write information about the implementation
    """
    def __init__(self, num_of_line, text_size, text_width):
        self.space = 10.0
        self.step = text_size + self.space
        self.text_size = text_size
        self.width = int(self.step*num_of_line + 2.0 * self.space)
        self.height = int(text_width)
        self.position = 0
        self.refresh_time = time.time()
        self.text_to_draw = [["None"]*11, ["None"]*4]
        self.surface = self.get_pygame_surface()
        self.mode_0 = False
        self.info_mode = 0
        self.paths_enabled = True
        self.pygame_quit = False
        self.button_size_x = 140
        self.button_size_y = 25
        self.button1 = pygame.Rect(self.height - 2*self.button_size_x - 5, self.width - self.button_size_y-2,
                                  self.button_size_x, self.button_size_y)
        self.button2 = pygame.Rect(self.height - self.button_size_x, self.width - self.button_size_y-2,
                                   self.button_size_x, self.button_size_y)
        # Buttons for position and direction
        self.p_d_button_down = pygame.Rect(self.height - self.button_size_x / 3 - 2, 3,
                                           self.button_size_x / 3, self.button_size_y)
        self.p_d_button_up = pygame.Rect(self.height - self.button_size_x*5/3 - 6, 3,
                                   self.button_size_x/3, self.button_size_y)
        self.p_d_button_noise = pygame.Rect(self.height - self.button_size_x * 4 / 3 - 4, 3,
                                            self.button_size_x, self.button_size_y)
        # Buttons for speed
        self.s_button_down = pygame.Rect(self.height - self.button_size_x / 3 - 2, self.button_size_y + 6,
                                        self.button_size_x / 3, self.button_size_y)
        self.s_button_up = pygame.Rect(self.height - self.button_size_x * 5 / 3 - 6, self.button_size_y + 6,
                                    self.button_size_x / 3, self.button_size_y)
        self.s_button_noise = pygame.Rect(self.height - self.button_size_x * 4 / 3 - 4, self.button_size_y + 6,
                                        self.button_size_x, self.button_size_y)
        self.button_objects = pygame.Rect(self.height - self.button_size_x - 2, 8 * self.button_size_y + 8,
                                         self.button_size_x, self.button_size_y)
        self.button_frenet = pygame.Rect(self.height - self.button_size_x - 2, 7 * self.button_size_y + 6,
                                         self.button_size_x,  self.button_size_y)
        self.button_optimal_path = pygame.Rect(self.height - self.button_size_x - 2, 6 * self.button_size_y + 4,
                                         self.button_size_x, self.button_size_y)
        self.button_prototype = pygame.Rect(self.height - self.button_size_x - 2, 5 * self.button_size_y + 2,
                                            self.button_size_x, self.button_size_y)
        self.frenet_enable = True
        self.optimal_enable = True
        self.prototype_enable = True
        self.objects_enable = True
        self.p_d_noise_enable = False
        self.p_d_std = 0.5
        self.s_noise_enable = False
        self.s_std = 0.5
        self.button_font = pygame.font.Font("freesansbold.ttf", int(self.button_size_x/15))
        # completely fill the surface object with white color
        self.surface.fill(local_map.COLORS["white"])

    def get_pygame_surface(self):
        pygame.init()
        # create the display surface object
        # of specific dimension..e(X, Y).
        display_surface = pygame.display.set_mode((self.height, self.width))
        # set the pygame window name
        pygame.display.set_caption('Information window')
        return display_surface

    def pygame_buttons_check(self):
        # pygame.init()
        # iterate over the list of Event objects
        # that was returned by pygame.event.get() method.
        for event in pygame.event.get():
            # if event object type is QUIT
            # then quitting the pygame
            # and program both.
            if event.type == pygame.QUIT:
                # deactivates the pygame library
                self.pygame_quit = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # 1 is the left mouse button, 2 is middle, 3 is right.
                if event.button == 1:
                    # `event.pos` is the mouse position.
                    if self.button1.collidepoint(event.pos):
                        # Change between mode 0 and 1
                        self.info_mode = 1 if self.info_mode == 0 else 0
                        self.mode_0 = False
                    elif self.button2.collidepoint(event.pos):
                        # Enable disable paths draw
                        self.paths_enabled = not self.paths_enabled
                    elif self.p_d_button_down.collidepoint(event.pos):
                        self.p_d_std += -0.1
                        self.p_d_std = 0.0 if self.p_d_std < 0 else self.p_d_std
                    elif self.p_d_button_up.collidepoint(event.pos):
                        self.p_d_std += 0.1
                    elif self.p_d_button_noise.collidepoint(event.pos):
                        self.p_d_noise_enable = not self.p_d_noise_enable
                    elif self.s_button_down.collidepoint(event.pos):
                        self.s_std += -0.1
                        self.s_std = 0.0 if self.s_std < 0 else self.s_std
                    elif self.s_button_up.collidepoint(event.pos):
                        self.s_std += 0.1
                    elif self.s_button_noise.collidepoint(event.pos):
                        self.s_noise_enable = not self.s_noise_enable
                    elif self.button_frenet.collidepoint(event.pos):
                        self.frenet_enable = not self.frenet_enable
                    elif self.button_optimal_path.collidepoint(event.pos):
                        self.optimal_enable = not self.optimal_enable
                    elif self.button_prototype.collidepoint(event.pos):
                        self.prototype_enable = not self.prototype_enable
                    elif self.button_objects.collidepoint(event.pos):
                        self.objects_enable = not self.objects_enable

    def write_text(self, text_line, text_color, bg_color, continue_line=False, refresh_window=False):
        if refresh_window:
            self.position = 0
            self.surface.fill(local_map.COLORS["white"])
        font = pygame.font.Font('freesansbold.ttf', int(self.text_size))
        text = font.render(text_line, True, local_map.COLORS[text_color], local_map.COLORS[bg_color])
        textRect = text.get_rect()
        if not continue_line:
            self.position += self.step
            textRect.center = (self.height/2, self.position)
        else:
            self.position += self.text_size
            textRect.center = (self.height / 2, self.position)
        self.surface.blit(text, textRect)

    def draw_text_info(self):
        self.pygame_buttons_check()
        current_time = time.time()
        if current_time - self.refresh_time > 0.00001:
            if self.info_mode == 1:
                self.refresh_time = current_time
                bg_color = "white"
                self.write_text("", "black", bg_color, refresh_window=True)
                self.write_text("", "black", bg_color)

                # ----------- Information for the maneuver -----------
                if self.text_to_draw[1][3] == "True":
                    self.write_text(" The vehicle reached its destination, this is the end of the route path : ", "black", "red")
                self.write_text(" maneuver_type : " + self.text_to_draw[0][0], "black", bg_color)
                self.write_text(" target_lateral_offset : " + self.text_to_draw[0][1], "black", bg_color)
                self.write_text(" Road speed limit : " + self.text_to_draw[1][2], "black", bg_color)
                self.write_text(" target_speed : " + (self.text_to_draw[0][2]), "black", bg_color)
                self.write_text(" direct_control : " + self.text_to_draw[0][3], "black", bg_color)
                self.write_text(" left_road_width : " + self.text_to_draw[0][4], "black", bg_color)
                self.write_text(" right_road_width : " + self.text_to_draw[0][5], "black", bg_color)
                self.write_text(" num_of_paths_samples : " + self.text_to_draw[0][6], "black", bg_color)
                self.write_text(" from_time : " + self.text_to_draw[0][7], "black", bg_color)
                self.write_text(" to_time : " + self.text_to_draw[0][8], "black", bg_color)
                self.write_text(" time_sample_step : " + self.text_to_draw[0][9], "black", bg_color)
                self.write_text(" dt : " + self.text_to_draw[0][10], "black", bg_color)
                self.write_text(" Collision with vehicle probability : " + self.text_to_draw[1][0], "black", bg_color)
                self.write_text(" Collision with pedestrian probability : " + self.text_to_draw[1][1], "black", bg_color)
                # Button 1
                pygame.draw.rect(self.surface, local_map.COLORS["blue"], self.button1)
                text_surf1 = self.button_font.render("> Change window <", True, local_map.COLORS["black"])
                text_rect1 = text_surf1.get_rect(center=(self.height - int(3*self.button_size_x/2)-2,
                                                         self.width - int(self.button_size_y/2)))
                self.surface.blit(text_surf1, text_rect1)
                # Button 2
                color_button = local_map.COLORS["green"] if self.paths_enabled else local_map.COLORS["red"]
                pygame.draw.rect(self.surface, color_button, self.button2)
                text_enable = "Enabled" if self.paths_enabled else "Disabled"
                text_surf = self.button_font.render("> Paths view " + text_enable + " <", True, local_map.COLORS["black"])
                text_rect = text_surf.get_rect(center=(self.height - int(self.button_size_x / 2),
                                                       self.width - int(self.button_size_y / 2)))
                self.surface.blit(text_surf, text_rect)
                # Button Position Direction Noise Enable
                # Button Down
                pygame.draw.rect(self.surface, local_map.COLORS["blue"], self.p_d_button_down)
                text_surf1 = self.button_font.render("Down", True, local_map.COLORS["black"])
                text_rect1 = text_surf1.get_rect(center=(self.height - int(self.button_size_x / 6) - 2,
                                                         int(self.button_size_y / 2)+3))
                self.surface.blit(text_surf1, text_rect1)
                # Button Up
                pygame.draw.rect(self.surface, local_map.COLORS["blue"], self.p_d_button_up)
                text_surf1 = self.button_font.render("Up", True, local_map.COLORS["black"])
                text_rect1 = text_surf1.get_rect(center=(self.height - int(9 * self.button_size_x / 6) - 4,
                                                         int(self.button_size_y / 2)+3))
                self.surface.blit(text_surf1, text_rect1)
                color_button = local_map.COLORS["green"] if self.p_d_noise_enable else local_map.COLORS["red"]
                text_enable = "Enabled" if self.p_d_noise_enable else "Disabled"
                pygame.draw.rect(self.surface, color_button, self.p_d_button_noise)
                text_surf1 = self.button_font.render("Noise "+text_enable, True, local_map.COLORS["black"])
                text_rect1 = text_surf1.get_rect(center=(self.height - int(5 * self.button_size_x / 6) - 6,
                                                         int(self.button_size_y / 2)+3))
                self.surface.blit(text_surf1, text_rect1)
                text_surf1 = self.button_font.render("Standard deviation of position and direction (m,rad): "
                                                     + str(round(self.p_d_std, 2)), True, local_map.COLORS["black"])
                text_rect1 = text_surf1.get_rect(center=(self.height - int(15 * self.button_size_x / 6) - 22,
                                                         int(self.button_size_y / 2) + 3))
                self.surface.blit(text_surf1, text_rect1)
                # Button Speed Noise Enable
                # Button Down
                pygame.draw.rect(self.surface, local_map.COLORS["blue"], self.s_button_down)
                text_surf1 = self.button_font.render("Down", True, local_map.COLORS["black"])
                text_rect1 = text_surf1.get_rect(center=(self.height - int(self.button_size_x / 6) - 2,
                                                         self.button_size_y + int(self.button_size_y / 2) + 6))
                self.surface.blit(text_surf1, text_rect1)
                # Button Up
                pygame.draw.rect(self.surface, local_map.COLORS["blue"], self.s_button_up)
                text_surf1 = self.button_font.render("Up", True, local_map.COLORS["black"])
                text_rect1 = text_surf1.get_rect(center=(self.height - int(9 * self.button_size_x / 6) - 4,
                                                         self.button_size_y + int(self.button_size_y / 2) + 6))
                self.surface.blit(text_surf1, text_rect1)
                color_button = local_map.COLORS["green"] if self.s_noise_enable else local_map.COLORS["red"]
                text_enable = "Enabled" if self.s_noise_enable else "Disabled"
                pygame.draw.rect(self.surface, color_button, self.s_button_noise)
                text_surf1 = self.button_font.render("Noise " + text_enable, True, local_map.COLORS["black"])
                text_rect1 = text_surf1.get_rect(center=(self.height - int(5 * self.button_size_x / 6) - 6,
                                                         self.button_size_y + int(self.button_size_y / 2) + 6))
                self.surface.blit(text_surf1, text_rect1)
                text_surf1 = self.button_font.render("Standard deviation of objects speed (km/h): "
                                                     + str(round(self.s_std, 2)), True, local_map.COLORS["black"])
                text_rect1 = text_surf1.get_rect(center=(self.height - int(15 * self.button_size_x / 6) - 8,
                                                         self.button_size_y + int(self.button_size_y / 2) + 6))
                self.surface.blit(text_surf1, text_rect1)
                # Objects Enable
                color_button = local_map.COLORS["green"] if self.objects_enable else local_map.COLORS["red"]
                text_enable = "Enabled" if self.objects_enable else "Disabled"
                pygame.draw.rect(self.surface, color_button, self.button_objects)
                text_surf1 = self.button_font.render("Objects info " + text_enable, True,
                                                     local_map.COLORS["black"])
                text_rect1 = text_surf1.get_rect(center=(self.height - int(self.button_size_x / 2) - 2,
                                                         8 * self.button_size_y + int(self.button_size_y / 2) + 8))
                self.surface.blit(text_surf1, text_rect1)
                # Frenet Paths Enable
                color_button = local_map.COLORS["green"] if self.frenet_enable else local_map.COLORS["red"]
                text_enable = "Enabled" if self.frenet_enable else "Disabled"
                pygame.draw.rect(self.surface, color_button, self.button_frenet)
                text_surf1 = self.button_font.render("Frenet "+text_enable, True, local_map.COLORS["black"])
                text_rect1 = text_surf1.get_rect(center=(self.height - int(self.button_size_x / 2) - 2,
                                                         7*self.button_size_y + int(self.button_size_y / 2) + 6))
                self.surface.blit(text_surf1, text_rect1)
                # Optimal Paths Enable
                color_button = local_map.COLORS["green"] if self.optimal_enable else local_map.COLORS["red"]
                text_enable = "Enabled" if self.optimal_enable else "Disabled"
                pygame.draw.rect(self.surface, color_button, self.button_optimal_path)
                text_surf1 = self.button_font.render("Optimal path " + text_enable, True, local_map.COLORS["black"])
                text_rect1 = text_surf1.get_rect(center=(self.height - int(self.button_size_x / 2) - 2,
                                                         6 * self.button_size_y + int(self.button_size_y / 2) + 4))
                self.surface.blit(text_surf1, text_rect1)
                # Prototype Trajectories Enable
                color_button = local_map.COLORS["green"] if self.prototype_enable else local_map.COLORS["red"]
                text_enable = "Enabled" if self.prototype_enable else "Disabled"
                pygame.draw.rect(self.surface, color_button, self.button_prototype)
                text_surf1 = self.button_font.render("Prototype trajectories " + text_enable, True, local_map.COLORS["black"])
                text_rect1 = text_surf1.get_rect(center=(self.height - int(self.button_size_x / 2) - 2,
                                                         5 * self.button_size_y + int(self.button_size_y / 2) + 2))
                self.surface.blit(text_surf1, text_rect1)
                pygame.display.update()
                # -----------------------------------------------------
            elif self.info_mode == 0:
                if not self.mode_0:
                    self.mode_0 = True
                    self.refresh_time = current_time
                    bg_color = "white"
                    self.write_text("", "black", bg_color, refresh_window=True)
                    if self.text_to_draw[1][3] == "True":
                        self.write_text(" The vehicle reached its destination, this is the end of route path : ", "black", "red")
                    # ---------------- Information for the paths ---------------
                    self.write_text(" 0 : Frenet Paths (Red points)", "black", "red")
                    self.write_text(" 0 : Optimal path to follow (Green points)", "black", "green")
                    self.write_text(" * : Traffic signs visibility (Darkgreen points)", "black", "darkgreen")
                    self.write_text(" ^ : Ego vehicle possible trajectory (Magenta points)", "black", "magenta")
                    self.write_text(" < : Route path (Blue points)", "black", "blue")
                    self.write_text(" <----------() : Predicted collision point (Magenta points)", "black", "magenta")
                    self.write_text(" <-------@ : Vehicles and pedestrians position (DarkBlue points)", "black", "darkblue")
                    self.write_text("Changes because of noise [In position(meters), In speed(Km/h)]", "black", "darkblue", True)
                    self.write_text(" Arrows between vehicles : Front and rear vehicles in left, current", "black", "white")
                    self.write_text("and right lane (Front vehicles magenta, rear vehicles orange)", "black", "white", True)
                    self.write_text("Arrows on the road:  The left, current and right lane", "black", "white")
                    self.write_text("(Front of the car magenta, rear of the car orange )", "black", "white", True)
                    # Button 1
                    pygame.draw.rect(self.surface, local_map.COLORS["blue"], self.button1)
                    text_surf = self.button_font.render("> Change window <", True, local_map.COLORS["black"])
                    text_rect = text_surf.get_rect(center=(self.height - int(3*self.button_size_x / 2)-2,
                                                           self.width - int(self.button_size_y / 2)))
                    self.surface.blit(text_surf, text_rect)
                # Button 2
                color_button = local_map.COLORS["green"] if self.paths_enabled else local_map.COLORS["red"]
                pygame.draw.rect(self.surface, color_button, self.button2)
                text_enable = "Enabled" if self.paths_enabled else "Disabled"
                text_surf = self.button_font.render("> Paths view "+text_enable+" <", True, local_map.COLORS["black"])
                text_rect = text_surf.get_rect(center=(self.height - int(self.button_size_x / 2),
                                                       self.width - int(self.button_size_y / 2)))
                self.surface.blit(text_surf, text_rect)
                pygame.display.update()
                # -----------------------------------------------------
        return self.paths_enabled, self.pygame_quit, self.p_d_noise_enable, self.p_d_std, self.s_noise_enable, \
                self.s_std, self.frenet_enable, self.optimal_enable, self.prototype_enable, self.objects_enable


class CarlaAndProgramBridge:
    def __init__(self):
        self.ego_vehicle_name = "*vehicle.ford.mustang*"
        client = local_map.carla.Client('localhost', 2000)
        client.set_timeout(10.0)  # seconds
        world = client.get_world()
        actors_list = world.get_actors()
        obj_veh = actors_list.filter(self.ego_vehicle_name)[0]
        self.local_map = local_map.LocalMap(obj_veh, 70)
        self.py_game_window = PygameDraw(18, 15, 510)
        self.draw_paths_enabled = True
        self.pygame_quit = False
        self.draw_paths_subscribers = []
        self.text_subscribers = []
        self.p_d_add_noise = True
        self.p_d_noise_std = 0.5
        self.s_add_noise = True
        self.s_noise_std = 0.5
        self.frenet_enable = True
        self.optimal_enable = True
        self.prototype_enable = True
        self.objects_enable = True
        # --- ROS ---
        rospy.init_node('Perception_node', anonymous=True)
        # Services
        rospy.Service('ego_vehicle_name_srv', EgoVehicleName, self.handle_ego_vehicle_name)
        self.client_get_route_path()
        rospy.Service('ego_vehicle_geometry_srv', EgoVehicleGeometry, self.handle_ego_vehicle_geometry)
        rospy.Service('get_vehicle_by_id_srv', VehicleByID, self.handle_vehicle_by_id)
        rospy.Service('driving_paths_srv', DrivingPaths, self.handle_driving_paths)

        self.pub_ego_vehicle = rospy.Publisher('ego_vehicle_msg', Object, queue_size=1)
        self.pub_sim_time = rospy.Publisher('simulator_time_instance_msg', Float64, queue_size=1)
        self.pub_traffic_signs = rospy.Publisher('traffic_signs_msg', TrafficSigns, queue_size=1)
        self.pub_ego_trajectory = rospy.Publisher('ego_trajectory_msg', WaypointsList, queue_size=1)
        self.pub_signs_location = rospy.Publisher('signs_location_msg', WaypointsList, queue_size=1)
        self.pub_vehicles = rospy.Publisher('vehicles_list_msg', ObjectsList, queue_size=1)
        self.pub_pedestrians = rospy.Publisher('pedestrians_list_msg', ObjectsList, queue_size=1)
        self.pub_curvature = rospy.Publisher('curvature_msg', Float64List, queue_size=1)
        self.pub_lanes_information = rospy.Publisher('lanes_information_msg', RoadLanes, queue_size=1)
        self.pub_current_lane_information = rospy.Publisher('current_lane_information_msg', Lane, queue_size=1)
        self.pub_global_draw = rospy.Publisher('global_draw_msg', Bool, queue_size=1)

        # Subscribers
        self.subscriber_vehicle_controller = rospy.Subscriber("vehicle_controller_msg", VehicleCmd,
                                                         self.callback_vehicle_controller, queue_size=1)
        self.subscriber_maneuver_data = rospy.Subscriber('maneuver_data_msg', ManeuverDataROS,
                                                    self.callback_maneuver_data_draw, queue_size=1)
        self.subscriber_text_draw_maneuver = rospy.Subscriber('text_draw_maneuver_msg', StringList,
                                                         self.callback_maneuver_text_draw, queue_size=1)
        self.subscriber_route_localization = rospy.Subscriber("localization_in_route_path_msg", RouteLocation,
                                                         self.callback_route_localization, queue_size=1)
        self.text_subscribers = [self.subscriber_maneuver_data, self.subscriber_text_draw_maneuver]
        if self.draw_paths_enabled:
            self.draw_paths_subscribers = self.subscribe_to_draw_topics()

    def talker(self):
        while not rospy.is_shutdown():
            sleep(0.01)
            self.publish_ego_vehicle(self.pub_ego_vehicle)
            self.publish_simulator_time_instance(self.pub_sim_time)
            self.publish_traffic_signs(self.pub_traffic_signs)
            self.publish_ego_trajectory(self.pub_ego_trajectory)
            self.publish_signs_location(self.pub_signs_location)
            self.publish_vehicles(self.pub_vehicles)
            self.publish_pedestrians(self.pub_pedestrians)
            self.publish_curvature(self.pub_curvature)
            self.publish_lanes_information(self.pub_lanes_information, self.pub_current_lane_information)

            if not self.pygame_quit:
                # Take and give information to pygame window
                self.draw_paths_enabled, self.pygame_quit, self.p_d_add_noise, self.p_d_noise_std, \
                    self.s_add_noise, self.s_noise_std, self.frenet_enable, self.optimal_enable,  \
                        self.prototype_enable, self.objects_enable = self.py_game_window.draw_text_info()
                # If the paths view is disabled or the window closed then delete subscribers and vice versa
                if self.draw_paths_enabled and len(self.draw_paths_subscribers) == 0 and not self.pygame_quit:
                    self.draw_paths_subscribers = self.subscribe_to_draw_topics()
                    local_map.GLOBAL_DRAW = True
                    self.publish_global_draw(self.pub_global_draw)
                elif (not self.draw_paths_enabled and len(self.draw_paths_subscribers) != 0) or \
                        (self.pygame_quit and len(self.draw_paths_subscribers) != 0):
                    local_map.GLOBAL_DRAW = False
                    self.publish_global_draw(self.pub_global_draw)
                    for subscriber in self.draw_paths_subscribers:
                        subscriber.unregister()
                    self.draw_paths_subscribers = []
                    if self.pygame_quit:
                        for subscriber in self.text_subscribers:
                            subscriber.unregister()
                        self.text_subscribers = []
                        pygame.quit()

    def subscribe_to_draw_topics(self):
        # Draw information
        subscriber_candidate_paths_draw = rospy.Subscriber("candidate_paths_draw_msg", TrajectoriesList,
                                                           self.callback_candidate_paths_draw, queue_size=1)
        subscriber_optimal_local_path_draw = rospy.Subscriber("optimal_local_path_msg", LocalOptimalPath,
                                                              self.callback_optimal_local_path_draw, queue_size=1)
        subscriber_prototype_trajectories_draw = rospy.Subscriber("prototype_trajectories_draw_msg",  TrajectoriesList,
                                                                  self.callback_prototype_trajectories_draw, queue_size=1)
        subscriber_collision_points_draw = rospy.Subscriber("collision_points_draw_msg", WaypointsList,
                                                            self.callback_collision_points_draw, queue_size=1)

        return [subscriber_candidate_paths_draw, subscriber_optimal_local_path_draw,
                subscriber_prototype_trajectories_draw, subscriber_collision_points_draw]

    # Publishers
    def publish_ego_vehicle(self, pub):
        ego_v = self.local_map.get_ego_vehicle()
        ego_vehicle = self.assign_object(ego_v)
        #rospy.loginfo(ego_vehicle)
        pub.publish(ego_vehicle)

    def publish_simulator_time_instance(self, pub):
        simulator_time = Float64()
        simulator_time.data = self.local_map.get_simulator_time()
        #rospy.loginfo(simulator_time)
        pub.publish(simulator_time)

    def publish_traffic_signs(self, pub):
        traffic_signs = TrafficSigns()
        tr_sings = self.local_map.get_traffic_signs(dist=50)
        traffic_signs.traffic_light_exist = tr_sings.traffic_light_exist
        traffic_signs.traffic_light_distance = tr_sings.traffic_light_distance
        traffic_signs.traffic_light_state = tr_sings.traffic_light_state
        traffic_signs.stop_sign_exist = tr_sings.stop_sign_exist
        traffic_signs.stop_sign_distance = tr_sings.stop_sign_distance
        traffic_signs.speed_sign_exist = tr_sings.speed_sign_exist
        traffic_signs.speed_sign_distance = tr_sings.speed_sign_distance
        traffic_signs.traffic_junction_exist = tr_sings.traffic_junction_exist
        traffic_signs.traffic_junction_distance = tr_sings.traffic_junction_distance
        traffic_signs.speed_limit = tr_sings.speed_limit
        #rospy.loginfo(traffic_signs)
        pub.publish(traffic_signs)

    def publish_ego_trajectory(self, pub):
        ego_trajectory = WaypointsList()
        trajectory = self.local_map.get_ego_vehicle_possible_trajectory()
        waypoint_list = []
        for waypoint in trajectory:
            w = Waypoint()
            w.x = waypoint.x
            w.y = waypoint.y
            w.yaw = waypoint.yaw
            waypoint_list.append(w)
        ego_trajectory.waypoints_list = waypoint_list
        #rospy.loginfo(ego_trajectory)
        pub.publish(ego_trajectory)

    def publish_signs_location(self, pub):
        signs_location = WaypointsList()
        signs_locations = self.local_map.get_traffic_stop_and_lights_objects_location()
        waypoint_list = []
        for waypoint in signs_locations:
            w = Waypoint()
            w.x = waypoint.x
            w.y = waypoint.y
            w.yaw = waypoint.yaw
            waypoint_list.append(w)
        signs_location.waypoints_list = waypoint_list
        #rospy.loginfo(signs_location)
        pub.publish(signs_location)

    def publish_vehicles(self, pub):
        self.local_map.add_noise_in_objects_position_direction_and_speed(p_d_add_noise=self.p_d_add_noise,
                                p_d_std=(self.p_d_noise_std/1.4142), s_add_noise=self.s_add_noise, s_std=(self.s_noise_std/(1.4142*3.6)))  # 1.4142 = sqrt(2)
        list_veh, values_change = self.local_map.get_dynamic_objects(obj_type_list="vehicle", get_diff=True)
        vehicles_list = []
        for veh in list_veh:
            vehicles_list.append(self.assign_object(veh))
        vehicles = ObjectsList()
        vehicles.objects_list = vehicles_list
        #rospy.loginfo(vehicles)
        pub.publish(vehicles)
        if self.draw_paths_enabled:
            self.draw_objects_position(vehicles_list, values_change)

    def publish_pedestrians(self, pub):
        self.local_map.add_noise_in_objects_position_direction_and_speed(p_d_add_noise=self.p_d_add_noise,
                                 p_d_std=(self.p_d_noise_std/1.4142), s_add_noise=self.s_add_noise, s_std=math.sqrt(self.s_noise_std/(1.4142*3.6)))
        list_pedestrians, values_change = self.local_map.get_dynamic_objects(obj_type_list="pedestrian",  get_diff=True)
        pedestrians_list = []
        for ped in list_pedestrians:
            pedestrians_list.append(self.assign_object(ped))
        pedestrians = ObjectsList()
        pedestrians.objects_list = pedestrians_list
        #rospy.loginfo(pedestrians)
        pub.publish(pedestrians)
        if self.draw_paths_enabled:
            self.draw_objects_position(pedestrians_list, values_change)

    def publish_curvature(self, pub):
        curvature = Float64List()
        curvature.float64_list = self.local_map.get_route_curve(dist=50, regions_num=4)
        #rospy.loginfo(curvature)
        pub.publish(curvature)

    def publish_lanes_information(self, pub, pub2):
        lane_information = RoadLanes()
        left_lane, current_lane, right_lane = self.local_map.get_lanes_information()
        lanes = [left_lane, current_lane, right_lane]
        lanes_list = []
        for lane_i in lanes:
            lane = Lane()
            lane.availability = lane_i.availability
            lane.lane_type = lane_i.lane_type
            lane.front_front_vehicle = self.assign_object(lane_i.front_front_vehicle)
            lane.front_vehicle = self.assign_object(lane_i.front_vehicle)
            lane.rear_vehicle = self.assign_object(lane_i.rear_vehicle)
            lane.rear_rear_vehicle = self.assign_object(lane_i.rear_rear_vehicle)
            lane.vehicle_num_front = lane_i.vehicle_num_front
            lane.vehicle_num_rear = lane_i.vehicle_num_rear
            lane.lateral_offset_front_front = lane_i.lateral_offset_front_front
            lane.lateral_offset_front = lane_i.lateral_offset_front
            lane.lateral_offset_rear = lane_i.lateral_offset_rear
            lane.lateral_offset_front_rear = lane_i.lateral_offset_front_rear
            lane.lane_width = lane_i.lane_width
            lane.opposite_direction = lane_i.opposite_direction
            lanes_list.append(lane)
        lane_information.left_lane = lanes_list[0]
        lane_information.current_lane = lanes_list[1]
        lane_information.right_lane = lanes_list[2]
        #rospy.loginfo(lane_information)
        pub.publish(lane_information)
        pub2.publish(lane_information.current_lane)

    def publish_global_draw(self, pub):
        global_draw = Bool()
        global_draw.data = self.draw_paths_enabled
        # rospy.loginfo(global_draw)
        pub.publish(global_draw)

    @staticmethod
    def assign_object(object_input):
        object_output = Object()
        if object_input is None:
            object_output.object_type = "None"
        else:
            object_output.object_type = object_input.object_type
            object_output.object_id = object_input.object_id
            object_output.acceleration = object_input.acceleration
            object_output.x = object_input.x
            object_output.y = object_input.y
            object_output.yaw = object_input.yaw
            object_output.vel_x = object_input.vel_x
            object_output.vel_y = object_input.vel_y
            object_output.speed = object_input.speed
            object_output.dimensions.width = object_input.dimensions["width"]
            object_output.dimensions.length = object_input.dimensions["length"]
        return object_output

    # Callbacks
    def callback_vehicle_controller(self, ros_data):
        self.local_map.apply_vehicle_control(ros_data.veh_throttle, ros_data.steer_angle, ros_data.veh_brake)

    # Draw paths, points and information in simulator
    def callback_candidate_paths_draw(self, ros_data):
        if self.frenet_enable:
            for trajectory in ros_data.trajectories_list:
                path_targ = [[]]
                for w in trajectory.waypoints_list:
                    path_targ[0].append(w)
                self.local_map.draw_paths(paths=path_targ, life_time=DRAW_TIME, color=[250, 0, 0], same_color=True)
            # rospy.loginfo(ros_data)

    def callback_optimal_local_path_draw(self, ros_data):
        if self.optimal_enable:
            path_targ = [[]]
            for ii in range(len(ros_data.x)):
                w = Waypoint()
                w.x = ros_data.x[ii]
                w.y = ros_data.y[ii]
                w.yaw = ros_data.yaw[ii]
                path_targ[0].append(w)
            self.local_map.draw_paths(paths=path_targ, life_time=DRAW_TIME, color=[0, 255, 0], same_color=True)
            #rospy.loginfo(ros_data)

    def callback_prototype_trajectories_draw(self, ros_data):
        if self.prototype_enable:
            for trajectory in ros_data.trajectories_list:
                path_targ = [[]]
                for w in trajectory.waypoints_list:
                    path_targ[0].append(w)
                self.local_map.draw_paths(paths=path_targ, life_time=DRAW_TIME, color=[0, 200, 0], same_color=True)
            # rospy.loginfo(ros_data)

    def draw_objects_position(self, objects_list, values_change):
        if self.objects_enable:
            path_targ = [[]]
            for i in range(len(objects_list)):
                waypoint = Waypoint()
                waypoint.x = objects_list[i].x
                waypoint.y = objects_list[i].y
                #path_targ[0].append(waypoint)
                self.local_map.draw_paths(paths=[[waypoint]], life_time=(DRAW_TIME), color=[0, 0, 255], same_color=True,
                                          symbol="<-------@ "+str(values_change[i]))

    def callback_collision_points_draw(self, ros_data):
        if self.prototype_enable:
            path_targ = [[]]
            for w in ros_data.waypoints_list:
                path_targ[0].append(w)
            self.local_map.draw_paths(paths=path_targ, life_time=DRAW_TIME, color=[255, 0, 255], same_color=True, symbol="<----------()")
            # rospy.loginfo(ros_data)

    def callback_maneuver_data_draw(self, ros_data):
        self.py_game_window.text_to_draw[0][0] = str(ros_data.maneuver_type)
        self.py_game_window.text_to_draw[0][1] = str(ros_data.target_lateral_offset)
        self.py_game_window.text_to_draw[0][2] = str(round(ros_data.target_speed*3.6, 2))
        self.py_game_window.text_to_draw[0][3] = str(ros_data.direct_control)
        self.py_game_window.text_to_draw[0][4] = str(ros_data.left_road_width)
        self.py_game_window.text_to_draw[0][5] = str(ros_data.right_road_width)
        self.py_game_window.text_to_draw[0][6] = str(ros_data.num_of_paths_samples)
        self.py_game_window.text_to_draw[0][7] = str(ros_data.from_time)
        self.py_game_window.text_to_draw[0][8] = str(ros_data.to_time)
        self.py_game_window.text_to_draw[0][9] = str(ros_data.time_sample_step)
        self.py_game_window.text_to_draw[0][10] = str(ros_data.dt)

    def callback_maneuver_text_draw(self, ros_data):
        for i in range(len(ros_data.string_list)):
            self.py_game_window.text_to_draw[1][i] = ros_data.string_list[i]

    def callback_route_localization(self, ros_data):
        self.local_map.path_index = ros_data.route_index

    # Services
    def handle_ego_vehicle_geometry(self, request):
        L, max_angle = self.local_map.get_ego_vehicle_length_and_max_steer_angle()
        geometry_info = EgoVehicleGeometryResponse()
        geometry_info.L = L
        geometry_info.max_angle = max_angle
        return geometry_info

    def handle_ego_vehicle_name(self, request):
        geometry_info = EgoVehicleNameResponse()
        geometry_info.name = self.ego_vehicle_name
        return geometry_info

    def handle_vehicle_by_id(self, request):
        vehicle = self.local_map.get_vehicle_by_id(request.vehicle_id)
        return self.assign_object(vehicle)

    def handle_driving_paths(self, request):
        veh_trajectories = []
        for i in range(len(request.x)):
            trajectories = self.local_map.find_all_driving_paths(request.x[i], request.y[i], request.path_length)
            trajectories_list = []
            for trajectory in trajectories:
                waypoint_list = []
                for waypoint in trajectory:
                    w = Waypoint()
                    w.x = waypoint.x
                    w.y = waypoint.y
                    w.yaw = waypoint.yaw
                    waypoint_list.append(w)
                ros_trajectory = WaypointsList()
                ros_trajectory.waypoints_list = waypoint_list
                trajectories_list.append(ros_trajectory)
                paths = TrajectoriesList()
                paths.trajectories_list = trajectories_list
                veh_trajectories.append(paths)
        ros_driving_paths = DrivingPathsResponse()
        ros_driving_paths.driving_paths = veh_trajectories
        return ros_driving_paths

    def client_get_route_path(self):
        rospy.wait_for_service('route_path_srv')
        try:
            route_path_srv = rospy.ServiceProxy('route_path_srv', RoutePath)
            resp1 = route_path_srv(0)
            #rospy.loginfo(resp1)
            self.local_map.set_route_path(resp1.route_path)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)


if __name__ == '__main__':
    bridge = CarlaAndProgramBridge()
    try:
        bridge.talker()
    except rospy.ROSInterruptException:
        pass



