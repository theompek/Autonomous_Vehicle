"""
- [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame](https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf)

- [Optimal trajectory generation for dynamic street scenarios in a Frenet Frame](https://www.youtube.com/watch?v=Cj6tAQe7UCY)

"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import math

# According the maneuver we can force the vehicle toward to a goal
LEFT_ROAD_WIDTH = -7.0  # maximum road width [m] <-- Perception / Maneuver
RIGHT_ROAD_WIDTH = 7.0  # maximum road width [m] <-- Perception / Maneuver
ROAD_SAMPLING_NUM = 11  # road width sampling length [m] <-- Compromise
# According the free space and the other vehicles,so the maneuver, of the ego vehicle's environment we can define
# how far ahead the sampling will be and the sampling density
TIME_STEP = 0.2  # time step [s] <-- Maneuver
TO_TIME = 3  # max prediction time [m] <-- Maneuver
FROM_TIME = 2.8  # min prediction time [m] <-- Maneuver
#
TARGET_SPEED = 30.0 / 3.6  # target speed [m/s] <-- Maneuver
SPEED_SAMPLING_LENGTH = 5.0 / 3.6  # target speed sampling length [m/s] <-- Compromise
N_S_SAMPLE = 1  # sampling number of target speed <-- Compromise

# cost weights
KJ = 0.1
KT = 0.1
KD = 1.0
KLAT = 1.0
KLON = 1.0

show_animation = True


class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, T):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * T ** 2, 4 * T ** 3],
                      [6 * T, 12 * T ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt


class QuinticPolynomial:

    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        # calc coefficient of quintic polynomial
        # See jupyter notebook document for derivation of this equation.
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[time ** 3, time ** 4, time ** 5],
                      [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
                      [6 * time, 12 * time ** 2, 20 * time ** 3]])
        b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
                      vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

        return xt


class FrenetPath:
    """
     The paths characterized by the the following parameters.
    """
    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []


def calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0, maneuver_data):
    frenet_paths = []
    if maneuver_data is not None:
        lateral_offset_points = maneuver_data.get_lateral_point_list()
        time_reach_points = maneuver_data.get_time_points_list()
        speed_points = maneuver_data.get_speed_points_list()
        dt = maneuver_data.dt  # <-- Sampling step,defines the points per path
        #print("lateral_offset_points", lateral_offset_points)
        #print("time_reach_points", time_reach_points)
        #print("speed_points", speed_points)
    else:
        dt = TIME_STEP  # <-- Sampling step,defines the points per path
        lateral_offset_points = [LEFT_ROAD_WIDTH + i * (RIGHT_ROAD_WIDTH - LEFT_ROAD_WIDTH) /
                                    (ROAD_SAMPLING_NUM - 1) for i in range(ROAD_SAMPLING_NUM)]
        time_reach_points = [FROM_TIME + i * TIME_STEP for i in range(int((TO_TIME - FROM_TIME) / TIME_STEP) + 1)]
        speed_points = [TARGET_SPEED - i * SPEED_SAMPLING_LENGTH for i in range(N_S_SAMPLE + 1)] + \
                       [TARGET_SPEED + i * SPEED_SAMPLING_LENGTH for i in range(1, N_S_SAMPLE + 1)]

    # Lateral offset points on which the paths ends
    for di in lateral_offset_points:
        # Time the vehicle reach the goal points
        for Ti in time_reach_points:
            fp = FrenetPath()
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            fp.t = [t for t in np.arange(0.0, Ti, dt)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]
            # The speed of the vehicle when reaches the goal points
            for tv in speed_points:
                tfp = copy.deepcopy(fp)
                lon_qp = QuarticPolynomial(s0, c_speed, 0.0, tv, 0.0, Ti)
                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]
                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk
                # square of diff from target speed
                ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2.0
                tfp.cd = KJ * Jp + KT * Ti + KD * tfp.d[-1] ** 2.0
                tfp.cv = KJ * Js + KT * Ti + KD * ds
                tfp.cf = KLAT * tfp.cd + KLON * tfp.cv
                frenet_paths.append(tfp)

    return frenet_paths


def calc_coordinates(fplist, csp):
    for fp in fplist:

        # calc global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            iyaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(iyaw + math.pi / 2.0)
            fy = iy + di * math.sin(iyaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        # calc yaw and ds
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            yaw_i = math.degrees(math.atan2(dy, dx))
            yaw_i = 360 + yaw_i if yaw_i < 0 else yaw_i
            fp.yaw.append(yaw_i)
            fp.ds.append(math.hypot(dx, dy))

        if len(fp.yaw) != 0:
            fp.yaw.append(fp.yaw[-1])
            fp.ds.append(fp.ds[-1])
            if len(fp.yaw) != len(fp.x):
                if len(fp.yaw) < len(fp.x):
                    for i in range(len(fp.x)-len(fp.yaw)):
                        fp.yaw.append(fp.yaw[-1])
        if len(fp.yaw) != len(fp.x):
            length = min(len(fp.yaw), len(fp.x), len(fp.y))
            fp.yaw = fp.yaw[0:length]
            fp.x = fp.x[0:length]
            fp.y = fp.y[0:length]

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

    return fplist


def check_paths(fplist, max_speed):
    MAX_ACCEL = 15.0  # maximum acceleration [m/ss]
    MAX_CURVATURE = 1480.0  # maximum curvature [1/m]
    okind = []
    for i, _ in enumerate(fplist):
        if any([v > max_speed for v in fplist[i].s_d]):  # Max speed check
            continue
        elif any([abs(a) > MAX_ACCEL for a in fplist[i].s_dd]):  # Max accel check
            continue
        elif any([abs(c) > MAX_CURVATURE for c in fplist[i].c]):  # Max curvature check
            continue


        okind.append(i)

    return [fplist[i] for i in okind]


def frenet_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, maneuver_data=None):
    fplist = calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0, maneuver_data)
    fplist = calc_coordinates(fplist, csp)
    if maneuver_data is not None:
        max_speed = maneuver_data.target_speed + 5
    else:
        max_speed = 30/3.6
    fplist2 = check_paths(fplist, max_speed)
    if len(fplist2) < 2:
        fplist2 = fplist

    return fplist2


def main():
    pass


if __name__ == '__main__':
    main()
