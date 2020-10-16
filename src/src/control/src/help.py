#!/usr/bin/env python
import os
import math
import sys


def main():
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data_info_save.txt")
    file_path2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data_distance_save.txt")
    with open(file_path2, "a+") as file:
        file.truncate()
    file = open(file_path, "r")
    Lines = file.readlines()
    count = 0
    r_x = []
    r_y = []
    v_x = []
    v_y = []
    for line in Lines:
        data = line.split(",")
        r_x.append(float(data[0]))
        r_y.append(float(data[1]))
        v_x.append(float(data[2]))
        v_y.append(float(data[3]))
    min_dist = []
    for vx_i, vy_i in zip(v_x, v_y):
        min_dist.append(min([math.hypot(rx_i-vx_i, ry_i-vy_i) for rx_i, ry_i in zip(r_x, r_y)]))
    print(sum(min_dist) / len(min_dist))

    for dist in min_dist:
        with open(file_path2, "a+") as file:
            file.write(str(dist) + "\n")





if __name__ == '__main__':
    main()



