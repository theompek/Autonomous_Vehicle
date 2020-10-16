#!/usr/bin/env python

# ==============================================================================
# --General imports ------------------------------------------------------------
# ==============================================================================
import time
import math
import os
import sys


def main(f_name):
    # Using readlines()
    file1 = open(f_name, 'r')
    Lines = file1.readlines()[1:]
    penalties = [0.6, 0.5, 0.65, 0.8, 0.7, 0.9, 0.7]
    maps = [Lines[:10], Lines[10:20], Lines[20:30], Lines[30:40], Lines[40:50], Lines]
    for map_line in maps:
        Pi = []
        Ri = []
        for line in map_line:
            violations = [float(i) for i in line.rstrip('\n').split(",")]
            Ri.append(violations[0])
            pi = float(violations[-1])
            for i, violation in enumerate(violations[1:-1]):
                pi = pi*(penalties[i]**int(violation))
            Pi.append(pi)

        N = float(len(Ri))
        driving_score = 0.0
        for pi, ri in zip(Pi, Ri):
            driving_score += float(pi)*float(ri)
        driving_score = driving_score/N
        route_completion = sum(Ri)/N
        infraction_penalty = sum(Pi)/N
        print(driving_score)
        print(route_completion)
        print(infraction_penalty)
        print(maps.index(map_line)+1, "----------map------------")


def errors_num(f_name):
    # Using readlines()
    file1 = open(f_name, 'r')
    Lines = file1.readlines()[1:]
    penalties = [0.6, 0.5, 0.65, 0.8, 0.7, 0.9, 0.7]
    maps = [Lines[:10], Lines[10:20], Lines[20:30], Lines[30:40], Lines[40:50], Lines]
    for map_line in maps:
        violations_num = [float(i) for i in map_line[0].rstrip('\n').split(",")]
        N = 1.0
        for line in map_line[1:]:
            N += 1.0
            violations = [float(i) for i in line.rstrip('\n').split(",")]
            for i in range(len(violations)):
                violations_num[i] += float(violations[i])
        violations_num = [v for v in violations_num]

        violations_num[0] = violations_num[0]/N
        violations_num[-1] = violations_num[-1]/N
        print(violations_num)
        print(maps.index(map_line)+1, "----------map------------")


if __name__ == '__main__':
    #files_name = ['data_info_save0.txt','data_info_save30.txt','data_info_save50.txt','data_info_save80.txt','data_info_save110.txt']
    oemfiles_name = ['data_info_save_noise50.txt','data_info_save_noise80.txt','data_info_save_noise110.txt']
    for file_name in files_name:
        errors_num(file_name)
        print("*************")
        print(" ")
        print(" ")
        print(" ")

    for file_name in files_name:
        main(file_name)
        print("*************")
        print(" ")
        print(" ")
        print(" ")
