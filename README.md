# Autonomous_Vehicle
In this project was developed a system for autonomous driving of a vehicle. The system developed is in the form of an ago-only and modular system.

The software is written in Python program language and also the Carla simulator and the ROS middleware were used for the implementation. The system is oriented more toward the behavior planning and the path planning processes of a system of autonomous driving. There are also modules for behavior prediction of vehicles and pedestrians and for autonomous vehicle control in terms of its speed and direction (low level control). The perception system uses algorithmic processes for extracting the information needed of the environment around the autonomous vehicle from the API of the Carla Simulator. Finally a route planning system was also developed for finding a route path that the vehicle has to follow in order to go from start point A to final position B, using the A* algorithm. For the overall evaluation of the system was also created a evaluation subsystem. So, the overall system consists of the following subsystems,

1. **Perception system.**
1. **Behaviour prediction system.**
1. **Route planning system.**
1. **Maneuver planning system.**
1. **Control system.**
1. **Evaluation system.**
<p align="center">
<img src="/images/image144.png" alt="drawing" width="700"/>
</p>
<p align="center">
<img src="/images/image73.png" alt="drawing" width="700"/>
</p>
<p align="center">α) ego-only systems, β) Connected systems</p>
<p align="center">
<img src="/images/image129.png" alt="drawing" width="700"/>
</p>
<p align="center">α) Modular systems, β) End-to-end systems</p>
