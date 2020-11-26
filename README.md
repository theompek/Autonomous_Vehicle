# Thesis Title

Optimal route planning and on-road autonomous vehicle navigation with
and without dynamic obstacles.

# Abstract

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The technology of autonomous driving has been extensively studied in recent years
and especially in the last decade. Both the scientific community and industry are making
significant efforts to develop the necessary sophisticated technology and ultimately to
achieve autonomous driving. Vehicles and transportation play a key role in the development
of trade and consequently in the development of societies as are known. Autonomous
driving will dramatically increase citizens’ safety in the coming years, reduce transportation
time and traffic congestion.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The technology of autonomous driving is quite complex and its development is a
challenge for the scientific community. The operation of autonomous vehicles requires
a very good understanding of their environment, immediate response to changes in it
and therefore, a reliable assessment of their position in space. The vehicle must have the
appropriate technology to enable it to navigate the road network and execute complex
driving scenarios. For this purpose, the vehicles are equipped with appropriate stateof-the-art sensors and also with the appropriate control and decision-making system.
Implementing such systems is a quite complicated process, since they consist of individual
systems that are specialized in solving specific problems of autonomous driving.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This dissertation focuses on the development of an autonomous driving system. The
purpose of the system is to navigate the vehicle optimally and safely from a starting
point to a destination point, within a city where there are vehicles and pedestrians, with
respect for traffic rules. The system was developed in the form of an ego-only system
and also the form of the modular system was chosen. The software was built using the
Python programming language and the ROS middleware. The Carla simulator was used
for the system development, with which the tests, the experiments and the simulation
of autonomous driving were performed. The system was developed in a modular form
and consequently consists of the following individual systems a) perception, b) behavior
selection, d) behavior prediction, e) construction of a basic path (or route), f) construction
of local paths, h) vehicle control. Each of these systems is responsible for performing
specific processes that are necessary for successful autonomous driving.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The behavior selection system uses multi-criteria decision analysis to evaluate and
select the appropriate behavior for the vehicle based on the environment in which it
moves. The prediction system, using the ”prototype trajectories” method and a Hidden
Markov Model (HMM), is used to predict the behavior of vehicles and pedestrians around
the autonomous vehicle. The perception system using the programming interface of the
Carla simulator was used to obtain the necessary data related to the environment of the
autonomous vehicle. The vehicle during its navigation must be able to follow alternative
paths at any time, for this purpose the system of local path planning was constructed.
Therefore, using the Frenet path construction methodology, the local alternative paths
generation system was constructed. Finally, in order to the vehicle be controlled at a
low-level, the control system was built. Using a suitable controller built for speed control
and using the Pure Pursuit method for direction control, the overall navigation of the
vehicle was achieved.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ultimately, the system was tested using the Carla simulator for various traffic environments
and various motion scenarios. The vehicle’s performance was assessed using criteria that
describe both the autonomy of the autonomous vehicle and the effectiveness of the
autonomous vehicle in terms of its ability to complete the route path. The system was
finally tested by adding noise to the perception system, where white noise was added
to the position and speed data of vehicles and pedestrians received by the perception
system.


# Autonomous_Vehicle
In this thesis a system for autonomous driving of a vehicle was developed . The system developed is in the form of an ago-only and modular system.

<p align="center">
<img src="/images/image144.png" alt="drawing" width="700"/>
</p>
<p align="center">
<img src="/images/image73.png" alt="drawing" width="700"/>
</p>
<p align="center">α) Ego-only systems, β) Connected systems</p>
<p align="center">
<img src="/images/image129.png" alt="drawing" width="700"/>
</p>
<p align="center">α) Modular systems, β) End-to-end systems</p>


The software is written in Python program language and also the Carla simulator and the ROS middleware were used for the implementation. The system is oriented more toward the behavior planning and the path planning processes of a system of autonomous driving. There are also modules for behavior prediction of vehicles and pedestrians and for autonomous vehicle control in terms of its speed and direction (low level control). The perception system uses algorithmic processes for extracting the information needed of the environment around the autonomous vehicle from the API of the Carla Simulator. Also a local path planning system for producing local alternatives path was created. Finally a route planning system was also developed for finding a route path that the vehicle has to follow in order to go from start point A to final position B, using the A* algorithm. For the overall evaluation of the system was also created a evaluation subsystem. So, the overall system consists of the following subsystems:

1. **Perception system.**
1. **Behaviour prediction system.**
1. **Route planning system.**
1. **Maneuver planning system.**
1. **Local path planning.**
1. **Control system.**
1. **Evaluation system.**

<p align="center">
<img src="/images/image135.png" alt="drawing" width="700"/>
</p>
<p align="center"> System connectivity and data types </p>

# Presentation
## Prediction of pedestrian motion and waiting for the pedestrian to cross the road.
<p align="center">
<img src="/images/vehicle_follow_wait_red_light.gif" alt="drawing" width="700"/>
</p>

[![IMAGE ALT TEXT HERE](https://www.youtube.com/watch?v=Nu7SIumINpE)](https://www.youtube.com/watch?v=Nu7SIumINpE)
