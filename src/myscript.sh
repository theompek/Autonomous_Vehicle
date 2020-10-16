#!/bin/bash

a=0
# define cleanup function
cleanup() {
  if [ $a -gt 0 ]
    then
      rosnode kill -a
      sleep 1
      killall -9 rosmaster
      pkill -f ros
  fi
  sleep 1
  pkill -f manual_control_spectator.py
  if [ -v pid_1 ]
  then
    sleep 2
    kill -9 "$pid_1"
  fi

  sleep 1
 pkill -f Carla
  exit 0

}

trap cleanup EXIT INT TERM ERR

~/CARLA_0.9.7/./CarlaUE4.sh &
sleep 3
cd ~/CARLA_0.9.7/PythonAPI/util/ || exit
if [ $# -gt 0 ]
  then
    arg="$1"
    town="Town0${arg}"
    ./config.py --m "$town"
fi
sleep 1
cd ~/CARLA_0.9.7/PythonAPI/examples/ || exit
./manual_control_spectator.py --spec_h 30 --filter vehicle.ford.mustang --res 800x600  & pid_1="$!"

if [ $# -gt 1 ]
  then
   ./spawn_npc.py -n "$2" -w "$2" &
fi
cd ~/catkinevaluation/

source devel/setup.bash || exit
sleep 2
PATHLENGTH="2000"
if [ $# -gt 2 ]
  then
   export PATHLENGTH="$3"
fi
echo $PATHLENGTH
#echo "Press any key to start the autonomous vehicle... "
#echo "After in order to stop the program press any key again... "
#read -p "---> " -n1 -s
cd ~/catkinevaluation/
roslaunch maneuver_generator maneuver_run.launch
#a=1
#read -p "---> " -n1 -s
#echo "Τhe application is terminated, please wait"




