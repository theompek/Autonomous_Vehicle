#!/bin/bash




for map in {3..5}
do
vehicles_num=110
  for i in {1..10}
  do
    ./myscript.sh $map $vehicles_num
    sleep 5
  done

done

mv src/evaluation/src/data_info_save.txt src/evaluation/src/data_info_save_noise110.txt
