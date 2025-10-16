#!/bin/bash
source /environment.sh
dt-launchfile-init
rosrun my_package publisher.py
dt-launchfile-join


