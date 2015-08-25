# ros_caffe
================
## Purpose
Integrate Caffe model into ROS

## Requirements
* Ubuntu 

* [Install ROS](http://wiki.ros.org/ROS/Installation) with [catkin build](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment). Create a catkin workspace.

## Grab the source
`$ cd $CATKIN_WS/src`
`$ git clone https://github.com/tzutalin/ros_caffe.git`
`$ cd $CATKIN_WS/src/data/`
`$ sh getCaffeWeight.sh`

## Build & Run
`$ cd $CATKIN_WS`
`$ catkin_make`

Open a terminal, execute
`$ roscore`
Open another terminal, execute
`$ rosrun ros_caffe ros_caffe_test`

## Note

* Optional: Download the model and install Caffe by following the [Blog](http://tzutalin.blogspot.tw/2015/06/setup-caffe.html) or [Caffe offical site](http://caffe.berkeleyvision.org/installation.html)
You can change your Caffe's include and lib path in CMakeLists.txt. 

	`set(CAFFE_INCLUDEDIR caffe/include caffe/distribute/include)`
	`set(CAFFE_LINK_LIBRARAY caffe/lib)`
