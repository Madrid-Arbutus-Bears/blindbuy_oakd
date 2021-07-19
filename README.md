# ROS Noetic - blindbuy_oakd

### ROS Noetic
#### Clone repository
Clone noetic-devel branch:
```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone -b noetic-devel --recursive https://github.com/Madrid-Arbutus-Bears/blindbuy_oakd.git
```
#### Install depthai
The following script will install depthai-core and update usb rules and install depthai devices

```
cd ~/catkin_ws/src
sudo wget -qO- https://raw.githubusercontent.com/luxonis/depthai-ros/noetic-devel/install_dependencies.sh | sudo bash
```
```
cd ~/catkin_ws
wget https://raw.githubusercontent.com/luxonis/depthai-ros/noetic-devel/underlay.repos
vcs import src < underlay.repos
rosdep install --from-paths src --ignore-src -r -y
catkin_make
```
#### JSK Visualization Plugin - (https://github.com/jsk-ros-pkg/jsk_visualization)
```
sudo apt-get install ros-noetic-jsk-visualization
```
