# blindbuy_oakd

## Installation
### Prerrequesites
```
sudo apt install libopencv-dev
sudo apt install python3-vcstool
```
```
sudo apt install python3-rosdep
sudo rosdep init
rosdep update
```
### ROS Noetic
Clone this repository (noetic-devel branch)
```
cd ~/catkin_ws/src
git clone -b noetic-devel --recursive https://github.com/Madrid-Arbutus-Bears/blindbuy_oakd.git
```
Add in ~/.bashrc     
```
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
```
The following script will install depthai-core and update usb rules and install depthai devices

```
sudo wget -qO- https://raw.githubusercontent.com/luxonis/depthai-ros/noetic-devel/install_dependencies.sh | sudo bash
```
```
mkdir -p ~/catkin_ws/src
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
### ROS Foxy
Clone this repository (foxy-devel branch)
```
cd ~/ros2_ws/src
git clone -b foxy-devel --recursive https://github.com/Madrid-Arbutus-Bears/blindbuy_oakd.git
```
Add in ~/.bashrc  
```
source /opt/ros/foxy/setup.bash
source /usr/share/colcon_cd/function/colcon_cd.sh
export _colcon_cd_root=~/ros2_ws
source ~/ros2_ws/install/setup.bash
```
The following script will install depthai-core and update usb rules and install depthai devices
```
sudo wget -qO- https://raw.githubusercontent.com/luxonis/depthai-ros/foxy-devel/install_dependencies.sh | sudo bash
```
```
mkdir -p ~/ros2_ws/src
cd  ~/ros2_ws
wget https://raw.githubusercontent.com/luxonis/depthai-ros/foxy-devel/underlay.repos
vcs import src < underlay.repos
rosdep install -y -r -q --from-paths src --ignore-src --rosdistro foxy
colcon build
```
### Install drivers

[Install CH340 driver](https://learn.sparkfun.com/tutorials/how-to-install-ch340-drivers/drivers-if-you-need-them)

Python libraries:
```
pip3 install pyserial
```
Copy rules for usb devices:
```
sudo cp ~/ros2_ws/src/blindbuy_oakd/cfg/99-usb.rules /etc/udev/rules.d/
```
[Find usb devices name:](https://unix.stackexchange.com/questions/144029/command-to-determine-ports-of-a-device-like-dev-ttyusb0)
```
#!/bin/bash

for sysdevpath in $(find /sys/bus/usb/devices/usb*/ -name dev); do
    (
        syspath="${sysdevpath%/dev}"
        devname="$(udevadm info -q name -p $syspath)"
        [[ "$devname" == "bus/"* ]] && exit
        eval "$(udevadm info -q property --export -p $syspath)"
        [[ -z "$ID_SERIAL" ]] && exit
        echo "/dev/$devname - $ID_SERIAL"
    )
done
```
    sudo chmod 666 /dev/ttyUSB0
    

