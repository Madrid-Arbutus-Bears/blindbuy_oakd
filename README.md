# blindbuy_oakd

## Installation

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
    
(NOETIC) Add in ~/.bashrc     
```
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
```
(FOXY) Add in ~/.bashrc  
```
source /opt/ros/foxy/setup.bash
source /usr/share/colcon_cd/function/colcon_cd.sh
export _colcon_cd_root=~/ros2_ws
source ~/ros2_ws/install/setup.bash
```
And then:
```
sudo wget -qO- https://raw.githubusercontent.com/luxonis/depthai-ros/foxy-devel/install_dependencies.sh | bash
```
```
mkdir -p ros2_ws/src
cd ros2_ws
wget https://raw.githubusercontent.com/luxonis/depthai-ros/foxy-devel/underlay.repos
vcs import src < underlay.repos
rosdep install -y -r -q --from-paths src --ignore-src --rosdistro foxy
colcon build
```
