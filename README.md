# ROS 2 Foxy - blindbuy_oakd

### ROS 2 Foxy
```
sudo apt install ros-foxy-image-pipeline
```
#### Clone repository
Clone foxy-devel branch:
```
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone -b foxy-devel --recursive https://github.com/Madrid-Arbutus-Bears/blindbuy_oakd.git
```
#### Install depthai
The following script will install depthai-core and update usb rules and install depthai devices
```
cd ~/ros2_ws/src
sudo wget -qO- https://raw.githubusercontent.com/luxonis/depthai-ros/foxy-devel/install_dependencies.sh | sudo bash
```
```
cd  ~/ros2_ws
wget https://raw.githubusercontent.com/luxonis/depthai-ros/foxy-devel/underlay.repos
vcs import src < underlay.repos
rosdep install -y -r -q --from-paths src --ignore-src --rosdistro foxy
colcon build
```
Source:
```
source /opt/ros/foxy/setup.bash
source /usr/share/colcon_cd/function/colcon_cd.sh
export _colcon_cd_root=~/ros2_ws
source ~/ros2_ws/install/setup.bash
```
