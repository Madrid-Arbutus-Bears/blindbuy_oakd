# BlindBuy: Shopping Assistant for Visually Impaired People

Shopping is a big problem for visually impaired people, especially in large self-service supermarkets. Not only must they  know how to guide themselves and detect obstacles, but also locate products and differentiate them from each other.

The aim of the project is to create an embedded system located in a shopping cart which assists visually impaired people with shop navigation, product localization and obstacle avoidance. 

### [Youtube Video](https://youtu.be/NxBS4PAIyDQ)
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/NxBS4PAIyDQ/maxresdefault.jpg)](https://youtu.be/NxBS4PAIyDQ)

## Demos
### Demo 1: Pan and tilt tracker
This demo shows how the pan and tilt base tracks the head of the user trying to keep it centered in the frame
```
ros2 launch blindbuy_oakd demo1.launch.py
```
![image](https://github.com/Madrid-Arbutus-Bears/blindbuy_oakd/blob/main/blindbuy_oakd/data/images/demo1.gif)

### Demo 2: Head pose and hand position
This demo shows a 3D model representation of the head with its orientation and the location of the hand. The pan and tilt base can be moved to different positions using two sliders.
```
ros2 launch blindbuy_oakd demo2.launch.py
```
![image](https://github.com/Madrid-Arbutus-Bears/blindbuy_oakd/blob/main/blindbuy_oakd/data/images/demo2.gif)

### Demo 3: Aural Augmented Reality guidance
This demo shows how the Aural Augmented Reality guidance works. There is a product in the virtual environment that can be moved freely in the 3D space. The camera detects the pose of our head and the position of our hand. When we move the head we can feel in our headphones how the sound changes according to the virtual environment and when we approach our hand to the product the frequency of the sound increases.
```
ros2 launch blindbuy_oakd demo3.launch.py
```
![image](https://github.com/Madrid-Arbutus-Bears/blindbuy_oakd/blob/main/blindbuy_oakd/data/images/demo3.gif)

### Demo 4: Automatic product detection
This demo shows how the automatic product detection works recognizing the name written in the labels and placing its position in the 3D world.
```
ros2 launch blindbuy_oakd demo4.launch.py
```
![image](https://github.com/Madrid-Arbutus-Bears/blindbuy_oakd/blob/main/blindbuy_oakd/data/images/demo4.gif)

## Installation (tested on Ubuntu 20.04 - ROS 2 Foxy)

[Install ROS2 Foxy](https://docs.ros.org/en/foxy/Installation/Linux-Install-Debians.html)

Don't forget to install colcon:
```
sudo apt install python3-colcon-common-extensions
```
Add following lines in ~/.bashrc:
```
source /opt/ros/foxy/setup.bash
source /usr/share/colcon_cd/function/colcon_cd.sh
export _colcon_cd_root=~/ros2_ws
source ~/ros2_ws/install/setup.bash
```
Clone repository:
```
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone -b python-api https://github.com/DaniGarciaLopez/blindbuy_oakd.git --recursive
```
[Official installation guide](https://docs.luxonis.com/projects/api/en/latest/install/):
```
sudo wget -qO- http://docs.luxonis.com/_static/install_dependencies.sh | bash
python3 -m pip install depthai
```
Install pip requirements:
```
cd ~/ros2_ws/src/blindbuy_oakd/blindbuy_oakd/depthai-python/examples
python3 install_requirements.py
```
Install ROS packages:
```
sudo apt install ros-foxy-joint-state-publisher
sudo apt install ros-foxy-joint-state-publisher-gui
```

Build:
```
cd ~/ros2_ws/
colcon build
```

## OAK-D Pipeline Diagram
![image](https://github.com/Madrid-Arbutus-Bears/blindbuy_oakd/blob/main/blindbuy_oakd/data/images/pipeline.jpeg)
