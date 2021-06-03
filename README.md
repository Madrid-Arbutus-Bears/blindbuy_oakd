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
