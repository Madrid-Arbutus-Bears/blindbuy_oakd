#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import serial
import serial.rs485
import time

ser = serial.Serial('/dev/ttyUSB0', 9600)
ser.rs485_mode = serial.rs485.RS485Settings(rts_level_for_tx=True, rts_level_for_rx=False, loopback=False,
                                            delay_before_tx=None, delay_before_rx=None,)

print(ser.isOpen())
#PELCO D protocol commands for left right and stop action
thestring = bytearray.fromhex('FF 01 00 04 3F FF 74')
thestring2 = bytearray.fromhex('FF 01 00 02 20 3F 62')
stop = bytearray.fromhex('FF 01 00 00 00 00 01')
print(thestring)


ser.write(thestring2)
time.sleep(0.5)
ser.write(stop)
time.sleep(1)
ser.write(thestring)
time.sleep(3)
ser.write(stop)


ser.close()