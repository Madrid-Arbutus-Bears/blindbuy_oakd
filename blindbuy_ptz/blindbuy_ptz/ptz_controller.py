#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import serial
import serial.rs485
import time

ser = serial.Serial('com3', 9600)
ser.rs485_mode = serial.rs485.RS485Settings(rts_level_for_tx=True, rts_level_for_rx=False, loopback=False,
                                            delay_before_tx=None, delay_before_rx=None,)

print(ser.isOpen())
#PELCO D protocol commands for left right and stop action
left = bytearray.fromhex('FF 01 00 04 3F 00 44')
right = bytearray.fromhex('FF 01 00 02 20 00 23')
up = bytearray.fromhex('FF 01 00 08 00 3F 48')
down = bytearray.fromhex('FF 01 00 10 20 00 31')
stop = bytearray.fromhex('FF 01 00 00 00 00 01') 

sync = 0xFF
address=0x01
command1=0x00
command2=0x12
data1=0x3F
data2=0x3F

#Calculate checksum: https://evileg.com/en/post/207/
checksum = address + command1 + command2 + data1 + data2
checksum %= 100

#Convert hex to string: https://stackoverflow.com/questions/40123901/python-integer-to-hex-string-with-padding
sync='{:02x}'.format(sync)
address='{:02x}'.format(address)
command1='{:02x}'.format(command1)
command2='{:02x}'.format(command2)
data1='{:02x}'.format(data1)
data2='{:02x}'.format(data2)
checksum='{:02x}'.format(checksum)

#Convert hex string command to bytearray
command=sync+address+command1+command2+data1+data2+checksum
command=bytearray.fromhex(command) 

#Publish command
ser.write(command)


ser.close()
