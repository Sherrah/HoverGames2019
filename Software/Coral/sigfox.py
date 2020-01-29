'''
Created on Aug 19, 2019

@author: msherrah
'''
from periphery import Serial
from periphery import time

#connect to serial port ttyUSB1 of Sigfox module connected to coral micro usb port with baudrate 115200
sigfox_serial_port = Serial("/dev/ttyACM0", 115200)

#send wakeup
sigfox_serial_port.write(b"1\n\r")
time.sleep(0.2)

#switch to RCZ1 for Europe
sigfox_serial_port.write(b"15\n\r")
time.sleep(0.2)

#transmit payload
sigfox_serial_port.write(b"4\n\r")
time.sleep(0.2)

#transmit payload
sigfox_serial_port.write(b"h3p1t182803\n\r")
