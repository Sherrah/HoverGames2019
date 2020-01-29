'''
Created on Aug 19, 2019

@author: msherrah
'''
from periphery import Serial
from periphery import time

def sig_transmit(count):
    #connect to serial port ttyUSB1 of Sigfox module connected to coral micro usb port with baudrate 115200
    print("Sigfox connecting...")

    #send wakeup
    print("Sigfox waking up...")
    time.sleep(0.2)

    #switch to RCZ1 for Europe
    print("Sigfox switching to European network...")
    time.sleep(0.2)

    #transmit payload
    print("Sigfox transmitting payload...")
    print("Count = %d" % (count))
