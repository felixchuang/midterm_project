import numpy as np
import serial
import time

waitTime = 0.1

# generate the waveform table
signalLength = 208
signalTable = np.int16(
    [
        # twinkle star
        261, 261, 392, 392, 440, 440, 392,
        349, 349, 330, 330, 294, 294, 261,
        392, 392, 349, 349, 330, 330, 294,
        392, 392, 349, 349, 330, 330, 294,
        261, 261, 392, 392, 440, 440, 392,
        349, 349, 330, 330, 294, 294, 261,
        # noteLength of twinkle star
        1, 1, 1, 1, 1, 1, 2,
        1, 1, 1, 1, 1, 1, 2,
        1, 1, 1, 1, 1, 1, 2,
        1, 1, 1, 1, 1, 1, 2,
        1, 1, 1, 1, 1, 1, 2,
        1, 1, 1, 1, 1, 1, 2,
        # butterfly
        261, 261, 294, 330, 330,
        294, 261, 294, 330, 261,
        330, 330, 349, 392, 392,
        349, 330, 349, 392, 330,
        523, 494, 440, 392, 330,
        523, 494, 440, 392,
        440, 494, 523, 392, 330,
        392, 349, 294, 261,
        # noteLength of butterfly
        2, 1, 1, 2, 2,
        1, 1, 1, 1, 2,
        2, 1, 1, 2, 2,
        1, 1, 1, 1, 2,
        2, 1, 1, 2, 2,
        2, 1, 1, 2,
        2, 1, 1, 2, 2,
        2, 1, 1, 2,
        # London bridge
        392, 440, 392, 349, 330, 349, 392,
        294, 330, 349, 330, 349, 392,
        392, 449, 392, 349, 330, 349, 392,
        294, 392, 330, 261,
        # nodeLength of London bridge
        2, 1, 1, 1, 1, 1, 2,
        1, 1, 2, 1, 1, 2,
        2, 1, 1, 1, 1, 1, 2,
        2, 2, 1, 2       
    ]
)
# output formatter
formatter = lambda x: "%3d" % x

# send the waveform table to K66F
serdev = '/dev/ttyACM0'
s = serial.Serial(serdev)
print("Sending signal ...")
print("It may take about %d seconds ..." % (int(signalLength * waitTime)))
for data in signalTable:
  s.write(bytes(formatter(data), 'UTF-8'))
  time.sleep(waitTime)
s.close()
print("Signal sended")