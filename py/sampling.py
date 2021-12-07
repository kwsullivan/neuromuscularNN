import numpy as np

# filename = '../emgOutput/sims10/test/healthy-001.dat'
filename = '/Users/kevinsullivan/Desktop/sim/trunk/data/run006/patient/emg/emg1.dat'
SAMPLE_RATE = 31250

with open(filename, 'rb') as fh:
        loaded_array = np.frombuffer(fh.read(), dtype=np.uint8)

counter = 0
for curr in loaded_array:
        print(curr)
        counter += 1

print(counter)