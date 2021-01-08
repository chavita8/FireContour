import matplotlib.pyplot as plt
from neurodsp.sim import sim_oscillation, sim_bursty_oscillation
from neurodsp.utils import set_random_seed
from neurodsp.utils import create_times
from neurodsp.plts.time_series import plot_time_series

time = 1.5
sampling_rate = 100
oscillation_freq = 1
set_random_seed(0)
oscillation_sine = sim_oscillation(time, sampling_rate, oscillation_freq, cycle='sine')
timeSequenceMatrix = create_times(time, sampling_rate)
print("TIME : "+ str(len(timeSequenceMatrix)))
print(timeSequenceMatrix)
print("OSCILACION : "+ str(len(oscillation_sine)))
print(oscillation_sine)
plt.scatter(timeSequenceMatrix,oscillation_sine)
plt.show()
plot_time_series(timeSequenceMatrix, oscillation_sine)