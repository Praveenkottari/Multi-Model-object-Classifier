import time
import lsm9ds1
from machine import Pin, I2C
import math

# I2C bus
bus = I2C(1, scl=Pin(15), sda=Pin(14))
lsm = lsm9ds1.LSM9DS1(bus)

# Time delay between sequences
delay = 500

# Function to calculate minimum value
def min(data):
    return sorted(data)[0]

# Function to calculate maximum value
def max(data):
    return sorted(data)[-1]

# Function to calculate mean
def mean(Buffer):
    return sum(Buffer) / len(Buffer)

# Function to calculate median
def median(data):
    n = len(data)
    sorted_data = sorted(data)
    mid1 = sorted_data[n // 2 - 1]
    mid2 = sorted_data[n // 2]
    return (mid1 + mid2) / 2

# Function to calculate variance
def variance(data):
    n = len(data)
    mean_val = sum(data) / n
    return sum((x - mean_val) ** 2 for x in data) / n

# Function to calculate standard deviation
def standard_deviation(data):
    return variance(data) ** 0.5

# Function to calculate signal range
def signal_range(data):
    return max(data) - min(data)

# Function to calculate signal energy
def signal_energy(data):
    energy = sum(x**2 for x in data)
    return energy

# Function to calculate skewness
def skewness(data):
    n = len(data)
    mean_val = mean(data)
    stddev = standard_deviation(data)
    if stddev == 0:
        return 0
    skew = sum((x - mean_val) ** 3 for x in data) / (n * (stddev ** 3))
    return skew

# Function to calculate kurtosis
def kurtosis(data):
    n = len(data)
    mean_val = mean(data)
    stddev = standard_deviation(data)
    if stddev == 0:
        return 0
    kurt = sum((x - mean_val) ** 4 for x in data) / (n * (stddev ** 4))
    return kurt

# Function to calculate root mean square
def rms(data):
    rms_val = math.sqrt(sum(x**2 for x in data) / len(data))
    return rms_val

# Function to calculate interquartile range
def iqr(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    q1 = sorted_data[int(0.25 * n)]
    q3 = sorted_data[int(0.75 * n)]
    return q3 - q1

seq_num = 0
while True:
    num_iterations = 100
    Ax_Buffer = []
    Ay_Buffer = []
    Az_Buffer = []
    Mx_Buffer = []
    My_Buffer = []
    Mz_Buffer = []
    Gx_Buffer = []
    Gy_Buffer = []
    Gz_Buffer = []
    iteration = 0

    while iteration < num_iterations:
        # Accelerometer data
        Ax, Ay, Az = lsm.accel()
        Mx, My, Mz = lsm.magnet()
        Gx, Gy, Gz = lsm.gyro()

        Ax_Buffer.append(Ax)
        Ay_Buffer.append(Ay)
        Az_Buffer.append(Az)
        Mx_Buffer.append(Mx)
        My_Buffer.append(My)
        Mz_Buffer.append(Mz)
        Gx_Buffer.append(Gx)
        Gy_Buffer.append(Gy)
        Gz_Buffer.append(Gz)

        iteration += 1

    # Accelerometer statistics
    print(seq_num, end=",")  # Print sequence number
    for data in [Ax_Buffer, Ay_Buffer, Az_Buffer]:
        print(min(data), max(data), mean(data), median(data), standard_deviation(data), signal_range(data), signal_energy(data), skewness(data), kurtosis(data), rms(data), iqr(data), sep=",", end=",")  # Print statistics for Ax, Ay, Az

    # Gyroscope statistics
    for data in [Gx_Buffer, Gy_Buffer, Gz_Buffer]:
        print(min(data), max(data), mean(data), median(data), standard_deviation(data), signal_range(data), signal_energy(data), skewness(data), kurtosis(data), rms(data), iqr(data), sep=",", end=",")  # Print statistics for Gx, Gy, Gz

    # Magnetometer statistics
    for data in [Mx_Buffer, My_Buffer, Mz_Buffer]:
        print(min(data), max(data), mean(data), median(data), standard_deviation(data), signal_range(data), signal_energy(data), skewness(data), kurtosis(data), rms(data), iqr(data), sep=",", end=",")  # Print statistics for Mx, My, Mz

    print("")  # Newline for next sequence

    seq_num += 1
    time.sleep_ms(delay)
