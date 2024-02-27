import serial
import csv

# COM port 
COM_PORT = 'COM9'
BAUD_RATE = 9600  

# CSV file
CSV_FILE_NAME = 'data_sequence11.csv'

ser = serial.Serial(COM_PORT, BAUD_RATE)

# Header row
HEADER = ['Sequence', 'Ax_min', 'Ay_min', 'Az_min', 'Ax_max', 'Ay_max', 'Az_max', 
          'Ax_mean', 'Ay_mean', 'Az_mean', 'Ax_median', 'Ay_median', 'Az_median', 
          'Ax_std_dev', 'Ay_std_dev', 'Az_std_dev', 'Ax_range', 'Ay_range', 'Az_range', 
          'Ax_energy', 'Ay_energy', 'Az_energy', 'Ax_skewness', 'Ay_skewness', 'Az_skewness', 
          'Ax_kurtosis', 'Ay_kurtosis', 'Az_kurtosis', 'Ax_rms', 'Ay_rms', 'Az_rms', 
          'Ax_iqr', 'Ay_iqr', 'Az_iqr', 'Gx_min', 'Gy_min', 'Gz_min', 'Gx_max', 'Gy_max', 
          'Gz_max', 'Gx_mean', 'Gy_mean', 'Gz_mean', 'Gx_median', 'Gy_median', 'Gz_median', 
          'Gx_std_dev', 'Gy_std_dev', 'Gz_std_dev', 'Gx_range', 'Gy_range', 'Gz_range', 
          'Gx_energy', 'Gy_energy', 'Gz_energy', 'Gx_skewness', 'Gy_skewness', 'Gz_skewness', 
          'Gx_kurtosis', 'Gy_kurtosis', 'Gz_kurtosis', 'Gx_rms', 'Gy_rms', 'Gz_rms', 
          'Gx_iqr', 'Gy_iqr', 'Gz_iqr', 'Mx_min', 'My_min', 'Mz_min', 'Mx_max', 'My_max', 
          'Mz_max', 'Mx_mean', 'My_mean', 'Mz_mean', 'Mx_median', 'My_median', 'Mz_median', 
          'Mx_std_dev', 'My_std_dev', 'Mz_std_dev', 'Mx_range', 'My_range', 'Mz_range', 
          'Mx_energy', 'My_energy', 'Mz_energy', 'Mx_skewness', 'My_skewness', 'Mz_skewness', 
          'Mx_kurtosis', 'My_kurtosis', 'Mz_kurtosis', 'Mx_rms', 'My_rms', 'Mz_rms', 
          'Mx_iqr', 'My_iqr', 'Mz_iqr']

with open(CSV_FILE_NAME, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write header row
    csv_writer.writerow(HEADER)
    
    # Data from serial port
    while True:
        line = ser.readline().decode().strip()
        print(line)
        
        # Split the received data by comma
        data = line.split(',')
        
        # Ensure that the received data has the correct number of fields
        if len(data) == len(HEADER):
            # Write data to CSV file
            csv_writer.writerow(data)
        else:
            print("Received data does not match expected format. Skipping...")
            continue

ser.close()
