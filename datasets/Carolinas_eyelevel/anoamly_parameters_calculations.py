import pandas as pd
import os
import numpy as np

folder_path = '/home/vkatariy/Datasets/Trajectory_Datasets/Carolinas_eyelevel/test'
time_per_frame = 1/5  # Assuming 5 frames per second

def compute_metrics(vehicle_data):
    velocities = []
    longitudinal_accs = []
    lateral_accs = []

    for i in range(len(vehicle_data) - 1):
        dx = vehicle_data['x-pixel cordinate'].iloc[i+1] - vehicle_data['x-pixel cordinate'].iloc[i]
        dy = vehicle_data['y-pixel cordinate'].iloc[i+1] - vehicle_data['y-pixel cordinate'].iloc[i]
        
        v = ((dx**2 + dy**2) ** 0.5) / time_per_frame
        velocities.append(v)

    for i in range(len(velocities) - 1):
        a_longitudinal = (velocities[i+1] - velocities[i]) / time_per_frame
        a_lateral = (dy / time_per_frame**2)
        
        longitudinal_accs.append(a_longitudinal)
        lateral_accs.append(a_lateral)

    da_longitudinal = [(longitudinal_accs[i+1] - longitudinal_accs[i]) / time_per_frame for i in range(len(longitudinal_accs) - 1)]
    da_lateral = [(lateral_accs[i+1] - lateral_accs[i]) / time_per_frame for i in range(len(lateral_accs) - 1)]

    return velocities, longitudinal_accs, lateral_accs, da_longitudinal, da_lateral

all_velocities = []
all_longitudinal_accs = []
all_lateral_accs = []
all_da_longitudinal = []
all_da_lateral = []

for file in os.listdir(folder_path):
    if file.endswith('.txt'):
        # Assign column names manually
        data = pd.read_csv(os.path.join(folder_path, file), delimiter='\t', header=None, names=['Frames', 'vehicle ID', 'x-pixel cordinate', 'y-pixel cordinate'])
        vehicles = data['vehicle ID'].unique()

        file_velocities = []
        file_longitudinal_accs = []
        file_lateral_accs = []
        file_da_longitudinal = []
        file_da_lateral = []

        for vehicle in vehicles:
            vehicle_data = data[data['vehicle ID'] == vehicle].sort_values(by='Frames')
            velocities, longitudinal_accs, lateral_accs, da_longitudinal, da_lateral = compute_metrics(vehicle_data)

            file_velocities.extend(velocities)
            file_longitudinal_accs.extend(longitudinal_accs)
            file_lateral_accs.extend(lateral_accs)
            file_da_longitudinal.extend(da_longitudinal)
            file_da_lateral.extend(da_lateral)

        # Print mean and standard deviation for each parameter for the current file
        print(f"File: {file}")
        print(f"Mean Scalar Velocity: {np.mean(file_velocities):.2f}, Standard Deviation: {np.std(file_velocities):.2f}")
        print(f"Mean Longitudinal Acceleration: {np.mean(file_longitudinal_accs):.2f}, Standard Deviation: {np.std(file_longitudinal_accs):.2f}")
        print(f"Mean Lateral Acceleration: {np.mean(file_lateral_accs):.2f}, Standard Deviation: {np.std(file_lateral_accs):.2f}")
        print(f"Mean Derivative of Longitudinal Acceleration: {np.mean(file_da_longitudinal):.2f}, Standard Deviation: {np.std(file_da_longitudinal):.2f}")
        print(f"Mean Derivative of Lateral Acceleration: {np.mean(file_da_lateral):.2f}, Standard Deviation: {np.std(file_da_lateral):.2f}")
        print("------------------------------------------------")

        # Append to overall lists
        all_velocities.extend(file_velocities)
        all_longitudinal_accs.extend(file_longitudinal_accs)
        all_lateral_accs.extend(file_lateral_accs)
        all_da_longitudinal.extend(file_da_longitudinal)
        all_da_lateral.extend(file_da_lateral)

# Print overall mean and standard deviation for each parameter
print("Overall Metrics:")
print(f"Mean Scalar Velocity: {np.mean(all_velocities):.2f}, Standard Deviation: {np.std(all_velocities):.2f}")
print(f"Mean Longitudinal Acceleration: {np.mean(all_longitudinal_accs):.2f}, Standard Deviation: {np.std(all_longitudinal_accs):.2f}")
print(f"Mean Lateral Acceleration: {np.mean(all_lateral_accs):.2f}, Standard Deviation: {np.std(all_lateral_accs):.2f}")
print(f"Mean Derivative of Longitudinal Acceleration: {np.mean(all_da_longitudinal):.2f}, Standard Deviation: {np.std(all_da_longitudinal):.2f}")
print(f"Mean Derivative of Lateral Acceleration: {np.mean(all_da_lateral):.2f}, Standard Deviation: {np.std(all_da_lateral):.2f}")
