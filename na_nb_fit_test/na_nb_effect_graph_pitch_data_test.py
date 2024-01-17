import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Read the CSV file containing your data
data = pd.read_csv('mse_values_test_pitch20.csv')

# Extracting the columns from the data
na = data['na']
nb = data['nb']
MSE_pitch = data['MSE_pitch']

# Create a meshgrid for na and nb
na_vals = sorted(set(na))
nb_vals = sorted(set(nb))
na_mesh, nb_mesh = np.meshgrid(na_vals, nb_vals)

# Reshape MSE_pitch to match the meshgrid shape
pitch_values = (MSE_pitch.values.reshape(len(nb_vals), len(na_vals)))  # Corrected reshape dimensions

# Create a figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting the 3D surface with corrected meshgrids
surf = ax.plot_surface(na_mesh, nb_mesh, pitch_values, cmap='Set2', edgecolor='none')

for i in range(len(nb_vals)):
    for j in range(len(na_vals)):
        ax.text(na_mesh[i, j], nb_mesh[i, j], pitch_values[i, j],
                '%.2f' % pitch_values[i, j], color='black', fontsize=8)

# Customize the plot
ax.set_xlabel('nb')
ax.set_ylabel('na')
ax.set_zlabel(r'MSE_pitch')
ax.set_title('MSE Pitch for Different na, nb in Data Testing', fontweight="bold")
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='MSE_pitch')

# Set viewing angle for better visualization
ax.view_init(elev=30, azim=135)

# Show the plot
plt.show()
