import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Read the CSV file containing your data
data = pd.read_csv('mse_values_test_depth_rate.csv')

# Extracting the columns from the data
na = data['na']
nb = data['nb']
MSE_depthrate = data['MSE_pitch']

# Create a meshgrid for na and nb
na_vals = sorted(set(na))
nb_vals = sorted(set(nb))
na_mesh, nb_mesh = np.meshgrid(na_vals, nb_vals)

# Reshape MSE_depthrate to match the meshgrid shape
depthrate_values = MSE_depthrate.values.reshape(len(na_vals), len(nb_vals))

# Create a single figure for the subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting the surface for MSE_depthrate
surf = ax.plot_surface(na_mesh, nb_mesh, depthrate_values, cmap='Set3', edgecolor='none')

# Add annotations for MSE values at each grid point
for i in range(len(na_vals)):
    for j in range(len(nb_vals)):
        ax.text(na_mesh[i, j], nb_mesh[i, j], depthrate_values[i, j],
                '%.2f' % depthrate_values[i, j], color='black', fontsize=8,
                ha='center', va='center')

# Customize the plot
ax.set_xlabel('nb')
ax.set_ylabel('na')
ax.set_zlabel('MSE_depthrate')
ax.set_title('MSE Depth Rate for Different na, nb on Data Testing', fontweight="bold")
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='MSE_depthrate')

# Set viewing angle for better visualization
ax.view_init(elev=30, azim=135)

# Show the plot
plt.show()
