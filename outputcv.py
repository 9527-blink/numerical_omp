import numpy as np
import matplotlib.pyplot as plt

# Load the data
filename = "/Users/don/sc_mywork/40_40.out"
data = np.loadtxt(filename)

# Plotting the data as a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(data, cmap='viridis', origin='lower')
plt.colorbar(label="Value")
plt.title("Solution Visualization from 80x90 Grid")
plt.xlabel("Y-axis")
plt.ylabel("X-axis")
plt.show()


