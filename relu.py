import numpy as np
import matplotlib.pyplot as plt

# Define the ReLU activation function
def relu(x):
    return max(0, x)

# Generate a range of values for x
x = np.linspace(-1, 1, 100)  # Adjust the range as needed

# Apply the ReLU function to each value of x
y = [relu(val) for val in x]

# Create the plot
plt.plot(x, y, label='ReLU')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('ReLU Activation Function')
plt.legend()
plt.show()
## Generar todas las actuvaciones de ReLU cap 4
### figura 4.17 pag 17