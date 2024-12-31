import numpy as np 

def convolve1d(data, kernel, stride=1, padding=0):
    # Add padding
    if padding > 0:
        data = np.pad(data, (padding, padding), mode='constant')

    # Determine output length
    kernel_size = len(kernel)
    data_length = len(data)
    output_length = (data_length - kernel_size) // stride + 1

    # Initialize output
    output = np.zeros(output_length)

    # Perform convolution
    for i in range(output_length):
        region = data[i * stride:i * stride + kernel_size]
        output[i] = np.sum(region * kernel)

    return output

# Example: 1D Convolution
time_series = np.array([1, 2, 3, 4, 5, 6, 7])  # Input data
kernel = np.array([1, 0, -1])  # Kernel to detect changes
output_1d = convolve1d(time_series, kernel, stride=1, padding=1)
print("1D Convolution Output:", output_1d)
