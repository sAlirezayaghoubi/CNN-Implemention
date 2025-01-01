import numpy as np

def max_pooling_1d(input_array, pool_size, stride):
    output_size = (len(input_array) - pool_size) // stride + 1
    pooled = np.zeros(output_size)
    for i in range(0, len(input_array) - pool_size + 1, stride):
        pooled[i // stride] = np.max(input_array[i:i + pool_size])
    return pooled

def avg_pooling_1d(input_array, pool_size, stride):
    output_size = (len(input_array) - pool_size) // stride + 1
    pooled = np.zeros(output_size)
    for i in range(0, len(input_array) - pool_size + 1, stride):
        pooled[i // stride] = np.mean(input_array[i:i + pool_size])
    return pooled

def max_pooling_2d(input_array, pool_size, stride):
    h, w = input_array.shape
    output_height = (h - pool_size) // stride + 1
    output_width = (w - pool_size) // stride + 1
    pooled = np.zeros((output_height, output_width))
    for i in range(0, h - pool_size + 1, stride):
        for j in range(0, w - pool_size + 1, stride):
            pooled[i // stride, j // stride] = np.max(input_array[i:i + pool_size, j:j + pool_size])
    return pooled

def avg_pooling_2d(input_array, pool_size, stride):
    h, w = input_array.shape
    output_height = (h - pool_size) // stride + 1
    output_width = (w - pool_size) // stride + 1
    pooled = np.zeros((output_height, output_width))
    for i in range(0, h - pool_size + 1, stride):
        for j in range(0, w - pool_size + 1, stride):
            pooled[i // stride, j // stride] = np.mean(input_array[i:i + pool_size, j:j + pool_size])
    return pooled

def max_pooling_3d(input_array, pool_size, stride):
    d, h, w = input_array.shape
    output_depth = (d - pool_size) // stride + 1
    output_height = (h - pool_size) // stride + 1
    output_width = (w - pool_size) // stride + 1
    pooled = np.zeros((output_depth, output_height, output_width))
    for z in range(0, d - pool_size + 1, stride):
        for i in range(0, h - pool_size + 1, stride):
            for j in range(0, w - pool_size + 1, stride):
                pooled[z // stride, i // stride, j // stride] = np.max(input_array[z:z + pool_size, i:i + pool_size, j:j + pool_size])
    return pooled

def avg_pooling_3d(input_array, pool_size, stride):
    d, h, w = input_array.shape
    output_depth = (d - pool_size) // stride + 1
    output_height = (h - pool_size) // stride + 1
    output_width = (w - pool_size) // stride + 1
    pooled = np.zeros((output_depth, output_height, output_width))
    for z in range(0, d - pool_size + 1, stride):
        for i in range(0, h - pool_size + 1, stride):
            for j in range(0, w - pool_size + 1, stride):
                pooled[z // stride, i // stride, j // stride] = np.mean(input_array[z:z + pool_size, i:i + pool_size, j:j + pool_size])
    return pooled
