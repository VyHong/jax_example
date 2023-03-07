import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import numpy as np

# Define a function to compute the Sobel kernel
def sobel_kernel():
    x_kernel = jnp.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    y_kernel = jnp.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return x_kernel, y_kernel

# Define a function to compute the convolution of an image with a kernel
def convolve(image, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    output = np.zeros((output_height, output_width))
    for i in range(output_height):
        for j in range(output_width):
            patch = image[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = jnp.dot(patch.flatten(), kernel.flatten())
    return output

# Define a function to compute the Sobel edges of an image
def sobel_edges(image):
    x_kernel, y_kernel = sobel_kernel()
    dx = convolve(image, x_kernel)
    dy = convolve(image, y_kernel)
    mag = jnp.sqrt(dx**2 + dy**2)
    return mag

# Define a function to compute the gradient of the Sobel edges with respect to the image
sobel_grad = grad(sobel_edges)

# Create some example input data
key = random.PRNGKey(0)
image = random.normal(key, (28, 28))

# Compute the Sobel edges and the gradients
edges = sobel_edges(image)
print(edges)
#grads = sobel_grad(image)