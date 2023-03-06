import cv2
import numpy as np
from jax import grad, vmap, jit
import jax.numpy as jnp
from jax.scipy.signal import convolve2d
from jax.scipy.ndimage import map_coordinates
import time
from jax.test_util import check_grads
import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio

from jax.scipy.optimize import minimize


# mpl.use('TkAgg')
def generate_image_stack(img, theta, number):
    jitted_shift = jit(shift_image)
    image_list = []
    for i in range(number):
        image_list.append(jitted_shift(img.astype(jnp.float32), i * theta))
    return jnp.array(image_list)


def shift_image(image, x_shift):
    y_shift = 0
    # Create a meshgrid from the image coordinates
    x, y = jnp.meshgrid(jnp.arange(image.shape[1]), jnp.arange(image.shape[0]))
    # Shift the coordinates by the given amount
    x_shifted = x + x_shift
    y_shifted = y + y_shift

    # Interpolate the image values at the shifted coordinates
    shifted_image = map_coordinates(image, [y_shifted, x_shifted], order=1, mode='constant')
    return shifted_image


def shift_stack(stack, t, theta):
    image_stack = []

    for i, pic in enumerate(stack):
        image_stack.append(shift_image(pic, theta * (i - t)))

    return jnp.array(image_stack)


def overlay(stack, t, theta):
    overlaid = jnp.sum(shift_stack(stack, t, theta), axis=0) / float(len(image_stack))

    return overlaid


def gradient_sum(image):
    # Calculate the horizontal and vertical gradients using Sobel filters
    # Define the Sobel kernels for the x and y directions.
    sobel_x = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = jnp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Define the convolution function using the Sobel kernels.
    conv_x = lambda x: jnp.sum(sobel_x * x)
    conv_y = lambda x: jnp.sum(sobel_y * x)

    # Calculate the x and y gradients using the convolution function.
    grad_x_image = grad(lambda x: jnp.sum(jnp.abs(jnp.apply_along_axis(conv_x, 1, x))))
    grad_y_image = grad(lambda x: jnp.sum(jnp.abs(jnp.apply_along_axis(conv_y, 0, x))))

    # Apply the gradients to the image.
    grad_x_image = grad_x_image(image)
    grad_y_image = grad_y_image(image)
    magnitude = jnp.sqrt(jnp.square(grad_x_image) + jnp.square(grad_y_image))

    return magnitude


def loss(stack, theta):
    overlaid = overlay(stack, 0, theta)
    return jnp.mean((stack[0] - overlaid) ** 2)


def gradient_loss(stack, t, theta):
    overlaid = overlay(stack, t, theta)
    return gradient_sum(overlaid)


def variance_loss(stack, t, theta):
    jitted_overlay = jit(overlay)
    overlaid = jitted_overlay(stack, t, theta)
    return -jnp.var(overlaid)


def pixel_variance_loss(stack, t, theta):
    stack = shift_stack(stack, t, theta)
    num_pixels = jnp.prod(jnp.array(stack.shape[1:]))
    pixel_values = stack.reshape((10, -1)).T
    layer_values = pixel_values.reshape(num_pixels, stack.shape[0])
    return jnp.sum(vmap(jnp.var)(layer_values))


def generate_data_points(stack, t, range_start, range_end, step_size, loss_function):
    x_data = []
    y_data = []
    for i in jnp.arange(range_start, range_end, step_size):
        x_data.append(i)
        theta = i
        if loss_function == "variance":
            y_data.append(variance_loss(stack, t, theta))
        if loss_function == "gradient":
            y_data.append(gradient_loss(stack, t, theta))
        if loss_function == "squared":
            y_data.append(loss(stack, t, theta))
        if loss_function == "pixel_variance":
            y_data.append(pixel_variance_loss(stack, t, theta))

    return x_data, y_data


def variance_gradient_descent(stack, t, theta, learning_rate):
    image_list = []
    for i in range(100):
        gradient = variance_derivative(stack, t, theta)

        theta -= learning_rate * gradient

        loss_temp = variance_loss(stack, t, theta)
        image_list.append(np.array(overlay(stack, t, theta)).astype(np.uint8))
        print("Iteration {}: Loss = {:.6f} theta ={}".format(i, loss_temp, theta))

    return image_list


def create_video(images, name):
    height, width = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    name = name + ".mp4"
    video_writer = cv2.VideoWriter(name, fourcc, 30.0, (width, height))
    for array in images:
        video_writer.write(array)
    video_writer.release()
    print("video created")


def create_gif(images, name):
    images = [imageio.core.util.Array(image) for image in images]
    name = name + ".gif"
    with imageio.get_writer(name, mode="I", duration=0.1) as writer:
        for image in images:
            writer.append_data(image)
    print("gif created")


img = cv2.imread("images/banana.jpg", cv2.IMREAD_GRAYSCALE)
img_rows, img_cols = img.shape[:2]
zero_cols = np.ones((img_rows, np.abs(50)), np.uint8) * 255
img = np.hstack((img, zero_cols))
img = 255 - img
cv2.imshow("inverted", img)

image_stack = generate_image_stack(img, -15.0, 10)
create_gif(np.array(image_stack).astype(np.uint8), "stack")

# Shift the image by a real value and interpolate the resulting image
cv2.imshow("interpolated", np.array(image_stack[9]).astype(np.uint8))
cv2.imshow("re-interpolated", np.array(shift_image(image_stack[9], 10 * 15)).astype(np.uint8))

x, y = generate_data_points(image_stack, 0.0, -100, 100, 0.1, "pixel_variance")
plt.plot(x, y)
plt.show()
print("generated graph")

variance_derivative = grad(variance_loss, 2)
variance_gradient = variance_derivative(image_stack, 0, 10.0)
print(variance_gradient)

gif_images = variance_gradient_descent(image_stack, 0.0, 0.0, 0.02)

create_gif(gif_images, "output")
'''
derivative = grad(loss, 2)
gradient = derivative(img, right_img, 40.0)
print(gradient)

gradient_derivative = grad(gradient_loss,2)
magnitude_gradient = gradient_derivative(img,right_img,40.0)
print(magnitude_gradient)

result = minimize(variance_loss, jnp.array([10.0]), args=(img, right_img), method="BFGS")

'''

key = cv2.waitKey(0)
if key == 27:  # Press ESC to exit
    cv2.destroyAllWindows()
