import cv2
import numpy as np
from jax import grad, vmap
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
img = cv2.imread("images/banana.jpg", cv2.IMREAD_GRAYSCALE)
img_rows, img_cols = img.shape[:2]
zero_cols = np.ones((img_rows, np.abs(50)), np.uint8) * 255
img = np.hstack((img, zero_cols))
img = 255 - img
cv2.imshow("inverted", img)


def shift_image(image, x_shift):
    # Create a meshgrid from the image coordinates
    x, y = jnp.meshgrid(jnp.arange(image.shape[1]), jnp.arange(image.shape[0]))
    # Shift the coordinates by the given amount
    x_shifted = x + x_shift

    # Interpolate the image values at the shifted coordinates
    shifted_image = map_coordinates(image, [y, x_shifted], order=1, mode='nearest')
    return shifted_image


def overlay(base_image, right_image, shift):
    shifted_image = shift_image(right_image, shift)
    overlaid = (base_image + shifted_image) / 2.0

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


def loss(base_image, shifted_image, shift):
    overlaid = overlay(base_image, shifted_image, shift)
    return jnp.mean((base_image - overlaid) ** 2)


def gradient_loss(base_image, shifted_image, shift):
    overlaid = overlay(base_image, shifted_image, shift)
    return gradient_sum(overlaid)


def variance_loss(base_image, shifted_image, shift):
    overlaid = overlay(base_image, shifted_image, shift)
    return -jnp.var(overlaid)


def generate_data_points(target, image, range_start, range_end, step_size, loss_function):
    x_data = []
    y_data = []
    for i in jnp.arange(range_start, range_end, step_size):
        x_data.append(i)
        theta = i
        if loss_function == "variance":
            y_data.append(variance_loss(target, image, theta))
        if loss_function == "gradient":
            y_data.append(gradient_loss(target, image, theta))
        if loss_function == "squared":
            y_data.append(loss(target, image, theta))

    return x_data, y_data


def variance_gradient_descent(img, shifted_img, theta, learning_rate):
    image_list = []
    for i in range(100):
        gradient = variance_derivative(img, shifted_img, theta)

        theta -= learning_rate * gradient

        loss_temp = variance_loss(img, shifted_img, theta)
        image_list.append(np.array(overlay(img, shifted_img, theta)).astype(np.uint8))
        print("Iteration {}: Loss = {:.6f} theta ={}".format(i, loss_temp, theta))

    return image_list


def create_video(images):
    height, width = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter("output.mp4", fourcc, 30.0, (width, height))
    for array in images:
        video_writer.write(array)
    video_writer.release()
    print("video created")


def create_gif(images):
    images = [imageio.core.util.Array(image) for image in images]

    with imageio.get_writer("output.gif", mode="I", duration=0.1) as writer:
        for image in images:
            writer.append_data(image)
    print("gif created")
# Shift the image by a real value and interpolate the resulting image
right_img = shift_image(img.astype(np.float32), -50.5)
cv2.imshow("interpolated", np.array(right_img).astype(np.uint8))

x, y = generate_data_points(img, right_img, -100, 100, 0.1, "variance")

plt.plot(x, y)
plt.show()

variance_derivative = grad(variance_loss, 2)
variance_gradient = variance_derivative(img, right_img, 10.0)
print(variance_gradient)

images = variance_gradient_descent(img, right_img, 0.0, 0.2)

create_video(images)
create_gif(images)
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
