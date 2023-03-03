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
    sobel_x = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = jnp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    convolved_x = convolve2d(image, sobel_x, mode='same')
    convolved_y = convolve2d(image, sobel_y, mode='same')

    # Compute gradient magnitude
    gradient_image = jnp.sqrt(jnp.square(convolved_x) + jnp.square(convolved_y))
    cv2.imshow("overlay_gradient" + str(time.time()), np.array(gradient_image.astype(np.uint8)))

    grad_sum = jnp.sum(gradient_image)
    return grad_sum


def loss(base_image, right_image, shift):
    overlaid = overlay(base_image, right_image, shift)
    return jnp.mean((base_image - overlaid) ** 2)


def generate_data_points(target, image, rangeStart, rangeEnd, stepsize):
    x_data = []
    y_data = []
    for i in jnp.arange(rangeStart, rangeEnd, stepsize):
        x_data.append(i)
        theta = i
        y_data.append(loss(target, image, theta))

    return x_data, y_data


# Shift the image by a real value and interpolate the resulting image
right_img = shift_image(img.astype(np.float32), -50.5)

cv2.imshow("interpolated", np.array(right_img).astype(np.uint8))

x, y = generate_data_points(img, right_img, -60, 60, 1)

print(x, y)
plt.plot(x, y)
plt.show()

derivative = grad(loss, 2)
gradient = derivative(img, right_img, 40.0)
# check_grads(derivative, (img.astype(np.float32), right_img.astype(np.float32), 2.0), 1)

print(gradient)

key = cv2.waitKey(0)
if key == 27:  # Press ESC to exit
    cv2.destroyAllWindows()
