import cv2
import numpy as np
from jax import grad, vmap
import jax.numpy as jnp
from jax.scipy.signal import convolve2d
from jax.scipy.ndimage import map_coordinates
from jax.test_util import check_grads
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')
img = cv2.imread("images/banana.jpg", cv2.IMREAD_GRAYSCALE)
img_rows, img_cols = img.shape[:2]
zero_cols = np.ones((img_rows, np.abs(50)), np.uint8) * 255
img = np.hstack((img, zero_cols))
img = 255-img
cv2.imshow("inverted", img)

def gradient_sum_overlay(base_image, right_image, shift):
    shifted_image = shift_image(right_image, shift)
    overlaid = (base_image + shifted_image) / 2.0

    sobel_x = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = jnp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    convolved_x = convolve2d(overlaid, sobel_x, mode='same')
    convolved_y = convolve2d(overlaid, sobel_y, mode='same')

    # Compute gradient magnitude
    grad = jnp.sqrt(jnp.square(convolved_x) + jnp.square(convolved_y))
    cv2.imshow("overlay_gradient", np.array(grad.astype(np.uint8)))
    grad = jnp.sum(grad)
    return grad


def generate_sum_data_points(base_image, number, space):
    x_data = []
    y_data = []

    for i in range(1, number):
        x_data.append(space * i)
        y_data.append(np.sum(shift_image(base_image,space*i)))

    return x_data, y_data


def shift_image(image, shift):
    # Create a meshgrid from the image coordinates
    x, y = jnp.meshgrid(jnp.arange(image.shape[1]), jnp.arange(image.shape[0]))
    # Shift the coordinates by the given amount
    x_shifted = x + shift
    # Interpolate the image values at the shifted coordinates
    shifted_image = map_coordinates(image, [y, x_shifted], order=1,mode='nearest')
    return shifted_image


def generate_data_points(base_image, shift_image, number, space):
    x_data = []
    y_data = []
    for i in reversed(range(number)):
        x_data.append(space * -i)
        y_data.append(gradient_sum_overlay(base_image, shift_image, space * -i))

    for i in range(1,number):
        x_data.append(space * i)
        y_data.append(gradient_sum_overlay(base_image, shift_image, space * i))


    return x_data, y_data

# Shift the image by a real value and interpolate the resulting image
right_img = shift_image(img.astype(np.float32), -50.5)
cv2.imshow("interpolated", np.array(right_img).astype(np.uint8))
x, y = generate_data_points(img, right_img, 1000, 0.5)
#x, y = generate_sum_data_points(img,50,0.2)
print(x,y)
plt.plot(x, y)
plt.show()
print(np.sum(shift_image(img,50)))
print(np.sum(shift_image(img,40)))
derivative = grad(gradient_sum_overlay, 2)
gradient = derivative(img.astype(np.float32),right_img.astype(np.float32), 20.0)
#check_grads(derivative,(img.astype(np.float32),right_img.astype(np.float32),20.0),1)

print(derivative)
print(gradient)


key = cv2.waitKey(0)
cv2.destroyAllWindows()
