import cv2
import numpy as np
from jax import grad, vmap
import jax.numpy as jnp
from jax.scipy.signal import convolve2d
from jax.scipy.ndimage import map_coordinates
import time
import matplotlib.pyplot as plt


# theta can be multi dimensional so different theta possible for different picture
def affine_grid_generator(height, width, theta):
    batch_size = theta.shape[0]
    x = jnp.linspace(-1, 1, width)
    y = jnp.linspace(-1, 1, height)
    x_t_flat = jnp.repeat(x, height)
    y_t_flat = jnp.repeat(y, width)
    all_ones = jnp.ones_like(x_t_flat)

    sampling_grid = jnp.vstack((x_t_flat, y_t_flat, all_ones))

    sampling_grid = jnp.tile(sampling_grid, (batch_size, 1))
    sampling_grid = jnp.reshape(sampling_grid, (batch_size, 3, height * width))

    batch_grids = jnp.matmul(theta, sampling_grid)
    y_s = jnp.reshape(batch_grids[:, 1, :], (batch_size, height, width))
    x_s = jnp.reshape(batch_grids[:, 0, :], (batch_size, width, height))
    return jnp.transpose(x_s, (0, 2, 1)), y_s


def bilinear_sampler(img, x, y):
    height = img.shape[0]
    width = img.shape[1]
    max_y = height - 1
    max_x = width - 1

    x = 0.5 * (x + 1.0) * (max_x)
    y = 0.5 * (y + 1.0) * (max_y)

    x0 = jnp.floor(x)
    x1 = x0 + 1
    y0 = jnp.floor(y)
    y1 = y0 + 1

    x0 = jnp.clip(x0, 0, max_x)
    x1 = jnp.clip(x1, 0, max_x)
    y0 = jnp.clip(y0, 0, max_y)
    y1 = jnp.clip(y1, 0, max_y)

    w1 = (x1 - x) * (y1 - y)
    w2 = (x1 - x) * (y - y0)
    w3 = (x - x0) * (y1 - y)
    w4 = (x - x0) * (y - y0)

    valA = get_pixel_values(img, x0, y0)
    valB = get_pixel_values(img, x0, y1)
    valC = get_pixel_values(img, x1, y0)
    valD = get_pixel_values(img, x1, y1)

    resultant = w1 * valA + w2 * valB + w3 * valC + w4 * valD

    return resultant


def get_pixel_values(img, x, y):
    x = jnp.floor(x).astype(jnp.int32)
    y = jnp.floor(y).astype(jnp.int32)
    return img[y, x]


def loss(target, image, theta):
    x_s, y_s = affine_grid_generator(image.shape[0], image.shape[1],
                                     theta)

    shifted = bilinear_sampler(jnp.array(image), x_s[0], y_s[0])

    return jnp.mean((target - overlay(target,shifted) ** 2))


def generate_data_points(target, image, rangeStart, rangeEnd, stepsize, theta):
    x_data = []
    y_data = []
    for i in jnp.arange(rangeStart, rangeEnd, stepsize):
        x_data.append(i)
        theta[0][0][2] = i
        y_data.append(loss(target, image, theta))

    return x_data, y_data


def overlay(base_image, right_image):
    overlaid = (base_image + right_image) / 2.0
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


def gradient_loss(base_image, shifted_image, theta):
    x_s, y_s = affine_grid_generator(shifted_image.shape[0], shifted_image.shape[1],
                                     theta)

    shifted = bilinear_sampler(jnp.array(shifted_image), x_s[0], y_s[0])

    overlaid = overlay(base_image, shifted)
    #cv2.imshow("overlaid", np.array(overlaid).astype(np.uint8))
    return gradient_sum(overlaid)


img = cv2.imread("images/banana.jpg", cv2.IMREAD_GRAYSCALE)
img_rows, img_cols = img.shape[:2]
zero_cols = np.ones((img_rows, np.abs(50)), np.uint8) * 255
img = np.hstack((img, zero_cols))
test_image = 255 - img
cv2.imshow("inverted", test_image)

theta_shift = np.array([[[1.0, 0.0, -0.50], [0.0, 1.0, -0.0]]])
x_s, y_s = affine_grid_generator(test_image.shape[0], test_image.shape[1],
                                 theta_shift)
interpolated = bilinear_sampler(jnp.array(test_image), x_s[0], y_s[0])

cv2.imshow("interpolated", np.array(interpolated).astype(np.uint8))
result_image = test_image

theta = np.array([[[1.0, 0.0, 0.60], [0.0, 1.0, -0.0]]])
print(gradient_loss(test_image,interpolated,theta))

x, y = generate_data_points(result_image, interpolated, -1, 1, 0.1, theta_shift)
plt.plot(x, y)
plt.show()

derivative = grad(loss, 2)
gradient = derivative(result_image, interpolated, theta)
print("Gradient von Theta" + str(gradient))

'''derivative =grad(gradient_loss,2)
gradient = derivative(test_image,interpolated,theta)
print(gradient)'''


key = cv2.waitKey(0)
if key == 27:  # Press ESC to exit
    cv2.destroyAllWindows()
