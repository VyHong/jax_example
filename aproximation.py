import cv2
import numpy as np
from jax import grad, vmap
import jax.numpy as jnp
from jax.scipy.signal import convolve2d
import matplotlib.pyplot as plt

img = cv2.imread("images/banana.jpg", cv2.IMREAD_GRAYSCALE)
img_rows, img_cols = img.shape[:2]
zero_cols = np.ones((img_rows, np.abs(50)), np.uint8) * 255
img = np.hstack((img, zero_cols))


def shift_image(image, shift_pixels):
    rows, cols = image.shape[:2]
    matrix = np.zeros((rows, cols), dtype=np.uint8)
    if shift_pixels < 0:
        matrix[:, -1] = 255
        diag = np.eye(cols, cols, k=-1)

    if shift_pixels >= 0:
        matrix[:, 0] = 255
        diag = np.eye(cols, cols, k=1)

    shift_pixels = jnp.abs(shift_pixels)
    result = image
    for i in range(int(shift_pixels)):
        result = np.dot(result, diag)
        result = result + matrix
    return result


def gradient_sum_overlay(base_image, right_image, shift):
    shifted_image = shift_image(right_image, shift)
    overlaid = (base_image + shifted_image) / 2.0

    sobel_x = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = jnp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    convolved_x = convolve2d(overlaid, sobel_x, mode='same')
    convolved_y = convolve2d(overlaid, sobel_y, mode='same')

    # Compute gradient magnitude
    grad = np.sqrt(jnp.square(convolved_x) + jnp.square(convolved_y))
    cv2.imshow("overlay_gradient", grad.astype(np.uint8))
    grad = np.sum(grad)
    return grad


image_right_shifted = shift_image(img, 50)
right_image = image_right_shifted.astype(np.float32)

base_image = img.astype(np.float32)

cv2.imshow("base", base_image.astype(np.uint8))
cv2.imshow("right", right_image.astype(np.uint8))


# grad_shift = grad(gradient_sum_overlay, 2, allow_int=True)
# gradient_x = grad_shift(base_image, right_image, 5.0)


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
# print(gradient_x)

x_data, y_data = generate_data_points(base_image, right_image,25, 5)
print(x_data, y_data)

coeffs = np.polyfit(x_data,y_data,25)
f = np.poly1d(coeffs)
x = np.linspace(-100, 100, 100)

# Evaluate the function at each point in the range
y = f(x)
plt.plot(x,y)
plt.show()
key = cv2.waitKey(0)

cv2.destroyAllWindows()
