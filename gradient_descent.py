import cv2
import numpy as np
from jax import grad, vmap
import jax.numpy as jnp
from jax.scipy.signal import convolve2d
img = cv2.imread("images/banana.jpg",cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape[:2]
zero_cols = np.ones((rows, np.abs(50)), np.uint8)* 255
img = np.hstack((img, zero_cols))

def shift_image(image, shift_pixels):
    shift_pixels = int(shift_pixels)
    if shift_pixels > 0:
        return np.concatenate((np.ones((image.shape[0], shift_pixels), dtype=np.uint8)*255, image[:, :-shift_pixels]), axis=1)
    elif shift_pixels < 0:
        return np.concatenate((image[:, -shift_pixels:], np.ones((image.shape[0], -shift_pixels), dtype=np.uint8)*255), axis=1)
    else:
        return image

def shift_image_left(image, shift_pixels):
    rows, cols = image.shape[:2]
    matrix = np.zeros((rows, cols), dtype=np.uint8)
    matrix[:, -1] = 255
    diag = np.eye(cols, cols, k=-1)
    result = image
    for i in range(int(shift_pixels)):
        result = np.dot(result,diag)
        result  = result + matrix
    return result

def gradient_sum_overlay(base_image,right_image,shift):
    shifted_image = shift_image_left(right_image,shift)
    overlaid = (base_image + shifted_image) / 2
    cv2.imshow("overlaid", overlaid.astype(np.uint8))

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    convolved_x = np.abs(convolve2d(img, sobel_x, mode='same'))
    convolved_y = np.abs(convolve2d(img, sobel_y, mode='same'))

    # Compute gradient magnitude
    grad = np.sqrt(np.square(convolved_x) + np.square(convolved_y))
    grad = np.sum(grad)
    return grad

image_right_shifted = shift_image(img,50)
base_image = img.astype(np.float32)
right_image = image_right_shifted.astype(np.float32)

grad_shift = grad(gradient_sum_overlay,2, allow_int=True)

gradient_x = grad_shift(base_image,right_image,5.0)
print(gradient_sum_overlay(base_image,right_image,5.0))

print(gradient_x)


cv2.imshow("base",base_image.astype(np.uint8))
cv2.imshow("right",right_image.astype(np.uint8))


key = cv2.waitKey(0)
if key == 27:  # Press ESC to exit
    cv2.destroyAllWindows()