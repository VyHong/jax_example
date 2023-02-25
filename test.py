'''

from jax import grad, vmap
import jax.numpy as jnp
img = cv2.imread("images/banana.jpg",cv2.IMREAD_GRAYSCALE)

def extended(image, shift_pixels):
    rows, cols = img.shape[:2]
    zero_cols = np.ones((rows, np.abs(shift_pixels)), np.uint8)* 255
    if(shift_pixels>0):
        extended_img = np.hstack((image, zero_cols))
    if(shift_pixels<=0):
        extended_img = np.hstack(( zero_cols,image))
    return extended_img

image_right_extended = extended(img,50)
image_left_extended = extended(img,-50)

img1 = image_right_extended.astype(np.float32)
img2 = image_left_extended.astype(np.float32)

overlaid = (img1 + img2) / 2

overlaid = overlaid.astype(np.uint8)

# Compute gradient in x direction
grad_x = cv2.Sobel(overlaid, cv2.CV_32F, 1, 0, ksize=3)

# Compute gradient in y direction
grad_y = cv2.Sobel(overlaid, cv2.CV_32F, 0, 1, ksize=3)

# Compute gradient magnitude
grad = np.sqrt(np.square(grad_x) + np.square(grad_y))
print(np.sum(grad))
cv2.imshow("right",image_right_extended)
cv2.imshow("left",image_left_extended)
cv2.imshow("overlaid",overlaid)
cv2.imshow("gradImage",grad)

key = cv2.waitKey(0)
if key == 27:  # Press ESC to exit
    cv2.destroyAllWindows()import numpy as np
import cv2



def shift_image_left(image, shift_pixels):
    rows, cols = image.shape[:2]
    matrix = np.zeros((rows, cols), dtype=np.uint8)
    matrix[:, -1] = 255
    diag = np.eye(cols, cols, k=-1)
    result = image
    for i in range(shift_pixels):
        result = np.dot(result,diag)
        result  = result + matrix
    return result

shifted_image = shift_image_left(img,50).astype(np.uint8)

cv2.imshow("base",img)
cv2.imshow("shift",shifted_image)

key = cv2.waitKey(0)
if key == 27:  # Press ESC to exit
    cv2.destroyAllWindows()'''
import cv2
import numpy as np
import jax

img = cv2.imread("images/banana.jpg",cv2.IMREAD_GRAYSCALE)

def sobel_gradient(image):
    image = np.array(image)
    image = cv2.UMat(image)
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return np.abs(grad_x) + np.abs(grad_y)

def loss(image):
    return np.sum(sobel_gradient(image))

grad_loss = jax.grad(loss)

print(grad_loss(img.astype(np.float32)))

