import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import numpy as np
import cv2

img = cv2.imread("images/fish.png", cv2.IMREAD_GRAYSCALE)
img_rows, img_cols = img.shape[:2]
zero_cols = np.ones((img_rows, np.abs(50)), np.uint8) * 255
img = np.hstack((img, zero_cols))
img = 255 - img
imgfloat = img.astype(jnp.float32)

noise = np.random.normal(loc=0, scale=50,size= img.shape)
noisy_img = np.clip( imgfloat + noise,0,255)

cv2.imshow("test",noisy_img)
cv2.imshow("test",noisy_img.astype(np.uint8))

key = cv2.waitKey(0)
if key == 27:  # Press ESC to exit
    cv2.destroyAllWindows()
