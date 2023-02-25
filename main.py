import jax.numpy as jnp
import numpy as np
import time
import jax

# Matrix multiplication in Jax
start = time.time()
for i in range(100):
  jax_result = jnp.dot(jnp.ones((1000, 1000)), jnp.ones((1000, 1000)))
jax_time = time.time() - start

# Matrix multiplication in Numpy
start = time.time()
for i in range(100):
  numpy_result = np.dot(np.ones((1000, 1000)), np.ones((1000, 1000)))
numpy_time = time.time() - start

# Comparison of computation times
print("Jax computation time: ", jax_time)
print("Numpy computation time: ", numpy_time)
print("Jax is {:.2f}% faster than Numpy.".format((1 - jax_time/numpy_time)*100))


def regular_function(x):
  return jnp.dot(x, x.T)


jitted_function = jax.jit(regular_function)

# Non-jitted function
start = time.time()
result = regular_function(jnp.ones((1000, 1000)))
non_jitted_time = time.time() - start

# Jitted function
start = time.time()
result = jitted_function(jnp.ones((1000, 1000)))
jitted_time = time.time() - start

# Comparison of computation times
print("Non-jitted computation time: ", non_jitted_time)
print("Jitted computation time: ", jitted_time)
print("The jitted function is {:.2f}% faster than the non-jitted function.".format((1 - jitted_time / non_jitted_time) * 100))


def add_constant(x,y):
  return jnp.dot(x,y)


# Define a vector
vector1 = jnp.array([1, 2, 3, 4, 5])
vector2 = jnp.array([[1, 2, 3, 4, 5],[1, 2, 3, 4, 5],[1, 2, 3, 4, 5],[1, 2, 3, 4, 5],[1,2,3,4,5]])


# Use vmap to apply add_constant to the vector in parallel
vmap_func = jax.vmap(add_constant)
result_vmap = vmap_func(vector1, vector2)

print("Vector: ", vector1)
print("Result: ", result_vmap )