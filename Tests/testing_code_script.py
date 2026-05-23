"""
from numba import njit
import numpy as np
import math
import time

s = 5
a = a = np.random.uniform(low=0,high=4.0,size=s)
b = np.random.uniform(low=0,high=4.0,size=s)

@njit
def dot_nb(a, b):
    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

@njit
def squared_distance_inplace(a, b, length):
    acc = 0.0
    for i in range(length):
        diff = a[i] - b[i]
        acc += diff * diff
    return acc

t = time.time()
d = np.linalg.norm(a-b)
e = time.time() - t
print("Distance: ", d, "Time taken: ", e)

t = time.time()
d = np.sum((a - b)**2)
e = time.time() - t
print("Distance: ", d, "Time taken: ", e)

t = time.time()
d = np.dot(a - b, a - b)
e = time.time() - t
print("Distance: ", d, "Time taken: ", e)

t = time.time()
d = np.einsum('i,i', a - b, a - b)
e = time.time() - t
print("Distance: ", d, "Time taken: ", e)

t = time.time()
g = a-b
d = np.einsum('i,i', g, g)
e = time.time() - t
print("Distance: ", d, "Time taken: ", e)

t = time.time()
g = a-b
d = dot_nb(g, g)
e = time.time() - t
print("Distance: ", d, "Time taken: ", e)

t = time.time()
d = squared_distance_inplace(a,b,s)
e = time.time() - t
print("Distance: ", d, "Time taken: ", e)

t = time.time()
l = len(a)
d = squared_distance_inplace(a,b,l)
e = time.time() - t
print("Distance: ", d, "Time taken: ", e)

"""


"""

import numpy as np
import time


# Create large test vectors
dim = 5
a = np.random.rand(dim)
b = np.random.rand(dim)

# Warm-up compilation
squared_distance_numba(a, b)
euclidean_distance_numba(a, b)
euclidean_distance_numba_with_l(a, b, dim)
euclidean_distance_satisfaction_numba(a, b, 4.0)

# Timing squared distance
start = time.time()
sq_dist = squared_distance_numba(a, b)
end = time.time()
print(f"Squared distance time: {end - start:.20f} s, result={sq_dist:.6f}")

# Timing euclidean distance (with sqrt)
start = time.time()
dist = euclidean_distance_numba(a, b)
end = time.time()
print(f"Euclidean distance time: {end - start:.20f} s, result={dist:.6f}")

# Timing euclidean distance with length parameter
start = time.time()
dist_with_l = euclidean_distance_numba_with_l(a, b, dim)
end = time.time()
print(f"Euclidean distance with length time: {end - start:.20f} s, result={dist_with_l:.6f}")

# Timing euclidean distance with length parameter
start = time.time()
dist_with_l = euclidean_distance_satisfaction_numba(a, b, 4.0)
end = time.time()
print(f"Euclidean distance with length time: {end - start:.20f} s, result={dist_with_l:.6f}")


"""