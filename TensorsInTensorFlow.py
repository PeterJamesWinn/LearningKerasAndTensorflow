
import tensorflow as tf
import matplotlib as plt
import numpy as np


x = tf.constant([[1.,2.,3.],[4., 5., 6.]])
print(x)
print("shape: {} data type: {}".format(x.shape, x.dtype))

double = x + x
quintle = 5 * x
print(x @ tf.transpose(x)) # matrix multiplication 
print(tf.concat([x, x, x], axis=0))
print("double {}: quintle: {}".format(double, quintle))


rank_0_tensor = tf.constant(4)
print(rank_0_tensor)

rank_1_tensor = tf.constant([2, 3, 4])
print(rank_1_tensor)

rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)

rank_1_tensor = tf.constant([2.0, 3.0, 4.0], dtype=tf.float16)
print(rank_1_tensor)

rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print(rank_2_tensor)

rank_3_tensor = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],])

print(rank_3_tensor)

rank_2_tensor_as_array = np.array(rank_2_tensor)
print("rank_2_tensor_as_array {}".format(rank_2_tensor_as_array) )


a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]]) # Could have also said `tf.ones([2,2])`

print(tf.add(a, b), "\n")
print(a + b, "\n")
print(tf.multiply(a, b), "\n") # elementwise multiplication
print(a* b, "\n")
print(tf.matmul(a, b), "\n") # matrix multiplication
print(a @ b, "\n")

c = tf.constant([[4.0, 5.0], [10.0, 1.0]])

# Find the largest value
print(tf.reduce_max(c))
# Find the index of the largest value
print(tf.argmax(c))
# Compute the softmax
print(tf.nn.softmax(c))