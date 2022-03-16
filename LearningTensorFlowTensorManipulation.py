
import tensorflow as tf
import numpy as np
import matplotlib as plt


#a=tf.Variable([1,2,3,4])
b=tf.range(1,7)
b=tf.reshape(b, (2,3))
print(b)
# above is same as b=tf.Variable([[1,2,3],[4,5,6]])


print("b[0,1:2]: {}".format(b[0,1:2]))
print("b[1,1:2]: {}".format(b[1,1:2]))
print(b[0,1].numpy())


x = tf.zeros([3,2])
y = tf.Variable(x)
print(x)
print(x[0,1])
print("X[0,1]: {}".format(x[0,1]))
#y[0,1].assign([1.0])
print(x)
#print(x, y)



#t12 = tf.tensor_scatter_nd_add(t11,
#                               indices=[[0, 2], [1, 1], [2, 0]],
#                               updates=[6, 5, 4])


#print(x[1:2,])