
import tensorflow as tf

tensor = [0, 0, 0, 0, 0, 0, 0, 0]    # tf.rank(tensor) == 1
indices = [[1], [3], [4], [7]]       # num_updates == 4, index_depth == 1
updates = [9, 10, 11, 12]            # num_updates == 4
print(tf.tensor_scatter_nd_update(tensor, indices, updates))


x = tf.zeros([3,2])
y = tf.Variable(x)
print(x)
print(x[0,1])
print("y[0,1]: {}".format(y[0,1]))
print(y)
print(tf.tensor_scatter_nd_update(y, [[0,1]], [1.0]))
#y[0,1].assign([1.0])
print(x)