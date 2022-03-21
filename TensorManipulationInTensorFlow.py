import tensorflow as tf
import numpy as np

# some basic tensor creation code followed by exercises based on Justin Johnson's Michigan deep learning machine vision course assignment A1 - originally for pytorch. 

x = tf.zeros([3,2])
y = tf.Variable(x)
z = tf.ones([3,2])
a = tf.eye(3,3)
b = tf.random.normal([3,2])

print(z,a,b)

print(y)
print(tf.tensor_scatter_nd_update(y, [[0,1]], [1.0]))
print(x)


#print(tf.tensor_scatter_nd_update(y, [[0,1],[1,0]], [10, 100]))
y=tf.tensor_scatter_nd_update(y, [[0,1],[1,0]], [10, 100])
print(y)
x=tf.tensor_scatter_nd_update(x, [[0,1],[1,0]], [10, 100])
print("x : {}".format(x))

def create_sample_tensor() -> "Tensor":
    """
    Return a  Tensor of shape (3, 2) which is filled with zeros, except
    for element (0, 1) which is set to 10 and element (1, 0) which is set to
    100.

    Returns:
        Tensor of shape (3, 2) as described above.
    """
    x = tf.zeros([3,2])
    #y = tf.Variable(x)
    x=tf.tensor_scatter_nd_update(x, [[0,1],[1,0]], [10, 100])
   
    return x



def mutate_tensor(  x, indices, values) -> "Tensor":
    """
    Mutate the tensor x according to indices and values. Specifically, indices
    is a list [(i0, j0), (i1, j1), ... ] of integer indices, and values is a
    list [v0, v1, ...] of values. This function should mutate x by setting:

    x[i0, j0] = v0
    x[i1, j1] = v1

    and so on.

    If the same index pair appears multiple times in indices, you should set x
    to the last one.

    Args:
        x: A Tensor of shape (H, W)
        indices: A list of N tuples [(i0, j0), (i1, j1), ..., ]
        values: A list of N values [v0, v1, ...]

    Returns:
        The input tensor x
    """
    ##########################################################################
    #                     TODO: Implement this function                      #
    ##########################################################################
    # Replace "pass" statement with your code
    print(x)
    print(indices)
    print(values)
    x=tf.tensor_scatter_nd_update(x, indices, values)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return x

def create_tensor_of_pi(M: int, N: int) -> "Tensor":
    """
    Returns a Tensor of shape (M, N) filled entirely with the value 3.14

    Args:
        M, N: Positive integers giving the shape of Tensor to create

    Returns:
        x: A tensor of shape (M, N) filled with the value 3.14
    """
    x = None
    ##########################################################################
    #         TODO: Implement this function. It should take one line.        #
    ##########################################################################
    # Replace "pass" statement with your code
    x=tf.ones([M,N], tf.float32)
    pi = np.pi
    x = pi*x  
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return x
# Create a sample tensor
x = create_sample_tensor()
print('Here is the sample tensor: \n {}'.format(x))

# Mutate the tensor by setting a few elements
indices = [[0, 0], [1, 0], [1, 1]]
values = [4, 5, 6]
#x=tf.tensor_scatter_nd_update(x, [[0,1],[1,0]], [10, 100])
#x=tf.tensor_scatter_nd_update(x, [[0, 0], [1, 0], [1, 1]], [4, 5, 6])
#x=tf.tensor_scatter_nd_update(x, indices, values)
x=mutate_tensor(x, indices, values)
print('\nAfter mutating:')
print(x)

x = create_tensor_of_pi(4, 5)
print(x)
print('x is a tensor:', tf.is_tensor(x))
print('x has correct shape: ', x.shape == (4, 5))
#print('x is filled with pi: ', (x == 3.14).all().item() == 1)
#tf.assertAllEqual()
#assertAllEqual(
#    a, b, msg=None
#)