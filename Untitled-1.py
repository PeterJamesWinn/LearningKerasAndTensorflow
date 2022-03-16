# %%


# %%
import tensorflow as tf
import matplotlib as plt
import numpy as np



# %%

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

# %%
rank_4_tensor = tf.zeros([3, 2, 4, 5])

print("Type of every element:", rank_4_tensor.dtype)
print("Number of axes:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())

# %%
rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
print(rank_1_tensor.numpy())

print("First:", rank_1_tensor[0])
print("First:", rank_1_tensor[0].numpy())  # .numpy() converts to numpy array which gives cleaner output c.f. above.
print("Fourth:", rank_1_tensor[3].numpy())
print("Last:", rank_1_tensor[-1].numpy())

# %%
print("Everything:", rank_1_tensor[:].numpy())
print("Before 4:", rank_1_tensor[:4].numpy())
print("From 4 to the end:", rank_1_tensor[4:].numpy())
print("From 2, before 7:", rank_1_tensor[2:7].numpy())
print("Every other item:", rank_1_tensor[::2].numpy())
print("Reversed:", rank_1_tensor[::-1].numpy())

print(rank_2_tensor.numpy())
print(rank_2_tensor[1, 1].numpy())


# %%
rank_2_tensor = tf.constant([[1, 2, 3, 4],[5, 6, 7, 8], [9, 10, 11, 12]])

# Get row and column tensors
print("Second row:", rank_2_tensor[1, :].numpy())
print("Second column:", rank_2_tensor[:,1].numpy())
print("First row:", rank_2_tensor[0, :].numpy())
print("First two rows:", rank_2_tensor[0: 2].numpy())
print("First two columns:", rank_2_tensor[: ,0:2].numpy())
print("Last row:", rank_2_tensor[-1, :].numpy())
print("First item in last column:", rank_2_tensor[0, -1].numpy())
print("Skip the first row:")
print(rank_2_tensor[1:, :].numpy(), "\n")

# %%
rank_3_tensor = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],])

print(rank_3_tensor[:, :, 4])



# %%
x = tf.constant([[1],[2],[3]])
print(x.shape)
print(x.shape.as_list())
print(x)

reshaped = tf.reshape(x, [1,3])

print(reshaped)
print(reshaped.shape)


# %%
print(rank_3_tensor)
print(tf.reshape(rank_3_tensor, [-1]))
print(rank_3_tensor)

# %%
x = tf.zeros(3,2)
y = tf.Variable(x)
print(x, y)

# %%
def create_sample_tensor() -> Tensor:
    """
    Return a  Tensor of shape (3, 2) which is filled with zeros, except
    for element (0, 1) which is set to 10 and element (1, 0) which is set to
    100.

    Returns:
        Tensor of shape (3, 2) as described above.
    """
    x = tf.zeros(3,2)
    x[0,1] = 10
    x[1,0] = 100
   
   
    return x


def mutate_tensor(
    x: Tensor, indices: List[Tuple[int, int]], values: List[float]
) -> Tensor:
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
    pass
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return x


def count_tensor_elements(x: Tensor) -> int:
    """
    Count the number of scalar elements in a tensor x.

    For example, a tensor of shape (10,) has 10 elements; a tensor of shape
    (3, 4) has 12 elements; a tensor of shape (2, 3, 4) has 24 elements, etc.

    You may not use the functions torch.numel or x.numel. The input tensor
    should not be modified.

    Args:
        x: A tensor of any shape

    Returns:
        num_elements: An integer giving the number of scalar elements in x
    """
    num_elements = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    #   You CANNOT use the built-in functions torch.numel(x) or x.numel().   #
    ##########################################################################
    # Replace "pass" statement with your code
    pass
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return num_elements


def create_tensor_of_pi(M: int, N: int) -> Tensor:
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
    pass
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return x


# Create a sample tensor
x = create_sample_tensor()
print('Here is the sample tensor:')
print(x)

# Mutate the tensor by setting a few elements
indices = [(0, 0), (1, 0), (1, 1)]
values = [4, 5, 6]
mutate_tensor(x, indices, values)
print('\nAfter mutating:')
print(x)
print('\nCorrect shape: ', x.shape == (3, 2))
print('x[0, 0] correct: ', x[0, 0].item() == 4)
print('x[1, 0] correct: ', x[1, 0].item() == 5)
print('x[1, 1] correct: ', x[1, 1].item() == 6)

# Check the number of elements in the sample tensor
num = count_tensor_elements(x)
print('\nNumber of elements in x: ', num)
print('Correctly counted: ', num == 6)


