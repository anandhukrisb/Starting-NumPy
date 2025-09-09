import numpy as np

arr = np.array([[[1, 2, 3, 4], [5, 6, 7, 8 ], [9, 10, 11, 12]],
                [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
                [[25, 26, 27, 28], [29, 30, 31, 32], [33, 34, 35, 36]]])  

# SLICING
# print(arr[0:3:2])
# print(arr[:, 0])
# print(arr[:, 1:2])
# print(arr[1:4, 1:4])
# print(arr[1:4, 1:4])

# ARITHMETIC 

# print(arr - 1)
# print(np.max(arr))
# print(np.pi * arr**2)

# arr[arr >= 4] = 0
# print(arr)

# BROADCASTING

a1 = np.array([[1, 2, 3, 4]])
a2 = np.array([[1], [2], [3], [4]])

# print(a1.shape)
# print(a2.shape)

# print(a1 * a2)

