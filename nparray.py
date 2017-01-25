#!/usr/bin/env python

from __future__ import print_function

import numpy as np

##
## basics
##

arr = np.arange(12).reshape(4, 3)
print(arr)
print('\n')
arr = arr.reshape((2, 6))
print(arr)
print('\n')
arr = arr.reshape((1, 12))
print(arr)
print('\n')

# create array with range of values
# along with a step val
range_val = np.arange(1, 10, 0.1)
print(range_val)
range_val = range_val.reshape(10, 9)
print(range_val)
print(list(range_val.shape))
print(range_val.dtype.name)
print(range_val.size)

# prints the dimensions of array in list format
print(list(arr.shape))

# prints the dimensions of array as a tuple
print(arr.shape)

# prints data type 
print(arr.dtype.name)

# prints size of data type
print(arr.itemsize)

# prints number of elements in array
print(arr.size)

# prints details of type : different than data type
print(type(arr))


##
## Creation of np.array
##

# creating a np.array
val = np.array([2, 3, 4])
print(list(val.shape))
print(val.dtype.name)
print(val.size)

# creating a 2d np.array
twod_val = np.array([(1.5, 2, 3), (4, 5, 6)])
print(list(twod_val.shape))
print(twod_val.dtype.name)
print(twod_val.size)

# creating a np.array with prespecified data type
sp_val = np.array([(1, 2), (3, 4)], dtype=complex)
print(list(sp_val.shape))
print(sp_val.dtype.name)
print(sp_val.size)

##
## basic operations
##

a_values = np.arange(0, 100, 1)
b_values = np.arange(0, 200, 2)

if(a_values.size == a_values.size):
	print(a_values ** 2)
	print('\n')
	print(b_values ** 2)

# print true/false according to filter
print(a_values < 50)

# element-wise product
print(a_values * b_values)

# matrix product
print(a_values.dot(b_values))

# another way to do matrix product
print(np.dot(a_values, b_values))

##
## tricky operation (important)
##

# +=, *= operations don't create a new array
# they make changes into available array
# hence, if both arrays aren't of same type;
# this operation will return an error

a = np.ones((2,3), dtype=int)
b = np.random.random((2,3))
# this is allowed
b += a
# this is not
# this will give the error: 
# TypeError: Cannot cast ufunc add output from dtype('float64') \
# to dtype('int64') with casting rule 'same_kind'
# a += b

# Gotta do this like this
a = a + b
print(a)
# type of array changes in this way
print(a.dtype)

# sums up all values of array
print(a.sum())

# prints min/max of all values
print(a.min())
print(a.max())

# in case of multi-dimensional arrays
# operations can be applied to specific axes
b = np.arange(12).reshape(3,4)
print(b)

# sum of each column
print(b.sum(axis=0))

# sum of each row
print(b.sum(axis=1))

# sums values uptil that position 
# in the reference axis
print(b.cumsum(axis=1))


# similarly axes in 3d array...
c = np.arange(0, 100).reshape(5, 2, 10)
print(c)
print('\n')
print(c.sum(axis=0))
print(c.sum(axis=1))
print(c.sum(axis=2))


##
## universal functions
##

d = np.arange(3)
print(np.exp(d))
print(np.sqrt(d))
e = np.array([2., -1., 4.])
print(np.add(d, e))

##
## indexing
##

a = np.arange(10)
# get values from 3rd to 6th position
print(a[2:5])

# reverse the array and print
print(a[: : -1])

b = np.arange(100).reshape(5, 2, 10)

# `...` mean rest of the values
# it could be any dimensional other than the ones mentioned
print(b[2, ...])
print(b[..., 2])


##
## shape manipulation
##

a = np.floor(10*np.random.random((3,4)))
print(a)
print(a.shape)

# flattens the array
print(a.ravel())
print(list(a.ravel().shape))

# reshape() returns array with different shape
# resize changes the shape of the array it is called by
a.resize((6, 2))
print(a)

# Transpose the matrix
print(a.T)
print(a.T.shape)


##
## stacking arrays
##

num_la = np.floor(10*np.random.random((2,2)))
num_lb = np.floor(10*np.random.random((2,2)))

# stack rows of both arrays vertically
print(np.vstack((num_la, num_lb)))

# stack rows of both arrays horizontally
print(np.hstack((num_la, num_lb)))

# in order to add 1d array into a 2d array stack
# column_stack can be used
from numpy import newaxis

a = np.array([4.,2.])
b = np.array([2.,8.])

# change the shape of the array
print(a[:, newaxis])

# combines 1d arrays that have been accessed using 
# newaxis so that they work like 2d arrays
print(np.column_stack((a[:, newaxis], b[:, newaxis])))

# same thing using vstack instead of column_stack would give 
# different results
print(np.vstack((a[:, newaxis], b[:, newaxis])))

# For arrays of with more than two dimensions, hstack stacks along their second axes, 
# vstack stacks along their first axes, and concatenate allows for an 
# optional arguments giving the number of the axis along which the concatenation should happen.

##
## split arrays into smaller arrays
##

val = np.floor(10*np.random.random((2,12)))

# split array into 3 parts
print(np.hsplit(val, 3))

# split array after third and fourth column
print(np.hsplit(val, (3, 4)))

##
## copies of arrays
##

# no copy
a = np.arange(12)
b = a

# this being `true` means that these two names point to same object
# hence nothing new is created, a and b are basically the same thing
print(b is a)

# create shallow copy
c = a.view()

# this is false, hence they aren't pointing to same object 
print(c is a)

# this being true means c is a view of the data owned by a
print(c.base is a)

# this being false tells that c doesn't have a new copy of the data held by a even
# though its not pointing to the same object a points to
print(c.flags.owndata)

# deep copy
d = a.copy()

# both are false
print(d is a)
print(d.base is a)

# it owns its own data now
print(d.flags.owndata)


##
## broadcasting
##



##
## indexing tricks
##

# indexing with arrays of indices
a = np.arange(12) ** 2
i = np.array([1, 1, 3, 8, 5])
print(a)
 
# this gives the values from a with indices from i 
print(a[i])

# multidimensional example
palette = np.array([
		[0, 0, 0],
		[255, 0, 0],
		[0, 255, 0],
		[0, 0, 255],
		[255, 255, 255]
	])

image = np.array([
		[0, 1, 2, 0],
		[0, 3, 4, 0]
	])

print(palette[image])

# indexing with boolean arrays
a = np.arange(12).reshape(3,4)
b = a > 4

# it will print all values from a where b is true
# this can be used in order to filter out values 
print(a[b])

# this code filters out the values in a according to 
# boolean expression used to create b
a[b] = 0
print(a)

# using this technique to create Mandelbrot set

import matplotlib.pyplot as plt
def mandelbrot(h, w, maxit=20) :
	"""
	Returns a mandelbrot fractal of size (h, w)
	"""
	y,x = np.ogrid[-1.4:1.4:h * 1j, -2:0.8:w * 1j]
	c = x+y*1j
	z = c
	divtime = maxit + np.zeros(z.shape, dtype=int)

	for i in range(maxit):
	  z = z ** 2 + c
	  diverge = z * np.conj(z) > 2 ** 2         # who is diverging
	  div_now = diverge & (divtime == maxit)  	# who is diverging now
	  divtime[div_now] = i                  		# note when
	  z[diverge] = 2                        		# avoid diverging too much

	return divtime

plt.imshow(mandelbrot(1000, 1000))
plt.show()

##
## Linear Algebra
##

# array operations
a = np.array([[1.0, 2.0], [3.0, 4.0]])

# transpose of a
print(a.transpose())

# inversion of a 
print(np.linalg.inv(a))

# create a unit matrix of given size
# eye refers to I (identity matrix; weird I know)
u = np.eye(5)

j = np.array([[0.0, -1.0], [1.0, 0.0]])
# dot product of 2 matrices
print(np.dot(j, j))

# sum of diagonals
print(np.trace(u))

# solve linear equations with the 2 matrices being coefficient matrix and ordinates
# solution for a*x = b
# matrices given are 'a' and 'b'
y = np.array([[5.], [7.]])
print(np.linalg.solve(a, y))

# eigen values of matrix
print(np.linalg.eig(j))

##
## Histograms
##

mu, sigma = 2, 0.5
v = np.random.normal(mu, sigma, 10000)

# plot histogram with 50 bins
plt.hist(v, bins=50, normed=1)
plt.show() 