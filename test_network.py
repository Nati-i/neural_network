from manual_neural_network import*

import numpy as np
import matplotlib.pyplot as plt
# Z = Ax + b
# A = 10
# b = 1
# x is a Placeholder
g = Graph()
g.set_as_default()
A = Variable([[10,20],[30,40]])
b = Variable([1,1])

x = Placeholder()
y = matmul(A,x)
z = add(y,b)
sess = Session()
result = sess.run(operation=z,feed_dict={x:10})
#print(result)

import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

sample_z = np.linspace(-10,10,100)
sample_a = sigmoid(sample_z)

plt.plot(sample_z,sample_a)

#plt.show()

g = Graph()

g.set_as_default()
x = Placeholder()

w = Variable([1,1])
c = Variable(-5)

z = add(matmul(w,x),c)

a = Sigmoid(z)

sess = Session()

l = sess.run(operation=a,feed_dict={x:[2,-10]})
print(l)