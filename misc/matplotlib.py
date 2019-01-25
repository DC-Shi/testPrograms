import matplotlib.pyplot as plt

x = [1,2,4,5,7,20]
y = [4.5,2.5,1.1,9.4,10,13]

plt.scatter(x, y, color='b', marker='o')
plt.scatter(y, x, color='r', marker='o')

plt.xlabel('x')
plt.ylabel('y')
plt.show()
