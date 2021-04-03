import numpy as np

a = np.array([[1,2],[3,4],[6,7]])
print(a)
b = a.tobytes()
print(b)
y = np.frombuffer(b,dtype=a.dtype)
y = y.reshape(3,2)
print(y)
