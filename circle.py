import numpy as np
import matplotlib.pyplot as plt

from Model import Model

l = [1, 20, 20, 20, 2]
m = Model("circle", l, 1.0, 50000)

def my_random(start, stop, num):
    step = (stop - start) / num
    lst = []
    for i in range(num):
        lst.append(np.random.uniform(start+i*step, start+(i+1)*step, 1)[0])
    return np.array(lst)

num_epoch = 0
batch_size = 100
max_epoch = 100
while True:
    num_epoch = num_epoch + 1
    print("Epoch =", num_epoch)
    m.train({
        m.varIn: np.array([[0.]]),
        m.varOut: np.array([[0., 1.]]),
        m.varAux: my_random(0.0, 1.0, batch_size).reshape(-1, 1),
    }, method="L-BFGS-B")
    c = m.sess.run(m.loss, {
        m.varIn: np.array([[0.]]),
        m.varOut: np.array([[0., 1.]]),
        m.varAux: my_random(0.0, 1.0, batch_size).reshape(-1, 1),
    })
    if c < 0.005:
        break
    elif num_epoch > max_epoch:
        print("Fail to converge.")
        break
    else: 
        print("Error =", c)

t = np.linspace(0.0, 1.0, 50)
xy = m.eval({m.varIn: t.reshape(-1, 1)})
x = xy[:, 0]
y = xy[:, 1]

print("Error =", 1.0 - y[-1])

fig, ax = plt.subplots(1, 2)
ax1, ax2 = ax
ax1.plot(t, x, 'o', label="NN predicted")
ax1.plot(t, np.sin(t*2.0*np.pi), label="Exact")
ax2.plot(x, y)
ax2.set_aspect('equal', 'datalim')
plt.legend()
plt.show()