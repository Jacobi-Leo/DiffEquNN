from Model import Model

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from BurgersData import getBody, getBoundary
import dataBurgers2 as dt

l = [2, 20, 20, 20, 20, 20, 1]
m = Model("burgers", layers=l, penalty=2.0, num_steps=50000)

num_epoch = 0
batch_size = 1000
max_epoch = 100
small = 50

while True:
    num_epoch = num_epoch + 1
    print("Epoch =", num_epoch)
    tmp = np.stack(getBoundary(small), axis=1)
    m.train({
        m.varIn: tmp[:, 0:2],
        m.varOut: tmp[:, 2:3],
        m.varAux: np.stack(getBody(batch_size), axis=1),
    }, method="L-BFGS-B")
    c = m.convergence[-1][0]

    # tmp = dt.surfaceData()
    # tmp1 = (tmp[0], tmp[1])
    # tmp2 = tmp[2]
    # m.train({
    #     m.varIn: np.concatenate(tmp1, axis=1),
    #     m.varOut: tmp2,
    #     m.varAux: np.concatenate(dt.insideCoordData(), axis=1),
    # })
    # c = m.convergence[-1][0]

    if c < 5e-4:
    #if c < 1e-7:
        print("Converged!")
        break
    elif num_epoch > max_epoch:
        print("Fail to converge.")
        break
    else:
        print("Error =", c)
       
t1 = 0.25; t2 = 0.5; t3 = 0.75
sample_size = 70
x = np.linspace(-1.0, 1.0, sample_size)
eval = lambda t: m.eval({
    m.varIn: np.stack(
        (t*np.ones(sample_size), x), 
        axis=1,
    )
}).reshape(-1)
u0 = eval(0)
u1 = eval(t1)
u2 = eval(t2)
u3 = eval(t3)


plt.plot(x, u0, label="t = 0")
plt.plot(x, u1, label="t = "+str(t1))
plt.plot(x, u2, label="t = "+str(t2))
plt.plot(x, u3, label="t = "+str(t3))
plt.legend()
plt.show()