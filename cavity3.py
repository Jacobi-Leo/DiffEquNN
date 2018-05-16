from Model import Model

import numpy as np
import matplotlib.pyplot as plt

def getBody(N):

    return np.random.uniform(size=N), np.random.uniform(size=N)

l = [2, 50, 50, 50, 2]
m = Model("cavity3", layers=l, penalty=2.0, num_steps=50000)

num_epoch = 0
batch_size = 2000
max_epoch = 3

while True:
    num_epoch = num_epoch + 1
    print("Epoch =", num_epoch)
    m.train({
        m.varAux: np.stack(getBody(batch_size), axis=1),
    }, method="L-BFGS-B")
    c = m.convergence[-1]

    if c[0] < 0.005:
        print("Converged!")
        break
    elif num_epoch > max_epoch:
        print("Fail to converge.")
        break
    else:
        print("Error =", c)