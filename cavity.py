from Model import Model

import numpy as np
import matplotlib.pyplot as plt

def getBody(N):

    return np.random.uniform(size=N), np.random.uniform(size=N)

l = [2, 40, 40, 40, 3]
m = Model("cavity", layers=l, penalty=200.0, num_steps=50000)

num_epoch = 0
batch_size = 5000
max_epoch = 100

while True:
    num_epoch = num_epoch + 1
    print("Epoch =", num_epoch)
    m.train({
        m.varAux: np.stack(getBody(batch_size), axis=1),
    }, method="L-BFGS-B")
    c = m.convergence[-1]

    if c[0] < 0.0005:
        print("Converged!")
        break
    elif num_epoch > max_epoch:
        print("Fail to converge.")
        break
    else:
        print("Error =", c)