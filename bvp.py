import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np

from Model import Model

l = [1, 20, 20, 20, 1]
m = Model("bvp", l, 0.01)

m.train({
    m.varIn: np.array([[0], [1]]),
    m.varOut: np.array([[1], [0]]),
    m.varAux: np.linspace(0.0, 1.0, 100).reshape(-1, 1),
})

def boundaryLayerFunction(x):
    nu = m.nu
    return 2.0/(1.0 + np.exp((x - 1.0)/nu)) - 1.0

x = np.linspace(0.0, 1.0, 50)
y = m.eval({m.varIn: x.reshape(-1, 1)}).reshape(-1)
ye = boundaryLayerFunction(x)
err = np.sqrt(np.mean(np.square(y-ye)))
plt.plot(x, y, label="NN predicted")
plt.plot(x, ye, label="Analytical solution")
plt.legend()
plt.text(0.2, 0.2, "Error = " + str(err))
plt.show()