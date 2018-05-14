import numpy as np
import matplotlib.pyplot as plt

from Model import Model

l = [1, 20, 20, 20, 1]
m = Model("bvp", l, 1.0, 50000)
    
epoches = 0
batch_size = 10
while True:
    epoches = epoches + 1
    m.train({
        m.varIn: np.array([[0], [1]]),
        m.varOut: np.array([[1], [0]]),
        m.varAux: np.random.uniform(0.0, 1.0, batch_size).reshape(-1, 1),
    })
    c = m.sess.run(m.loss, {
        m.varIn: np.array([[0], [1]]),
        m.varOut: np.array([[1], [0]]),
        m.varAux: np.random.uniform(0.0, 1.0, 10).reshape(-1, 1),
    })
    if c < 0.005:
        print("Converge with Error =", c)
        break
    elif epoches > 100:
        print("Fail to converge.")
        break
    else: 
        print("Error =", c)


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