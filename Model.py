import numpy as np
import tensorflow as tf

tfdtype = tf.float32
npdtype = np.float32

class Model:

    def __init__(self, name, layers, penalty=1.0, num_steps=100000, debug=False):

        self.reduceFunc = lambda x: tf.reduce_mean(tf.square(x))
        self.sess = tf.Session()
        self.name = name
        self._penalty = 0.5 * penalty
        self._layers = layers
        self._num_steps = num_steps
        self._setSmallNumber()
        self._num_output = self._layers[-1]
        self._num_input = self._layers[0]
        self._setPhysics()
        self._buildNet()
        self.debug = debug
        self.convergence = []
        self.sess.run(tf.global_variables_initializer())
        
        
    def _setPhysics(self):
        
        if self.name in ['bvp', 'BVP']:
            self.physics = self._bvpPhysics
        elif self.name in ['heat', 'Heat', 'HeatConduction']:
            self.physics = self._heatPhysics
        elif self.name in ['burgers', 'Burgers', "Burgers'"]:
            self.physics = self._burgersPhysics
        elif self.name in ['circle', 'Circle']:
            self.physics = self._circlePhysics
        elif self.name in ['cavity', 'Cavity', 'DrivenCavity']:
            self.physics = self._cavityPhysics
        elif self.name in ['cavity2']:
            self.physics = self._cavityPhysics2
        else:
            raise ValueError("Wrong name for model")
        
        
    def _setSmallNumber(self):
        
        a = 1.0 # aspect ratio
        if tfdtype == tf.float32:
            self.tol = a * np.finfo(np.float32).eps
        elif tfdtype == tf.float64:
            self.tol = a * np.finfo(np.float64).eps
        else:
            raise ValueError("Wrong data type selection.")


    def _setInputVariables(self):

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.varOut = tf.placeholder(dtype=tfdtype, shape=[None, self._num_output])
            self.varIn = tf.placeholder(dtype=tfdtype, shape=[None, self._num_input])
            self.varAux = tf.placeholder(dtype=tfdtype, shape=[None, self._num_input])


    def _setNetworkVariables(self):

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            
            self._weights = []
            for i in range(len(self._layers) - 1):
                self._weights.append(
                    tf.get_variable(
                        "w"+str(i), 
                        shape=[self._layers[i], self._layers[i+1]], 
                        dtype=tfdtype,
                        initializer=tf.contrib.layers.xavier_initializer(),
                    )
                )

            self._biases = []
            for i in range(len(self._layers) - 1):
                self._biases.append(
                    tf.get_variable(
                        "b"+str(i),
                        shape=[self._layers[i+1]],
                        dtype=tfdtype,
                        initializer=tf.contrib.layers.xavier_initializer(),
                    )
                )


    def _neuralNet(self, x):

        # acFun is activation function
        acFun = tf.tanh

        layer = x
        for i in range(len(self._layers) - 2):
            layer = acFun(tf.matmul(layer, self._weights[i]) + self._biases[i])
            
        layer = tf.matmul(layer, self._weights[-1]) + self._biases[-1]
        return layer


    def model(self, x):

        return self._neuralNet(x)
    

    def _circlePhysics(self):
                
        t = self.varAux
        u = self.model(t)
        x, y = tf.split(u, 2, 1)
        
        xt = tf.gradients(x, t)[0]
        yt = tf.gradients(y, t)[0]
        
        eq1 = xt - y * 2.0 * np.pi
        eq2 = yt + x * 2.0 * np.pi
        
        cost = self.hypothesis - self.varOut
        
        return self.reduceFunc(cost), self.reduceFunc(eq1) + self.reduceFunc(eq2)


    def _burgersPhysics(self):

        D = lambda y: tf.split(tf.gradients(y, self.varAux)[0], 2, 1)
        
        pi = tf.constant(np.pi, dtype=tfdtype)
        tx = self.varAux
        u = self.model(tx)

        ut, ux = D(u)
        _, uxx = D(ux)

        deviation = ut + u * ux - 0.01/pi * uxx

        cost = self.hypothesis - self.varOut

        return self.reduceFunc(cost), self.reduceFunc(deviation)

    
    def _bvpPhysics(self):

        self.nu = 0.1
        x = self.varAux
        u = self.model(x)

        ux = tf.gradients(u, x)[0]
        uxx = tf.gradients(ux, x)[0]

        deviation = u * ux - self.nu * uxx

        cost = self.hypothesis - self.varOut

        return self.reduceFunc(cost), self.reduceFunc(deviation)
    
    
    def _heatPhysics(self):
        
        u = self.model(self.varAux)
        D = lambda y: tf.split(tf.gradients(y, self.varAux)[0], 2, 1)
        
        ux, uy = D(u)
        uxx, _ = D(ux)
        _, uyy = D(uy)
        
        deviation = uxx + uyy

        cost = self.hypothesis - self.varOut
       
        return self.reduceFunc(cost), self.reduceFunc(deviation)

    
    def _cavityPhysics(self):

        D = lambda y: tf.split(tf.gradients(y, self.varAux)[0], 2, 1)

        self.nu = 1.0 / 200.0
        u, v, p = tf.split(self.model(self.varAux), 3, 1)

        ux, uy = D(u)
        vx, vy = D(v)
        px, py = D(p)

        uxx, _ = D(ux)
        _, uyy = D(uy)

        vxx, _ = D(vx)
        _, vyy = D(vy)

        deviation0 = ux + vy
        deviation1 = u * ux + v * uy + px - self.nu * (uxx + uyy)
        deviation2 = u * vx + v * vy + py - self.nu * (vxx + vyy)

        tmp = np.linspace(0.0, 1.0, 100, dtype=npdtype)

        x = tf.concat((
            np.linspace(0.0, 1.0, 100, dtype=npdtype),
            np.ones(100, dtype=npdtype),
            np.linspace(0.0, 1.0, 100, dtype=npdtype),
            np.zeros(100, dtype=npdtype),
        ), axis=0)

        y = tf.concat((
            np.ones(100, dtype=npdtype),
            np.linspace(0.0, 1.0, 100, dtype=npdtype),
            np.zeros(100, dtype=npdtype),
            np.linspace(0.0, 1.0, 100, dtype=npdtype),
        ), axis=0)

        u_ref = tf.concat((
            16.0 * np.square(tmp) * np.square(1.0 - tmp),
            np.zeros(300, dtype=npdtype)
        ), axis=0)

        v_ref = tf.zeros(400, dtype=tfdtype)

        var = tf.stack([x, y], axis=1)
        uu, vv, _ = tf.split(self.model(var), 3, 1)

        cost1 = uu - u_ref
        cost2 = vv - v_ref

        return (
            self.reduceFunc(cost1) + self.reduceFunc(cost2), 
            self.reduceFunc(deviation0) + self.reduceFunc(deviation1) + \
                self.reduceFunc(deviation2),
            )


    def _cavityPhysics2(self):

        self.nu = 1.0 / 200.0
        nu = self.nu

        num_x_grid = 100
        num_y_grid = 100

        dx = 1.0 / num_x_grid
        dy = 1.0 / num_y_grid

        x_tmp = tf.linspace(0.0+dx/2.0, 1.0-dx/2.0, num_x_grid)
        y_tmp = tf.linspace(0.0+dy/2.0, 1.0-dy/2.0, num_y_grid)

        X, Y = tf.meshgrid(x_tmp, y_tmp, indexing='ij')

        x_set = tf.reshape(X, [-1])
        y_set = tf.reshape(Y, [-1])

        u_set, v_set, p_set = tf.split(
            self.model(tf.stack([x_set, y_set], axis=1)), 3, 1)
        U = tf.reshape(u_set, [num_x_grid, num_y_grid])
        V = tf.reshape(v_set, [num_x_grid, num_y_grid])
        P = tf.reshape(p_set, [num_x_grid, num_y_grid])

        L0 = 0.0; L1 = 0.0; L2 = 0.0
        for i in range(num_x_grid):
            for j in range(num_y_grid):
                if i == 0 and j == 0:
                    uyN = (U[i, j+1] - U[i, j]) / dy
                    vyN = (V[i, j+1] - V[i, j]) / dy
                    uyS = (U[i, j]) / dy / 2.0
                    vyS = (V[i, j]) / dy / 2.0
                    uxE = (U[i+1, j] - U[i, j]) / dx
                    vxE = (V[i+1, j] - V[i, j]) / dx
                    uxW = (U[i, j]) / dx / 2.0
                    vxW = (V[i, j]) / dx / 2.0

                    if V[i, j] + V[i, j+1] > 0:
                        uN = U[i, j]
                        vN = V[i, j]
                        pN = P[i, j]
                    elif V[i, j] + V[i, j+1] < 0:
                        uN = U[i, j+1]
                        vN = V[i, j+1]
                        pN = P[i, j+1]
                    else:
                        uN = 0.0
                        vN = 0.0
                        pN = 0.0

                    uS = 0.0
                    vS = 0.0
                    pS = 0.0

                    if U[i, j] + U[i+1, j] > 0:
                        uE = U[i, j]
                        vE = V[i, j]
                        pE = P[i, j]
                    elif U[i, j] + U[i+1, j] < 0:
                        uE = U[i+1, j]
                        vE = V[i+1, j]
                        pE = P[i+1, j]
                    else:
                        uE = 0.0
                        vE = 0.0
                        pE = 0.0

                    uW = 0.0
                    vW = 0.0
                    pW = 0.0
                elif i == 0 and j == num_y_grid - 1:
                    uyN = (0.0 - U[i, j]) / dy / 2.0
                    vyN = (0.0 - V[i, j]) / dy / 2.0
                    uyS = (U[i, j] - U[i, j-1]) / dy
                    vyS = (V[i, j] - V[i, j-1]) / dy
                    uxE = (U[i+1, j] - U[i, j]) / dx
                    vxE = (V[i+1, j] - V[i, j]) / dx
                    uxW = (U[i, j] - 0.0) / dx / 2.0
                    vxW = (V[i, j] - 0.0) / dx / 2.0

                    uN = 0.0
                    vN = 0.0
                    pN = P[i, j]

                    if V[i, j] + V[i, j-1] < 0:
                        uS = U[i, j]
                        vS = V[i, j]
                        pS = P[i, j]
                    elif V[i, j] + V[i, j-1] > 0:
                        uS = U[i, j-1]
                        vS = V[i, j-1]
                        pS = P[i, j-1]
                    else:
                        uS = 0.0
                        vS = 0.0
                        pS = 0.0

                    if U[i, j] + U[i+1, j] > 0:
                        uE = U[i, j]
                        vE = V[i, j]
                        pE = P[i, j]
                    elif U[i, j] + U[i+1, j] < 0:
                        uE = U[i+1, j]
                        vE = V[i+1, j]
                        pE = P[i+1, j]
                    else:
                        uE = 0.0
                        vE = 0.0
                        pE = 0.0

                    uW = 0.0
                    vW = 0.0
                    pW = P[i, j]
                elif i == 0:
                    uyN = (U[i, j+1] - U[i, j]) / dy
                    vyN = (V[i, j+1] - V[i, j]) / dy
                    uyS = (U[i, j] - U[i, j-1]) / dy
                    vyS = (V[i, j] - V[i, j-1]) / dy
                    uxE = (U[i+1, j] - U[i, j]) / dx
                    vxE = (V[i+1, j] - V[i, j]) / dx
                    uxW = (U[i, j] - 0.0) / dx / 2.0
                    vxW = (V[i, j] - 0.0) / dx / 2.0

                    if V[i, j] + V[i, j+1] > 0:
                        uN = U[i, j]
                        vN = V[i, j]
                        pN = P[i, j]
                    elif V[i, j] + V[i, j+1] < 0:
                        uN = U[i, j+1]
                        vN = V[i, j+1]
                        pN = P[i, j+1]
                    else:
                        uN = 0.0
                        vN = 0.0
                        pN = 0.0

                    if V[i, j] + V[i, j-1] < 0:
                        uS = U[i, j]
                        vS = V[i, j]
                        pS = P[i, j]
                    elif V[i, j] + V[i, j-1] > 0:
                        uS = U[i, j-1]
                        vS = V[i, j-1]
                        pS = P[i, j-1]
                    else:
                        uS = 0.0
                        vS = 0.0
                        pS = 0.0

                    if U[i, j] + U[i+1, j] > 0:
                        uE = U[i, j]
                        vE = V[i, j]
                        pE = P[i, j]
                    elif U[i, j] + U[i+1, j] < 0:
                        uE = U[i+1, j]
                        vE = V[i+1, j]
                        pE = P[i+1, j]
                    else:
                        uE = 0.0
                        vE = 0.0
                        pE = 0.0

                    uW = 0.0
                    vW = 0.0
                    pW = P[i, j]
                elif j == num_y_grid - 1 and i == num_x_grid - 1:
                    uyN = (0.0 - U[i, j]) / dy / 2.0
                    vyN = (0.0 - V[i, j]) / dy / 2.0
                    uyS = (U[i, j] - U[i, j-1]) / dy
                    vyS = (V[i, j] - V[i, j-1]) / dy
                    uxE = (0.0 - U[i, j]) / dx / 2.0
                    vxE = (0.0 - V[i, j]) / dx / 2.0
                    uxW = (U[i, j] - U[i-1, j]) / dx
                    vxW = (V[i, j] - V[i-1, j]) / dx

                    uN = 0.0
                    vN = 0.0
                    pN = P[i, j]

                    if V[i, j] + V[i, j-1] < 0:
                        uS = U[i, j]
                        vS = V[i, j]
                        pS = P[i, j]
                    elif V[i, j] + V[i, j-1] > 0:
                        uS = U[i, j-1]
                        vS = V[i, j-1]
                        pS = P[i, j-1]
                    else:
                        uS = 0.0
                        vS = 0.0
                        pS = 0.0

                    uE = 0.0
                    vE = 0.0
                    pE = P[i, j]

                    if U[i, j] + U[i-1, j] < 0:
                        uW = U[i, j]
                        vW = V[i, j]
                        pW = P[i, j]
                    elif U[i, j] + U[i-1, j] > 0:
                        uW = U[i-1, j]
                        vW = V[i-1, j]
                        pW = P[i-1, j]
                    else:
                        uW = 0.0
                        vW = 0.0
                        pW = 0.0
                elif j == 0 and i == num_x_grid - 1:
                    uyN = (U[i, j+1] - U[i, j]) / dy
                    vyN = (V[i, j+1] - V[i, j]) / dy
                    uyS = (U[i, j] - 0.0) / dy / 2.0
                    vyS = (V[i, j] - 0.0) / dy / 2.0
                    uxE = (0.0 - U[i, j]) / dx / 2.0
                    vxE = (0.0 - V[i, j]) / dx / 2.0
                    uxW = (U[i, j] - U[i-1, j]) / dx
                    vxW = (V[i, j] - V[i-1, j]) / dx

                    if V[i, j] + V[i, j+1] > 0:
                        uN = U[i, j]
                        vN = V[i, j]
                        pN = P[i, j]
                    elif V[i, j] + V[i, j+1] < 0:
                        uN = U[i, j+1]
                        vN = V[i, j+1]
                        pN = P[i, j+1]
                    else:
                        uN = 0.0
                        vN = 0.0
                        pN = 0.0

                    uS = 0.0
                    vS = 0.0
                    pS = P[i, j]

                    uE = 0.0
                    vE = 0.0
                    pE = P[i, j]

                    if U[i, j] + U[i-1, j] < 0:
                        uW = U[i, j]
                        vW = V[i, j]
                        pW = P[i, j]
                    elif U[i, j] + U[i-1, j] > 0:
                        uW = U[i-1, j]
                        vW = V[i-1, j]
                        pW = P[i-1, j]
                    else:
                        uW = 0.0
                        vW = 0.0
                        pW = 0.0
                elif j == 0:
                    uyN = (U[i, j+1] - U[i, j]) / dy
                    vyN = (V[i, j+1] - V[i, j]) / dy
                    uyS = (U[i, j] - 0.0) / dy / 2.0
                    vyS = (V[i, j] - 0.0) / dy / 2.0
                    uxE = (U[i+1, j] - U[i, j]) / dx
                    vxE = (V[i+1, j] - V[i, j]) / dx
                    uxW = (U[i, j] - U[i-1, j]) / dx
                    vxW = (V[i, j] - V[i-1, j]) / dx

                    if V[i, j] + V[i, j+1] > 0:
                        uN = U[i, j]
                        vN = V[i, j]
                        pN = P[i, j]
                    elif V[i, j] + V[i, j+1] < 0:
                        uN = U[i, j+1]
                        vN = V[i, j+1]
                        pN = P[i, j+1]
                    else:
                        uN = 0.0
                        vN = 0.0
                        pN = 0.0

                    uS = 0.0
                    vS = 0.0
                    pS = P[i, j]

                    if U[i, j] + U[i+1, j] > 0:
                        uE = U[i, j]
                        vE = V[i, j]
                        pE = P[i, j]
                    elif U[i, j] + U[i+1, j] < 0:
                        uE = U[i+1, j]
                        vE = V[i+1, j]
                        pE = P[i+1, j]
                    else:
                        uE = 0.0
                        vE = 0.0
                        pE = 0.0

                    if U[i, j] + U[i-1, j] < 0:
                        uW = U[i, j]
                        vW = V[i, j]
                        pW = P[i, j]
                    elif U[i, j] + U[i-1, j] > 0:
                        uW = U[i-1, j]
                        vW = V[i-1, j]
                        pW = P[i-1, j]
                    else:
                        uW = 0.0
                        vW = 0.0
                        pW = 0.0
                elif j == num_y_grid - 1:
                    uyN = (0.0 - U[i, j]) / dy / 2.0
                    vyN = (0.0 - V[i, j]) / dy / 2.0
                    uyS = (U[i, j] - U[i, j-1]) / dy
                    vyS = (V[i, j] - V[i, j-1]) / dy
                    uxE = (U[i+1, j] - U[i, j]) / dx
                    vxE = (V[i+1, j] - V[i, j]) / dx
                    uxW = (U[i, j] - U[i-1, j]) / dx
                    vxW = (V[i, j] - V[i-1, j]) / dx

                    uN = 0.0
                    vN = 0.0
                    pN = P[i, j]

                    if V[i, j] + V[i, j-1] < 0:
                        uS = U[i, j]
                        vS = V[i, j]
                        pS = P[i, j]
                    elif V[i, j] + V[i, j-1] > 0:
                        uS = U[i, j-1]
                        vS = V[i, j-1]
                        pS = P[i, j-1]
                    else:
                        uS = 0.0
                        vS = 0.0
                        pS = 0.0

                    if U[i, j] + U[i+1, j] > 0:
                        uE = U[i, j]
                        vE = V[i, j]
                        pE = P[i, j]
                    elif U[i, j] + U[i+1, j] < 0:
                        uE = U[i+1, j]
                        vE = V[i+1, j]
                        pE = P[i+1, j]
                    else:
                        uE = 0.0
                        vE = 0.0
                        pE = 0.0

                    if U[i, j] + U[i-1, j] < 0:
                        uW = U[i, j]
                        vW = V[i, j]
                        pW = P[i, j]
                    elif U[i, j] + U[i-1, j] > 0:
                        uW = U[i-1, j]
                        vW = V[i-1, j]
                        pW = P[i-1, j]
                    else:
                        uW = 0.0
                        vW = 0.0
                        pW = 0.0
                elif i == num_x_grid - 1:
                    uyN = (U[i, j+1] - U[i, j]) / dy
                    vyN = (V[i, j+1] - V[i, j]) / dy
                    uyS = (U[i, j] - U[i, j-1]) / dy
                    vyS = (V[i, j] - V[i, j-1]) / dy
                    uxE = (0.0 - U[i, j]) / dx / 2.0
                    vxE = (0.0 - V[i, j]) / dx / 2.0
                    uxW = (U[i, j] - U[i-1, j]) / dx
                    vxW = (V[i, j] - V[i-1, j]) / dx

                    if V[i, j] + V[i, j+1] > 0:
                        uN = U[i, j]
                        vN = V[i, j]
                        pN = P[i, j]
                    elif V[i, j] + V[i, j+1] < 0:
                        uN = U[i, j+1]
                        vN = V[i, j+1]
                        pN = P[i, j+1]
                    else:
                        uN = 0.0
                        vN = 0.0
                        pN = 0.0

                    if V[i, j] + V[i, j-1] < 0:
                        uS = U[i, j]
                        vS = V[i, j]
                        pS = P[i, j]
                    elif V[i, j] + V[i, j-1] > 0:
                        uS = U[i, j-1]
                        vS = V[i, j-1]
                        pS = P[i, j-1]
                    else:
                        uS = 0.0
                        vS = 0.0
                        pS = 0.0

                    uE = 0.0
                    vE = 0.0
                    pE = P[i, j]

                    if U[i, j] + U[i-1, j] < 0:
                        uW = U[i, j]
                        vW = V[i, j]
                        pW = P[i, j]
                    elif U[i, j] + U[i-1, j] > 0:
                        uW = U[i-1, j]
                        vW = V[i-1, j]
                        pW = P[i-1, j]
                    else:
                        uW = 0.0
                        vW = 0.0
                        pW = 0.0
                else:
                    uyN = (U[i, j+1] - U[i, j]) / dy
                    vyN = (V[i, j+1] - V[i, j]) / dy
                    uyS = (U[i, j] - U[i, j-1]) / dy
                    vyS = (V[i, j] - V[i, j-1]) / dy
                    uxE = (U[i+1, j] - U[i, j]) / dx
                    vxE = (V[i+1, j] - V[i, j]) / dx
                    uxW = (U[i, j] - U[i-1, j]) / dx
                    vxW = (V[i, j] - V[i-1, j]) / dx

                    if V[i, j] + V[i, j+1] > 0:
                        uN = U[i, j]
                        vN = V[i, j]
                        pN = P[i, j]
                    elif V[i, j] + V[i, j+1] < 0:
                        uN = U[i, j+1]
                        vN = V[i, j+1]
                        pN = P[i, j+1]
                    else:
                        uN = 0.0
                        vN = 0.0
                        pN = 0.0

                    if V[i, j] + V[i, j-1] < 0:
                        uS = U[i, j]
                        vS = V[i, j]
                        pS = P[i, j]
                    elif V[i, j] + V[i, j-1] > 0:
                        uS = U[i, j-1]
                        vS = V[i, j-1]
                        pS = P[i, j-1]
                    else:
                        uS = 0.0
                        vS = 0.0
                        pS = 0.0

                    if U[i, j] + U[i+1, j] > 0:
                        uE = U[i, j]
                        vE = V[i, j]
                        pE = P[i, j]
                    elif U[i, j] + U[i+1, j] < 0:
                        uE = U[i+1, j]
                        vE = V[i+1, j]
                        pE = P[i+1, j]
                    else:
                        uE = 0.0
                        vE = 0.0
                        pE = 0.0

                    if U[i, j] + U[i-1, j] < 0:
                        uW = U[i, j]
                        vW = V[i, j]
                        pW = P[i, j]
                    elif U[i, j] + U[i-1, j] > 0:
                        uW = U[i-1, j]
                        vW = V[i-1, j]
                        pW = P[i-1, j]
                    else:
                        uW = 0.0
                        vW = 0.0
                        pW = 0.0
                
                L0tmp = (vN - vS) * dx + (uE - uW) * dy
                L0 = L0 + L0tmp*L0tmp

                L1tmp = dx * (uN*vN - uS*vS + nu*uyS - nu*uyN) + \
                        dy * (uE*uE - uW*uW + nu*uxW - nu*uxE + pE - pW)
                L1 = L1 + L1tmp*L1tmp

                L2tmp = dx * (vN*vN - vS*vS + nu*vyS - nu*vyN + pN - pS) + \
                        dy * (uE*vE - uW*vW + nu*vxW - nu*vxE)
                L2 = L2 + L2tmp*L2tmp
        
        n = num_x_grid * num_y_grid
        L0, L1, L2 = L0 / n, L1 / n, L2 / n

        tmp = np.linspace(0.0, 1.0, 100, dtype=npdtype)

        x = tf.concat((
            np.linspace(0.0, 1.0, 100, dtype=npdtype),
            np.ones(100, dtype=npdtype),
            np.linspace(0.0, 1.0, 100, dtype=npdtype),
            np.zeros(100, dtype=npdtype),
        ), axis=0)

        y = tf.concat((
            np.ones(100, dtype=npdtype),
            np.linspace(0.0, 1.0, 100, dtype=npdtype),
            np.zeros(100, dtype=npdtype),
            np.linspace(0.0, 1.0, 100, dtype=npdtype),
        ), axis=0)

        u_ref = tf.concat((
            16.0 * np.square(tmp) * np.square(1.0 - tmp),
            np.zeros(300, dtype=npdtype)
        ), axis=0)

        v_ref = tf.zeros(400, dtype=tfdtype)

        var = tf.stack([x, y], axis=1)
        uu, vv, _ = tf.split(self.model(var), 3, 1)

        cost1 = uu - u_ref
        cost2 = vv - v_ref

        return self.reduceFunc(cost1) + self.reduceFunc(cost2), L0 + L1 + L2


    def _buildNet(self):

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self._setInputVariables()
            self._setNetworkVariables()
            self.hypothesis = self.model(self.varIn)
            self.loss_model, self.regularization = self.physics()
            a = self.loss_model
            b = self.regularization
            w = self._penalty
            self.loss = a + b * w
            # self.loss = tf.sqrt(tf.square(a) + tf.square(b))
            # self.loss = tf.square(a) + tf.square(b)
            # self.loss = a * b / (a + b)

    
    def eval(self, feed):

        return self.sess.run(self.hypothesis, feed_dict=feed)


    def train(self, feed, learning_rate=0.01, method='L-BFGS-B'):

        if method == 'L-BFGS-B':
            self._fast_train(feed)
        elif method == 'GradientDescent':
            self._gradient_train(feed, learning_rate)
        elif method == 'SLSQP':
            self._constrained_train(feed)
        return

    
    def _fast_train(self, feed):

        if not hasattr(self, 'optimizer'):
            self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                self.loss,
                var_list=self._weights+self._biases,
                options={
                    #'maxfun': self._num_steps,
                    'maxiter': self._num_steps,
                    'maxls': 50,
                    'maxcor': 50,
                    'disp': False,
                },
            )
            
        if self.debug:
            self.optimizer.minimize(
                session=self.sess, 
                feed_dict=feed,
                fetches=[self.loss, self.loss_model, self.regularization],
                loss_callback = self.callback_debug,
            )
        else:
            self.optimizer.minimize(
                session=self.sess, 
                feed_dict=feed,
                fetches=[self.loss, self.loss_model, self.regularization],
                loss_callback = self.callback,
            )
        return


    def _gradient_train(self, feed, learning_rate=0.01):
        
        if not hasattr(self, 'optimizer'):
            self.optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate,
            ).minimize(self.loss)
        for step in range(self._num_steps + 1):
            c, _ = self.sess.run([self.loss, self.optimizer], feed_dict=feed)
            if step % 100 == 0:
                print("step =", step, "cost =", c)
            if c < self.tol:
                break
        return


    def _constrained_train(self, feed):

        equalities = [self.loss_model]
        if not hasattr(self, 'optimizer'):
            self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                self.regularization, 
                equalities=equalities, 
                method='SLSQP', 
                options={
                    'maxiter': self._num_steps,
                    'disp': False,
                },
            )
        self.optimizer.minimize(session=self.sess, feed_dict=feed)
        
        return


    def callback(self, loss, loss_model, regularization):
        self.convergence.append((loss, loss_model, regularization))

    
    def callback_debug(self, loss, loss_model, regularization):
        self.convergence.append((loss, loss_model, regularization))
        print(
            "loss = ", loss, "loss_model = ", loss_model,
            "regularization = ", regularization)
