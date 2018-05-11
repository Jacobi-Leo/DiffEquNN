import numpy as np
import tensorflow as tf

dtype = tf.float32

class Model:

    def __init__(self, name, layers, penalty=1.0, num_steps=100000):

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
        self.sess.run(tf.global_variables_initializer())
        
        
    def _setPhysics(self):
        
        if self.name == 'bvp' or self.name == 'BVP':
            self.physics = self._bvpPhysics
        elif self.name == 'heat' or self.name == 'Heat' or self.name == 'HeatConduction':
            self.physics = self._heatPhysics
        else:
            raise ValueError("Wrong name for model")
        
        
    def _setSmallNumber(self):
        
        a = 1.0 # aspect ratio
        if dtype == tf.float32:
            self.tol = a * np.finfo(np.float32).eps
        elif dtype == tf.float64:
            self.tol = a * np.finfo(np.float64).eps
        else:
            raise ValueError("Wrong data type selection.")


    def _setInputVariables(self):

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.varOut = tf.placeholder(dtype=dtype, shape=[None, self._num_output])
            self.varIn = tf.placeholder(dtype=dtype, shape=[None, self._num_input])
            self.varAux = tf.placeholder(dtype=dtype, shape=[None, self._num_input])


    def _setNetworkVariables(self):

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            
            self._weights = []
            for i in range(len(self._layers) - 1):
                initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
                self._weights.append(tf.get_variable("w"+str(i), 
                                                     shape=[self._layers[i], self._layers[i+1]], 
                                                     initializer=initializer))

            self._biases = []
            for i in range(len(self._layers) - 1):
                initializer = tf.zeros_initializer(dtype=dtype)
                self._biases.append(tf.get_variable("b"+str(i),
                                    shape=[self._layers[i+1]],
                                    initializer=initializer))


    def _neuralNet(self, x):

        # acFun is activation function
        acFun = tf.tanh

        layers = []
        layers.append(x)
        for i in range(len(self._layers) - 1):
            layers.append(acFun(tf.matmul(layers[-1], self._weights[i]) + self._biases[i]))
            
        return layers[-1]


    def model(self, x):

        return self._neuralNet(x)
    
    def _circlePhysics(self):
        
        pass

    
    def _bvpPhysics(self):

        func = lambda x: tf.reduce_mean(tf.square(x))
        self.nu = 0.1
        x = self.varAux
        u = self.model(x)

        ux = tf.gradients(u, x)[0]
        uxx = tf.gradients(ux, x)[0]

        deviation = u * ux - self.nu * uxx

        cost = self.hypothesis - self.varOut

        return func(cost), func(deviation)
    
    
    def _heatPhysics(self):
        
        func = lambda x: tf.reduce_mean(tf.square(x))
        u = self.model(self.varAux)
        D = lambda y: tf.split(tf.gradients(y, self.varAux)[0], 2, 1)
        
        ux, uy = D(u)
        uxx, _ = D(ux)
        _, uyy = D(uy)
        
        deviation = uxx + uyy

        cost = self.hypothesis - self.varOut
       
        return func(cost), func(deviation)


    def _buildNet(self):

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self._setInputVariables()
            self._setNetworkVariables()
            self.hypothesis = self.model(self.varIn)
            self.loss_model, self.regularization = self.physics()
            self.loss = self.loss_model + self.regularization * self._penalty
        return

    
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
        '''Cautious: optimizer is local'''

        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            self.loss,
            options={
                'maxfun': self._num_steps,
                'maxiter': self._num_steps,
                'maxls': 50,
                'maxcor': 50,
            },
        )
        optimizer.minimize(session=self.sess, feed_dict=feed)
        return

    def _gradient_train(self, feed, learning_rate=0.01):
        '''Cautious: optimizer is local'''

        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate).minimize(self.loss)
        for step in range(self._num_steps + 1):
            c, _ = self.sess.run([self.loss, optimizer], feed_dict=feed)
            if step % 100 == 0:
                print("step =", step, "cost =", c)
            if c < self.tol:
                break
        return

    def _constrained_train(self, feed):
        '''Cautious: optimizer is local'''

        equalities = [self.loss_model]
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            self.regularization, 
            equalities=equalities, 
            method='SLSQP', 
            options={
                'maxiter': self._num_steps,
                'ftol': self.tol,
                'disp': False,
            },
        )
        optimizer.minimize(session=self.sess, feed_dict=feed)
        
        return
