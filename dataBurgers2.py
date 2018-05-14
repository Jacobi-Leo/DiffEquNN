import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

fsg = np.float64
N_x = 50
N_t = 50
N_f = 1000

def load():
    '''
    sur : surface_data
    ins : inside_data
    '''

    t_sur, x_sur, U_sur = surfaceData()
    t_ins, x_ins = insideCoordData()
    return (t_sur, x_sur, U_sur, t_ins, x_ins)


def surfaceData():

    t_ini = np.zeros([N_x, 1], dtype=fsg)
    # x_ini = np.linspace(-1.0, 1.0, N_x, dtype=fsg).reshape([N_x, 1])
    x_ini = np.random.uniform(-1.0, 1.0, (N_x, 1))
    u_ini = -np.sin(x_ini * np.pi)

    # t_b1 = np.linspace(0.0, 1.0, N_t, dtype=fsg).reshape([N_t, 1])
    t_b1 = np.random.uniform(0.0, 1.0, (N_t, 1))
    x_b1 = -1.0 * np.ones([N_t, 1], dtype=fsg)
    u_b1 = np.zeros([N_t, 1], dtype=fsg)

    # t_b2 = t_b1
    t_b2 = np.random.uniform(0.0, 1.0, (N_t, 1))
    x_b2 = -1.0 * x_b1
    u_b2 = np.zeros([N_t, 1], dtype=fsg)

    p = lambda a, b, c: np.concatenate((np.concatenate((a, b), axis=0), c), axis=0)

    return (p(t_ini, t_b1, t_b2),
            p(x_ini, x_b1, x_b2),
            p(u_ini, u_b1, u_b2))


def insideCoordData():

    return np.random.uniform(0.0, 1.0, (N_f, 1)), np.random.uniform(-1.0, 1.0, (N_f, 1))


def testData(time=0.5):

    return (time * np.ones((50, 1), dtype=fsg), np.linspace(-1, 1, 50, dtype=fsg).reshape(50, 1))


u = 1.19
l = -1.19
def normalize(x):
    ## This is for sigmoid function
    # return (x - l) / (u - l)

    ## This is for tanh, but not work
    # return (x - (u + l) * 0.5) / u
    
    return x

def restore(x):
    ## This is for sigmoid function
    # return l + (u - l) * x

    ## This is for tanh, but not work
    # return u * x + 0.5 * (u + l)

    return x

def show(u):
    return


def plot(x, y):
    return
