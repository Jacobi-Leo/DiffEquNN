import numpy as np

dtype = np.float64

def my_random(start, stop, num):
    step = (stop - start) / num
    lst = []
    for i in range(num):
        lst.append(np.random.uniform(start+i*step, start+(i+1)*step, 1)[0])
    return np.array(lst)
    #return np.linspace(start, stop, num, dtype=dtype)

def getBody(N, random=True):

    if random:
        t_data = np.random.uniform(0.0, 1.0, N)
        x_data = np.random.uniform(-1.0, 1.0, N)
    else:
        raise ValueError
        # the following two lines of code need bug-fixing.
        t_data = np.linspace(0.0, 1.0, N, dtype=dtype)
        x_data = np.linspace(-1.0, 1.0, N, dtype=dtype)

    return t_data, x_data

def getBoundary(N, random=True):

    if random:
        t_data_1 = my_random(0.0, 1.0, N)
        x_data_1 = 1.0 * np.ones(N, dtype=dtype)
        u_data_1 = np.zeros(N, dtype=dtype)

        t_data_2 = np.zeros(N, dtype=dtype)
        x_data_2 = my_random(-1.0, 1.0, N)
        u_data_2 = -np.sin(np.pi * x_data_2)

        t_data_3 = my_random(0.0, 1.0, N)
        #t_data_3 = t_data_1
        x_data_3 = -1.0 * np.ones(N, dtype=dtype)
        u_data_3 = np.zeros(N, dtype=dtype)
    else:
        t_data_1 = np.linspace(0.0, 1.0, N, dtype=dtype)
        x_data_1 = 1.0 * np.ones(N, dtype=dtype)
        u_data_1 = np.zeros(N, dtype=dtype)

        t_data_2 = np.zeros(N, dtype=dtype)
        x_data_2 = np.linspace(-1.0, 1.0, N, dtype=dtype)
        u_data_2 = -np.sin(np.pi*x_data_2)

        t_data_3 = np.linspace(0.0, 1.0, N, dtype=dtype)
        x_data_3 = -1.0 * np.ones(N, dtype=dtype)
        u_data_3 = np.zeros(N, dtype=dtype)

    return (
        np.concatenate([t_data_1, t_data_2, t_data_3]),
        np.concatenate([x_data_1, x_data_2, x_data_3]),
        np.concatenate([u_data_1, u_data_2, u_data_3]),
    )