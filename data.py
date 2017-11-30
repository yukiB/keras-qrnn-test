from scipy.integrate import odeint, simps
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import random

# 乱数の種として、0を渡す
random.seed(0)
random_factor = 0.05
# sin曲線 1周期あたりのステップ数
steps_per_cycle = 100
# 生成する周期数
number_of_cycles = 1000


def get_n_sequence_dataset_and_m_peripd_later_data(data, n_prev=100, m=1):
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data) - n_prev * m):

        docX.append(data.iloc[i:i + n_prev].as_matrix())
        docY.append(data.iloc[i + n_prev + (m - 1)].as_matrix())

    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY


def create_train_data_and_test_data(df, test_size=0.1, n_prev=100, m=1):
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = get_n_sequence_dataset_and_m_peripd_later_data(df.iloc[0:ntrn], n_prev, m)
    X_test, y_test = get_n_sequence_dataset_and_m_peripd_later_data(df.iloc[ntrn:], n_prev, m)

    return (X_train, y_train), (X_test, y_test)


def duffing(var, t, gamma, a, b, F0, omega, delta):
    """
    var = [x, p]
    dx/dt = p
    dp/dt = -gamma*p + 2*a*x - 4*b*x**3 + F0*cos(omega*t + delta)
    """
    x_dot = var[1]
    p_dot = -gamma * var[1] + 2 * a * var[0] - 4 * b * var[0]**3 + F0 * np.cos(omega * t + delta)

    return np.array([x_dot, p_dot])


def chaos_logistic():
    x = [0.8]
    for i in range(10000):
        x.append(3.7 * x[-1] * (1 - x[-1]))
    return x


def chaos_duffing():
    F0, gamma, omega, delta = 10, 0.1, np.pi / 3, 1.5 * np.pi
    a, b = 1 / 4, 1 / 2
    var, var_lin = [[0.1, 1]] * 2

    # timescale
    t = np.arange(0, 20000, 2 * np.pi / omega)
    t_lin = np.linspace(0, 1000, 10000)

    # solve
    var = odeint(duffing, var, t, args=(gamma, a, b, F0, omega, delta))
    var_lin = odeint(duffing, var_lin, t_lin, args=(gamma, a, b, F0, omega, delta))

    x_lin, p_lin = var_lin.T[0], var_lin.T[1]
    return x_lin, t_lin, p_lin


def create_sin_data():
    # 区間 -1.0 〜 +1.0 の 一様乱数 付き の sin曲線
    df = pd.DataFrame(np.arange(steps_per_cycle * number_of_cycles + 1), columns=["t"])
    df["data"] = df.t.apply(lambda x: math.sin(x * (2 * math.pi / steps_per_cycle)
                                               + random.uniform(-1.0, +1.0) * random_factor) + math.cos(x * (2 * math.pi / (steps_per_cycle / 3))
                                                                                                        + random.uniform(-1.0, +1.0) * random_factor) + 2 * math.cos(x * (2 * math.pi / (steps_per_cycle * 5))
                                                                                                                                                                     + random.uniform(-1.0, +1.0) * random_factor))
    # 2周期分、描画
    return df


def create_chaos_data(data_type="duffing"):
    if data_type == "duffing":
        x_lin, t_lin, p_lin = chaos_duffing()
    else:
        x_lin = chaos_logistic()
    print(len(x_lin))
    df = pd.DataFrame(np.arange(len(x_lin)), columns=["t"])
    df["data"] = df.t.apply(lambda x: x_lin[x])
    df["data"].head(500).plot()
    return df
