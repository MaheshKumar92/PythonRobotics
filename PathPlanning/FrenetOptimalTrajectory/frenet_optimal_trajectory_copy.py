import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import sys
import pathlib
from scipy.spatial import KDTree
from joblib import Parallel, delayed

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from QuinticPolynomialsPlanner.quintic_polynomials_planner import QuinticPolynomial
from CubicSpline import cubic_spline_planner

SIM_LOOP = 500

# Parameter
MAX_SPEED = 50.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 2.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 1.0  # maximum curvature [1/m]
MAX_ROAD_WIDTH = 7.0  # maximum road width [m]
D_ROAD_W = 1.0  # road width sampling length [m]
DT = 0.2  # time tick [s]
MAX_T = 5.0  # max prediction time [m]
MIN_T = 4.0  # min prediction time [m]
TARGET_SPEED = 30.0 / 3.6  # target speed [m/s]
D_T_S = 5.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target speed
ROBOT_RADIUS = 2.0  # robot radius [m]

# cost weights
K_J = 0.1
K_T = 0.1
K_D = 1.0
K_LAT = 1.0
K_LON = 1.0

show_animation = True


class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt


class FrenetPath:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []


def calc_frenet_paths_parallel(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0):
    results = Parallel(n_jobs=-1)(delayed(generate_path)(di, Ti, c_speed, c_accel, c_d, c_d_d, c_d_dd, s0)
                                  for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W)
                                  for Ti in np.arange(MIN_T, MAX_T, DT))
    frenet_paths = [item for sublist in results for item in sublist]
    return frenet_paths

def generate_path(di, Ti, c_speed, c_accel, c_d, c_d_d, c_d_dd, s0):
    paths = []
    lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
    t_values = np.arange(0.0, Ti, DT)
    d_values = lat_qp.calc_point(t_values)
    d_d_values = lat_qp.calc_first_derivative(t_values)
    d_dd_values = lat_qp.calc_second_derivative(t_values)
    d_ddd_values = lat_qp.calc_third_derivative(t_values)
    for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE, TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
        fp = FrenetPath()
        lon_qp = QuarticPolynomial(s0, c_speed, c_accel, tv, 0.0, Ti)
        fp.t = t_values
        fp.d = d_values
        fp.d_d = d_d_values
        fp.d_dd = d_dd_values
        fp.d_ddd = d_ddd_values
        fp.s = lon_qp.calc_point(t_values)
        fp.s_d = lon_qp.calc_first_derivative(t_values)
        fp.s_dd = lon_qp.calc_second_derivative(t_values)
        fp.s_ddd = lon_qp.calc_third_derivative(t_values)
        Jp = np.sum(np.power(fp.d_ddd, 2))  # square of jerk
        Js = np.sum(np.power(fp.s_ddd, 2))  # square of jerk
        ds = (TARGET_SPEED - fp.s_d[-1]) ** 2  # square of diff from target speed
        fp.cd = K_J * Jp + K_T * Ti + K_D * fp.d[-1] ** 2
        fp.cv = K_J * Js + K_T * Ti + K_D * ds
        fp.cf = K_LAT * fp.cd + K_LON * fp.cv
        paths.append(fp)
    return paths


def calc_global_paths(fplist, csp):
    for fp in fplist:
        # calc global positions
        positions = np.array([csp.calc_position(s) for s in fp.s], dtype=float)
        valid_positions = positions[~np.any(np.isnan(positions), axis=1)]
        for i, (ix, iy) in enumerate(valid_positions):
            i_yaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
            fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        # calc yaw and ds
        deltas = np.diff(valid_positions, axis=0)
        fp.yaw = np.arctan2(deltas[:, 1], deltas[:, 0]).tolist()
        fp.ds = np.hypot(deltas[:, 0], deltas[:, 1]).tolist()

        if len(fp.yaw) > 0:
            fp.yaw.append(fp.yaw[-1])
            fp.ds.append(fp.ds[-1])

        # calc curvature
        fp.c = np.diff(fp.yaw) / fp.ds[:-1]

    return fplist


def check_collision(fp, ob_tree):
    distances, _ = ob_tree.query(list(zip(fp.x, fp.y)))
    return all(d > ROBOT_RADIUS for d in distances)


def check_paths(fplist, ob):
    ob_tree = KDTree(ob)
    okind = []
    for i, _ in enumerate(fplist):
        if all(v <= MAX_SPEED for v in fplist[i].s_d) and \
                all(abs(a) <= MAX_ACCEL for a in fplist[i].s_dd) and \
                all(abs(c) <= MAX_CURVATURE for c in fplist[i].c):
            if check_collision(fplist[i], ob_tree):
                okind.append(i)
    return [fplist[i] for i in okind]


def frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, ob):
    fplist = calc_frenet_paths_parallel(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0)
    fplist = calc_global_paths(fplist, csp)
    fplist = check_paths(fplist, ob)

    # find minimum cost path
    min_cost = float("inf")
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    return best_path


def generate_target_course(x, y):
    csp = cubic_spline_planner.CubicSpline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp


def main():
    print(__file__ + " start!!")

    # way points
    wx = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    wy = [0.0, -6.0, 5.0, 6.5, 0.0, -3.0, 5.0]

    # obstacle positions
    ob = np.array([
        [20.0, 10.0],
        [30.0, 6.0],
        [30.0, 8.0],
        [35.0, 8.0],
        [50.0, 3.0]
    ])

    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)

    c_speed = 10.0 / 3.6  # current speed [m/s]
    c_accel = 0.1  # current acceleration [m/ss]
    c_d = 2.0  # current lateral position [m]
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current lateral acceleration [m/ss]
    s0 = 0.0  # current course position

    for i in range(SIM_LOOP):
        path = frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, ob)

        s0 = path.s[1]
        c_d = path.d[1]
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]
        c_speed = path.s_d[1]
        c_accel = path.s_dd[1]

        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0:
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            plt.plot(tx, ty, "-r", label="course")
            plt.plot(ob[:, 0], ob[:, 1], "ok")
            plt.plot(path.x[1:], path.y[1:], "-g", label="trajectory")
            plt.plot(path.x[1], path.y[1], "vc")
            plt.xlim(path.x[1] - 10, path.x[1] + 10)
            plt.ylim(path.y[1] - 10, path.y[1] + 10)
            plt.title("v[km/h]:" + str(c_speed * 3.6)[0:4])
            plt.grid(True)
            plt.pause(0.0001)

    print("Finish")
    if show_animation:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()


if __name__ == '__main__':
    main()
