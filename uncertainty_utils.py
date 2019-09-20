import numpy as np
import controlpy
import matplotlib.pyplot as plt
from matplotlib import animation

def regression(x, u, lamb):
    """Estimates linear system dynamics
    x, u: date used in the regression
    lamb: regularization coefficient
    """

    # Want to solve W^* = argmin sum_i ||W^T z_i - y_i ||_2^2 + lamb ||W||_F,
    # with z_i = [x_i u_i] and W \in R^{n + d} x n
    N = x.shape[1]
    M = u.shape[1]

    Y = x[1:x.shape[0], :]
    X = np.hstack((x[0:(x.shape[0] - 1), :], u[0:(x.shape[0] - 1), :]))

    Q = np.linalg.inv(np.dot(X.T, X) + lamb * np.eye(X.shape[1]))
    b = np.dot(X.T, Y)
    W = np.dot(Q, b)
    A = W.T[:, 0:N]
    B = W.T[:, N:N+M]

    ErrorMatrix = np.dot(X, W) - Y

    return A, B, ErrorMatrix

class KF:

    def __init__(self, P, process_noise, measurement_noise, A, B, C, x_init):
        self.P = P
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.A = A
        self.B = B
        self.C = C
        self.x = x_init

    def update_estimate(self, y, u):
        # predict
        x_priori = self.A @ self.x + self.B @ u
        P_priori = self.A @ self.P @ self.A.T + self.process_noise

        # update
        S = self.C @ P_priori @ self.C.T + self.measurement_noise
        K = P_priori @ self.C.T @ np.linalg.inv(S)
        P_post = (np.eye(self.P.shape[0]) - K @ self.C) @ P_priori
        x_post = x_priori + K @ (y - self.C @ x_priori)

        #store results
        self.x = x_post
        self.P = P_post

        return self.x

def sim_traj(A, B, C, K, Q, R, process_noise, measurement_noise, kf, x_init, N=30, input_limits=np.array([-1e9, 1e9])):
    x_traj = [x_init]
    x_est_traj = [np.copy(kf.x)]
    u_traj = []

    h = lambda x,u: x.T @ Q @ x + u.T @ R @ u
    est_cost = 0
    true_cost = 0

    for _ in range(N):
        u = np.minimum(np.maximum(input_limits[0], - K @ x_est_traj[-1]), input_limits[1])
        x_next = A @ x_traj[-1] + B @ u + np.random.multivariate_normal(np.zeros(x_init.shape[0]), process_noise)
        y_next = C @ x_next + np.random.multivariate_normal(np.zeros(C.shape[0]), measurement_noise)
        x_est_next = kf.update_estimate(y_next, u)
        est_cost += h(x_est_traj[-1], u)
        true_cost += h(x_traj[-1], u)

        x_traj.append(x_next)
        x_est_traj.append(x_est_next)
        u_traj.append(u)

    return np.array(x_traj), np.array(x_est_traj), np.array(u_traj), est_cost, true_cost

def get_LQG(A_est, B_est, C, Q, R, P_init, process_noise, measurement_noise, x_init_est):
    K, _, _= controlpy.synthesis.controller_lqr_discrete_time(A_est, B_est, Q, R)
    kf = KF(P_init, process_noise, measurement_noise, A_est, B_est, C, x_init_est)
    return K, kf

def animate_trajectories(data, file):
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(xlim=(-3, 15), ylim=(-7, 5))
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    line1, = ax.plot([], [], "k", label="estimated")
    line2, = ax.plot([], [], "r", label="true")
    ax.legend(loc="upper right")

    # initialization function: plot the background of each frame
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1,line2,

    # animation function.  This is called sequentially
    def animate(i):
        line1.set_data(data["est_traj"][i][:,0], data["est_traj"][i][:,1])
        line2.set_data(data["true_traj"][i][:,0], data["true_traj"][i][:,1])
        ax.set_title("trajectory {}".format(i))
        return line1, line2

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(data["est_traj"]), blit=True)
    anim.save(file, fps=5, extra_args=['-vcodec', 'libx264'])






