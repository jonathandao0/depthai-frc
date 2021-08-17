import numpy as np
from filterpy.kalman import KalmanFilter


# https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html
# https://automaticaddison.com/extended-kalman-filter-ekf-with-python-code-example/#Python_Code_for_the_Extended_Kalman_Filter
class extended_kalman_filter:

    def __init__(self, ):
        self.ekf = KalmanFilter(dim_x=4, dim_z=8)
        # inital measurement state
        self.ekf.x = np.array([0., 0., 0., 0.])
        # x1, y1, z1, t1, x2, y2, z2, t2
        self.ekf.F = np.array([[1., 0., 0., 0., 1., 0., 0., 0.],
                               [0., 1., 0., 0., 0., 1., 0., 0.],
                               [0., 0., 1., 0., 0., 0., 1., 0.],
                               [0., 0., 0., 1., 0., 0., 0., 1.],
                               [1., 0., 0., 0., 1., 0., 0., 0.],
                               [0., 1., 0., 0., 0., 1., 0., 0.],
                               [0., 0., 1., 0., 0., 0., 1., 0.],
                               [0., 0., 0., 1., 0., 0., 0., 1.]])
        # Measurement function
        self.ekf.H = np.array([[1., 1., 1., 1., 1., 1., 1., 1.]])
        # Covariance * uncertainty
        self.ekf.P *= 10.
        # Measurement noise
        self.ekf.R *= np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])

    def update(self, measurement, timestep):

class AlphaBetaFiiter:

    def __init__(self, a=0, b=0):