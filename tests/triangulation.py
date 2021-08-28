from __future__ import print_function
import numpy as np
import cv2
import time

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


def triangulate_nviews(P, ip):
    """
    Triangulate a point visible in n camera views.
    P is a list of camera projection matrices.
    ip is a list of homogenised image points. eg [ [x, y, 1], [x, y, 1] ], OR,
    ip is a 2d array - shape nx3 - [ [x, y, 1], [x, y, 1] ]
    len of ip must be the same as len of P
    """
    if not len(ip) == len(P):
        raise ValueError('Number of points and number of cameras not equal.')
    n = len(P)
    M = np.zeros([3*n, 4+n])
    for i, (x, p) in enumerate(zip(ip, P)):
        M[3*i:3*i+3, :4] = p
        M[3*i:3*i+3, 4+i] = -x
    V = np.linalg.svd(M)[-1]
    X = V[-1, :4]
    return X / X[3]


def triangulate_points(P1, P2, x1, x2):
    """
    Two-view triangulation of points in
    x1,x2 (nx3 homog. coordinates).
    Similar to openCV triangulatePoints.
    """
    if not len(x2) == len(x1):
        raise ValueError("Number of points don't match.")
    X = [triangulate_nviews([P1, P2], [x[0], x[1]]) for x in zip(x1, x2)]
    return np.array(X)


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

# 3 camera projection matrices
P1 = np.array([[5.010e+03, 0.000e+00, 3.600e+02, 0.000e+00],
               [0.000e+00, 5.010e+03, 6.400e+02, 0.000e+00],
               [0.000e+00, 0.000e+00, 1.000e+00, 0.000e+00]])

P2 = np.array([[5.037e+03, -9.611e+01, -1.756e+03, 4.284e+03],
               [2.148e+02,  5.354e+03,  1.918e+02, 8.945e+02],
               [3.925e-01,  7.092e-02,  9.169e-01, 4.930e-01]])

P3 = np.array([[5.217e+03,  2.246e+02,  2.366e+03, -3.799e+03],
               [-5.734e+02,  5.669e+03,  8.233e+02, -2.567e+02],
               [-3.522e-01, -5.839e-02,  9.340e-01,  6.459e-01]])

# 3 corresponding image points - nx2 arrays, n=1
x1 = np.array([[274.128, 624.409]])
x2 = np.array([[239.571, 533.568]])
x3 = np.array([[297.574, 549.260]])

# 3 corresponding homogeneous image points - nx3 arrays, n=1
x1h = np.array([[274.128, 624.409, 1.0]])
x2h = np.array([[239.571, 533.568, 1.0]])
x3h = np.array([[297.574, 549.260, 1.0]])

# 3 corresponding homogeneous image points - nx3 arrays, n=2
x1h2 = np.array([[274.129, 624.409, 1.0], [322.527, 624.869, 1.0]])
x2h2 = np.array([[239.572, 533.568, 1.0], [284.507, 534.572, 1.0]])
x3h2 = np.array([[297.575, 549.260, 1.0], [338.942, 546.567, 1.0]])


# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------

print('Triangulate 3d points - units in meters')
# triangulatePoints requires 2xn arrays, so transpose the points
p = cv2.triangulatePoints(P1, P2, x1.T, x2.T)
# however, homgeneous point is returned
p /= p[3]
print('Projected point from openCV:',  p.T)

p = triangulate_nviews([P1, P2], [x1h, x2h])
print('Projected point from 2 camera views:',  p)

p = triangulate_nviews([P1, P2, P3], [x1h, x2h, x3h])
print('Projected point from 3 camera views:',  p)

# cv2 two image points - not homgeneous on input
p = cv2.triangulatePoints(P1, P2, x1h2[:, :2].T, x2h2[:, :2].T)
p /= p[3]
print('Projected points from openCV:\n', p.T)

p = triangulate_points(P1, P2, x1h2, x2h2)
print('Projected point from code:\n',  p)

# -----------------------------------------------------------------------------
# Timing
# -----------------------------------------------------------------------------

t1 = time.time()
for i in range(10000):
    p = cv2.triangulatePoints(P1, P2, x1.T, x2.T)
    p /= p[3]
t2 = time.time()
print('Elapsed time cv2:', t2-t1)

t1 = time.time()
for i in range(10000):
    p = triangulate_nviews([P1, P2], [x1h, x2h])
t2 = time.time()
print('Elapsed time sfm:', t2-t1)