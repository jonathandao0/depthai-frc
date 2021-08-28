import cv2
import numpy as np

from common.camera_info import OAK_L_CAMERA_LEFT, OAK_L_CAMERA_RIGHT

LR_Translation = np.array(list(OAK_L_CAMERA_LEFT['extrinsics']['translation'].values()))
LR_Rotation = np.array(list(OAK_L_CAMERA_LEFT['extrinsics']['rotationMatrix']))

L_DISTORTION = np.array(OAK_L_CAMERA_LEFT['distortionCoeff'])
L_INTRINSIC = np.array(OAK_L_CAMERA_LEFT['intrinsicMatrix'])

R_DISTORTION = np.array(OAK_L_CAMERA_RIGHT['distortionCoeff'])
R_INTRINSIC = np.array(OAK_L_CAMERA_RIGHT['intrinsicMatrix'])

R1, R2, L_Projection, R_Projection, Q, L_ROI, R_ROI = cv2.stereoRectify(L_INTRINSIC, L_DISTORTION, R_INTRINSIC, R_DISTORTION, (1280, 720), LR_Rotation, LR_Translation)

pointsL = np.array([( 700, 250),
           ( 200, 300),
           ( 600, 350),
           ( 400, 400),
           ( 500, 450),
           ( 600, 500),
           ( 700, 550),
           ( 800, 600),
           ( 150, 650),
           (1000, 700)])

pointsR = np.array([( 690, 250),
           ( 180, 300),
           ( 590, 350),
           ( 385, 400),
           ( 495, 450),
           ( 575, 500),
           ( 691, 550),
           ( 782, 600),
           ( 120, 650),
           ( 960, 700)])

points3D = cv2.triangulatePoints(L_Projection, R_Projection, pointsL.T, pointsR.T)

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

print('Triangulate 3d points - units in meters')
# triangulatePoints requires 2xn arrays, so transpose the points
p = cv2.triangulatePoints(L_Projection, R_Projection, x1.T, x2.T)
# however, homgeneous point is returned
p /= p[3]
print('Projected point from openCV:',  p.T)

p = triangulate_nviews([L_Projection, R_Projection], [x1h, x2h])
print('Projected point from 2 camera views:',  p)

# cv2 two image points - not homgeneous on input
p = cv2.triangulatePoints(L_Projection, R_Projection, x1h2[:, :2].T, x2h2[:, :2].T)
p /= p[3]
print('Projected points from openCV:\n', p.T)

p = triangulate_points(L_Projection, R_Projection, x1h2, x2h2)
print('Projected point from code:\n',  p)