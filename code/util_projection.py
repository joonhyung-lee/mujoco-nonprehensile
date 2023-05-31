import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.spatial import distance
import math
import sys
sys.path.append('..')

# mujoco intrinsics

# Camera intrinsic
img_height = 800  # depth_img.shape[0]
img_width = 1200  # depth_img.shape[1]
fovy = 45
focal_scaling = 0.5*img_height/np.tan(fovy*np.pi/360)
cam_matrix = np.array(((focal_scaling, 0, img_width/2),
                       (0, focal_scaling, img_height/2),
                       (0, 0, 1)))


def get_pointcloud(depth, cam_matrix):
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - cam_matrix[0, 2]) * (depth / cam_matrix[0, 0])
    py = (py - cam_matrix[1, 2]) * (depth / cam_matrix[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    return points


def get_heightmap(points, bounds, pixel_size):
    width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
    height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
    heightmap = np.zeros((height, width), dtype=np.float32)
    # Filter out 3D points that are outside of the predefined bounds.
    ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (
        points[Ellipsis, 0] < bounds[0, 1])  # Range of the X
    iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (
        points[Ellipsis, 1] < bounds[1, 1])  # Range of the Y
    iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (
        points[Ellipsis, 2] < bounds[2, 1])  # Range of the Z
    valid = ix & iy & iz
    points = points[valid]
    points.shape
    # Sort 3D points by z-value, which works with array assignment to simulate
    # z-buffering for rendering the heightmap image.
    iz = np.argsort(points[:, -1])
    points = points[iz]
    px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
    py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)
    heightmap[py, px] = points[:, 2] - bounds[2, 0]  # Depth value
    return heightmap


def transform_pointcloud(points, transform):
    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(points.copy(), padding,
                            'constant', constant_values=1)
    for i in range(3):
        points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
    return points

def ortho_projection(image, cam_matrix):
    width = 800
    height = 1200
    # Preprocessing 
    image = image.reshape((width, height)) # Image shape [Width x Height]
    image = np.flipud(image) # flip image
    # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Orthographic projection 
    points = get_pointcloud(image*1, cam_matrix)
    # Homogeneous Matrix 
    # rot_mat = Rotation_X(-np.deg2rad(40))    
    # trans_mat = Translation(x=-0., y=0.3, z=1.25-0.79)
    # homo_mat  = HT_matrix(Rotation=rot_mat, Position=trans_mat) 
    # homo_inv_mat = np.linalg.inv(homo_mat)

    # homo_inv_mat =np.array([[ 1.,          0.,          0.,          0.        ],
    #                         [ 0.,          0.76604444, -0.64278761,  0.06586897],
    #                         [ 0.,          0.64278761,  0.76604444, -0.54521673],
    #                         [ 0.,          0.,          0.,          1.        ]])

    homo_inv_mat = np.array([[0., -1., -0.,  0.],
                            [-0.71, -0., -0.71,  1.13],
                             [0.71,  0., -0.71,  0.76],
                             [0.,  0.,  0.,  1.]])

    new_points = transform_pointcloud(points, homo_inv_mat)
    new_bounds = np.array([[-0.45,0.45], 
                        [-1.05,-0.35],
                        [-1.,2]])
    ortho_image = get_heightmap(new_points, new_bounds, pixel_size=0.00455)/2

    return ortho_image 