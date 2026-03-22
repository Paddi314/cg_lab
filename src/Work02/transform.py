# src/Work02/transform.py
# Created: 2026-03-22
# Computer Graphics Lab 02: Rotation and Transformation
# Implementation of MVP (Model-View-Projection) transformation matrices

import taichi as ti
import math

@ti.func
def get_model_matrix(angle: float) -> ti.Matrix:
    """
    Return a 4x4 homogeneous rotation matrix around Z-axis.
    
    Args:
        angle: Rotation angle in degrees (clockwise when looking from positive Z)
    
    Returns:
        4x4 rotation matrix
    """
    # Convert degrees to radians
    rad = angle * math.pi / 180.0
    
    # Rotation matrix around Z-axis
    # [cosθ  -sinθ  0  0]
    # [sinθ   cosθ  0  0]
    # [0      0     1  0]
    # [0      0     0  1]
    cos_a = ti.cos(rad)
    sin_a = ti.sin(rad)
    
    return ti.Matrix([
        [cos_a, -sin_a, 0.0, 0.0],
        [sin_a,  cos_a, 0.0, 0.0],
        [0.0,     0.0,  1.0, 0.0],
        [0.0,     0.0,  0.0, 1.0]
    ])

@ti.func
def get_view_matrix(eye_pos) -> ti.Matrix:
    """
    Return a 4x4 homogeneous view transformation matrix.
    
    Args:
        eye_pos: Camera position in world coordinates (3D vector)
    
    Returns:
        4x4 view matrix that translates camera to origin
    """
    # View matrix translates the world so that camera is at origin
    # Since camera is at eye_pos looking at -Z direction,
    # we need to translate by -eye_pos
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix(eye_fov: float, aspect_ratio: float, zNear: float, zFar: float) -> ti.Matrix:
    """
    Return a 4x4 homogeneous perspective projection matrix.
    
    Args:
        eye_fov: Field of view in Y direction (in degrees)
        aspect_ratio: width / height
        zNear: distance to near clipping plane (positive value)
        zFar: distance to far clipping plane (positive value)
    
    Returns:
        4x4 perspective projection matrix
    """
    # Convert fov to radians
    fov_rad = eye_fov * math.pi / 180.0
    
    # Perspective to orthographic matrix (M_persp->ortho)
    # In right-handed coordinate system with camera looking at -Z:
    # n = -zNear, f = -zFar
    n = -zNear
    f = -zFar
    
    # Calculate top, bottom, left, right boundaries
    t = ti.tan(fov_rad / 2.0) * abs(n)  # top = tan(fov/2) * |n|
    b = -t                              # bottom = -top
    r = aspect_ratio * t                # right = aspect * top
    l = -r                              # left = -right
    
    # Orthographic projection matrix (M_ortho)
    # First scale to [-1, 1] cube, then translate to origin
    ortho_scale = ti.Matrix([
        [2.0/(r-l), 0.0,        0.0,        0.0],
        [0.0,       2.0/(t-b),  0.0,        0.0],
        [0.0,       0.0,        2.0/(n-f),  0.0],
        [0.0,       0.0,        0.0,        1.0]
    ])
    
    ortho_translate = ti.Matrix([
        [1.0, 0.0, 0.0, -(r+l)/2.0],
        [0.0, 1.0, 0.0, -(t+b)/2.0],
        [0.0, 0.0, 1.0, -(n+f)/2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    M_ortho = ortho_scale @ ortho_translate
    
    # Perspective to orthographic matrix (M_persp->ortho)
    M_persp_to_ortho = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n+f, -n*f],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    # Final projection matrix: M_proj = M_ortho @ M_persp_to_ortho
    return M_ortho @ M_persp_to_ortho