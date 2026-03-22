# src/Work02/main.py
# Created: 2026-03-22
# Computer Graphics Lab 02: Rotation and Transformation
# Main program to demonstrate MVP transformation with rotating triangle

import taichi as ti
import math
from transform import get_model_matrix, get_view_matrix, get_projection_matrix

# Initialize Taichi
ti.init(arch=ti.gpu)

# Window settings
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 700
WINDOW_TITLE = "Experiment 2: Rotation and Transformation"

# Triangle vertices in 3D space (as given in the experiment)
# v0: (2.0, 0.0, -2.0)
# v1: (0.0, 2.0, -2.0)
# v2: (-2.0, 0.0, -2.0)
triangle_vertices = ti.Vector.field(3, dtype=float, shape=3)
triangle_vertices[0] = ti.Vector([2.0, 0.0, -2.0])
triangle_vertices[1] = ti.Vector([0.0, 2.0, -2.0])
triangle_vertices[2] = ti.Vector([-2.0, 0.0, -2.0])

# Camera parameters
camera_pos = ti.Vector([0.0, 0.0, 5.0])  # Camera at (0, 0, 5) looking at -Z
fov = 45.0  # Field of view in degrees
aspect_ratio = WINDOW_WIDTH / WINDOW_HEIGHT
z_near = 0.1
z_far = 50.0

# Rotation angle (in degrees)
rotation_angle = ti.field(float, shape=())

@ti.kernel
def update_vertices(angle: float, vertices: ti.template(), screen_coords: ti.template()):
    """
    Apply MVP transformation to triangle vertices and convert to screen coordinates.
    
    Args:
        angle: Rotation angle in degrees
        vertices: Input 3D vertices
        screen_coords: Output 2D screen coordinates
    """
    # Get transformation matrices
    M_model = get_model_matrix(angle)
    M_view = get_view_matrix(camera_pos)
    M_proj = get_projection_matrix(fov, aspect_ratio, z_near, z_far)
    
    # Combined MVP matrix: M_proj @ M_view @ M_model
    MVP = M_proj @ M_view @ M_model
    
    for i in range(vertices.shape[0]):
        # Get vertex in homogeneous coordinates (x, y, z, 1)
        v = ti.Vector([vertices[i][0], vertices[i][1], vertices[i][2], 1.0])
        
        # Apply MVP transformation
        v_transformed = MVP @ v
        
        # Perspective division (divide by w)
        v_ndc = ti.Vector([0.0, 0.0, 0.0, 1.0])
        if abs(v_transformed[3]) > 1e-6:  # Avoid division by zero
            v_ndc = v_transformed / v_transformed[3]
        else:
            v_ndc = v_transformed
        
        # Convert from NDC [-1, 1] to screen coordinates [0, 1]
        # Taichi GUI uses coordinate system where (0, 0) is bottom-left
        x_screen = (v_ndc[0] + 1.0) * 0.5
        y_screen = (v_ndc[1] + 1.0) * 0.5
        
        # Clamp to [0, 1] range
        x_screen = ti.max(0.0, ti.min(1.0, x_screen))
        y_screen = ti.max(0.0, ti.min(1.0, y_screen))
        
        screen_coords[i] = ti.Vector([x_screen, y_screen])

def run():
    """Main program loop."""
    print("Experiment 2: Rotation and Transformation")
    print("Press A/D to rotate triangle, ESC to exit")
    
    # Initialize GUI
    gui = ti.GUI(WINDOW_TITLE, res=(WINDOW_WIDTH, WINDOW_HEIGHT))
    
    # Field for screen coordinates
    screen_coords = ti.Vector.field(2, dtype=float, shape=3)
    
    # Initial rotation angle
    rotation_angle[None] = 0.0
    
    # Main loop
    while gui.running:
        # Process all pending events
        gui.get_event()
        
        # Check for ESC key to exit
        if gui.is_pressed(ti.GUI.ESCAPE):
            break
        
        # Check for A/D key presses
        if gui.is_pressed('a'):
            rotation_angle[None] += 1.0  # Rotate clockwise
        if gui.is_pressed('d'):
            rotation_angle[None] -= 1.0  # Rotate counter-clockwise
        
        # Update vertices with current rotation angle
        update_vertices(rotation_angle[None], triangle_vertices, screen_coords)
        
        # Clear screen
        gui.clear(0x112F41)  # Dark blue background
        
        # Draw triangle edges (wireframe)
        # Edge 0-1
        gui.line(screen_coords[0], screen_coords[1], color=0xFF6B6B, radius=2)
        # Edge 1-2
        gui.line(screen_coords[1], screen_coords[2], color=0x4ECDC4, radius=2)
        # Edge 2-0
        gui.line(screen_coords[2], screen_coords[0], color=0xFFE66D, radius=2)
        
        # Draw vertices as small circles
        for i in range(3):
            color = 0xFFFFFF  # White
            if i == 0:
                color = 0xFF6B6B  # Red
            elif i == 1:
                color = 0x4ECDC4  # Cyan
            else:
                color = 0xFFE66D  # Yellow
            gui.circle(screen_coords[i], color=color, radius=4)
        
        # Display rotation angle
        gui.text(f"Rotation: {rotation_angle[None]:.1f} deg", (0.05, 0.95), font_size=20, color=0xFFFFFF)
        gui.text("Press A/D to rotate, ESC to exit", (0.05, 0.90), font_size=16, color=0xCCCCCC)
        
        # Show frame
        gui.show()

if __name__ == "__main__":
    run()