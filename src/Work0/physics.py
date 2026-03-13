# src/Work0/physics.py
import taichi as ti
from .config import *

# 数据结构定义：在显存中开辟空间
pos = ti.Vector.field(2, dtype=float, shape=NUM_PARTICLES)
vel = ti.Vector.field(2, dtype=float, shape=NUM_PARTICLES)

@ti.kernel
def init_particles():
    """初始化每一个粒子的随机坐标"""
    for i in range(NUM_PARTICLES):
        pos[i] = [ti.random(), ti.random()]
        vel[i] = [0.0, 0.0]

@ti.kernel
def update_particles(mouse_x: float, mouse_y: float):
    """物理更新：由 GPU 并行执行"""
    for i in range(NUM_PARTICLES):
        mouse_pos = ti.Vector([mouse_x, mouse_y])
        direction = mouse_pos - pos[i]
        dist = direction.norm()
        
        # --- 核心改进：平方反比引力公式 ---
        # 1. 使用 direction / (dist**2 + epsilon)
        # 2. 0.002 是软化因子，防止粒子离鼠标太近时受力无穷大而“炸飞”
        strength = GRAVITY_STRENGTH / (dist**2 + 0.002)
        
        # 施加加速度：力的大小与距离平方成反比
        vel[i] += direction * strength
            
        # 施加阻力并更新位置
        vel[i] *= DRAG_COEF  
        pos[i] += vel[i]

        # 边界碰撞检测与反馈
        for j in ti.static(range(2)):
            if pos[i][j] < 0:
                pos[i][j] = 0.0
                vel[i][j] *= BOUNCE_COEF
            elif pos[i][j] > 1:
                pos[i][j] = 1.0
                vel[i][j] *= BOUNCE_COEF