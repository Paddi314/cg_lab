# src/Work0/main.py
import taichi as ti

# 1. ！！！核心修复：ti.init 必须在导入 physics 之前执行 ！！！
ti.init(arch=ti.gpu)

# 2. 导入自定义模块（此时 physics.py 里的 ti.field 才能被正确创建）
from .config import WINDOW_RES, PARTICLE_COLOR, PARTICLE_RADIUS
from .physics import init_particles, update_particles, pos

def run():
    print("Taichi Gravity Swarm 启动中...")
    init_particles()
    
    # 3. 使用 ti.GUI (对应你的课程参考代码风格)
    gui = ti.GUI("Experiment 0: Gravity Swarm", res=WINDOW_RES)
    
    while gui.running:
        mouse_x, mouse_y = gui.get_cursor_pos()
        update_particles(mouse_x, mouse_y)
        
        # 渲染
        gui.circles(pos.to_numpy(), color=PARTICLE_COLOR, radius=PARTICLE_RADIUS)
        gui.show()

if __name__ == "__main__":
    run()