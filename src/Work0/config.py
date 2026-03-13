# src/Work0/config.py

# --- 物理系统参数 ---
NUM_PARTICLES = 800      # 粒子总数
GRAVITY_STRENGTH = 0.005   # 鼠标引力强度（配合新公式进行了微调）
DRAG_COEF = 0.96           # 能量损耗系数（越小粒子越“粘”，越大粒子越“滑”）
BOUNCE_COEF = -0.5         # 边界反弹能量损耗

# --- 渲染系统参数 ---
WINDOW_RES = (800, 600)    # 窗口分辨率
PARTICLE_RADIUS = 1.2      # 稍微调小一点，会让 10000 个粒子看起来更细腻
PARTICLE_COLOR = 0x00BFFF  # 粒子颜色 (天蓝色)