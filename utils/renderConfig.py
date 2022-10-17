import pybullet as p

'''
初始化渲染设置
'''
def resetBaseRender():
    # 禁用tinyrenderer，不让CPU上的集成显卡来参与渲染工作。
    p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
    # 关闭渲染
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
'''
关闭渲染
'''
def setRenderClose():
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # 关闭渲染

'''
开启渲染
'''
def setRenderOpen():
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # 开启渲染

'''
取消渲染时候周围的控制面板
'''
def setRenderCloseControlPanel():
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
'''
打开渲染时候周围的控制面板
'''
def setRenderOpenControlPanel():
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)