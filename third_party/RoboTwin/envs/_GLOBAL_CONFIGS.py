import os

ROOT_PATH = os.path.abspath(__file__)
ROOT_PATH = ROOT_PATH[: ROOT_PATH.rfind("/")]
ROOT_PATH = ROOT_PATH[: ROOT_PATH.rfind("/") + 1]
ASSETS_PATH = os.path.join(ROOT_PATH, "assets/")
EMBODIMENTS_PATH = os.path.join(ASSETS_PATH, "embodiments/")
TEXTURES_PATH = os.path.join(ASSETS_PATH, "background_texture/")
CONFIGS_PATH = os.path.join(ROOT_PATH, "task_config/")
SCRIPT_PATH = os.path.join(ROOT_PATH, "script/")
DESCRIPTION_PATH = os.path.join(ROOT_PATH, "description/")
GRASP_DIRECTION_DIC = {
    "left": [0, 0, 0, -1],
    "front_left": [-0.383, 0, 0, -0.924],
    "front": [-0.707, 0, 0, -0.707],
    "front_right": [-0.924, 0, 0, -0.383],
    "right": [-1, 0, 0, 0],
    "top_down": [-0.5, 0.5, -0.5, -0.5],
    "down_right": [-0.707, 0, -0.707, 0],
    "down_left": [0, 0.707, 0, -0.707],
    "top_down_little_left": [-0.353523, 0.61239, -0.353524, -0.61239],
    "top_down_little_right": [-0.61239, 0.353523, -0.61239, -0.353524],
    "left_arm_perf": [-0.853532, 0.146484, -0.353542, -0.3536],
    "right_arm_perf": [-0.353518, 0.353564, -0.14642, -0.853568],
}
WORLD_DIRECTION_DIC = {
    "left": [0, -0.707, 0, 0.707],
    "front": [0.5, -0.5, 0.5, 0.5],
    "right": [0.707, 0, 0.707, 0],
    "top_down": [0, 0.707, -0.707, 0],
}
ROTATE_NUM = 10
