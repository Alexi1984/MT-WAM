import gymnasium as gym
import sapien.core as sapien
import toppra as ta


class Sapien_TEST(gym.Env):
    def __init__(self):
        super().__init__()
        ta.setup_logging("CRITICAL")
        try:
            self.setup_scene()
            print("\x1b[32m" + "Render Well" + "\x1b[0m")
        except Exception as error:
            raise RuntimeError("SAPIEN renderer initialization failed.") from error

    def setup_scene(self, **kwargs):
        self.engine = sapien.Engine()
        from sapien.render import set_global_config

        set_global_config(max_num_materials=50000, max_num_textures=50000)
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)


if __name__ == "__main__":
    a = Sapien_TEST()
