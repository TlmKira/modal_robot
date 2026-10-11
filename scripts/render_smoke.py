"""Validate real SAPIEN GPU rendering without a display server."""
from pathlib import Path
import sapien
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parents[1]
scene = sapien.Scene()
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [1, 1, 1])
builder = scene.create_actor_builder()
builder.add_box_visual(half_size=[0.2, 0.2, 0.2], material=[1, 0, 0, 1])
actor = builder.build_static()
camera = scene.add_camera("smoke", 320, 240, 1, 0.1, 10)
camera.entity.set_pose(sapien.Pose([-1, 0, 0]))
scene.step()
scene.update_render()
camera.take_picture()
rgb = np.clip(camera.get_picture("Color")[..., :3] * 255, 0, 255).astype(np.uint8)
assert rgb.shape == (240, 320, 3) and rgb.std() > 1, "Blank camera image"
Image.fromarray(rgb).save(root / "results/screenshots/sapien_smoke.png")
print("SAPIEN", sapien.__version__, "render OK", rgb.shape, "std", rgb.std())
