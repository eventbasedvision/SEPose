import glob
import os
import sys
# Import CARLA from the egg
try:
    sys.path.append("carla-0.9.15-py3.7-linux-x86_64.egg")
except Exception as e:    
    print(e)


import carla
client = carla.Client('localhost', 2000)
world = client.get_world()
print(world.get_map().name)

spectator = world.get_spectator()


cam_loc, cam_rot =\
    carla.Location(x=-35.280499, y=-0.017508, z=1.532597), \
    carla.Rotation(pitch=4.182043, yaw=130.544815, roll=0.000170)

spectator.set_transform(
    carla.Transform(location=cam_loc, rotation=cam_rot)
)

transform = spectator.get_transform()
print(f"Spectator location: x={transform.location.x}, y={transform.location.y}, z={transform.location.z}")
    
# Spawn RGB camera at spectator location
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')

camera = world.spawn_actor(camera_bp, transform)

# Store image data
image_data = None

def process_image(image):
    global image_data
    image_data = image

camera.listen(process_image)

# Let it capture a few frames
for _ in range(10):
    world.tick()

# Save and display image
if image_data:
    image_data.save_to_disk('camera_view.png')
    print("Image saved as camera_view.png")

camera.destroy()
