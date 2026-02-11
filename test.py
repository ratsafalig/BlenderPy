from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
import random

# Initialize the application
app = Ursina()

# --- Visual Settings for "Cinematic" Feel ---
window.fps_counter.enabled = False
window.vsync = False
scene.fog_density = 0.001
scene.fog_color = color.rgb(20, 20, 40) # Dark blue atmospheric fog
camera.fov = 90  # Start with a wide lens

# --- The Environment ---

# 1. The Sky
sky = Sky(texture='sky_sunset') # Built-in texture

# 2. The Ground (far below)
ground = Entity(
    model='plane',
    scale=(1000, 1, 1000),
    color=color.dark_gray,
    texture='grass',
    texture_scale=(50, 50),
    collider='box'
)

# 3. The Skyscraper
building_height = 300
building = Entity(
    model='cube',
    scale=(20, building_height, 20),
    y=building_height/2, # Center the pivot
    texture='brick',
    texture_scale=(5, 50),
    color=color.rgb(50, 50, 50),
    collider='box'
)

# 4. The Roof (Platform to stand on)
roof = Entity(
    model='cube',
    scale=(21, 1, 21),
    y=building_height,
    color=color.black,
    collider='box',
    texture='white_cube'
)

# --- The Player ---
player = FirstPersonController(
    model='cube',
    z=-5, # Start back a bit from the edge
    y=building_height + 2, # Start on top of the roof
    origin_y=-.5,
    speed=8
)
player.cursor.visible = False
player.gravity = 1 # Standard gravity

# --- Cinematic Logic ---
is_falling = False
fall_speed = 0

def update():
    global is_falling, fall_speed
    
    # Check if player has walked off the building
    # The roof is at y=300. If we drop below 295, we are falling.
    if player.y < (building_height - 5):
        is_falling = True
    else:
        is_falling = False
        camera.fov = 90 # Reset FOV if we land (or restart)

    if is_falling:
        # 1. Dynamic FOV: Widen the view to simulate air rushing past
        if camera.fov < 140:
            camera.fov += 20 * time.dt
        
        # 2. Camera Shake: Simulate wind buffeting
        shake_amount = 0.05
        camera.x += random.uniform(-shake_amount, shake_amount)
        camera.y += random.uniform(-shake_amount, shake_amount)
        
        # 3. Motion Blur simulation (Darken the screen edges slightly)
        # Note: True motion blur requires shaders, but we simulate stress via FOV.

    # Respawn mechanic if you hit the ground
    if player.y < 1:
        player.position = (0, building_height + 2, -5)
        camera.fov = 90
        print("Respawned on roof")

# --- Run the Game ---
app.run()