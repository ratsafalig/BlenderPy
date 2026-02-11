import pygame
import math
import random

# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Colors
COLOR_BG = (20, 20, 25)          # Very dark grey floor
COLOR_WALL = (100, 100, 100)     # Visible static objects
COLOR_LIGHT = (255, 230, 200)    # Warm torch light
COLOR_PLAYER = (200, 200, 255)   # Player skin/helmet color
COLOR_ENEMY = (255, 50, 50)      # Enemy color
COLOR_BULLET = (255, 255, 100)
COLOR_ITEM = (50, 255, 50)
SHADOW_ALPHA = 240               # 0-255 (255 is pitch black darkness)

# Game Settings
PLAYER_SPEED = 3
PLAYER_RADIUS = 12
TORCH_RADIUS_BASE = 250
BULLET_SPEED = 10

# --- Helper Math Function ---
def rotate_point(center, point, angle_degrees):
    """Rotates a point around a center (for animating arms/legs)."""
    angle_rad = math.radians(-angle_degrees) # Negative for pygame coord system
    ox, oy = center
    px, py = point

    qx = ox + math.cos(angle_rad) * (px - ox) - math.sin(angle_rad) * (py - oy)
    qy = oy + math.sin(angle_rad) * (px - ox) + math.cos(angle_rad) * (py - oy)
    return qx, qy

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Top-Down Shooter: Tactical Light")
        self.clock = pygame.time.Clock()
        self.running = True
        self.font = pygame.font.SysFont("Arial", 18)

        # Assets / Entities
        self.player_pos = pygame.math.Vector2(SCREEN_WIDTH//2, SCREEN_HEIGHT//2)
        self.player_angle = 0
        self.torch_radius = TORCH_RADIUS_BASE
        
        # Animation state
        self.walk_cycle = 0 
        
        self.walls = self.generate_walls(15)
        self.enemies = self.generate_enemies(8)
        self.items = self.generate_items(3)
        self.bullets = []
        
        # Shader Surface (The Darkness)
        self.light_layer = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        self.light_poly = [] 

    def generate_walls(self, count):
        walls = []
        # Borders
        walls.append(pygame.Rect(0, -10, SCREEN_WIDTH, 10))
        walls.append(pygame.Rect(0, SCREEN_HEIGHT, SCREEN_WIDTH, 10))
        walls.append(pygame.Rect(-10, 0, 10, SCREEN_HEIGHT))
        walls.append(pygame.Rect(SCREEN_WIDTH, 0, 10, SCREEN_HEIGHT))
        
        # Random boxes
        for _ in range(count):
            w, h = random.randint(40, 100), random.randint(40, 100)
            x = random.randint(50, SCREEN_WIDTH - 50 - w)
            y = random.randint(50, SCREEN_HEIGHT - 50 - h)
            walls.append(pygame.Rect(x, y, w, h))
        return walls

    def generate_enemies(self, count):
        enemies = []
        for _ in range(count):
            pos = pygame.math.Vector2(random.randint(50, SCREEN_WIDTH-50), 
                                      random.randint(50, SCREEN_HEIGHT-50))
            # No spawn in walls
            collides = True
            while collides:
                pos = pygame.math.Vector2(random.randint(50, SCREEN_WIDTH-50), 
                                          random.randint(50, SCREEN_HEIGHT-50))
                collides = False
                rect = pygame.Rect(pos.x - 15, pos.y - 15, 30, 30)
                for w in self.walls:
                    if rect.colliderect(w):
                        collides = True
            enemies.append({'pos': pos, 'alive': True, 'angle': random.randint(0, 360)})
        return enemies

    def generate_items(self, count):
        items = []
        for _ in range(count):
             pos = pygame.math.Vector2(random.randint(50, SCREEN_WIDTH-50), 
                                      random.randint(50, SCREEN_HEIGHT-50))
             items.append({'pos': pos, 'rect': pygame.Rect(pos.x-5, pos.y-5, 10, 10)})
        return items

    def handle_input(self):
        keys = pygame.key.get_pressed()
        move = pygame.math.Vector2(0, 0)
        if keys[pygame.K_w]: move.y = -1
        if keys[pygame.K_s]: move.y = 1
        if keys[pygame.K_a]: move.x = -1
        if keys[pygame.K_d]: move.x = 1

        if move.length() > 0:
            self.walk_cycle += 0.4 # Animate feet
            move = move.normalize() * PLAYER_SPEED
            new_pos = self.player_pos + move
            
            # Wall Collision
            player_rect = pygame.Rect(new_pos.x - PLAYER_RADIUS, new_pos.y - PLAYER_RADIUS, PLAYER_RADIUS*2, PLAYER_RADIUS*2)
            collision = False
            for w in self.walls:
                if player_rect.colliderect(w):
                    collision = True
                    break
            if not collision:
                self.player_pos = new_pos
        else:
            self.walk_cycle = 0 # Reset feet

        # Mouse Rotation
        mx, my = pygame.mouse.get_pos()
        rel_x, rel_y = mx - self.player_pos.x, my - self.player_pos.y
        self.player_angle = -math.degrees(math.atan2(rel_y, rel_x))

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: 
                    self.shoot()

    def shoot(self):
        rad = math.radians(self.player_angle)
        direction = pygame.math.Vector2(math.cos(rad), -math.sin(rad)) 
        
        # Spawn bullet at gun tip (approx)
        spawn_pos = self.player_pos + direction * 20
        
        self.bullets.append({
            'pos': spawn_pos,
            'dir': direction
        })

    def update_bullets(self):
        for b in self.bullets[:]:
            b['pos'] += b['dir'] * BULLET_SPEED
            
            # Remove if out of screen
            if not (0 <= b['pos'].x <= SCREEN_WIDTH and 0 <= b['pos'].y <= SCREEN_HEIGHT):
                self.bullets.remove(b)
                continue

            # Wall Collision
            hit_wall = False
            b_rect = pygame.Rect(b['pos'].x-2, b['pos'].y-2, 4, 4)
            for w in self.walls:
                if w.colliderect(b_rect):
                    self.bullets.remove(b)
                    hit_wall = True
                    break
            if hit_wall: continue

            # Enemy Collision
            for e in self.enemies:
                if e['alive']:
                    dist = b['pos'].distance_to(e['pos'])
                    if dist < 20: # Hitbox size
                        e['alive'] = False
                        if b in self.bullets: self.bullets.remove(b)
                        break

    def check_item_pickup(self):
        player_rect = pygame.Rect(self.player_pos.x-15, self.player_pos.y-15, 30, 30)
        for i in self.items[:]:
            if player_rect.colliderect(i['rect']):
                self.items.remove(i)
                self.torch_radius += 40 
                print("Light Radius Increased!")

    def calculate_lighting(self):
        # 1. Collect points (corners)
        points = []
        for w in self.walls:
            points.append((w.left, w.top))
            points.append((w.right, w.top))
            points.append((w.left, w.bottom))
            points.append((w.right, w.bottom))
        
        points.append((0, 0))
        points.append((SCREEN_WIDTH, 0))
        points.append((0, SCREEN_HEIGHT))
        points.append((SCREEN_WIDTH, SCREEN_HEIGHT))

        unique_angles = set()
        
        # 2. Calculate angles
        for p in points:
            dx = p[0] - self.player_pos.x
            dy = p[1] - self.player_pos.y
            angle = math.atan2(dy, dx)
            unique_angles.add(angle)
            unique_angles.add(angle - 0.0001) 
            unique_angles.add(angle + 0.0001)

        # 3. Raycast
        intersections = []
        sorted_angles = sorted(list(unique_angles))
        
        for angle in sorted_angles:
            ray_dx = math.cos(angle)
            ray_dy = math.sin(angle)
            
            closest_dist = self.torch_radius 
            closest_pt = None
            
            r_px, r_py = self.player_pos.x, self.player_pos.y
            r_dx, r_dy = ray_dx * self.torch_radius, ray_dy * self.torch_radius

            for w in self.walls:
                # Wall segments
                segments = [
                    ((w.left, w.top), (w.right, w.top)),
                    ((w.right, w.top), (w.right, w.bottom)),
                    ((w.right, w.bottom), (w.left, w.bottom)),
                    ((w.left, w.bottom), (w.left, w.top))
                ]
                
                for p1, p2 in segments:
                    rx, ry = r_px, r_py
                    sx, sy = r_dx, r_dy
                    qx, qy = p1[0], p1[1]
                    sdx, sdy = p2[0]-p1[0], p2[1]-p1[1]
                    
                    r_cross_s = sx * sdy - sy * sdx
                    if r_cross_s == 0: continue 
                    
                    q_minus_p_x = qx - rx
                    q_minus_p_y = qy - ry
                    
                    t = (q_minus_p_x * sdy - q_minus_p_y * sdx) / r_cross_s
                    u = (q_minus_p_x * sy - q_minus_p_y * sx) / r_cross_s
                    
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        dist = t * self.torch_radius
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_pt = (rx + t * sx, ry + t * sy)
            
            if closest_pt:
                intersections.append(closest_pt)
            else:
                intersections.append((r_px + r_dx, r_py + r_dy))

        self.light_poly = intersections

    def draw_character(self, surface, color, pos, angle, is_moving=False):
        """ Draw detailed character with gun, arms, and feet """
        x, y = pos.x, pos.y
        
        # --- FEET ---
        foot_offset = 0
        if is_moving:
            foot_offset = math.sin(self.walk_cycle) * 5

        # Local feet positions
        foot_l_pos = (x - 5 + foot_offset, y - 8)
        foot_r_pos = (x - 5 - foot_offset, y + 8)
        
        r_foot_l = rotate_point((x,y), foot_l_pos, angle)
        r_foot_r = rotate_point((x,y), foot_r_pos, angle)

        pygame.draw.circle(surface, (50, 50, 50), (int(r_foot_l[0]), int(r_foot_l[1])), 5)
        pygame.draw.circle(surface, (50, 50, 50), (int(r_foot_r[0]), int(r_foot_r[1])), 5)

        # --- GUN ---
        gun_color = (10, 10, 10)
        # Gun shape relative to center
        gun_poly = [
            (x + 10, y - 3), (x + 35, y - 3), 
            (x + 35, y + 3), (x + 10, y + 3)
        ]
        rot_gun_poly = [rotate_point((x,y), p, angle) for p in gun_poly]
        pygame.draw.polygon(surface, gun_color, rot_gun_poly)

        # --- ARMS ---
        shoulder_l = (x, y - 9)
        shoulder_r = (x, y + 9)
        hand_pos = (x + 20, y)

        r_shoulder_l = rotate_point((x,y), shoulder_l, angle)
        r_shoulder_r = rotate_point((x,y), shoulder_r, angle)
        r_hand = rotate_point((x,y), hand_pos, angle)

        # Draw arm lines
        pygame.draw.line(surface, color, r_shoulder_l, r_hand, 4)
        pygame.draw.line(surface, color, r_shoulder_r, r_hand, 4)
        
        # Hands
        pygame.draw.circle(surface, (180, 160, 140), (int(r_hand[0]), int(r_hand[1])), 3)

        # --- HEAD ---
        pygame.draw.circle(surface, color, (int(x), int(y)), 11) 
        
        # Goggles/Eyes
        eye_pos = (x + 6, y - 3)
        eye_pos_2 = (x + 6, y + 3)
        r_eye = rotate_point((x,y), eye_pos, angle)
        r_eye_2 = rotate_point((x,y), eye_pos_2, angle)
        pygame.draw.circle(surface, (0, 0, 0), (int(r_eye[0]), int(r_eye[1])), 2)
        pygame.draw.circle(surface, (0, 0, 0), (int(r_eye_2[0]), int(r_eye_2[1])), 2)

    def draw(self):
        # 1. Background
        self.screen.fill(COLOR_BG)

        # 2. Static Objects (Always faintly visible)
        for w in self.walls:
            pygame.draw.rect(self.screen, COLOR_WALL, w)
            pygame.draw.rect(self.screen, (0,0,0), w, 2)

        # 3. Items (Glowing dots)
        for i in self.items:
            pygame.draw.circle(self.screen, COLOR_ITEM, (int(i['pos'].x), int(i['pos'].y)), 5)

        # 4. Fog of War (Shadow Layer)
        self.light_layer.fill((0, 0, 0, SHADOW_ALPHA))
        if len(self.light_poly) > 2:
            pygame.draw.polygon(self.light_layer, (0,0,0,0), self.light_poly)
        
        # Calculate Visible Enemies
        visible_enemies = []
        if len(self.light_poly) > 2:
             for e in self.enemies:
                 if not e['alive']: continue
                 dist = self.player_pos.distance_to(e['pos'])
                 if dist > self.torch_radius: continue
                 
                 # Line of sight check
                 blocked = False
                 line = (self.player_pos, e['pos'])
                 for w in self.walls:
                     if w.clipline(line): 
                         blocked = True
                         break
                 if not blocked:
                     visible_enemies.append(e)

        # 5. Apply Darkness
        self.screen.blit(self.light_layer, (0,0))

        # 6. Draw Entities (Player, Enemies, Bullets)
        # Enemies
        for e in visible_enemies:
            # Random angle for enemy or facing player
            self.draw_character(self.screen, COLOR_ENEMY, e['pos'], e['angle'], is_moving=False)

        # Player
        keys = pygame.key.get_pressed()
        is_moving = keys[pygame.K_w] or keys[pygame.K_s] or keys[pygame.K_a] or keys[pygame.K_d]
        self.draw_character(self.screen, COLOR_PLAYER, self.player_pos, self.player_angle, is_moving)

        # Bullets
        for b in self.bullets:
            pygame.draw.circle(self.screen, COLOR_BULLET, (int(b['pos'].x), int(b['pos'].y)), 3)
            
        # HUD
        text = self.font.render(f"Torch Power: {int(self.torch_radius)} | Enemies Left: {len([e for e in self.enemies if e['alive']])}", True, (200, 200, 200))
        self.screen.blit(text, (10, 10))

        pygame.display.flip()

    def run(self):
        while self.running:
            self.clock.tick(FPS)
            self.handle_input()
            self.update_bullets()
            self.check_item_pickup()
            self.calculate_lighting()
            self.draw()
        pygame.quit()

if __name__ == "__main__":
    game = Game()
    game.run()