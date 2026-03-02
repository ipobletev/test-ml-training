import pygame
import math
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from collections import deque

class Car:
    def __init__(self, x, y, angle=90):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 0
        self.acceleration = 0.2
        self.friction = 0.1
        self.max_speed = 4
        self.width = 18
        self.height = 32
        self.sensors = [0] * 5
        self.alive = True
        self.distance_traveled = 0
        self.position_history = deque(maxlen=100) # Track last 100 positions

    def get_points(self):
        cos_val = math.cos(math.radians(self.angle))
        sin_val = math.sin(math.radians(self.angle))
        p = [
            (-self.width/2, -self.height/2),
            (self.width/2, -self.height/2),
            (self.width/2, self.height/2),
            (-self.width/2, self.height/2)
        ]
        points = []
        for px, py in p:
            nx = self.x + px * cos_val - py * sin_val
            ny = self.y + px * sin_val + py * cos_val
            points.append((nx, ny))
        return points

    def update(self, action, track_mask):
        if not self.alive: return
        if action == 1: self.speed += self.acceleration
        elif action == 2: self.speed -= self.acceleration
        if self.speed != 0:
            turn_dir = 1 if self.speed > 0 else -1
            if action == 3: self.angle -= 5 * turn_dir
            elif action == 4: self.angle += 5 * turn_dir
        if self.speed > self.max_speed: self.speed = self.max_speed
        if self.speed < -self.max_speed/2: self.speed = -self.max_speed/2
        if self.speed > 0: self.speed -= self.friction
        elif self.speed < 0: self.speed += self.friction
        if abs(self.speed) < self.friction: self.speed = 0
        
        self.x += self.speed * math.sin(math.radians(self.angle))
        self.y -= self.speed * math.cos(math.radians(self.angle))
        self.distance_traveled += abs(self.speed)
        
        # Save position to history
        self.position_history.append((self.x, self.y))
        
        self.update_sensors(track_mask)
        self.check_collision(track_mask)

    def is_stuck(self):
        if len(self.position_history) < 100: return False
        # Calculate max movement in history
        start_pos = self.position_history[0]
        max_dist_sq = 0
        for pos in self.position_history:
            dist_sq = (pos[0] - start_pos[0])**2 + (pos[1] - start_pos[1])**2
            if dist_sq > max_dist_sq: max_dist_sq = dist_sq
        
        # If moved less than 10 pixels total in 100 steps, it's stuck
        return max_dist_sq < 100 # 10^2

    def update_sensors(self, track_mask):
        sensor_angles = [-45, -22.5, 0, 22.5, 45]
        max_dist = 160
        for i, angle in enumerate(sensor_angles):
            val = self._cast_ray(self.angle + angle, max_dist, track_mask)
            self.sensors[i] = val / max_dist

    def _cast_ray(self, angle, max_dist, track_mask):
        w, h = track_mask.get_size()
        for dist in range(0, max_dist, 3):
            rx = self.x + dist * math.sin(math.radians(angle))
            ry = self.y - dist * math.cos(math.radians(angle))
            if rx < 0 or rx >= w or ry < 0 or ry >= h: return dist
            if track_mask.get_at((int(rx), int(ry))) == (0, 0, 0, 255): return dist
        return max_dist

    def check_collision(self, track_mask):
        points = self.get_points()
        w, h = track_mask.get_size()
        for p in points:
            if p[0] < 0 or p[0] >= w or p[1] < 0 or p[1] >= h or track_mask.get_at((int(p[0]), int(p[1]))) == (0, 0, 0, 255):
                self.alive = False
                return

    def get_state(self):
        return self.sensors + [self.speed / self.max_speed]

class Environment:
    TRACKS = {
        "Professional": {
            "points": [(100, 100), (450, 80), (850, 120), (920, 350), (850, 580), (500, 620), (120, 580), (80, 350)],
            "width": 80, "start": (100, 100, 100)
        },
        "Oval": {
            "points": [(200, 200), (800, 200), (800, 500), (200, 500)],
            "width": 100, "start": (200, 200, 90)
        },
        "Labyrinth": {
            "points": [(100, 100), (900, 100), (900, 600), (500, 600), (500, 300), (100, 300)],
            "width": 70, "start": (100, 100, 90)
        }
    }

    def __init__(self, track_name="Professional", render_mode=True):
        pygame.init()
        self.width, self.height = 1000, 700
        self.track_name = track_name
        self.render_mode = render_mode
        if self.render_mode:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(f"Auto RL - {track_name}")
            self.font = pygame.font.SysFont("Inter", 20, bold=True)
            self.big_font = pygame.font.SysFont("Inter", 24, bold=True)
            self.title_font = pygame.font.SysFont("Inter", 32, bold=True)
        self.clock = pygame.time.Clock()
        self.setup_track()

    def setup_track(self):
        self.track_surface = pygame.Surface((self.width, self.height))
        self.track_surface.fill((0, 0, 0))
        track_info = self.TRACKS[self.track_name]
        pygame.draw.lines(self.track_surface, (255, 255, 255), True, track_info["points"], track_info["width"])
        for p in track_info["points"]:
            pygame.draw.circle(self.track_surface, (255, 255, 255), p, track_info["width"] // 2)

    def reset(self, num_cars=1):
        start_x, start_y, start_angle = self.TRACKS[self.track_name]["start"]
        self.cars = [Car(start_x, start_y, start_angle) for _ in range(num_cars)]
        return [car.get_state() for car in self.cars]

    def step(self, actions, stuck_detection=True):
        states = []
        rewards = []
        dones = []
        
        all_done = True
        for i, car in enumerate(self.cars):
            if not car.alive:
                states.append(car.get_state())
                rewards.append(0)
                dones.append(True)
                continue
            
            all_done = False
            prev_dist = car.distance_traveled
            car.update(actions[i], self.track_surface)
            state = car.get_state()
            reward = -0.1
            done = False
            
            if not car.alive:
                reward = -100
                done = True
            elif stuck_detection and car.is_stuck():
                reward = -20
                done = True
                car.alive = False
            else:
                reward += (car.distance_traveled - prev_dist) * 2.0
                if car.speed > 0.1: reward += 0.2
            
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            
        return states, rewards, all_done

    def render(self, episode, total_reward, epsilon, output_active, training_enabled, session_duration=0):
        if not self.render_mode: return
        self.screen.fill((34, 139, 34))
        asphalt = self.track_surface.copy()
        asphalt.set_colorkey((0,0,0))
        asphalt.fill((45, 45, 50), special_flags=pygame.BLEND_RGB_ADD)
        self.screen.blit(asphalt, (0,0))
        pygame.draw.lines(self.screen, (255, 255, 255, 100), True, self.TRACKS[self.track_name]["points"], 2)
        
        alive_count = 0
        for car in self.cars:
            if car.alive: alive_count += 1
            car_points = car.get_points()
            pygame.draw.polygon(self.screen, (0, 255, 120) if car.alive else (255, 50, 50), car_points)
            if car.alive:
                for i, angle in enumerate([-45, -22.5, 0, 22.5, 45]):
                    dist = car.sensors[i] * 160
                    rx = car.x + dist * math.sin(math.radians(car.angle + angle))
                    ry = car.y - dist * math.cos(math.radians(car.angle + angle))
                    pygame.draw.line(self.screen, (255, 255, 0), (car.x, car.y), (rx, ry), 1)

        train_status = "ON" if training_enabled else "OFF"
        # Calculate time string
        mins = int(session_duration // 60)
        secs = int(session_duration % 60)
        time_str = f"{mins:02d}:{secs:02d}"
        
        status_text = f"Ep: {episode} | Alive: {alive_count}/{len(self.cars)} | Eps: {epsilon:.2f} | Training: {train_status} | Time: {time_str}"
        self._draw_badge(status_text, (20, 20))
        
        # Output Toggle Button
        out_color = (0, 150, 255) if output_active else (100, 100, 110)
        out_text = f"Generate Output: {'ON' if output_active else 'OFF'}"
        self.out_toggle_rect = self._draw_button(out_text, (20, self.height - 70), out_color, width=280)
        
        self.btn_rect = self._draw_button("Finish and Exit", (self.width - 250, self.height - 70), (150, 30, 30), width=230)
        pygame.display.flip()

    def _draw_badge(self, text, pos):
        txt = self.font.render(text, True, (255, 255, 255))
        rect = txt.get_rect(topleft=pos).inflate(20, 10)
        pygame.draw.rect(self.screen, (40, 40, 50), rect, border_radius=8)
        self.screen.blit(txt, pos)

    def _draw_button(self, text, pos, color, width=None):
        txt = self.font.render(text, True, (255, 255, 255))
        rect = txt.get_rect(topleft=pos).inflate(40, 20)
        if width: rect.width = width
        m_pos = pygame.mouse.get_pos()
        if rect.collidepoint(m_pos): color = tuple(min(255, c + 30) for c in color)
        pygame.draw.rect(self.screen, color, rect, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255), rect, 2, border_radius=10)
        self.screen.blit(txt, (rect.centerx - txt.get_width()//2, rect.centery - txt.get_height()//2))
        return rect

    def open_file_dialog(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            initialdir=os.path.join(os.getcwd(), ".output"),
            title="Select weights file",
            filetypes=(("PyTorch Files", "*.pth"), ("All Files", "*.*"))
        )
        root.destroy()
        return file_path

    def show_start_menu(self):
        selected_track = "Professional"
        selected_model = None
        training_enabled = True
        stuck_detection = True
        num_cars = 1
        active_input = False
        running = True
        
        while running:
            self.screen.fill((25, 25, 30))
            title = self.title_font.render("Auto RL - Initial Configuration", True, (255, 255, 255))
            self.screen.blit(title, (self.width//2 - title.get_width()//2, 40))
            
            # Scenario Selection
            y_off = 100
            self.screen.blit(self.font.render("1. Select Track:", True, (200, 200, 200)), (100, y_off))
            track_btns = {}
            for i, t_name in enumerate(self.TRACKS.keys()):
                color = (50, 100, 200) if selected_track == t_name else (60, 60, 70)
                track_btns[t_name] = self._draw_button(t_name, (100 + i*160, y_off + 35), color, width=150)

            # Car Quantity
            y_off = 210
            self.screen.blit(self.font.render("2. Number of Cars:", True, (200, 200, 200)), (100, y_off))
            
            # Helper to draw smaller counter buttons
            minus_btn = self._draw_button("-", (100, y_off + 35), (150, 50, 50), width=60)
            
            # Input box with active state feedback
            box_color = (100, 100, 150) if active_input else (60, 60, 70)
            count_box = self._draw_button(str(num_cars), (170, y_off + 35), box_color, width=100)
            if active_input:
                pygame.draw.rect(self.screen, (255, 255, 255), count_box, 2, border_radius=10)
            
            plus_btn = self._draw_button("+", (280, y_off + 35), (50, 150, 50), width=60)
            
            # Quick presets
            presets = [1, 10, 50, 100]
            preset_btns = {}
            for i, p in enumerate(presets):
                color = (200, 50, 150) if num_cars == p else (60, 60, 70)
                preset_btns[p] = self._draw_button(f"x{p}", (370 + i*90, y_off + 35), color, width=80)

            # Model Selection
            y_off = 320
            self.screen.blit(self.font.render("3. Load Intelligence:", True, (200, 200, 200)), (100, y_off))
            new_btn_color = (100, 100, 110) if selected_model is None else (60, 60, 70)
            new_btn = self._draw_button("New Training", (100, y_off + 35), new_btn_color, width=280)
            load_btn_color = (200, 150, 50) if selected_model is not None else (60, 60, 70)
            load_btn = self._draw_button("Load Model (.pth)...", (400, y_off + 35), load_btn_color, width=280)
            if selected_model:
                model_name = os.path.basename(selected_model)
                self.screen.blit(self.font.render(f"Selected: {model_name}", True, (0, 255, 150)), (400, y_off + 95))

            # Toggles
            y_off = 450
            self.screen.blit(self.font.render("4. Simulation Options:", True, (200, 200, 200)), (100, y_off))
            
            # Training Toggle
            train_color = (40, 180, 100) if training_enabled else (150, 50, 50)
            train_text = f"Learning: {'ON' if training_enabled else 'OFF'}"
            train_btn = self._draw_button(train_text, (100, y_off + 35), train_color, width=330)
            
            # Stuck Toggle
            stuck_color = (200, 100, 40) if stuck_detection else (60, 60, 70)
            stuck_text = f"Stagnation: {'ON' if stuck_detection else 'OFF'}"
            stuck_btn = self._draw_button(stuck_text, (450, y_off + 35), stuck_color, width=330)

            start_btn = self._draw_button("START!", (self.width - 300, self.height - 90), (40, 180, 100), width=250)
            
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return None
                
                if event.type == pygame.KEYDOWN and active_input:
                    if event.key == pygame.K_BACKSPACE:
                        num_str = str(num_cars)[:-1]
                        num_cars = int(num_str) if num_str else 0
                    elif event.unicode.isdigit():
                        num_str = str(num_cars) + event.unicode
                        val = int(num_str)
                        if val <= 500: # Limit to 500 for performance
                            num_cars = val
                    elif event.key == pygame.K_RETURN:
                        active_input = False
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    active_input = count_box.collidepoint(event.pos)
                    
                    for t_name, btn in track_btns.items():
                        if btn.collidepoint(event.pos): selected_track = t_name
                    
                    if minus_btn.collidepoint(event.pos): num_cars = max(1, num_cars - 1)
                    if plus_btn.collidepoint(event.pos): num_cars = min(500, num_cars + 1)
                    for p, btn in preset_btns.items():
                        if btn.collidepoint(event.pos): num_cars = p
                        
                    if new_btn.collidepoint(event.pos): selected_model = None
                    if load_btn.collidepoint(event.pos): 
                        path = self.open_file_dialog()
                        if path: selected_model = path
                    if train_btn.collidepoint(event.pos):
                        training_enabled = not training_enabled
                    if stuck_btn.collidepoint(event.pos):
                        stuck_detection = not stuck_detection
                    if start_btn.collidepoint(event.pos): 
                        self.track_name = selected_track
                        self.setup_track()
                        return {
                            "track_name": selected_track,
                            "model_path": selected_model,
                            "training_enabled": training_enabled,
                            "stuck_detection": stuck_detection,
                            "num_cars": num_cars
                        }
        return None

    def is_finish_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if hasattr(self, 'btn_rect') and self.btn_rect.collidepoint(event.pos):
                return True
        return False

    def is_out_toggle_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if hasattr(self, 'out_toggle_rect') and self.out_toggle_rect.collidepoint(event.pos):
                return True
        return False
