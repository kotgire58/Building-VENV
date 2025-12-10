import pygame
import numpy as np
import pickle
import time
import math
import random
from collections import deque
import os

class AdaptiveClickGame:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Adaptive Difficulty Click Game")
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 100, 255)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (128, 128, 128)
        
        # Fonts
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.big_font = pygame.font.Font(None, 72)
        
        # Game state
        self.clock = pygame.time.Clock()
        self.running = True
        self.game_started = False
        self.paused = False
        
        # Target properties
        self.target_x = width // 2
        self.target_y = height // 2
        self.target_size = 50
        self.target_base_speed = 3
        self.target_speed = self.target_base_speed
        self.target_dx = random.choice([-1, 1]) * self.target_speed
        self.target_dy = random.choice([-1, 1]) * self.target_speed
        
        # Performance metrics
        self.score = 0
        self.clicks = 0
        self.hits = 0
        self.misses = 0
        self.current_streak = 0
        self.max_streak = 0
        self.combo_multiplier = 1.0
        self.reaction_times = deque(maxlen=10)
        self.recent_hits = deque(maxlen=10)
        self.start_time = time.time()
        self.last_target_time = time.time()
        
        # Difficulty parameters
        self.difficulty_level = 0.5
        self.min_target_size = 20
        self.max_target_size = 80
        self.min_speed = 1
        self.max_speed = 10
        
        # ML Model
        self.load_model()
        
        # Visual effects
        self.hit_effects = []
        self.miss_effects = []
        self.streak_animation_time = 0
        
    def load_model(self):
        """Load the trained DDA model"""
        try:
            with open('dda_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_columns = model_data['feature_columns']
                print("Model loaded successfully!")
        except FileNotFoundError:
            print("Warning: Model file not found. Running without adaptive difficulty.")
            self.model = None
            
    def get_current_features(self):
        """Extract current game features for ML prediction"""
        total_attempts = self.hits + self.misses
        hit_accuracy = self.hits / max(1, total_attempts)
        avg_reaction_time = np.mean(self.reaction_times) if self.reaction_times else 0.5
        time_played = time.time() - self.start_time
        
        # Recent performance
        recent_hits_count = sum(self.recent_hits) if self.recent_hits else 0
        recent_misses_count = len(self.recent_hits) - recent_hits_count
        
        features = {
            'hit_accuracy': hit_accuracy,
            'avg_reaction_time': avg_reaction_time,
            'current_streak': self.current_streak,
            'max_streak': self.max_streak,
            'score': self.score,
            'target_size': self.target_size,
            'target_speed': self.target_speed,
            'difficulty_level': self.difficulty_level,
            'recent_hits': recent_hits_count,
            'recent_misses': recent_misses_count,
            'combo_multiplier': self.combo_multiplier,
            'time_played': time_played,
            'total_clicks': self.clicks
        }
        
        # Add engineered features with safe division
        features['hit_miss_ratio'] = recent_hits_count / (recent_misses_count + 1)
        features['score_per_click'] = self.score / (self.clicks + 1)
        features['streak_ratio'] = self.current_streak / (self.max_streak + 1)
        features['difficulty_size_interaction'] = self.difficulty_level * (50 / max(1, self.target_size))
        features['difficulty_speed_interaction'] = self.difficulty_level * self.target_speed
        features['accuracy_reaction_product'] = hit_accuracy * avg_reaction_time
        
        # Ensure no infinite or NaN values
        for key, value in features.items():
            if np.isnan(value) or np.isinf(value):
                features[key] = 0.0
        
        return features
    
    def predict_reaction_time(self):
        """Predict player's next reaction time using ML model"""
        if not self.model or len(self.reaction_times) < 3:
            return 0.5  # Default prediction
            
        features = self.get_current_features()
        feature_array = np.array([[features[col] for col in self.feature_columns]])
        
        try:
            scaled_features = self.scaler.transform(feature_array)
            prediction = self.model.predict(scaled_features)[0]
            return np.clip(prediction, 0.15, 2.0)
        except:
            return 0.5
    
    def adjust_difficulty(self):
        """Dynamically adjust game difficulty based on ML predictions"""
        if len(self.reaction_times) < 3:
            return
            
        predicted_reaction = self.predict_reaction_time()
        current_avg_reaction = np.mean(self.reaction_times)
        
        # Calculate performance ratio
        performance_ratio = predicted_reaction / current_avg_reaction
        
        # Adjust difficulty based on performance
        if performance_ratio < 0.8:  # Player is performing better than expected
            self.difficulty_level = min(1.0, self.difficulty_level + 0.05)
        elif performance_ratio > 1.2:  # Player is struggling
            self.difficulty_level = max(0.1, self.difficulty_level - 0.05)
        
        # Apply difficulty adjustments
        self.target_size = int(self.max_target_size - (self.max_target_size - self.min_target_size) * self.difficulty_level)
        self.target_speed = self.min_speed + (self.max_speed - self.min_speed) * self.difficulty_level
        
        # Update target velocity
        speed_magnitude = math.sqrt(self.target_dx**2 + self.target_dy**2)
        if speed_magnitude > 0:
            self.target_dx = (self.target_dx / speed_magnitude) * self.target_speed
            self.target_dy = (self.target_dy / speed_magnitude) * self.target_speed
    
    def move_target(self):
        """Move the target with bouncing behavior"""
        self.target_x += self.target_dx
        self.target_y += self.target_dy
        
        # Bounce off walls
        if self.target_x - self.target_size < 0 or self.target_x + self.target_size > self.width:
            self.target_dx *= -1
            self.target_x = np.clip(self.target_x, self.target_size, self.width - self.target_size)
            
        if self.target_y - self.target_size < 0 or self.target_y + self.target_size > self.height:
            self.target_dy *= -1
            self.target_y = np.clip(self.target_y, self.target_size, self.height - self.target_size)
        
        # Add some randomness to movement
        if random.random() < 0.02:  # 2% chance to change direction slightly
            angle_change = random.uniform(-0.3, 0.3)
            cos_angle = math.cos(angle_change)
            sin_angle = math.sin(angle_change)
            new_dx = self.target_dx * cos_angle - self.target_dy * sin_angle
            new_dy = self.target_dx * sin_angle + self.target_dy * cos_angle
            self.target_dx = new_dx
            self.target_dy = new_dy
    
    def handle_click(self, pos):
        """Handle mouse click events"""
        if not self.game_started:
            self.game_started = True
            self.start_time = time.time()
            self.last_target_time = time.time()
            return
            
        self.clicks += 1
        mouse_x, mouse_y = pos
        
        # Calculate distance from click to target center
        distance = math.sqrt((mouse_x - self.target_x)**2 + (mouse_y - self.target_y)**2)
        
        # Record reaction time
        reaction_time = time.time() - self.last_target_time
        self.reaction_times.append(reaction_time)
        
        if distance <= self.target_size:
            # Hit!
            self.hits += 1
            self.recent_hits.append(1)
            self.current_streak += 1
            self.max_streak = max(self.max_streak, self.current_streak)
            
            # Update combo multiplier
            self.combo_multiplier = 1 + (self.current_streak / 10)
            
            # Calculate score
            accuracy_bonus = max(0, 1 - (distance / self.target_size)) * 100
            speed_bonus = max(0, (2 - reaction_time)) * 50
            points = int((accuracy_bonus + speed_bonus) * self.combo_multiplier * self.difficulty_level)
            self.score += points
            
            # Add hit effect
            self.hit_effects.append({
                'x': mouse_x,
                'y': mouse_y,
                'time': time.time(),
                'points': points
            })
            
            # Move target to new position
            self.spawn_new_target()
            
        else:
            # Miss!
            self.misses += 1
            self.recent_hits.append(0)
            self.current_streak = 0
            self.combo_multiplier = 1.0
            
            # Add miss effect
            self.miss_effects.append({
                'x': mouse_x,
                'y': mouse_y,
                'time': time.time()
            })
        
        # Adjust difficulty after each click
        self.adjust_difficulty()
        self.last_target_time = time.time()
    
    def spawn_new_target(self):
        """Spawn target at new random position"""
        margin = self.target_size + 20
        self.target_x = random.randint(margin, self.width - margin)
        self.target_y = random.randint(margin, self.height - margin)
        
        # Random initial direction
        angle = random.uniform(0, 2 * math.pi)
        self.target_dx = math.cos(angle) * self.target_speed
        self.target_dy = math.sin(angle) * self.target_speed
    
    def draw_target(self):
        """Draw the target with visual indicators"""
        # Draw target circles
        pygame.draw.circle(self.screen, self.RED, 
                         (int(self.target_x), int(self.target_y)), 
                         self.target_size, 3)
        pygame.draw.circle(self.screen, self.WHITE, 
                         (int(self.target_x), int(self.target_y)), 
                         self.target_size // 2, 2)
        pygame.draw.circle(self.screen, self.RED, 
                         (int(self.target_x), int(self.target_y)), 
                         5)
    
    def draw_effects(self):
        """Draw visual effects for hits and misses"""
        current_time = time.time()
        
        # Draw hit effects
        for effect in self.hit_effects[:]:
            age = current_time - effect['time']
            if age > 1.0:
                self.hit_effects.remove(effect)
                continue
                
            alpha = 255 * (1 - age)
            radius = int(20 + age * 30)
            
            # Draw expanding circle
            surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*self.GREEN, int(alpha)), (radius, radius), radius, 3)
            self.screen.blit(surf, (effect['x'] - radius, effect['y'] - radius))
            
            # Draw points text
            points_surf = self.font.render(f"+{effect['points']}", True, self.GREEN)
            points_rect = points_surf.get_rect(center=(effect['x'], effect['y'] - age * 50))
            self.screen.blit(points_surf, points_rect)
        
        # Draw miss effects
        for effect in self.miss_effects[:]:
            age = current_time - effect['time']
            if age > 0.5:
                self.miss_effects.remove(effect)
                continue
                
            # Draw X mark
            size = 20
            thickness = 3
            color = (*self.RED, int(255 * (1 - age * 2)))
            
            pygame.draw.line(self.screen, color,
                           (effect['x'] - size, effect['y'] - size),
                           (effect['x'] + size, effect['y'] + size), thickness)
            pygame.draw.line(self.screen, color,
                           (effect['x'] + size, effect['y'] - size),
                           (effect['x'] - size, effect['y'] + size), thickness)
    
    def draw_ui(self):
        """Draw UI elements"""
        # Score
        score_text = self.font.render(f"Score: {self.score}", True, self.WHITE)
        self.screen.blit(score_text, (10, 10))
        
        # Accuracy
        accuracy = (self.hits / max(1, self.clicks)) * 100 if self.clicks > 0 else 0
        accuracy_text = self.small_font.render(f"Accuracy: {accuracy:.1f}%", True, self.WHITE)
        self.screen.blit(accuracy_text, (10, 50))
        
        # Streak
        streak_color = self.YELLOW if self.current_streak >= 5 else self.WHITE
        streak_text = self.small_font.render(f"Streak: {self.current_streak}", True, streak_color)
        self.screen.blit(streak_text, (10, 80))
        
        # Combo multiplier
        if self.combo_multiplier > 1:
            combo_text = self.small_font.render(f"Combo: x{self.combo_multiplier:.1f}", True, self.YELLOW)
            self.screen.blit(combo_text, (10, 110))
        
        # Difficulty level
        diff_color = (255, int(255 * (1 - self.difficulty_level)), 0)
        diff_text = self.small_font.render(f"Difficulty: {self.difficulty_level:.2f}", True, diff_color)
        self.screen.blit(diff_text, (self.width - 150, 10))
        
        # Target size indicator
        size_text = self.small_font.render(f"Target Size: {self.target_size}", True, self.WHITE)
        self.screen.blit(size_text, (self.width - 150, 40))
        
        # Speed indicator
        speed_text = self.small_font.render(f"Speed: {self.target_speed:.1f}", True, self.WHITE)
        self.screen.blit(speed_text, (self.width - 150, 70))
        
        # Average reaction time
        if self.reaction_times:
            avg_reaction = np.mean(self.reaction_times)
            reaction_text = self.small_font.render(f"Avg Reaction: {avg_reaction:.3f}s", True, self.WHITE)
            self.screen.blit(reaction_text, (self.width - 200, 100))
        
        # ML prediction indicator
        if self.model and len(self.reaction_times) >= 3:
            predicted = self.predict_reaction_time()
            pred_text = self.small_font.render(f"Predicted: {predicted:.3f}s", True, self.BLUE)
            self.screen.blit(pred_text, (self.width - 200, 130))
    
    def draw_start_screen(self):
        """Draw the start screen"""
        title_text = self.big_font.render("ADAPTIVE CLICK GAME", True, self.WHITE)
        title_rect = title_text.get_rect(center=(self.width // 2, self.height // 3))
        self.screen.blit(title_text, title_rect)
        
        subtitle_text = self.font.render("ML-Powered Difficulty Adjustment", True, self.BLUE)
        subtitle_rect = subtitle_text.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(subtitle_text, subtitle_rect)
        
        start_text = self.font.render("Click anywhere to start!", True, self.GREEN)
        start_rect = start_text.get_rect(center=(self.width // 2, self.height * 2 // 3))
        self.screen.blit(start_text, start_rect)
        
        # Instructions
        instructions = [
            "Click the moving target as fast and accurately as possible",
            "The game adapts to your skill level in real-time",
            "Build streaks for combo multipliers!"
        ]
        
        y_offset = self.height * 3 // 4
        for instruction in instructions:
            inst_text = self.small_font.render(instruction, True, self.GRAY)
            inst_rect = inst_text.get_rect(center=(self.width // 2, y_offset))
            self.screen.blit(inst_text, inst_rect)
            y_offset += 30
    
    def run(self):
        """Main game loop"""
        while self.running:
            dt = self.clock.tick(60) / 1000.0  # 60 FPS
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(pygame.mouse.get_pos())
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
            
            # Clear screen
            self.screen.fill(self.BLACK)
            
            if not self.game_started:
                self.draw_start_screen()
            else:
                if not self.paused:
                    # Update game state
                    self.move_target()
                
                # Draw everything
                self.draw_target()
                self.draw_effects()
                self.draw_ui()
                
                if self.paused:
                    pause_text = self.big_font.render("PAUSED", True, self.WHITE)
                    pause_rect = pause_text.get_rect(center=(self.width // 2, self.height // 2))
                    self.screen.blit(pause_text, pause_rect)
            
            pygame.display.flip()
        
        pygame.quit()
        self.print_final_stats()
    
    def print_final_stats(self):
        """Print final game statistics"""
        print("\n=== GAME OVER ===")
        print(f"Final Score: {self.score}")
        print(f"Total Clicks: {self.clicks}")
        print(f"Hits: {self.hits}")
        print(f"Misses: {self.misses}")
        print(f"Accuracy: {(self.hits / max(1, self.clicks)) * 100:.1f}%")
        print(f"Max Streak: {self.max_streak}")
        if self.reaction_times:
            print(f"Average Reaction Time: {np.mean(self.reaction_times):.3f}s")
        print(f"Final Difficulty Level: {self.difficulty_level:.2f}")

if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists('dda_model.pkl'):
        print("Model file not found. Please run train.py first to generate the model.")
        print("The game will run without adaptive difficulty.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            exit()
    
    game = AdaptiveClickGame()
    game.run()