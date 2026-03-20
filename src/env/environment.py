"""
DoublePendulumEnv: A custom Gym environment for double inverted pendulum control.

Physics-based environment using pymunk for 2D rigid body simulation.
Agent controls a cart to balance two stacked poles.

Observation Space: [cart_x, cart_vx, pole1_angle, pole1_angvel, pole2_angle, pole2_angvel]
Action Space: Continuous force [-1.0, 1.0] applied to cart
"""

import numpy as np
import pymunk
import pymunk.pygame_util
import pygame
import csv
import os
from gymnasium import Env, spaces
from typing import Tuple, Dict, Any


class DoublePendulumEnv(Env):
    """
    Double Inverted Pendulum environment.
    
    A cart on a track with two poles linked together. The agent applies horizontal
    forces to the cart to balance both poles upright simultaneously.
    
    Attributes:
        observation_space: Box(6,) - [cart_x, cart_vx, pole1_θ, pole1_ω, pole2_θ, pole2_ω]
        action_space: Box(1,) - continuous force in [-1, 1]
    """
    
    # Physics parameters
    GRAVITY = 9.81
    CART_MASS = 1.0
    POLE_MASS = 0.1
    POLE_LENGTH = 0.5  # Length from pivot to center of mass
    POLE_RADIUS = 0.05
    CART_WIDTH = 0.4
    CART_HEIGHT = 0.2
    MAX_FORCE = 20.0  # Max force magnitude (action is scaled)
    
    # Environment parameters
    MAX_CART_POS = 2.5
    MAX_ANGLE = np.pi / 2  # Pole angle threshold for failure
    DT = 1.0 / 60.0  # Physics timestep (60 FPS)
    MAX_STEPS = 500  # Episode length
    
    # Rendering parameters
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 400
    SCALE = 100  # pixels per meter
    
    def __init__(
        self,
        reward_type: str = 'shaped',
        render_mode: str = None,
        enable_csv_logging: bool = False,
        log_path: str = 'logs/training_metrics.csv',
    ):
        """
        Initialize the Double Pendulum environment.
        
        Args:
            reward_type: 'baseline' or 'shaped'
            render_mode: 'human' or None
            enable_csv_logging: If True, writes timesteps and mean_reward to CSV.
            log_path: Path for CSV output when logging is enabled.
        """
        super().__init__()
        
        if reward_type not in ('baseline', 'shaped'):
            raise ValueError("reward_type must be either 'baseline' or 'shaped'")

        self.reward_type = reward_type
        self.render_mode = render_mode
        self.enable_csv_logging = enable_csv_logging
        self.log_path = log_path
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(6,), 
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Pygame setup
        self.screen = None
        self.clock = None
        self.font = None
        
        # Physics space
        self.space = pymunk.Space()
        self.space.gravity = (0, -self.GRAVITY)
        self.space.damping = 0.9
        
        # Physics bodies and shapes
        self.cart_body = None
        self.cart_shape = None
        self.pole1_body = None
        self.pole1_shape = None
        self.pole2_body = None
        self.pole2_shape = None
        
        # Constraints
        self.groove_joint = None
        self.pivot1 = None
        self.pivot2 = None
        
        # Episode tracking
        self.steps = 0
        self.global_timestep = 0
        self.episode_rewards = []
        self.last_obs = None

        if self.enable_csv_logging:
            self._initialize_csv_logger()
        
        # Initialize environment
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation array of shape (6,)
        """
        # Clear old space by creating a new one
        self.space = pymunk.Space()
        self.space.gravity = (0, -self.GRAVITY)
        self.space.damping = 0.9
        
        # === Create Cart ===
        self.cart_body = pymunk.Body(self.CART_MASS, float('inf'))
        self.cart_body.position = (0.0, 0.5)  # Cart center slightly above ground
        self.cart_body.velocity = (0.0, 0.0)
        
        self.cart_shape = pymunk.Poly.create_box(
            self.cart_body, 
            (self.CART_WIDTH, self.CART_HEIGHT)
        )
        self.cart_shape.friction = 0.5
        self.space.add(self.cart_body, self.cart_shape)
        
        # === Create Track (Groove Joint) ===
        # Groove joint constrains cart to horizontal movement only
        groove_a = (0, self.cart_body.position.y)
        groove_b = (self.MAX_CART_POS * 2, self.cart_body.position.y)
        groove_body = self.space.static_body
        self.groove_joint = pymunk.GrooveJoint(
            groove_body, 
            self.cart_body, 
            groove_a, 
            groove_b, 
            (0, 0)
        )
        self.groove_joint.collide_bodies = False
        self.space.add(self.groove_joint)
        
        # === Create Pole 1 (Bottom) ===
        # Pivot at cart center, extends upward
        pivot1_pos = self.cart_body.position
        self.pole1_body = pymunk.Body(
            self.POLE_MASS, 
            pymunk.moment_for_box(self.POLE_MASS, (self.POLE_RADIUS, self.POLE_LENGTH))
        )
        # Position pole1's center of mass just above pivot
        self.pole1_body.position = (pivot1_pos.x, pivot1_pos.y + self.POLE_LENGTH / 2)
        self.pole1_body.angle = np.random.uniform(-0.05, 0.05)
        
        self.pole1_shape = pymunk.Poly.create_box(
            self.pole1_body, 
            (self.POLE_RADIUS, self.POLE_LENGTH)
        )
        self.pole1_shape.friction = 0.1
        self.space.add(self.pole1_body, self.pole1_shape)
        
        # Joint connecting cart to pole1
        self.pivot1 = pymunk.PivotJoint(
            self.cart_body, 
            self.pole1_body, 
            self.cart_body.position
        )
        self.pivot1.collide_bodies = False
        self.space.add(self.pivot1)
        
        # === Create Pole 2 (Top) ===
        # Pivot at top of pole1, extends further upward
        pivot2_x = pivot1_pos.x
        pivot2_y = pivot1_pos.y + self.POLE_LENGTH
        pivot2_pos = (pivot2_x, pivot2_y)
        
        self.pole2_body = pymunk.Body(
            self.POLE_MASS, 
            pymunk.moment_for_box(self.POLE_MASS, (self.POLE_RADIUS, self.POLE_LENGTH))
        )
        # Position pole2's center of mass
        self.pole2_body.position = (pivot2_x, pivot2_y + self.POLE_LENGTH / 2)
        self.pole2_body.angle = np.random.uniform(-0.05, 0.05)
        
        self.pole2_shape = pymunk.Poly.create_box(
            self.pole2_body, 
            (self.POLE_RADIUS, self.POLE_LENGTH)
        )
        self.pole2_shape.friction = 0.1
        self.space.add(self.pole2_body, self.pole2_shape)
        
        # Joint connecting pole1 to pole2
        self.pivot2 = pymunk.PivotJoint(
            self.pole1_body, 
            self.pole2_body, 
            pivot2_pos
        )
        self.pivot2.collide_bodies = False
        self.space.add(self.pivot2)
        
        # Reset episode tracking
        self.steps = 0
        self.episode_rewards = []
        obs = self._get_observation()
        self.last_obs = obs
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step of the environment.
        
        Args:
            action: Array of shape (1,) with value in [-1.0, 1.0]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        action = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Apply force to cart
        force_magnitude = action * self.MAX_FORCE
        self.cart_body.force = (force_magnitude, 0.0)
        
        # Step physics simulation
        self.space.step(self.DT)
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(obs, action)
        
        # Check termination conditions
        terminated = self._is_failed()
        
        # Count steps for max_steps truncation
        self.steps += 1
        self.global_timestep += 1
        truncated = self.steps >= self.MAX_STEPS

        # Track and optionally log episode running mean reward
        self.episode_rewards.append(reward)
        running_mean_reward = float(np.mean(self.episode_rewards))
        if self.enable_csv_logging:
            self._log_metrics(self.global_timestep, running_mean_reward)
        
        # Info dictionary
        info = {
            'steps': self.steps,
            'timesteps': self.global_timestep,
            'cart_pos': float(self.cart_body.position.x),
            'pole1_angle': obs[2],
            'pole2_angle': obs[4],
            'mean_reward': running_mean_reward,
        }
        
        self.last_obs = obs
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Extract observation from current physics state.
        
        Returns:
            array [cart_x, cart_vx, pole1_θ, pole1_ω, pole2_θ, pole2_ω]
        """
        cart_x = float(self.cart_body.position.x)
        cart_vx = float(self.cart_body.velocity.x)
        
        pole1_angle = float(self.pole1_body.angle)  # Radians
        pole1_angvel = float(self.pole1_body.angular_velocity)
        
        pole2_angle = float(self.pole2_body.angle)  # Radians
        pole2_angvel = float(self.pole2_body.angular_velocity)
        
        obs = np.array(
            [cart_x, cart_vx, pole1_angle, pole1_angvel, pole2_angle, pole2_angvel],
            dtype=np.float32
        )
        
        return obs
    
    def _calculate_reward(self, obs: np.ndarray, action: float) -> float:
        """
        Calculate reward based on observation and reward type.
        
        Args:
            obs: Current observation array
            
        Returns:
            Scalar reward value
        """
        pole1_angle = obs[2]
        pole1_angvel = obs[3]
        pole2_angle = obs[4]
        pole2_angvel = obs[5]
        cart_x = obs[0]
        pole1_angle = self._wrap_angle(pole1_angle)
        pole2_angle = self._wrap_angle(pole2_angle)
        
        if self.reward_type == 'baseline':
            # Baseline: Reward only for upright poles
            # Maximum reward when both poles are perfectly upright (angle ≈ 0)
            reward = np.cos(pole1_angle) + np.cos(pole2_angle)
            
        elif self.reward_type == 'shaped':
            # Shaped reward with multiple components
            
            # 1. Upright Bonus (core goal)
            upright_bonus = np.cos(pole1_angle) + np.cos(pole2_angle)
            
            # 2. Center Penalty (discourages running off-screen)
            center_penalty = -np.abs(cart_x) * 0.1
            
            # 3. Velocity Penalty (encourages stability and smooth movements)
            velocity_penalty = -(np.abs(pole1_angvel) + np.abs(pole2_angvel)) * 0.01
            
            # 4. Action Penalty (discourages excessive force, saves energy)
            action_penalty = -(action ** 2) * 0.001
            
            reward = upright_bonus + center_penalty + velocity_penalty + action_penalty
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")
        
        return float(reward)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi] for stable trigonometric reward computation."""
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def _initialize_csv_logger(self) -> None:
        """Create CSV file with required evaluator columns."""
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        with open(self.log_path, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['timesteps', 'mean_reward'])

    def _log_metrics(self, timesteps: int, mean_reward: float) -> None:
        """Append a single metrics row to CSV."""
        with open(self.log_path, mode='a', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([timesteps, mean_reward])
    
    def _is_failed(self) -> bool:
        """
        Check if episode should terminate (failure condition).
        
        Returns:
            True if either pole fell over too far or cart went off-screen
        """
        cart_x = self.cart_body.position.x
        pole1_angle = self.pole1_body.angle
        pole2_angle = self.pole2_body.angle
        
        # Pole fell over (angle too large)
        if np.abs(pole1_angle) > self.MAX_ANGLE:
            return True
        if np.abs(pole2_angle) > self.MAX_ANGLE:
            return True
        
        # Cart went off-screen
        if np.abs(cart_x) > self.MAX_CART_POS:
            return True
        
        return False
    
    def render(self) -> None:
        """
        Render the environment state using pygame.
        """
        if self.render_mode is None:
            return
        
        # Initialize pygame if needed
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
            )
            pygame.display.set_caption("Double Pendulum - PPO Agent")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
        
        # Fill background
        self.screen.fill((255, 255, 255))
        
        # Draw ground line
        ground_y = int(self.SCREEN_HEIGHT - 50)
        pygame.draw.line(self.screen, (0, 0, 0), (0, ground_y), (self.SCREEN_WIDTH, ground_y), 2)
        
        # Draw track limits
        center_x = self.SCREEN_WIDTH // 2
        limit_left = center_x - int(self.MAX_CART_POS * self.SCALE)
        limit_right = center_x + int(self.MAX_CART_POS * self.SCALE)
        pygame.draw.line(self.screen, (200, 0, 0), (limit_left, ground_y - 10), (limit_left, ground_y + 10), 2)
        pygame.draw.line(self.screen, (200, 0, 0), (limit_right, ground_y - 10), (limit_right, ground_y + 10), 2)
        
        # Draw pymunk objects
        offset = (center_x, ground_y)
        self._draw_pymunk_space(offset)
        
        # Draw information text
        info_text = f"Step: {self.steps} | Reward Type: {self.reward_type}"
        text_surface = self.font.render(info_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))
        
        # Update display
        pygame.display.flip()
        self.clock.tick(60)  # 60 FPS
    
    def _draw_pymunk_space(self, offset: Tuple[int, int]) -> None:
        """
        Draw all bodies and shapes in the pymunk space.
        
        Args:
            offset: (x, y) offset for screen center
        """
        offset_x, offset_y = offset
        
        # Draw shapes using pymunk.pygame_util
        for shape in self.space.shapes:
            body = shape.body
            
            if isinstance(shape, pymunk.Poly):
                # Convert vertices to screen coordinates
                vertices = []
                for vertex in shape.get_vertices():
                    world_pos = body.local_to_world(vertex)
                    screen_x = int(offset_x + world_pos.x * self.SCALE)
                    screen_y = int(offset_y - world_pos.y * self.SCALE)
                    vertices.append((screen_x, screen_y))
                
                if len(vertices) > 0:
                    pygame.draw.polygon(self.screen, (100, 150, 255), vertices)
                    # Draw outline
                    pygame.draw.polygon(self.screen, (50, 100, 200), vertices, 2)
        
        # Draw pivot joints as circles
        for body in self.space.bodies:
            if body == self.space.static_body:
                continue
            
            pos_x = int(offset_x + body.position.x * self.SCALE)
            pos_y = int(offset_y - body.position.y * self.SCALE)
            
            # Draw body center
            pygame.draw.circle(self.screen, (0, 0, 0), (pos_x, pos_y), 3)
    
    def close(self) -> None:
        """
        Clean up resources.
        """
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None
    
    def __del__(self):
        """Ensure cleanup on object deletion."""
        self.close()
