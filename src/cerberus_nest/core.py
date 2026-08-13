import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import csv
import os
import math
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Optional, Any

# Try to import pygame for rendering (optional)
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Pygame not installed. Running headless.")

# =============================================================================
# CONFIGURATION
# =============================================================================

class ExperimentConfig:
    def __init__(self,
                 name="baseline",
                 seed=42,
                 max_steps=10000,
                 curiosity_weight=0.5,
                 memory_capacity=20000,
                 social_interaction=True,
                 novelty_factor=1.0,
                 day_night_cycle=True,
                 reward_structure="default",
                 render=False):
        self.name = name
        self.seed = seed
        self.max_steps = max_steps
        self.curiosity_weight = curiosity_weight
        self.memory_capacity = memory_capacity
        self.social_interaction = social_interaction
        self.novelty_factor = novelty_factor
        self.day_night_cycle = day_night_cycle
        self.reward_structure = reward_structure
        self.render = render

# =============================================================================
# 2D ENVIRONMENT
# =============================================================================

OBJECT_TYPE_NAMES = {0:'empty',1:'food',2:'water',3:'toy',4:'obstacle',5:'caregiver',6:'door'}
N_OBJECT_TYPES = 7
COLORS = {
    0: (200,200,200),
    1: (255,0,0),    # food red
    2: (0,0,255),    # water blue
    3: (255,255,0),  # toy yellow
    4: (100,100,100),# obstacle grey
    5: (0,255,0),    # caregiver green
    6: (139,69,19),  # door brown
}
AGENT_COLOR = (255,255,255)

class CerberusNest2D:
    """2D grid world with raw pixel observations."""
    def __init__(self, grid_size=(12,12), cell_size=30, seed=42, config=None):
        self.width, self.height = grid_size
        self.cell_size = cell_size
        self.rng = np.random.default_rng(seed)
        self.config = config if config else ExperimentConfig()
        self.action_space_n = 8

        self.screen = None
        if PYGAME_AVAILABLE and self.config.render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width*self.cell_size, self.height*self.cell_size))
            pygame.display.set_caption("CERBERUS NEST - Developmental AI")

        self.agent_pos = [self.width//2, self.height//2]
        self.facing = [0,-1]
        self.last_action = 4
        self.internal = self._init_internal()
        self.objects = []
        self.time = 0
        self.day_time = 0
        self._init_objects()

    def _init_internal(self):
        return {
            'hunger': 0.2,
            'thirst': 0.2,
            'fatigue': 0.1,
            'discomfort': 0.0,
            'curiosity': 0.8,
            'arousal': 0.5,
            'social_drive': 0.5,
            'safety': 0.9,
        }

    def _init_objects(self):
        self.objects = []
        occupied = {tuple(self.agent_pos)}
        types = [1,1,2,2,3,3,5]  # 2 food, 2 water, 2 toys, 1 caregiver
        for t in types:
            pos = self._random_empty(occupied)
            self.objects.append({'type':t, 'pos':list(pos), 'active':True, 'respawn_timer':0})
            occupied.add(pos)
        for _ in range(6):  # obstacles
            pos = self._random_empty(occupied)
            self.objects.append({'type':4, 'pos':list(pos), 'active':True, 'respawn_timer':0})
            occupied.add(pos)
        for _ in range(2):  # doors
            pos = self._random_empty(occupied)
            self.objects.append({'type':6, 'pos':list(pos), 'active':True, 'respawn_timer':0})
            occupied.add(pos)

    def _random_empty(self, occupied):
        while True:
            pos = (int(self.rng.integers(0,self.width)), int(self.rng.integers(0,self.height)))
            if pos not in occupied:
                return pos

    def _random_empty_objects(self):
        occupied = {tuple(self.agent_pos)}
        for o in self.objects:
            if o['active']:
                occupied.add(tuple(o['pos']))
        while True:
            pos = (int(self.rng.integers(0,self.width)), int(self.rng.integers(0,self.height)))
            if pos not in occupied and not self._is_obstacle(pos[0],pos[1]):
                return pos

    def reset(self):
        self.agent_pos = [self.width//2, self.height//2]
        self.facing = [0,-1]
        self.internal = self._init_internal()
        self.time = 0
        self.day_time = 0
        self.last_action = 4
        self._init_objects()
        return self._get_obs(), self._get_info(4, 0.0)

    def step(self, action):
        self.time += 1
        self.last_action = action
        if self.config.day_night_cycle and self.time % 20 == 0:
            self.day_time = (self.day_time + 1) % 24

        moved = False
        interaction_reward = 0.0
        if action in [0,1,2,3]:
            moved = self._move(action)
        elif action == 5:
            interaction_reward = self._interact()
        elif action == 6:
            self.internal['fatigue'] = max(0.0, self.internal['fatigue'] - 0.25)
        elif action == 7:
            pass

        self._update_caregiver(action)
        self._update_needs(moved)
        self._update_objects()

        discomfort = self._calculate_discomfort()
        self.internal['discomfort'] = discomfort

        external_reward = interaction_reward - 0.01 * (
            self.internal['hunger']**2 +
            self.internal['thirst']**2 +
            self.internal['fatigue']**2
        )
        if self.config.reward_structure == "sparse":
            external_reward = 1.0 if interaction_reward > 0 else 0.0
        elif self.config.reward_structure == "curiosity_only":
            external_reward = 0.0

        done = (
            self.internal['hunger'] >= 0.99 or
            self.internal['thirst'] >= 0.99 or
            self.internal['fatigue'] >= 0.99
        )

        obs = self._get_obs()
        info = self._get_info(action, external_reward)
        return obs, external_reward, done, info

    def _move(self, action):
        dxdy = {0:(0,-1),1:(0,1),2:(-1,0),3:(1,0)}
        dx,dy = dxdy[action]
        nx = self.agent_pos[0]+dx
        ny = self.agent_pos[1]+dy
        if 0<=nx<self.width and 0<=ny<self.height:
            if not self._is_obstacle(nx,ny):
                self.agent_pos = [nx,ny]
                self.facing = [dx,dy]
                return True
        return False

    def _interact(self):
        reward = 0.0
        target_positions = [self.agent_pos]
        front = [self.agent_pos[0]+self.facing[0], self.agent_pos[1]+self.facing[1]]
        if 0<=front[0]<self.width and 0<=front[1]<self.height:
            target_positions.append(front)

        for obj in self.objects:
            if not obj['active']:
                continue
            if obj['pos'] in target_positions:
                if obj['type'] == 1:  # food
                    if self.internal['hunger'] > 0.05:
                        reduction = min(0.4, self.internal['hunger'])
                        self.internal['hunger'] -= reduction
                        reward += 1.0*(reduction/0.4)
                    obj['active'] = False
                    obj['respawn_timer'] = 30
                elif obj['type'] == 2:  # water
                    if self.internal['thirst'] > 0.05:
                        reduction = min(0.4, self.internal['thirst'])
                        self.internal['thirst'] -= reduction
                        reward += 1.0*(reduction/0.4)
                    obj['active'] = False
                    obj['respawn_timer'] = 30
                elif obj['type'] == 3:  # toy
                    reward += 0.15
                elif obj['type'] == 5:  # caregiver
                    if self.internal['social_drive'] > 0.1:
                        reduction = min(0.3, self.internal['social_drive'])
                        self.internal['social_drive'] -= reduction
                        reward += 0.5*(reduction/0.3)
                    else:
                        reward += 0.05
        return reward

    def _update_needs(self, moved):
        move_cost = 0.002 if moved else 0.0
        self.internal['hunger'] = min(1.0, self.internal['hunger'] + 0.005 + move_cost)
        self.internal['thirst'] = min(1.0, self.internal['thirst'] + 0.004 + move_cost*0.5)
        self.internal['fatigue'] = min(1.0, self.internal['fatigue'] + 0.003 + move_cost*0.5)

        caregiver_dist = self._distance_to_type(5)
        if caregiver_dist is not None and caregiver_dist < 3:
            self.internal['social_drive'] = max(0.0, self.internal['social_drive'] - 0.01)
        else:
            self.internal['social_drive'] = min(1.0, self.internal['social_drive'] + 0.005)

        self.internal['arousal'] = 0.5 + 0.5*(self.internal['hunger']+self.internal['thirst'])/2.0
        self.internal['safety'] = 0.9 if caregiver_dist is not None and caregiver_dist < 3 else 0.7

    def _update_caregiver(self, action):
        if not self.config.social_interaction:
            return
        caregiver = None
        for obj in self.objects:
            if obj['type']==5 and obj['active']:
                caregiver = obj
                break
        if caregiver is None:
            return
        if action == 7 or self.rng.random() < 0.2:
            dx = self.agent_pos[0] - caregiver['pos'][0]
            dy = self.agent_pos[1] - caregiver['pos'][1]
            move = [0,0]
            if abs(dx) > abs(dy):
                move[0] = 1 if dx>0 else -1 if dx<0 else 0
            else:
                move[1] = 1 if dy>0 else -1 if dy<0 else 0
            nx = caregiver['pos'][0]+move[0]
            ny = caregiver['pos'][1]+move[1]
        else:
            nx = caregiver['pos'][0] + int(self.rng.integers(-1,2))
            ny = caregiver['pos'][1] + int(self.rng.integers(-1,2))
        if 0<=nx<self.width and 0<=ny<self.height and not self._is_obstacle(nx,ny):
            caregiver['pos'] = [nx,ny]

    def _update_objects(self):
        for obj in self.objects:
            if not obj['active'] and obj['type'] in [1,2]:
                if obj['respawn_timer'] > 0:
                    obj['respawn_timer'] -= 1
                else:
                    pos = self._random_empty_objects()
                    obj['pos'] = list(pos)
                    obj['active'] = True

    # ------------------------------------------------------------------
    # Raw Pixel Observation
    # ------------------------------------------------------------------
    def _get_obs(self):
        if self.screen is not None:
            # Use pygame surface for actual pixel data
            surf = pygame.Surface((self.width, self.height))
            self._draw(surf)
            rgb_array = pygame.surfarray.array3d(surf).transpose(1,0,2)  # HxWx3
        else:
            rgb_array = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for x in range(self.width):
                for y in range(self.height):
                    obj_type = self._get_object_type_at(x,y)
                    rgb_array[y,x] = COLORS[obj_type]
            rgb_array[self.agent_pos[1], self.agent_pos[0]] = AGENT_COLOR
        return rgb_array.astype(np.float32) / 255.0

    def _draw(self, surf):
        surf.fill(COLORS[0])
        for obj in self.objects:
            if obj['active']:
                rect = pygame.Rect(obj['pos'][0], obj['pos'][1], 1, 1)
                surf.fill(COLORS[obj['type']], rect)
        rect = pygame.Rect(self.agent_pos[0], self.agent_pos[1], 1, 1)
        surf.fill(AGENT_COLOR, rect)

    def _get_object_type_at(self, x, y):
        for obj in self.objects:
            if obj['active'] and obj['pos'] == [x,y]:
                return obj['type']
        return 0

    def _calculate_discomfort(self):
        d = 0.0
        d += max(0.0, self.internal['hunger'] - 0.7) * 0.5
        d += max(0.0, self.internal['thirst'] - 0.7) * 0.5
        d += max(0.0, self.internal['fatigue'] - 0.7) * 0.3
        return min(1.0, d)

    def _distance_to_type(self, obj_type):
        for obj in self.objects:
            if obj['active'] and obj['type'] == obj_type:
                return abs(self.agent_pos[0]-obj['pos'][0]) + abs(self.agent_pos[1]-obj['pos'][1])
        return None

    def _is_obstacle(self, x, y):
        for obj in self.objects:
            if obj['active'] and obj['type']==4 and obj['pos']==[x,y]:
                return True
        return False

    def _get_info(self, action, reward):
        internal_vec = np.array([
            self.internal['hunger'],
            self.internal['thirst'],
            self.internal['fatigue'],
            self.internal['discomfort'],
            self.internal['curiosity'],
            self.internal['arousal'],
            self.internal['social_drive'],
            self.internal['safety'],
        ], dtype=np.float32)
        return {
            'internal_state': internal_vec,
            'position': np.array(self.agent_pos, dtype=np.float32),
            'last_action': action,
            'hunger': self.internal['hunger'],
            'thirst': self.internal['thirst'],
            'fatigue': self.internal['fatigue'],
            'discomfort': self.internal['discomfort'],
            'social_drive': self.internal['social_drive'],
            'curiosity': self.internal['curiosity'],
            'reward': reward,
            'day_time': self.day_time,
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self, mode='human'):
        if not PYGAME_AVAILABLE or self.screen is None:
            return
        self._draw(self.screen)
        # Draw grid lines
        for x in range(self.width):
            pygame.draw.line(self.screen, (0,0,0), (x*self.cell_size,0), (x*self.cell_size,self.height*self.cell_size))
        for y in range(self.height):
            pygame.draw.line(self.screen, (0,0,0), (0,y*self.cell_size), (self.width*self.cell_size,y*self.cell_size))
        pygame.display.flip()
        # handle events to keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

# =============================================================================
# NEURAL NETWORK MODULES
# =============================================================================

class VisionEncoder(nn.Module):
    """CNN to process raw pixel observations (HxWx3) into feature vector."""
    def __init__(self, input_channels=3, output_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        # compute feature size automatically
        dummy = torch.zeros(1, input_channels, 12, 12)
        dummy = F.relu(self.conv1(dummy))
        dummy = F.relu(self.conv2(dummy))
        dummy = F.relu(self.conv3(dummy))
        self.feature_size = dummy.view(1, -1).size(1)
        self.fc = nn.Linear(self.feature_size, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return F.relu(self.fc(x))

class QNetwork(nn.Module):
    """Q-network combining vision features + internal state + action one-hot + focus one-hot."""
    def __init__(self, vision_dim, internal_dim, n_actions, n_focus, hidden=128):
        super().__init__()
        input_dim = vision_dim + internal_dim + n_actions + n_focus
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class RNDNetwork(nn.Module):
    """Simple MLP for Random Network Distillation."""
    def __init__(self, input_dim, embed_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, embed_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SelfModel(nn.Module):
    """Predicts next internal state and position delta from current internal + action."""
    def __init__(self, internal_dim, n_actions):
        super().__init__()
        self.internal_dim = internal_dim
        self.fc1 = nn.Linear(internal_dim + n_actions, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, internal_dim + 2)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, internal, action_onehot):
        x = torch.cat([internal, action_onehot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def update(self, internal, action_onehot, next_internal, position_delta):
        internal_t = torch.tensor(internal, dtype=torch.float32).unsqueeze(0)
        action_t = torch.tensor(action_onehot, dtype=torch.float32).unsqueeze(0)
        target = torch.cat([
            torch.tensor(next_internal, dtype=torch.float32).unsqueeze(0),
            torch.tensor(position_delta, dtype=torch.float32).unsqueeze(0),
        ], dim=-1)
        pred = self.forward(internal_t, action_t)
        loss = F.mse_loss(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

class WorldModel(nn.Module):
    """Predicts next vision feature given current vision feature + action."""
    def __init__(self, vision_dim, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim + n_actions, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, vision_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, vision_feat, action_onehot):
        x = torch.cat([vision_feat, action_onehot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def update(self, vision_feat, action_onehot, next_vision_feat):
        vf = torch.tensor(vision_feat, dtype=torch.float32).unsqueeze(0)
        a = torch.tensor(action_onehot, dtype=torch.float32).unsqueeze(0)
        nvf = torch.tensor(next_vision_feat, dtype=torch.float32).unsqueeze(0)
        pred = self.forward(vf, a)
        loss = F.mse_loss(pred, nvf)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

class RNDCuriosity:
    def __init__(self, input_dim, embed_dim=64):
        self.target = RNDNetwork(input_dim, embed_dim)
        self.predictor = RNDNetwork(input_dim, embed_dim)
        for p in self.target.parameters():
            p.requires_grad = False
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=1e-3)

    def intrinsic_reward(self, obs_vec):
        obs_t = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            t = self.target(obs_t)
            p = self.predictor(obs_t)
            return ((t - p)**2).mean().item()

    def update(self, obs_vec):
        obs_t = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
        t = self.target(obs_t)
        p = self.predictor(obs_t)
        loss = F.mse_loss(p, t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# =============================================================================
# MEMORY SYSTEMS
# =============================================================================

class EpisodicMemory:
    def __init__(self, capacity=20000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []

    def __len__(self):
        return len(self.buffer)

    def push(self, obs_vec, action, reward, next_obs_vec, done, td_error):
        if len(self.buffer) >= self.capacity:
            idx = int(np.argmin(self.priorities))
            self.buffer.pop(idx)
            self.priorities.pop(idx)
        self.buffer.append((obs_vec, action, reward, next_obs_vec, done))
        priority = (abs(td_error) + 1e-5) ** self.alpha
        self.priorities.append(priority)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None, None, None
        priorities = np.array(self.priorities, dtype=np.float64)
        probs = priorities / priorities.sum()
        probs = probs / probs.sum()  # ensure exact normalisation
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]
        weights = probs[indices]
        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for i, e in zip(indices, td_errors):
            self.priorities[i] = (abs(e) + 1e-5) ** self.alpha

    def get_average_priority(self):
        return float(np.mean(self.priorities)) if self.priorities else 0.0

    def consolidate(self, fraction=0.1):
        if len(self.buffer) == 0:
            return
        n_keep = max(1, int(len(self.buffer) * (1 - fraction)))
        sorted_idx = np.argsort(self.priorities)[::-1]
        keep = sorted_idx[:n_keep]
        self.buffer = [self.buffer[i] for i in keep]
        self.priorities = [self.priorities[i] for i in keep]

class SemanticMemory:
    def __init__(self, feature_dim, n_clusters=20):
        self.feature_dim = feature_dim
        self.n_clusters = n_clusters
        self.centroids = np.random.randn(n_clusters, feature_dim).astype(np.float32)
        self.counts = np.zeros(n_clusters)
        self.lr = 0.01

    def update(self, obs_vec):
        dists = np.linalg.norm(self.centroids - obs_vec, axis=1)
        idx = int(np.argmin(dists))
        self.counts[idx] += 1
        self.centroids[idx] += self.lr * (obs_vec - self.centroids[idx])

    def get_cluster_id(self, obs_vec):
        dists = np.linalg.norm(self.centroids - obs_vec, axis=1)
        return int(np.argmin(dists))

    def get_cluster_counts(self):
        return self.counts.copy()

class ProceduralMemory:
    def __init__(self):
        self.sequences = {}
    def update(self, state_hash, action, reward):
        if reward > 0:
            if state_hash not in self.sequences:
                self.sequences[state_hash] = []
            self.sequences[state_hash].append((action, reward))
            if len(self.sequences[state_hash]) > 10:
                self.sequences[state_hash] = self.sequences[state_hash][-10:]

# =============================================================================
# ATTENTION / GLOBAL WORKSPACE
# =============================================================================

class AttentionWorkspace:
    def __init__(self, temperature=0.2):
        self.temperature = temperature
        self.focus = None
        self.scores = {}
        self.history = []

    def compute_focus(self, signals: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        names = list(signals.keys())
        vals = np.array([signals[k] for k in names], dtype=np.float32)
        if np.max(vals) - np.min(vals) < 1e-8:
            probs = np.ones(len(names)) / len(names)
        else:
            exp = np.exp((vals - np.max(vals)) / self.temperature)
            probs = exp / exp.sum()
        focus_idx = int(np.random.choice(len(names), p=probs))
        self.focus = names[focus_idx]
        self.scores = dict(zip(names, map(float, probs)))
        self.history.append({'focus': self.focus, 'scores': self.scores})
        return self.focus, self.scores

# =============================================================================
# DEVELOPMENTAL TIMELINE
# =============================================================================

class DevelopmentalTimeline:
    STAGES = ['Birth','Sensorimotor','Object Discovery','Spatial Understanding','Social Learning','Communication','Self Representation']
    def __init__(self, thresholds=None):
        if thresholds is None:
            self.thresholds = [0,500,1000,2000,3500,5000,8000]
        else:
            self.thresholds = thresholds
    def get_stage(self, step):
        stage = 0
        for i, th in enumerate(self.thresholds):
            if step >= th:
                stage = i
        return stage
    def get_available_actions(self, step):
        stage = self.get_stage(step)
        if stage == 0:
            return [0,1,2,3,4,6]
        else:
            return [0,1,2,3,4,5,6,7]

# =============================================================================
# LANGUAGE ASSOCIATOR
# =============================================================================

class LanguageAssociator:
    def __init__(self, n_object_types=N_OBJECT_TYPES, window=10):
        self.window = window
        self.sound_history = deque(maxlen=window)
        self.object_history = deque(maxlen=window)
        self.associations = np.zeros(n_object_types)
    def observe(self, action, local_obj_type):
        if action == 7:
            self.sound_history.append(1)
        else:
            self.sound_history.append(0)
        self.object_history.append(local_obj_type)
        if len(self.sound_history)==self.window and sum(self.sound_history)>0:
            for t in range(self.window):
                if self.sound_history[t]==1:
                    obj = self.object_history[t]
                    self.associations[obj] += 1.0
    def get_strongest_association(self):
        return int(np.argmax(self.associations)), float(np.max(self.associations))

# =============================================================================
# AGENT (FULL SYSTEM)
# =============================================================================

class Agent:
    def __init__(self, vision_dim, internal_dim, n_actions, n_focus, config, device='cpu'):
        self.vision_dim = vision_dim
        self.internal_dim = internal_dim
        self.n_actions = n_actions
        self.n_focus = n_focus
        self.config = config
        self.device = device

        self.vision_encoder = VisionEncoder(input_channels=3, output_dim=vision_dim).to(device)
        self.qnet = QNetwork(vision_dim, internal_dim, n_actions, n_focus).to(device)
        self.target_qnet = QNetwork(vision_dim, internal_dim, n_actions, n_focus).to(device)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.optimizer = optim.Adam(list(self.qnet.parameters())+list(self.vision_encoder.parameters()), lr=3e-4)

        self.memory = EpisodicMemory(capacity=config.memory_capacity)
        self.rnd = RNDCuriosity(input_dim=vision_dim+internal_dim+n_actions+n_focus)
        self.self_model = SelfModel(internal_dim, n_actions)
        self.world_model = WorldModel(vision_dim, n_actions)
        self.attention = AttentionWorkspace()
        self.timeline = DevelopmentalTimeline()
        self.semantic = SemanticMemory(feature_dim=vision_dim+internal_dim+n_actions+n_focus, n_clusters=20)
        self.procedural = ProceduralMemory()
        self.language = LanguageAssociator()

        self.last_intrinsic_reward = 1.0
        self.last_self_prediction_error = 0.0
        self.last_world_prediction_error = 0.0

        self.gamma = 0.99
        self.curiosity_weight = config.curiosity_weight
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.update_every = 4
        self.target_update_every = 100
        self.consolidation_every = 500

        self.q_value_history = deque(maxlen=100)

    def preprocess(self, pixel_obs, internal_state, last_action, focus_idx):
        pixel_t = torch.tensor(pixel_obs, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vision_feat = self.vision_encoder(pixel_t).squeeze(0).cpu().numpy()
        action_onehot = np.zeros(self.n_actions, dtype=np.float32)
        action_onehot[last_action] = 1.0
        focus_onehot = np.zeros(self.n_focus, dtype=np.float32)
        focus_onehot[focus_idx] = 1.0
        obs_vec = np.concatenate([vision_feat, internal_state, action_onehot, focus_onehot])
        return obs_vec.astype(np.float32), vision_feat

    def get_attention_signals(self, info):
        memory_retrieval = self.memory.get_average_priority()
        return {
            'need_urgency': max(info['hunger'], info['thirst'], info['fatigue'], info['discomfort']),
            'visual_novelty': self.last_intrinsic_reward,
            'memory_retrieval': memory_retrieval,
            'prediction_error': self.last_self_prediction_error,
            'social_drive': info['social_drive'],
        }

    def select_action(self, obs_vec, available_actions):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        obs_t = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.qnet(obs_t).squeeze(0)
            mask = torch.full_like(q, -1e9)
            mask[available_actions] = 0.0
            q = q + mask
            return int(q.argmax())

    def update_q(self, batch):
        states = torch.tensor(np.array([b[0] for b in batch], dtype=np.float32)).to(self.device)
        actions = torch.tensor([b[1] for b in batch]).unsqueeze(1).to(self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array([b[3] for b in batch], dtype=np.float32)).to(self.device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32).to(self.device)

        q_values = self.qnet(states).gather(1, actions).squeeze()
        with torch.no_grad():
            next_q = self.target_qnet(next_states)
            max_next_q, _ = next_q.max(dim=1)
            target = rewards + (1-dones)*self.gamma*max_next_q
        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        td_errors = (target - q_values).detach().cpu().numpy()
        return td_errors

    def update_target(self):
        self.target_qnet.load_state_dict(self.qnet.state_dict())

    def consolidate_memory(self):
        self.memory.consolidate(fraction=0.1)

    def get_metacognition(self):
        if len(self.q_value_history) < 2:
            return 0.0
        return float(np.var(self.q_value_history))

# =============================================================================
# METRICS LOGGER
# =============================================================================

class MetricsLogger:
    def __init__(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.file = open(filename, 'w', newline='')
        self.writer = None
        self.fieldnames = None

    def log(self, data):
        if self.writer is None:
            self.fieldnames = list(data.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
            self.writer.writeheader()
        self.writer.writerow(data)

    def close(self):
        self.file.close()

# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(config: ExperimentConfig):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    env = CerberusNest2D(grid_size=(12,12), cell_size=30, seed=config.seed, config=config)
    focus_names = ['need_urgency','visual_novelty','memory_retrieval','prediction_error','social_drive']

    # Determine dimensions
    pixel_obs, info = env.reset()
    vision_dim = 64
    internal_dim = 8
    n_actions = env.action_space_n
    n_focus = len(focus_names)

    agent = Agent(vision_dim=vision_dim, internal_dim=internal_dim, n_actions=n_actions, n_focus=n_focus, config=config)

    logger = MetricsLogger(f"logs/{config.name}_{config.seed}.csv")
    batch_size = 32

    # Initial observation
    focus, scores = agent.attention.compute_focus(agent.get_attention_signals(info))
    obs_vec, vision_feat = agent.preprocess(pixel_obs, info['internal_state'], info['last_action'], focus_names.index(focus))

    episode_reward = 0.0
    last_position = info['position'].copy()
    last_internal = info['internal_state'].copy()

    for step in range(config.max_steps):
        available_actions = agent.timeline.get_available_actions(step)
        action = agent.select_action(obs_vec, available_actions)

        next_pixel_obs, reward, done, next_info = env.step(action)

        if config.render and PYGAME_AVAILABLE:
            env.render()
            pygame.time.delay(50)

        # Attention for next state
        next_focus, next_scores = agent.attention.compute_focus(agent.get_attention_signals(next_info))
        next_obs_vec, next_vision_feat = agent.preprocess(
            next_pixel_obs, next_info['internal_state'], next_info['last_action'], focus_names.index(next_focus)
        )

        # Curiosity
        intrinsic_reward = agent.rnd.intrinsic_reward(obs_vec)
        agent.last_intrinsic_reward = intrinsic_reward
        total_reward = reward + agent.curiosity_weight * intrinsic_reward

        # Self-model prediction error
        action_onehot = np.eye(n_actions)[action]
        position_delta = next_info['position'] - last_position
        self_pred_error = agent.self_model.update(
            last_internal, action_onehot, next_info['internal_state'], position_delta
        )
        agent.last_self_prediction_error = self_pred_error

        # World model prediction error
        world_pred_error = agent.world_model.update(vision_feat, action_onehot, next_vision_feat)
        agent.last_world_prediction_error = world_pred_error

        # RND update
        agent.rnd.update(obs_vec)

        # Semantic memory update
        agent.semantic.update(obs_vec)

        # Language association (use center cell object type)
        obj_type = env._get_object_type_at(env.agent_pos[0], env.agent_pos[1])
        agent.language.observe(action, obj_type)

        # Procedural memory
        state_hash = hash(obs_vec[:150].tobytes())
        agent.procedural.update(state_hash, action, reward)

        # TD error for prioritized replay
        with torch.no_grad():
            obs_t = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
            q_val = agent.qnet(obs_t)[0, action].item()
            agent.q_value_history.append(q_val)

            next_obs_t = torch.tensor(next_obs_vec, dtype=torch.float32).unsqueeze(0)
            next_q = agent.target_qnet(next_obs_t).detach()
            next_available = agent.timeline.get_available_actions(step+1)
            mask = torch.full_like(next_q, -1e9)
            mask[0, next_available] = 0.0
            next_q_masked = next_q + mask
            max_next_q = next_q_masked.max().item()

            target = total_reward + (0.0 if done else agent.gamma * max_next_q)
            td_error = target - q_val

        agent.memory.push(obs_vec, action, total_reward, next_obs_vec, done, td_error)

        # Periodic Q update
        if step % agent.update_every == 0:
            batch, indices, weights = agent.memory.sample(batch_size)
            if batch is not None:
                td_errors = agent.update_q(batch)
                agent.memory.update_priorities(indices, td_errors)

        if step % agent.target_update_every == 0:
            agent.update_target()

        if step % agent.consolidation_every == 0 and step > 0:
            agent.consolidate_memory()

        # Consciousness-related metrics (simplified)
        if action in [0,1,2,3]:
            moved = (position_delta[0]!=0 or position_delta[1]!=0)
            self_recognition = 1.0 if moved else 0.0
        else:
            self_recognition = 0.0

        body_ownership = 1.0 / (1.0 + self_pred_error)
        temporal_continuity = min(1.0, len(agent.memory)/1000.0)
        metacognition = agent.get_metacognition()
        introspection_error = abs(next_info['hunger'] - next_info['hunger'])  # placeholder
        strongest_obj, assoc_strength = agent.language.get_strongest_association()

        metrics = {
            'step': step,
            'stage': agent.timeline.get_stage(step),
            'episode_reward': episode_reward,
            'total_reward': total_reward,
            'external_reward': reward,
            'intrinsic_reward': intrinsic_reward,
            'hunger': next_info['hunger'],
            'thirst': next_info['thirst'],
            'fatigue': next_info['fatigue'],
            'discomfort': next_info['discomfort'],
            'social_drive': next_info['social_drive'],
            'attention_focus': focus,
            'need_urgency_score': scores.get('need_urgency',0.0),
            'visual_novelty_score': scores.get('visual_novelty',0.0),
            'memory_retrieval_score': scores.get('memory_retrieval',0.0),
            'prediction_error_score': scores.get('prediction_error',0.0),
            'social_score': scores.get('social_drive',0.0),
            'self_prediction_error': self_pred_error,
            'world_prediction_error': world_pred_error,
            'memory_size': len(agent.memory),
            'epsilon': agent.epsilon,
            'action': action,
            'self_recognition': self_recognition,
            'body_ownership': body_ownership,
            'temporal_continuity': temporal_continuity,
            'metacognition': metacognition,
            'introspection_error': introspection_error,
            'strongest_language_object': strongest_obj,
            'language_association_strength': assoc_strength,
            'semantic_cluster_0_count': agent.semantic.get_cluster_counts()[0],
        }
        logger.log(metrics)

        episode_reward += total_reward

        if done:
            pixel_obs, info = env.reset()
            focus, scores = agent.attention.compute_focus(agent.get_attention_signals(info))
            obs_vec, vision_feat = agent.preprocess(pixel_obs, info['internal_state'], info['last_action'], focus_names.index(focus))
            episode_reward = 0.0
            last_position = info['position'].copy()
            last_internal = info['internal_state'].copy()
        else:
            pixel_obs = next_pixel_obs
            info = next_info
            focus = next_focus
            scores = next_scores
            obs_vec = next_obs_vec
            vision_feat = next_vision_feat
            last_position = info['position'].copy()
            last_internal = info['internal_state'].copy()

    logger.close()
    print(f"[{config.name}] Seed {config.seed} — {config.max_steps} steps completed. CSV saved.")
    if PYGAME_AVAILABLE and config.render:
        pygame.quit()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run baseline headless
    run_experiment(ExperimentConfig(name="baseline", seed=42, max_steps=5000, render=False))

    # To run with rendering, uncomment the following line:
    # run_experiment(ExperimentConfig(name="baseline_render", seed=42, max_steps=2000, render=True))