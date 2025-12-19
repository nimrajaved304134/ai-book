---
title: "Advanced VLA Architectures and Training"
sidebar_label: "Lesson 2: Advanced VLA Architectures and Training"
---

# Lesson 2: Advanced VLA Architectures and Training

## Introduction

In this lesson, we'll dive deeper into advanced Vision-Language-Action (VLA) architectures and training methodologies. After understanding the basic components of VLA models in Lesson 1, we'll now explore more sophisticated architectures that better integrate vision, language, and action components. We'll also cover advanced training techniques that leverage large-scale datasets, reinforcement learning, and simulation environments to create more capable and robust humanoid robots.

Advanced VLA models go beyond simple fusion mechanisms to implement more sophisticated architectures that enable richer multimodal understanding and more complex action sequences. These models often utilize transformer-based architectures enhanced with specialized components for robotics tasks, enabling humanoid robots to perform complex, multi-step tasks based on natural language instructions while adapting to their environment in real-time.

## Key Concepts and Theory (Humanoid Robotics Principles and AI Integration)

### Advanced Architectural Patterns

Modern VLA systems employ several advanced architectural patterns:

- **Hierarchical Representations**: Multi-level understanding from low-level features to high-level concepts
- **Memory-Augmented Networks**: External memory to store and retrieve relevant information across episodes
- **Graph Neural Networks**: Modeling relationships between objects in the environment
- **Diffusion Models**: Generating complex action sequences guided by vision-language inputs

### Temporal Reasoning in VLA Models

For humanoid robots, temporal reasoning is crucial:

- **Action Sequences**: Understanding multi-step tasks and breaking them down into temporal steps
- **State Tracking**: Maintaining internal state representation across time steps
- **Intention Inference**: Understanding the long-term goal from partial observations
- **Temporal Consistency**: Maintaining coherent behavior across time steps

### Reinforcement Learning Integration

Advanced VLA models often incorporate RL:

- **Policy Networks**: Mapping VLA inputs directly to action policies
- **Value Functions**: Estimating expected future rewards for decision making
- **Exploration Strategies**: Safe exploration in real environments
- **Reward Shaping**: Designing rewards that align with human intentions

### Sim2Real Transfer

Critical for humanoid robotics:

- **Domain Randomization**: Varying visual properties to improve real-world performance
- **Simulation-to-Reality Gap**: Techniques to bridge differences between sim and real
- **Adaptive Policies**: Policies that adapt to new environments
- **Online Adaptation**: Learning during deployment

## Detailed Technical Explanations

### Transformer-Based VLA Architectures

Advanced VLA models often use transformer architectures with robotics-specific modifications:

1. **Perceiver Architecture**: Efficient processing of variable-length, multimodal inputs
2. **Cross-Attention Mechanisms**: Explicit mechanisms for vision-language fusion
3. **Robotics-Specific Tokens**: Special tokens encoding robot state and action spaces
4. **Memory Mechanisms**: Storing and retrieving past experiences

### Training Methodologies

- **Behavior Cloning**: Learning from human demonstrations
- **Inverse RL**: Learning reward functions from demonstrations
- **Multi-Task Learning**: Training on diverse tasks simultaneously
- **Curriculum Learning**: Progressively increasing task complexity

### Large-Scale Pretraining

- **Vision-Language Pretraining**: Using large vision-language datasets
- **Robotics-Specific Pretraining**: Pretraining on robotics datasets before fine-tuning
- **Continual Learning**: Adding new capabilities without forgetting old ones

## Code Examples

### Advanced VLA Model with Memory and Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPVisionConfig, AutoTokenizer
import numpy as np

class CrossModalAttention(nn.Module):
    """Cross-modal attention for fusing vision and language"""
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for queries, keys, values
        self.W_q_vision = nn.Linear(d_model, d_model)
        self.W_k_language = nn.Linear(d_model, d_model)
        self.W_v_language = nn.Linear(d_model, d_model)
        
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, vision_features, language_features):
        # vision_features: (batch_size, vision_seq_len, d_model)
        # language_features: (batch_size, lang_seq_len, d_model)
        
        batch_size, v_seq_len, _ = vision_features.shape
        _, l_seq_len, _ = language_features.shape
        
        # Generate queries from vision, keys and values from language
        Q = self.W_q_vision(vision_features).view(batch_size, v_seq_len, self.num_heads, self.d_k)
        K = self.W_k_language(language_features).view(batch_size, l_seq_len, self.num_heads, self.d_k)
        V = self.W_v_language(language_features).view(batch_size, l_seq_len, self.num_heads, self.d_k)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, v_seq_len, d_k)
        K = K.transpose(1, 2)  # (batch_size, num_heads, l_seq_len, d_k)
        V = V.transpose(1, 2)  # (batch_size, num_heads, l_seq_len, d_k)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)  # (batch_size, num_heads, v_seq_len, l_seq_len)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, v_seq_len, d_k)
        
        # Concatenate heads and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, v_seq_len, self.d_model)
        output = self.W_o(output)
        
        return output, attention_weights

class VLABlock(nn.Module):
    """A complete VLA processing block with cross-modal attention"""
    def __init__(self, d_model=512, num_heads=8, d_ff=2048):
        super().__init__()
        
        self.cross_attention = CrossModalAttention(d_model, num_heads)
        
        # Self-attention for vision modality
        self.vision_self_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
        # Feed-forward networks
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
    
    def forward(self, vision_features, language_features):
        # Cross-modal attention
        cross_attention_out, attention_weights = self.cross_attention(vision_features, language_features)
        
        # Add and norm
        vision_out = self.norm1(vision_features + self.dropout1(cross_attention_out))
        
        # Self-attention
        self_attn_out, _ = self.vision_self_attention(vision_out, vision_out, vision_out)
        vision_out = self.norm2(vision_out + self.dropout2(self_attn_out))
        
        # Feed-forward
        ffn_out = self.ffn(vision_out)
        vision_out = self.norm3(vision_out + self.dropout3(ffn_out))
        
        return vision_out, attention_weights

class ExternalMemory(nn.Module):
    """External memory component for VLA models"""
    def __init__(self, memory_size=100, memory_dim=512):
        super().__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        
        # Initialize memory with zeros
        self.register_buffer('memory', torch.zeros(memory_size, memory_dim))
        self.register_buffer('memory_usage', torch.zeros(memory_size))
        
        # Memory read/write networks
        self.read_key_generator = nn.Linear(512, memory_dim)
        self.write_key_generator = nn.Linear(512, memory_dim)
        self.write_value_generator = nn.Linear(512, memory_dim)
        
    def forward(self, query, write_content=None):
        # Generate query key
        query_key = self.read_key_generator(query)  # (batch_size, memory_dim)
        
        # Compute similarities with memory
        similarities = torch.matmul(query_key, self.memory.t())  # (batch_size, memory_size)
        attention_weights = F.softmax(similarities, dim=-1)  # (batch_size, memory_size)
        
        # Read from memory
        read_content = torch.matmul(attention_weights, self.memory)  # (batch_size, memory_dim)
        
        # Write to memory if content is provided
        if write_content is not None:
            write_key = self.write_key_generator(query_key)
            write_value = self.write_value_generator(write_content)
            
            # Find least used memory slot
            write_idx = torch.argmin(self.memory_usage)
            
            # Update memory
            self.memory[write_idx] = write_value
            self.memory_usage[write_idx] = 1.0
            
            # Reset oldest entry if memory is full
            if torch.sum(self.memory_usage) > self.memory_size * 0.8:  # Reset when 80% full
                oldest_idx = torch.argmin(self.memory_usage)
                self.memory[oldest_idx] = 0.0
                self.memory_usage[oldest_idx] = 0.0
        
        return read_content, attention_weights

class AdvancedVLAModel(nn.Module):
    """Advanced VLA model with memory and multi-step action generation"""
    def __init__(self, 
                 num_actions=20, 
                 vision_model_name="openai/clip-vit-base-patch32",
                 max_seq_len=512):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        
        # Vision encoder (frozen CLIP)
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        self.vision_proj = nn.Linear(768, 512)  # Project from CLIP's output to internal size
        
        # Language encoder (simple transformer)
        self.token_embedding = nn.Embedding(50257, 512)  # GPT-2 vocab size
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, 512))
        
        # VLA processing blocks
        self.vla_blocks = nn.ModuleList([
            VLABlock(d_model=512, num_heads=8) for _ in range(6)
        ])
        
        # External memory
        self.memory = ExternalMemory(memory_size=50, memory_dim=512)
        
        # Action generation head
        self.action_generator = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_actions * 3)  # For multi-step action prediction
        )
        
        # Temporal modeling
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512,
                nhead=8,
                batch_first=True
            ),
            num_layers=2
        )
        
    def forward(self, images, text_tokens, return_attention_weights=False):
        # Process images through vision encoder
        vision_outputs = self.vision_encoder(pixel_values=images)
        vision_features = vision_outputs.last_hidden_state  # (batch, patches, 768)
        
        # Project to internal dimension
        vision_features = self.vision_proj(vision_features)  # (batch, patches, 512)
        
        # Process text
        token_embeddings = self.token_embedding(text_tokens)  # (batch, seq_len, 512)
        
        # Add positional encoding to text
        seq_len = token_embeddings.size(1)
        pos_enc = self.pos_encoding[:, :seq_len, :]  # (1, seq_len, 512)
        language_features = token_embeddings + pos_enc  # (batch, seq_len, 512)
        
        # Apply VLA blocks for fusion
        attention_weights = []
        for block in self.vla_blocks:
            vision_features, attn_w = block(vision_features, language_features)
            if return_attention_weights:
                attention_weights.append(attn_w)
        
        # Pool vision features across patches
        pooled_vision = torch.mean(vision_features, dim=1)  # (batch, 512)
        
        # Add to memory and retrieve relevant information
        memory_content, memory_attention = self.memory(pooled_vision, pooled_vision)
        combined_features = pooled_vision + 0.1 * memory_content
        
        # Apply temporal modeling
        combined_features = combined_features.unsqueeze(1)  # (batch, 1, 512)
        temporal_out = self.temporal_transformer(combined_features)
        final_features = temporal_out.squeeze(1)  # (batch, 512)
        
        # Generate action sequence
        action_logits = self.action_generator(final_features)
        
        # Reshape to (batch, num_steps, num_actions) - assuming 3 steps
        action_logits = action_logits.view(-1, 3, 20)  # (batch, 3, num_actions)
        
        if return_attention_weights:
            return action_logits, attention_weights, memory_attention
        else:
            return action_logits

# Example training function
def train_advanced_vla_model():
    """Example training loop for the advanced VLA model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = AdvancedVLAModel(num_actions=20)
    model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=0.01
    )
    
    # Loss function for multi-step action prediction
    criterion = nn.CrossEntropyLoss()
    
    # Dummy training loop
    model.train()
    for epoch in range(10):  # Number of epochs
        total_loss = 0
        
        # In a real implementation, you would have a dataloader here
        # For this example, we'll use random dummy data
        batch_size = 4
        num_batches = 10
        
        for batch_idx in range(num_batches):
            # Create dummy inputs
            dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
            dummy_text = torch.randint(0, 50257, (batch_size, 32)).to(device)  # 32-token sequences
            dummy_actions = torch.randint(0, 20, (batch_size, 3)).to(device)  # 3-step action sequences
            
            # Forward pass
            action_logits = model(dummy_images, dummy_text)
            
            # Calculate loss (flatten for cross-entropy)
            loss = 0
            for step in range(3):  # For each step in the sequence
                step_loss = criterion(action_logits[:, step, :], dummy_actions[:, step])
                loss += step_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')

def main():
    print("Training advanced VLA model...")
    train_advanced_vla_model()
    print("Training completed!")

if __name__ == "__main__":
    main()
```

### VLA Reinforcement Learning Environment

```python
import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import cv2
from PIL import Image

class VLAEnvironment(gym.Env):
    """
    Custom environment for training VLA models with reinforcement learning
    """
    def __init__(self):
        super(VLAEnvironment, self).__init__()
        
        # Define action and observation space
        # Actions: [move_forward, move_backward, turn_left, turn_right, raise_arm, lower_arm, stop]
        self.action_space = spaces.Discrete(7)
        
        # Observations: concatenated vision (flattened image) and language (embedded)
        # For image: 64x64x3 -> flattened to 12288
        # For language: 32-dim embedding (simplified)
        self.observation_space = spaces.Dict({
            'vision': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            'language': spaces.Box(low=-1, high=1, shape=(32,), dtype=np.float32)
        })
        
        # Environment state
        self.robot_pos = np.array([0.0, 0.0])  # Robot position
        self.goal_pos = np.array([5.0, 5.0])   # Goal position
        self.robot_orientation = 0.0           # Robot orientation in radians
        self.current_task = "navigate_to_goal"
        
        # Task-specific parameters
        self.task_descriptions = {
            "navigate_to_goal": "Go to the red circle",
            "avoid_obstacle": "Go around the blue square",
            "follow_path": "Follow the yellow line"
        }
        
        self.language_embeddings = {
            "navigate_to_goal": np.random.randn(32).astype(np.float32),
            "avoid_obstacle": np.random.randn(32).astype(np.float32),
            "follow_path": np.random.randn(32).astype(np.float32)
        }
    
    def reset(self, seed=None):
        """Reset the environment to an initial state"""
        super().reset(seed=seed)
        
        # Reset robot position
        self.robot_pos = np.array([0.0, 0.0])
        self.robot_orientation = 0.0
        
        # Randomly select a task
        task_idx = np.random.randint(0, len(self.task_descriptions))
        self.current_task = list(self.task_descriptions.keys())[task_idx]
        
        # Generate observation
        vision_obs = self._get_vision_observation()
        language_obs = self.language_embeddings[self.current_task]
        
        return {
            'vision': vision_obs,
            'language': language_obs
        }, {}
    
    def step(self, action):
        """Execute one time step within the environment"""
        # Execute action
        self._take_action(action)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        terminated = self._check_termination()
        truncated = False  # Time limit not implemented
        
        # Generate next observation
        vision_obs = self._get_vision_observation()
        language_obs = self.language_embeddings[self.current_task]
        
        obs = {
            'vision': vision_obs,
            'language': language_obs
        }
        
        info = {
            'distance_to_goal': np.linalg.norm(self.robot_pos - self.goal_pos),
            'current_task': self.current_task
        }
        
        return obs, reward, terminated, truncated, info
    
    def _take_action(self, action):
        """Execute the given action in the environment"""
        # Define action effects
        action_effects = {
            0: np.array([0.1, 0.0]),  # Move forward
            1: np.array([-0.1, 0.0]), # Move backward
            2: np.array([0.0, 0.1]),  # Turn left (change orientation)
            3: np.array([0.0, -0.1]), # Turn right (change orientation)
            4: np.array([0.05, 0.05]), # Move upper right
            5: np.array([0.05, -0.05]), # Move upper left
            6: np.array([0.0, 0.0])    # Stop
        }
        
        effect = action_effects.get(action, np.array([0.0, 0.0]))
        
        if action in [2, 3]:  # Turning actions
            self.robot_orientation += effect[1]
        else:  # Movement actions
            # Convert movement to robot's local frame
            local_move = np.array([
                effect[0] * np.cos(self.robot_orientation) - effect[1] * np.sin(self.robot_orientation),
                effect[0] * np.sin(self.robot_orientation) + effect[1] * np.cos(self.robot_orientation)
            ])
            self.robot_pos += local_move
        
        # Keep robot in bounds
        self.robot_pos = np.clip(self.robot_pos, -10, 10)
    
    def _get_vision_observation(self):
        """Generate a vision observation (simplified 2D representation)"""
        # Create a simple 2D representation of the environment
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Scale positions to image coordinates (0-63)
        robot_x = int((self.robot_pos[0] + 10) * 3.175)  # Scale factor: 63/20
        robot_y = int((self.robot_pos[1] + 10) * 3.175)
        
        goal_x = int((self.goal_pos[0] + 10) * 3.175)
        goal_y = int((self.goal_pos[1] + 10) * 3.175)
        
        # Draw robot (green)
        cv2.circle(img, (robot_x, robot_y), 2, (0, 255, 0), -1)
        
        # Draw goal (red)
        cv2.circle(img, (goal_x, goal_y), 3, (255, 0, 0), -1)
        
        # Draw orientation indicator
        orient_x = int(robot_x + 3 * np.cos(self.robot_orientation))
        orient_y = int(robot_y + 3 * np.sin(self.robot_orientation))
        cv2.line(img, (robot_x, robot_y), (orient_x, orient_y), (0, 255, 0), 1)
        
        return img
    
    def _calculate_reward(self):
        """Calculate reward based on current state"""
        # Distance-based reward
        distance_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)
        distance_reward = -distance_to_goal / 10.0  # Normalize
        
        # Goal reached reward
        goal_reward = 10.0 if distance_to_goal < 0.5 else 0.0
        
        # Penalty for inefficient paths
        path_penalty = -0.01  # Small time penalty
        
        total_reward = distance_reward + goal_reward + path_penalty
        return total_reward
    
    def _check_termination(self):
        """Check if the episode should terminate"""
        distance_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)
        return distance_to_goal < 0.5  # Goal reached

class VLACustomFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for VLA observations in reinforcement learning
    """
    def __init__(self, observation_space, features_dim=256):
        super(VLACustomFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Vision processing
        self.vision_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU()
        )
        
        # Calculate conv output size
        conv_out_size = 64 * 8 * 8  # After convolutions: 64 channels, 8x8 spatial
        
        # Language processing
        self.lang_linear = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Combined processing
        self.combined_mlp = nn.Sequential(
            nn.Linear(conv_out_size + 64, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        # Extract vision and language components
        vision_obs = observations['vision']
        lang_obs = observations['language']
        
        # Process vision (normalize from [0,255] to [0,1] then to [-1,1])
        vision_processed = (vision_obs / 255.0 - 0.5) * 2.0
        vision_processed = vision_processed.permute(0, 3, 1, 2)  # NHWC to NCHW
        
        # Apply conv layers
        vision_features = self.vision_conv(vision_processed)
        vision_features = vision_features.view(vision_features.size(0), -1)  # Flatten
        
        # Process language
        lang_features = self.lang_linear(lang_obs)
        
        # Combine features
        combined_features = torch.cat([vision_features, lang_features], dim=1)
        
        # Final processing
        final_features = self.combined_mlp(combined_features)
        
        return final_features

def train_vla_rl_agent():
    """
    Train a VLA agent using reinforcement learning in the custom environment
    """
    # Create environment
    env = VLAEnvironment()
    
    # Create policy with custom feature extractor
    policy_kwargs = {
        "features_extractor_class": VLACustomFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": [256, 256]  # Two hidden layers of 256 units
    }
    
    # Create RL agent
    model = PPO(
        "MultiInputPolicy",  # Supports dict observations
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=512,  # Number of steps to run for each environment per update
        batch_size=64,
        n_epochs=10,
        gamma=0.99,  # Discount factor
        learning_rate=3e-4,
        clip_range=0.2
    )
    
    # Train the agent
    model.learn(total_timesteps=10000)
    
    # Save the trained model
    model.save("vla_rl_agent")
    print("Model saved as 'vla_rl_agent'")
    
    # Test the trained agent
    obs, _ = env.reset()
    for i in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished after {i+1} steps")
            obs, _ = env.reset()
    
    return model

def main():
    print("Training VLA RL agent...")
    trained_model = train_vla_rl_agent()
    print("Training completed!")

if __name__ == "__main__":
    main()
```

### VLA Training with Simulation Data

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from PIL import Image
import torchvision.transforms as transforms

class SimVLADataset(Dataset):
    """
    Dataset for training VLA models with simulation data
    """
    def __init__(self, data_dir, transform=None, max_length=64):
        self.data_dir = data_dir
        self.transform = transform
        self.max_length = max_length
        
        # Load the manifest
        manifest_path = os.path.join(data_dir, 'manifest.json')
        with open(manifest_path, 'r') as f:
            self.samples = json.load(f)
        
        # Tokenizer setup (simplified - would use actual tokenizer in practice)
        self.tokenizer = {
            'vocab': self._build_vocab(),
            'stoi': {},  # String to index
            'itos': {}   # Index to string
        }
        
        for i, word in enumerate(self.tokenizer['vocab']):
            self.tokenizer['stoi'][word] = i
            self.tokenizer['itos'][i] = word
        
        # Set default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _build_vocab(self):
        """Build vocabulary from all instructions in the dataset"""
        vocab = set(['<PAD>', '<UNK>', '<START>', '<END>'])
        
        for sample in self.samples:
            instruction = sample['instruction'].lower()
            tokens = instruction.split()
            vocab.update(tokens)
        
        return sorted(list(vocab))
    
    def tokenize(self, text):
        """Convert text to token indices"""
        tokens = text.lower().split()
        indices = [self.tokenizer['stoi'].get(token, self.tokenizer['stoi']['<UNK>']) 
                  for token in tokens]
        
        # Add start and end tokens
        indices = [self.tokenizer['stoi']['<START>']] + indices + [self.tokenizer['stoi']['<END>']]
        
        # Pad or truncate to max_length
        if len(indices) < self.max_length:
            indices.extend([self.tokenizer['stoi']['<PAD>']] * (self.max_length - len(indices)))
        else:
            indices = indices[:self.max_length]
        
        return torch.tensor(indices, dtype=torch.long)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and process image
        img_path = os.path.join(self.data_dir, sample['image_path'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Tokenize instruction
        instruction = sample['instruction']
        text_tokens = self.tokenize(instruction)
        
        # Get action sequence
        action_sequence = torch.tensor(sample['action_sequence'], dtype=torch.long)
        
        # Get robot state (optional, used by some models)
        robot_state = torch.tensor(sample.get('robot_state', np.zeros(10)), dtype=torch.float32)
        
        return {
            'image': image,
            'text_tokens': text_tokens,
            'action_sequence': action_sequence,
            'robot_state': robot_state
        }

def create_vla_training_loop():
    """
    Create a complete training loop for VLA models using simulation data
    """
    # Initialize dataset and dataloader
    dataset = SimVLADataset(
        data_dir='/path/to/simulation/vla/data',
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvancedVLAModel(num_actions=20)  # Using the model from previous example
    model.to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # For action sequence prediction
    
    # Training loop
    model.train()
    num_epochs = 10
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            images = batch['image'].to(device)
            text_tokens = batch['text_tokens'].to(device)
            action_sequences = batch['action_sequence'].to(device)
            
            # Forward pass
            action_logits = model(images, text_tokens)  # Shape: (batch, seq_len, num_actions)
            
            # Calculate loss for each step in sequence
            loss = 0
            for step in range(action_logits.size(1)):  # sequence length
                step_logits = action_logits[:, step, :]  # (batch, num_actions)
                step_targets = action_sequences[:, step]  # (batch,) - correct action for this step
                loss += criterion(step_logits, step_targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}')
        
        # Save model checkpoint
        if (epoch + 1) % 2 == 0:  # Save every 2 epochs
            checkpoint_path = f'vla_model_checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
    
    # Save final model
    final_model_path = 'final_vla_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved: {final_model_path}')

def main():
    print("Creating VLA training loop with simulation data...")
    create_vla_training_loop()
    print("Training completed!")

if __name__ == "__main__":
    main()
```

## Diagrams

```
[Advanced VLA Architecture with Memory and Attention]

[Input]                              [Processing]                             [Output]
Vision ──┐                           VLA Blocks
         │     ┌─────────────────┐    (6x)      ┌─────────────────┐
         ├────►│ Vision-Language ├──────────────►│ Action Sequence ├────► Robot Actions
Language ──┘    │    Fusion      │               │   Generator     │
                └─────────────────┘               └─────────────────┘
                       │                                  │
                       ▼                                  ▼
                 Cross-Attention                    Multi-Step Actions
                 Mechanisms                          (3-step prediction)
                       │
                       ▼
                   External
                   Memory
                (Read/Write)

```

> **Note:** For the actual textbook, you would include actual image files in the static/images directory and reference them using markdown syntax like `![Advanced VLA Architecture](/img/advanced-vla-architecture.png)`.

## Hands-on Exercises

1. **Exercise 1**: Implement the cross-modal attention mechanism from the code example and visualize the attention weights between vision and language components.

2. **Exercise 2**: Create a custom RL environment for a humanoid robot task and train a VLA policy using the PPO algorithm with the custom feature extractor.

3. **Exercise 3**: Extend the VLA model to handle multi-step action sequences and implement a teacher-forcing training approach.

4. **Exercise 4**: Implement domain randomization techniques in your simulation environment to improve sim-to-real transfer of your VLA model.

## Quiz

1. What is the purpose of external memory in advanced VLA models?
   - a) Storing large vision models
   - b) Storing and retrieving relevant information across episodes
   - c) Storing language tokens only
   - d) Storing action sequences only

2. Which attention mechanism enables fusion of vision and language inputs?
   - a) Self-attention
   - b) Cross-attention
   - c) Intra-attention
   - d) Local attention

3. What does "sim-to-real transfer" refer to?
   - a) Converting simulation to real data
   - b) Transferring models trained in simulation to real robots
   - c) Converting real data to simulation
   - d) Real-time simulation

4. True/False: VLA models can only learn from demonstration data, not from reinforcement learning.
   - Answer: _____

5. What is the primary purpose of temporal modeling in VLA systems?
   - a) Processing static images
   - b) Understanding multi-step tasks and maintaining coherent behavior over time
   - c) Speeding up computation
   - d) Reducing model size

## Summary

In this lesson, we explored advanced VLA architectures and training methodologies that enable more sophisticated multimodal reasoning and action planning in humanoid robots. We covered key components like cross-modal attention, external memory systems, and temporal modeling that allow VLA models to process complex, multi-step instructions while maintaining context across time.

We also examined advanced training approaches, including reinforcement learning integration and simulation-based training, which are crucial for developing robust VLA systems for humanoid robotics. These techniques enable robots to learn complex behaviors that would be difficult to program manually and to adapt to new situations through interaction with their environment.

The combination of advanced architectures and training methodologies represents the cutting edge of AI for robotics and provides the foundation for truly autonomous, context-aware humanoid robots.

## Key Terms

- **Cross-Modal Attention**: Attention mechanism fusing different input modalities
- **External Memory**: Memory component storing information across episodes
- **Temporal Modeling**: Processing sequential information over time
- **Sim-to-Real Transfer**: Applying simulation-trained models to real robots
- **Reinforcement Learning**: Learning through environmental interaction and rewards
- **Domain Randomization**: Varying simulation parameters to improve real-world performance
- **Multi-Step Action Sequences**: Predicting multiple actions for complex tasks
- **Teacher Forcing**: Training technique using ground truth during training
- **Policy Networks**: Networks mapping states to actions in RL
- **Behavior Cloning**: Learning by mimicking expert demonstrations