---
sidebar_position: 2
---

# Machine Learning for Robot Control

## Introduction

Machine learning has revolutionized robotics by enabling robots to learn complex behaviors, adapt to new situations, and improve their performance through experience. This lesson explores how various machine learning techniques can be applied to robot control, from low-level motor control to high-level decision making.

## Types of Machine Learning in Robotics

### 1. Supervised Learning for Robot Control

Supervised learning is used when we have labeled training data to teach robots specific behaviors:

```python
# Example: Supervised learning for robot trajectory following
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import pickle

class SupervisedTrajectoryLearner:
    def __init__(self):
        self.model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            random_state=42
        )
        self.is_trained = False
        self.training_data = []

    def collect_training_data(self, sensor_state, desired_action):
        """Collect training data from demonstrations"""
        training_sample = {
            'input': sensor_state,  # Sensor readings, current position, etc.
            'output': desired_action  # Desired motor commands
        }
        self.training_data.append(training_sample)

    def train_model(self):
        """Train the model on collected data"""
        if len(self.training_data) < 10:
            print("Not enough training data")
            return False

        X = np.array([sample['input'] for sample in self.training_data])
        y = np.array([sample['output'] for sample in self.training_data])

        self.model.fit(X, y)
        self.is_trained = True
        return True

    def predict_action(self, current_state):
        """Predict action based on current state"""
        if not self.is_trained:
            return self.default_action()  # Fallback to default behavior

        state_array = np.array(current_state).reshape(1, -1)
        predicted_action = self.model.predict(state_array)[0]
        return predicted_action

    def default_action(self):
        """Default action when model is not available"""
        return np.zeros(6)  # Zero velocity for all joints
```

### 2. Reinforcement Learning for Robot Control

Reinforcement learning is particularly powerful for robotics as it allows robots to learn through trial and error:

```python
# Example: Q-Learning for robot navigation
class QLearningNavigator:
    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # Exploration rate

        # Initialize Q-table (for discrete states)
        self.q_table = np.zeros((state_space_size, action_space_size))

    def discretize_state(self, continuous_state):
        """Convert continuous state to discrete state index"""
        # This is a simplified example - in practice, you'd have a more sophisticated discretization
        # For now, we'll assume the state is already discrete or use simple binning
        if isinstance(continuous_state, (int, np.integer)):
            return min(continuous_state, self.state_space_size - 1)
        else:
            # Simple discretization for continuous state
            return int(np.clip(continuous_state, 0, self.state_space_size - 1))

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        state_idx = self.discretize_state(state)

        if np.random.random() < self.epsilon:
            # Explore: choose random action
            return np.random.randint(0, self.action_space_size)
        else:
            # Exploit: choose best known action
            return np.argmax(self.q_table[state_idx])

    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Bellman equation"""
        state_idx = self.discretize_state(state)
        action_idx = action
        next_state_idx = self.discretize_state(next_state)

        # Current Q-value
        current_q = self.q_table[state_idx, action_idx]

        # Max Q-value for next state
        max_next_q = np.max(self.q_table[next_state_idx])

        # Update Q-value
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[state_idx, action_idx] = new_q

# Example: Deep Q-Network for continuous state spaces
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DeepQNavigator:
    def __init__(self, state_size, action_size, learning_rate=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate

        # Neural networks
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Update target network
        self.update_target_network()

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action based on current state"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.95 * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

### 3. Imitation Learning

Imitation learning allows robots to learn by observing human demonstrations:

```python
# Example: Imitation learning for robotic manipulation
class ImitationLearner:
    def __init__(self):
        self.demonstrations = []
        self.learner = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)
        self.is_trained = False

    def record_demonstration(self, state_trajectory, action_trajectory):
        """Record a human demonstration"""
        demonstration = {
            'states': state_trajectory,
            'actions': action_trajectory
        }
        self.demonstrations.append(demonstration)

    def train_from_demonstrations(self):
        """Train the model using behavioral cloning"""
        if len(self.demonstrations) == 0:
            return False

        # Prepare training data
        X = []  # States
        y = []  # Actions

        for demo in self.demonstrations:
            for state, action in zip(demo['states'], demo['actions']):
                X.append(state)
                y.append(action)

        X = np.array(X)
        y = np.array(y)

        # Train the model
        self.learner.fit(X, y)
        self.is_trained = True
        return True

    def predict_action(self, current_state):
        """Predict action based on current state using trained model"""
        if not self.is_trained:
            return np.zeros_like(current_state)  # Default action

        state_array = np.array(current_state).reshape(1, -1)
        predicted_action = self.learner.predict(state_array)[0]
        return predicted_action

    def dagger_update(self, current_state, expert_action):
        """Update model using DAgger algorithm"""
        # Add current state with expert action to training data
        X_new = np.array([current_state])
        y_new = np.array([expert_action])

        # Get all previous training data
        X_all = []
        y_all = []
        for demo in self.demonstrations:
            for state, action in zip(demo['states'], demo['actions']):
                X_all.append(state)
                y_all.append(action)

        X_all = np.array(X_all + [current_state])
        y_all = np.array(y_all + [expert_action])

        # Retrain the model
        self.learner.fit(X_all, y_all)
```

## Robot Control Applications

### Motor Control and Movement Learning

Learning to control robot joints and movements:

```python
# Example: Learning motor control policies
class MotorControlLearner:
    def __init__(self, joint_count):
        self.joint_count = joint_count
        self.control_policy = self.initialize_policy()
        self.experience_buffer = []

    def initialize_policy(self):
        """Initialize control policy (could be neural network, etc.)"""
        # For this example, we'll use a simple neural network
        return nn.Sequential(
            nn.Linear(self.joint_count * 3, 64),  # State: positions, velocities, efforts
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.joint_count)  # Output: desired joint commands
        )

    def learn_movement_pattern(self, desired_trajectory, actual_trajectory):
        """Learn to reproduce a movement pattern"""
        # Create training data from the trajectory
        training_data = self.create_training_data(desired_trajectory, actual_trajectory)

        # Update the control policy
        self.update_policy(training_data)

    def create_training_data(self, desired_trajectory, actual_trajectory):
        """Create training data from trajectory comparison"""
        training_data = []

        for i in range(len(actual_trajectory) - 1):
            current_state = actual_trajectory[i]
            desired_state = desired_trajectory[i]
            next_state = actual_trajectory[i + 1]

            # Input: current state and desired state
            input_state = np.concatenate([current_state, desired_state])

            # Output: control command to reach desired state
            control_command = desired_state - current_state  # Simple proportional control

            training_data.append((input_state, control_command))

        return training_data

    def update_policy(self, training_data):
        """Update the control policy with new training data"""
        # Implementation would involve training the neural network
        pass

    def execute_control(self, current_state, desired_state):
        """Execute learned control policy"""
        input_tensor = torch.FloatTensor(
            np.concatenate([current_state, desired_state])
        ).unsqueeze(0)

        with torch.no_grad():
            control_output = self.control_policy(input_tensor)

        return control_output.squeeze(0).numpy()
```

### Adaptive Control Systems

Learning to adapt to changing conditions:

```python
# Example: Adaptive control that learns to compensate for environmental changes
class AdaptiveController:
    def __init__(self):
        self.environment_model = self.initialize_environment_model()
        self.compensation_model = self.initialize_compensation_model()
        self.adaptation_enabled = True

    def initialize_environment_model(self):
        """Model of environmental conditions"""
        return nn.Sequential(
            nn.Linear(10, 32),  # Environmental inputs
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Environmental state prediction
        )

    def initialize_compensation_model(self):
        """Model to compensate for environmental effects"""
        return nn.Sequential(
            nn.Linear(20, 64),  # Robot state + environment state
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)   # Compensation for 6 DOF
        )

    def adapt_to_environment(self, robot_state, environmental_state):
        """Adapt control based on environmental conditions"""
        if not self.adaptation_enabled:
            return np.zeros(6)  # No compensation

        # Predict environmental effect
        env_tensor = torch.FloatTensor(environmental_state).unsqueeze(0)
        env_effect = self.environment_model(env_tensor)

        # Combine robot state and environmental state
        combined_state = torch.cat([
            torch.FloatTensor(robot_state),
            env_effect
        ]).unsqueeze(0)

        # Calculate compensation
        compensation = self.compensation_model(combined_state)

        return compensation.squeeze(0).numpy()

    def update_models(self, experience_data):
        """Update models based on experience"""
        # Train environment model to predict environmental changes
        # Train compensation model to counteract environmental effects
        pass
```

## ROS2 Implementation: Machine Learning for Robot Control

Here's a comprehensive ROS2 implementation combining multiple ML techniques:

```python
# ml_robot_control.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import String, Float32
from builtin_interfaces.msg import Time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class MLRobotControl(Node):
    def __init__(self):
        super().__init__('ml_robot_control')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.learning_status_pub = self.create_publisher(String, '/learning_status', 10)
        self.reward_pub = self.create_publisher(Float32, '/episode_reward', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10
        )

        # Robot state
        self.joint_states = None
        self.imu_data = None
        self.laser_data = None

        # ML components
        self.dqn_agent = DQNAgent(state_size=20, action_size=4)  # 4 discrete actions
        self.imitation_learner = ImitationLearner()
        self.adaptive_controller = AdaptiveController()

        # Learning parameters
        self.learning_enabled = True
        self.exploration_rate = 0.3
        self.episode_reward = 0.0
        self.episode_step = 0
        self.max_episode_steps = 1000

        # Experience buffer
        self.experience_buffer = deque(maxlen=10000)

        # Control loop
        self.control_timer = self.create_timer(0.05, self.ml_control_loop)  # 20 Hz

        # Learning state
        self.learning_state = {
            'total_episodes': 0,
            'total_steps': 0,
            'average_reward': 0.0,
            'learning_progress': 0.0
        }

    def joint_state_callback(self, msg):
        """Handle joint state updates"""
        self.joint_states = msg

    def imu_callback(self, msg):
        """Handle IMU data"""
        self.imu_data = msg

    def laser_callback(self, msg):
        """Handle laser scan data"""
        self.laser_data = msg

    def ml_control_loop(self):
        """Main machine learning control loop"""
        if not all([self.joint_states, self.imu_data, self.laser_data]):
            return

        # 1. STATE OBSERVATION
        current_state = self.get_current_state()

        # 2. ACTION SELECTION (using learned policy)
        if np.random.random() < self.exploration_rate:
            # Exploration: random action
            action = self.get_random_action()
            action_type = 'exploration'
        else:
            # Exploitation: learned action
            action = self.dqn_agent.act(current_state)
            action_type = 'exploitation'

        # 3. ACTION EXECUTION
        command = self.action_to_command(action)
        self.cmd_vel_pub.publish(command)

        # 4. REWARD CALCULATION
        reward = self.calculate_reward(action, current_state)
        self.episode_reward += reward

        # 5. NEXT STATE OBSERVATION
        next_state = self.get_current_state()

        # 6. LEARNING UPDATE
        if self.learning_enabled:
            # Store experience
            self.dqn_agent.remember(current_state, action, reward, next_state, False)

            # Train the agent
            if len(self.dqn_agent.memory) > 32:
                self.dqn_agent.replay()

        # 7. EPISODE MANAGEMENT
        self.episode_step += 1

        if self.episode_step >= self.max_episode_steps:
            self.end_episode()

        # 8. STATUS REPORTING
        self.report_learning_status()

    def get_current_state(self):
        """Get current robot state as input to ML models"""
        state = []

        # Joint positions (normalize to [-1, 1])
        if self.joint_states:
            for pos in self.joint_states.position:
                state.append(np.tanh(pos))  # Normalize

        # Joint velocities (normalize to [-1, 1])
        if self.joint_states and self.joint_states.velocity:
            for vel in self.joint_states.velocity:
                state.append(np.tanh(vel))  # Normalize

        # IMU data
        if self.imu_data:
            state.extend([
                self.imu_data.orientation.x,
                self.imu_data.orientation.y,
                self.imu_data.orientation.z,
                self.imu_data.angular_velocity.z,  # Yaw rate
            ])

        # Laser data (sample every 10th point to reduce dimensionality)
        if self.laser_data:
            for i in range(0, len(self.laser_data.ranges), 10):
                range_val = self.laser_data.ranges[i]
                if 0 < range_val < self.laser_data.range_max:
                    state.append(range_val / self.laser_data.range_max)  # Normalize
                else:
                    state.append(1.0)  # Max range for invalid readings

        # Pad or truncate to fixed size
        while len(state) < 20:
            state.append(0.0)
        state = state[:20]

        return np.array(state)

    def get_random_action(self):
        """Get a random discrete action"""
        return random.randint(0, 3)  # 4 possible actions

    def action_to_command(self, action):
        """Convert discrete action to continuous command"""
        cmd = Twist()

        if action == 0:  # Move forward
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0
        elif action == 1:  # Turn left
            cmd.linear.x = 0.1
            cmd.angular.z = 0.5
        elif action == 2:  # Turn right
            cmd.linear.x = 0.1
            cmd.angular.z = -0.5
        elif action == 3:  # Move backward
            cmd.linear.x = -0.2
            cmd.angular.z = 0.0

        return cmd

    def calculate_reward(self, action, state):
        """Calculate reward based on action and state"""
        reward = 0.0

        # Positive reward for moving forward
        if action == 0:  # Moving forward
            reward += 0.1

        # Negative reward for being too close to obstacles
        if self.laser_data:
            min_distance = min([r for r in self.laser_data.ranges if r > 0], default=float('inf'))
            if min_distance < 0.5:
                reward -= 1.0  # Penalty for being too close to obstacles
            elif min_distance > 1.0:
                reward += 0.2  # Bonus for maintaining safe distance

        # Reward for maintaining balance (from IMU)
        if self.imu_data:
            orientation = self.imu_data.orientation
            tilt = abs(orientation.x) + abs(orientation.y)
            if tilt < 0.2:  # Good balance
                reward += 0.1
            else:  # Penalty for being unbalanced
                reward -= 0.1

        return reward

    def end_episode(self):
        """End the current learning episode"""
        # Update learning statistics
        self.learning_state['total_episodes'] += 1
        self.learning_state['total_steps'] += self.episode_step
        self.learning_state['average_reward'] = (
            (self.learning_state['average_reward'] * (self.learning_state['total_episodes'] - 1) + self.episode_reward) /
            self.learning_state['total_episodes']
        )

        # Publish episode reward
        reward_msg = Float32()
        reward_msg.data = self.episode_reward
        self.reward_pub.publish(reward_msg)

        # Reset episode
        self.episode_reward = 0.0
        self.episode_step = 0

        # Decay exploration rate over time
        self.exploration_rate = max(0.05, self.exploration_rate * 0.999)

    def report_learning_status(self):
        """Report current learning status"""
        status_msg = String()
        status_msg.data = (
            f"Episodes: {self.learning_state['total_episodes']}, "
            f"Avg Reward: {self.learning_state['average_reward']:.2f}, "
            f"Steps: {self.learning_state['total_steps']}, "
            f"Explore: {self.exploration_rate:.2f}"
        )

        self.learning_status_pub.publish(status_msg)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate

        # Neural networks
        self.q_network = self.build_model()
        self.target_network = self.build_model()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Update target network
        self.update_target_network()

    def build_model(self):
        """Build the neural network model"""
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action based on current state"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.95 * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class PolicyGradientAgent:
    """Policy gradient agent for continuous action spaces"""
    def __init__(self, state_size, action_size, learning_rate=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        # Policy network (outputs action probabilities for discrete) or action parameters for continuous
        self.policy_network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

        # Value network for advantage estimation
        self.value_network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.optimizer_policy = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.optimizer_value = optim.Adam(self.value_network.parameters(), lr=learning_rate)

    def get_action(self, state):
        """Get action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_params = self.policy_network(state_tensor)

        # For continuous actions, we might add noise or use a distribution
        # For discrete actions, we can use argmax or sampling
        action = torch.argmax(action_params, dim=1).item()
        return action

    def update(self, states, actions, rewards, next_states):
        """Update the policy using collected experiences"""
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)

        # Calculate value predictions
        values = self.value_network(states_tensor).squeeze()

        # Calculate advantages (simplified)
        next_values = self.value_network(torch.FloatTensor(next_states)).squeeze()
        advantages = rewards_tensor + 0.95 * next_values - values

        # Update value network
        value_loss = nn.MSELoss()(values, rewards_tensor + 0.95 * next_values)
        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()

        # Update policy network
        log_probs = torch.log_softmax(self.policy_network(states_tensor), dim=1)
        selected_log_probs = log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
        policy_loss = -(selected_log_probs * advantages.detach()).mean()

        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()

def main(args=None):
    rclpy.init(args=args)
    ml_control = MLRobotControl()

    try:
        rclpy.spin(ml_control)
    except KeyboardInterrupt:
        pass
    finally:
        ml_control.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced ML Techniques for Robotics

### Deep Reinforcement Learning

Deep RL combines deep learning with reinforcement learning for complex robot behaviors:

```python
# Example: Actor-Critic method for continuous control
class ActorCriticAgent:
    def __init__(self, state_size, action_size, learning_rate=1e-4):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
            nn.Tanh()  # Output actions in [-1, 1]
        )

        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

    def get_action(self, state):
        """Get action from actor network with noise for exploration"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor)

        # Add noise for exploration
        noise = torch.randn_like(action) * 0.1
        noisy_action = torch.clamp(action + noise, -1, 1)

        return noisy_action.squeeze(0).detach().numpy()

    def update(self, state, action, reward, next_state, done):
        """Update actor and critic networks"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        reward_tensor = torch.FloatTensor([reward])
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        done_tensor = torch.BoolTensor([done])

        # Calculate value of current state
        current_value = self.critic(state_tensor)

        # Calculate target value
        next_value = self.critic(next_state_tensor).detach()
        target_value = reward_tensor + 0.99 * next_value * ~done_tensor

        # Critic loss
        critic_loss = nn.MSELoss()(current_value, target_value)

        # Actor loss (policy gradient)
        advantage = target_value - current_value
        action_probs = self.actor(state_tensor)
        actor_loss = -(action_probs * advantage.detach()).mean()

        # Update networks
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
```

### Transfer Learning for Robotics

Transfer learning allows robots to apply knowledge from one task to another:

```python
# Example: Transfer learning between similar robotic tasks
class TransferLearningRobot:
    def __init__(self):
        self.base_network = self.create_base_network()
        self.task_specific_heads = {}
        self.shared_features = True

    def create_base_network(self):
        """Create shared feature extraction network"""
        return nn.Sequential(
            nn.Linear(50, 128),  # Input size may vary
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def add_task_head(self, task_name, output_size):
        """Add task-specific output head"""
        task_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        self.task_specific_heads[task_name] = task_head

    def forward(self, state, task_name):
        """Forward pass for specific task"""
        features = self.base_network(state)
        output = self.task_specific_heads[task_name](features)
        return output

    def transfer_knowledge(self, source_task, target_task):
        """Transfer knowledge from source task to target task"""
        # Copy weights from base network
        # Initialize target head based on source head if similar
        pass
```

## Lab: Implementing ML-Based Robot Control

In this lab, you'll implement a machine learning system for robot control:

```python
# lab_ml_robot_control.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class MLLabControl(Node):
    def __init__(self):
        super().__init__('ml_lab_control')

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.reward_pub = self.create_publisher(Float32, '/episode_reward', 10)
        self.status_pub = self.create_publisher(String, '/ml_status', 10)

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        # Data storage
        self.joint_data = None
        self.scan_data = None

        # ML components
        self.q_network = nn.Sequential(
            nn.Linear(24, 64),  # 24 input features
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)   # 4 discrete actions
        )
        self.target_network = nn.Sequential(
            nn.Linear(24, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)

        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Episode tracking
        self.episode_reward = 0.0
        self.step_count = 0
        self.episode_count = 0

        # Control loop
        self.control_timer = self.create_timer(0.05, self.ml_control_loop)

    def joint_callback(self, msg):
        """Handle joint state data"""
        self.joint_data = msg

    def scan_callback(self, msg):
        """Handle laser scan data"""
        self.scan_data = msg

    def ml_control_loop(self):
        """Main ML control loop"""
        if not all([self.joint_data, self.scan_data]):
            return

        # Get current state
        current_state = self.get_state_vector()

        # Choose action (epsilon-greedy)
        if random.random() < self.epsilon:
            action = random.randint(0, 3)  # Explore
            action_type = 'explore'
        else:
            action = self.select_action(current_state)  # Exploit
            action_type = 'exploit'

        # Execute action
        command = self.action_to_command(action)
        self.cmd_pub.publish(command)

        # Get reward and next state
        reward = self.calculate_reward(action, current_state)
        next_state = self.get_state_vector()
        done = False  # Simplified - no terminal state

        # Store experience
        self.memory.append((current_state, action, reward, next_state, done))

        # Update networks
        if len(self.memory) > self.batch_size:
            self.train_network()

        # Update episode stats
        self.episode_reward += reward
        self.step_count += 1

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # End episode every 500 steps
        if self.step_count % 500 == 0:
            self.end_episode()

        # Publish status
        status_msg = String()
        status_msg.data = f"Episode: {self.episode_count}, Step: {self.step_count}, "
        status_msg.data += f"Reward: {self.episode_reward:.2f}, Epsilon: {self.epsilon:.3f}, "
        status_msg.data += f"Action: {action_type}"
        self.status_pub.publish(status_msg)

    def get_state_vector(self):
        """Create state vector from sensor data"""
        state = []

        # Joint positions (first 12 joints, normalized)
        if self.joint_data and len(self.joint_data.position) >= 12:
            for i in range(12):
                pos = self.joint_data.position[i]
                state.append(np.tanh(pos))  # Normalize to [-1, 1]

        # Joint velocities (first 12 joints, normalized)
        if self.joint_data and len(self.joint_data.velocity) >= 12:
            for i in range(12):
                vel = self.joint_data.velocity[i]
                state.append(np.tanh(vel))  # Normalize to [-1, 1]

        # Validate state length
        while len(state) < 24:
            state.append(0.0)
        state = state[:24]

        return np.array(state, dtype=np.float32)

    def select_action(self, state):
        """Select action using trained network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values).item())

    def action_to_command(self, action):
        """Convert discrete action to Twist command"""
        cmd = Twist()

        if action == 0:  # Move forward
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0
        elif action == 1:  # Turn left
            cmd.linear.x = 0.1
            cmd.angular.z = 0.5
        elif action == 2:  # Turn right
            cmd.linear.x = 0.1
            cmd.angular.z = -0.5
        elif action == 3:  # Stop
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        return cmd

    def calculate_reward(self, action, state):
        """Calculate reward based on action and state"""
        reward = 0.0

        # Positive reward for forward movement
        if action == 0:  # Moving forward
            reward += 0.1

        # Negative reward for being too close to obstacles
        if self.scan_data:
            min_distance = min([r for r in self.scan_data.ranges if 0 < r < self.scan_data.range_max], default=float('inf'))
            if min_distance < 0.5:
                reward -= 1.0  # Penalty for obstacles
            elif min_distance > 1.0:
                reward += 0.2  # Bonus for safe distance

        # Small time penalty to encourage efficiency
        reward -= 0.01

        return reward

    def train_network(self):
        """Train the Q-network using experience replay"""
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def end_episode(self):
        """End current episode and start new one"""
        # Publish episode reward
        reward_msg = Float32()
        reward_msg.data = self.episode_reward
        self.reward_pub.publish(reward_msg)

        # Update statistics
        self.episode_count += 1
        self.episode_reward = 0.0

        # Update target network periodically
        if self.episode_count % 10 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

def main(args=None):
    rclpy.init(args=args)
    lab = MLLabControl()

    try:
        rclpy.spin(lab)
    except KeyboardInterrupt:
        pass
    finally:
        lab.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercise: Design Your Own ML Robot Controller

Consider the following design challenge:

1. What specific robot behavior do you want to learn?
2. What type of machine learning approach would be most suitable (supervised, reinforcement, imitation)?
3. What sensor data would be most relevant for your task?
4. How would you define the reward function for reinforcement learning?
5. What action space would you use (discrete or continuous)?
6. How would you handle the exploration vs. exploitation trade-off?
7. What safety measures would you implement during learning?

## Summary

Machine learning has become essential for modern robotics, enabling robots to:

- **Learn Complex Behaviors**: Through experience rather than explicit programming
- **Adapt to New Situations**: Adjust behavior based on environmental changes
- **Improve Performance**: Continuously refine actions based on feedback
- **Handle Uncertainty**: Make decisions under uncertain conditions
- **Generalize Skills**: Apply learned knowledge to new but similar tasks

The integration of machine learning with ROS2 enables the development of intelligent robots that can learn and adapt to perform complex tasks in dynamic environments. Understanding these concepts is crucial for developing next-generation robotic systems.

In the next lesson, we'll explore path planning and navigation algorithms that incorporate machine learning techniques for more intelligent and adaptive robot movement.