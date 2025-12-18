---
sidebar_position: 2
---

# Perception-Action Loops in Embodied Intelligence

## Introduction

Perception-action loops form the core mechanism by which embodied agents interact with their environment. Unlike traditional AI systems that process information in discrete steps, embodied agents continuously perceive their environment and act upon it, creating dynamic feedback loops that enable complex behaviors to emerge from simple rules.

## The Perception-Action Framework

The perception-action loop can be described as:

```
Environment → Perception → Cognition → Action → Environment
     ↑__________________________________________|
```

This framework differs from classical AI approaches in several key ways:

- **Continuous Processing**: Rather than batch processing, the system operates in real-time
- **Embodied Cognition**: Cognitive processes are grounded in physical interaction
- **Dynamic Adaptation**: Behavior adapts continuously based on environmental feedback
- **Emergent Properties**: Complex behaviors arise from simple perception-action rules

## Types of Perception-Action Loops

### 1. Reactive Loops

The simplest form of perception-action loop responds directly to sensory input:

```python
# Example: Reactive perception-action loop
class ReactiveAgent:
    def __init__(self):
        self.sensors = ['proximity', 'light', 'sound']
        self.actuators = ['wheels', 'gripper', 'speaker']

    def perceive(self, sensor_data):
        """Process immediate sensory input"""
        return sensor_data

    def act(self, perception):
        """Generate immediate response"""
        action = {}

        # Simple reactive rules
        if perception.get('proximity', float('inf')) < 0.3:
            action['wheels'] = {'linear': 0.0, 'angular': 0.5}  # Turn away
        elif perception.get('light', 0) > 0.8:
            action['wheels'] = {'linear': 0.3, 'angular': 0.0}  # Move toward light
        else:
            action['wheels'] = {'linear': 0.2, 'angular': 0.0}  # Explore

        return action

    def run_loop(self, sensor_data):
        """Execute one perception-action cycle"""
        perception = self.perceive(sensor_data)
        action = self.act(perception)
        return action
```

### 2. Predictive Loops

More sophisticated agents use predictive models to anticipate outcomes:

```python
# Example: Predictive perception-action loop
import numpy as np

class PredictiveAgent:
    def __init__(self):
        self.internal_model = self.initialize_model()
        self.belief_state = np.zeros(10)  # Internal representation

    def initialize_model(self):
        """Initialize internal world model"""
        return {
            'transition_model': {},  # How world changes with actions
            'observation_model': {}, # How observations relate to state
            'reward_model': {}       # Expected outcomes
        }

    def perceive(self, observation, action_taken):
        """Update internal model based on observation"""
        # Update belief state using Bayes rule or similar
        self.belief_state = self.update_belief(
            self.belief_state, observation, action_taken
        )
        return self.belief_state

    def predict_outcome(self, candidate_action):
        """Predict outcome of a potential action"""
        predicted_state = self.internal_model['transition_model'].predict(
            self.belief_state, candidate_action
        )
        predicted_observation = self.internal_model['observation_model'].predict(
            predicted_state
        )
        expected_reward = self.internal_model['reward_model'].predict(
            predicted_state
        )

        return predicted_state, predicted_observation, expected_reward

    def select_action(self, belief_state):
        """Select action based on internal model"""
        best_action = None
        best_value = float('-inf')

        # Evaluate potential actions
        for action in self.get_possible_actions():
            predicted_state, pred_obs, pred_reward = self.predict_outcome(action)
            value = self.calculate_expected_value(pred_reward, predicted_state)

            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def get_possible_actions(self):
        """Return possible actions"""
        return ['move_forward', 'turn_left', 'turn_right', 'stop']

    def calculate_expected_value(self, reward, state):
        """Calculate expected value of action"""
        return reward  # Simplified for example

    def update_belief(self, current_belief, observation, action):
        """Update belief state based on new information"""
        # In practice, this would use more sophisticated methods
        return current_belief
```

### 3. Hierarchical Loops

Complex behaviors emerge from multiple interacting loops at different temporal and spatial scales:

```python
# Example: Hierarchical perception-action system
class HierarchicalAgent:
    def __init__(self):
        # Different temporal scales
        self.reflexive_layer = ReflexiveLayer()      # Fast (100Hz)
        self.reactive_layer = ReactiveLayer()        # Medium (10Hz)
        self.planning_layer = PlanningLayer()        # Slow (1Hz)

    def perceive(self, sensor_data, dt):
        """Process perception at multiple levels"""
        # Process at different temporal scales
        reflex_action = self.reflexive_layer.process(sensor_data, dt)
        reactive_action = self.reactive_layer.process(sensor_data, dt)
        planned_action = self.planning_layer.process(sensor_data, dt)

        # Integrate actions hierarchically
        integrated_action = self.integrate_actions(
            reflex_action, reactive_action, planned_action
        )

        return integrated_action

    def integrate_actions(self, reflex, reactive, planned):
        """Integrate actions from different layers"""
        # Safety reflexes override other actions
        if reflex['priority'] == 'high':
            return reflex

        # Otherwise, blend reactive and planned actions
        final_action = {}
        for key in set(reflex.keys()) | set(reactive.keys()) | set(planned.keys()):
            if key == 'priority':
                final_action[key] = 'medium'  # Default priority
            else:
                # Weighted combination based on temporal scale
                final_action[key] = (
                    0.1 * reflex.get(key, 0) +
                    0.3 * reactive.get(key, 0) +
                    0.6 * planned.get(key, 0)
                )

        return final_action

class ReflexiveLayer:
    def process(self, sensor_data, dt):
        """Fast reflexive responses (collision avoidance, balance)"""
        action = {'priority': 'high'}

        # Immediate safety responses
        if sensor_data.get('collision_imminent', False):
            action['linear_vel'] = 0.0
            action['angular_vel'] = 0.0

        return action

class ReactiveLayer:
    def process(self, sensor_data, dt):
        """Medium-term reactive behaviors"""
        action = {'priority': 'medium'}

        # Obstacle avoidance
        if sensor_data.get('obstacle_distance', float('inf')) < 0.5:
            action['angular_vel'] = 0.3  # Turn away

        return action

class PlanningLayer:
    def process(self, sensor_data, dt):
        """Long-term goal-directed behavior"""
        action = {'priority': 'low'}

        # Goal-directed navigation
        action['linear_vel'] = 0.2  # Continue toward goal

        return action
```

## Real-World Implementation with ROS2

Here's how to implement perception-action loops using ROS2:

```python
# perception_action_loop_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
import numpy as np
import cv2
import math

class PerceptionActionLoopNode(Node):
    def __init__(self):
        super().__init__('perception_action_loop')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/system_status', 10)

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )

        # Sensor data storage
        self.scan_data = None
        self.image_data = None
        self.imu_data = None

        # Processing components
        self.cv_bridge = CvBridge()
        self.perception_processor = PerceptionProcessor()
        self.action_selector = ActionSelector()
        self.predictive_model = PredictiveModel()

        # Loop timing
        self.loop_timer = self.create_timer(0.05, self.perception_action_loop)  # 20 Hz
        self.last_loop_time = self.get_clock().now()

        # System state
        self.system_state = {
            'position': np.array([0.0, 0.0, 0.0]),
            'orientation': 0.0,
            'velocity': np.array([0.0, 0.0, 0.0]),
            'goal': np.array([5.0, 5.0, 0.0])
        }

    def scan_callback(self, msg):
        """Handle laser scan data"""
        self.scan_data = msg

    def image_callback(self, msg):
        """Handle camera image data"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_data = cv_image
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def imu_callback(self, msg):
        """Handle IMU data"""
        self.imu_data = msg
        # Update system state with IMU data
        self.update_orientation_from_imu(msg)

    def perception_action_loop(self):
        """Main perception-action loop"""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_loop_time).nanoseconds / 1e9
        self.last_loop_time = current_time

        # Ensure we have sensor data
        if not all([self.scan_data, self.image_data, self.imu_data]):
            return

        # 1. PERCEPTION PHASE
        perceptual_state = self.process_perception()

        # 2. COGNITION/PREDICTION PHASE
        cognitive_state = self.process_cognition(perceptual_state, dt)

        # 3. ACTION SELECTION PHASE
        action = self.select_action(cognitive_state)

        # 4. ACTION EXECUTION PHASE
        self.execute_action(action)

        # 5. UPDATE SYSTEM STATE
        self.update_system_state(action, dt)

        # 6. PUBLISH STATUS
        self.status_pub.publish(
            String(data=f"Loop executing - Position: {self.system_state['position']}")
        )

    def process_perception(self):
        """Process all sensor data into unified perceptual state"""
        perceptual_state = {
            'environment_map': self.create_environment_map(),
            'object_detections': self.detect_objects_in_image(),
            'obstacle_distances': self.extract_obstacle_distances(),
            'current_pose': self.system_state['position'],
            'current_orientation': self.system_state['orientation']
        }

        return perceptual_state

    def create_environment_map(self):
        """Create occupancy grid from laser data"""
        if self.scan_data:
            # Convert laser scan to simple occupancy representation
            angles = np.linspace(
                self.scan_data.angle_min,
                self.scan_data.angle_max,
                len(self.scan_data.ranges)
            )
            ranges = np.array(self.scan_data.ranges)

            # Filter out invalid ranges
            valid_mask = (ranges > 0) & (ranges < self.scan_data.range_max)
            valid_angles = angles[valid_mask]
            valid_ranges = ranges[valid_mask]

            # Convert to Cartesian coordinates relative to robot
            x_points = valid_ranges * np.cos(valid_angles)
            y_points = valid_ranges * np.sin(valid_angles)

            return np.column_stack([x_points, y_points])

        return np.array([])

    def detect_objects_in_image(self):
        """Detect objects in camera image"""
        if self.image_data is not None:
            # Simple color-based object detection for example
            hsv = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2HSV)

            # Detect red objects (example)
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)

            # Also check for high red values (hue > 170)
            lower_red2 = np.array([170, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

            mask = mask1 + mask2
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            objects = []
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Minimum size threshold
                    # Calculate centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        objects.append({'x': cx, 'y': cy, 'type': 'red_object'})

            return objects

        return []

    def extract_obstacle_distances(self):
        """Extract minimum distances in different directions"""
        if self.scan_data:
            ranges = np.array(self.scan_data.ranges)
            valid_ranges = ranges[(ranges > 0) & (ranges < self.scan_data.range_max)]

            if len(valid_ranges) > 0:
                return {
                    'front': min(valid_ranges[300:600]) if len(valid_ranges[300:600]) > 0 else float('inf'),
                    'left': min(valid_ranges[0:180]) if len(valid_ranges[0:180]) > 0 else float('inf'),
                    'right': min(valid_ranges[540:720]) if len(valid_ranges[540:720]) > 0 else float('inf'),
                    'min': min(valid_ranges) if len(valid_ranges) > 0 else float('inf')
                }

        return {'front': float('inf'), 'left': float('inf'), 'right': float('inf'), 'min': float('inf')}

    def process_cognition(self, perceptual_state, dt):
        """Process perceptual state to make decisions"""
        cognitive_state = {
            'threat_level': self.assess_threats(perceptual_state),
            'navigation_state': self.assess_navigation(perceptual_state),
            'object_interest': self.assess_object_interest(perceptual_state),
            'predicted_environment': self.predict_environment(perceptual_state, dt)
        }

        return cognitive_state

    def assess_threats(self, perceptual_state):
        """Assess potential threats in environment"""
        threat_level = 0.0

        # Check for close obstacles
        if perceptual_state['obstacle_distances']['min'] < 0.5:
            threat_level += 0.8
        elif perceptual_state['obstacle_distances']['min'] < 1.0:
            threat_level += 0.3

        # Check for unstable orientation (if available)
        if hasattr(self, 'tilt_angle') and abs(self.tilt_angle) > 0.5:
            threat_level += 0.9

        return min(threat_level, 1.0)  # Clamp between 0 and 1

    def assess_navigation(self, perceptual_state):
        """Assess navigation state and goal progress"""
        # Calculate direction to goal
        current_pos = self.system_state['position']
        goal_pos = self.system_state['goal']

        direction_to_goal = goal_pos - current_pos
        distance_to_goal = np.linalg.norm(direction_to_goal)

        # Check if path to goal is clear
        path_clear = self.is_path_clear(perceptual_state, direction_to_goal)

        return {
            'distance_to_goal': distance_to_goal,
            'direction_to_goal': direction_to_goal,
            'path_clear': path_clear,
            'progress': 1.0 / (1.0 + distance_to_goal)  # Higher value = closer to goal
        }

    def is_path_clear(self, perceptual_state, direction):
        """Check if path in given direction is clear of obstacles"""
        # Simplified check - in reality would use more sophisticated path planning
        front_distance = perceptual_state['obstacle_distances']['front']
        return front_distance > 1.0

    def assess_object_interest(self, perceptual_state):
        """Assess interest in detected objects"""
        if perceptual_state['object_detections']:
            # For this example, any detected object is interesting
            return {
                'object_count': len(perceptual_state['object_detections']),
                'nearest_object': self.find_nearest_object(perceptual_state['object_detections'])
            }
        return {'object_count': 0, 'nearest_object': None}

    def find_nearest_object(self, objects):
        """Find the nearest object to the robot"""
        # Simplified - assumes objects have position information
        if objects:
            # In a real system, this would convert image coordinates to world coordinates
            return objects[0]  # Return first object as example
        return None

    def predict_environment(self, perceptual_state, dt):
        """Predict how environment will change"""
        # Simple prediction based on current state
        predicted_state = perceptual_state.copy()

        # Predict that obstacles might move closer if robot is moving toward them
        if self.system_state['velocity'][0] > 0:  # Moving forward
            for key in ['front', 'left', 'right', 'min']:
                if key in predicted_state['obstacle_distances']:
                    predicted_state['obstacle_distances'][key] = max(
                        0.1, predicted_state['obstacle_distances'][key] -
                        self.system_state['velocity'][0] * dt
                    )

        return predicted_state

    def select_action(self, cognitive_state):
        """Select action based on cognitive state"""
        # Action selection hierarchy
        # 1. Safety (highest priority)
        if cognitive_state['threat_level'] > 0.7:
            return self.emergency_action(cognitive_state)

        # 2. Navigation toward goal
        elif cognitive_state['navigation_state']['path_clear']:
            return self.navigate_toward_goal(cognitive_state)

        # 3. Obstacle avoidance
        elif not cognitive_state['navigation_state']['path_clear']:
            return self.avoid_obstacles(cognitive_state)

        # 4. Exploration
        else:
            return self.explore_environment(cognitive_state)

    def emergency_action(self, cognitive_state):
        """High-priority emergency action"""
        cmd = Twist()
        cmd.linear.x = 0.0  # Stop immediately
        cmd.angular.z = 0.5  # Turn to avoid threat

        self.get_logger().warn('EMERGENCY ACTION: High threat detected')
        return cmd

    def navigate_toward_goal(self, cognitive_state):
        """Navigate toward goal"""
        cmd = Twist()

        # Move toward goal
        direction = cognitive_state['navigation_state']['direction_to_goal']
        cmd.linear.x = 0.3  # Forward speed

        # Simple proportional controller for direction
        desired_angle = math.atan2(direction[1], direction[0])
        current_angle = self.system_state['orientation']
        angle_error = desired_angle - current_angle

        # Normalize angle error to [-π, π]
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi

        cmd.angular.z = 1.0 * angle_error  # Proportional control

        return cmd

    def avoid_obstacles(self, cognitive_state):
        """Avoid obstacles"""
        cmd = Twist()

        obstacle_distances = cognitive_state['perceptual_state']['obstacle_distances']

        # Turn away from closest obstacle
        if obstacle_distances['front'] < 0.8:
            # Obstacle in front - turn
            if obstacle_distances['left'] > obstacle_distances['right']:
                cmd.angular.z = 0.5  # Turn left
            else:
                cmd.angular.z = -0.5  # Turn right
        elif obstacle_distances['front'] < 1.5:
            # Slow down when approaching obstacles
            cmd.linear.x = 0.1
        else:
            # No immediate obstacles
            cmd.linear.x = 0.3

        return cmd

    def explore_environment(self, cognitive_state):
        """Explore environment"""
        cmd = Twist()
        cmd.linear.x = 0.2  # Slow exploration
        cmd.angular.z = 0.1  # Gentle turning

        return cmd

    def execute_action(self, action):
        """Execute the selected action"""
        if isinstance(action, Twist):
            self.cmd_vel_pub.publish(action)
        else:
            # If action is not already a Twist message, convert it
            cmd = Twist()
            if 'linear' in action:
                cmd.linear.x = action['linear']
            if 'angular' in action:
                cmd.angular.z = action['angular']
            self.cmd_vel_pub.publish(cmd)

    def update_system_state(self, action, dt):
        """Update internal system state based on action and time"""
        # Update position based on velocity
        if isinstance(action, Twist):
            linear_vel = action.linear.x
            angular_vel = action.angular.z
        else:
            linear_vel = action.get('linear', 0.0)
            angular_vel = action.get('angular', 0.0)

        # Update orientation
        self.system_state['orientation'] += angular_vel * dt

        # Update position (simple dead reckoning)
        self.system_state['position'][0] += linear_vel * math.cos(self.system_state['orientation']) * dt
        self.system_state['position'][1] += linear_vel * math.sin(self.system_state['orientation']) * dt

        # Update velocity
        self.system_state['velocity'][0] = linear_vel * math.cos(self.system_state['orientation'])
        self.system_state['velocity'][1] = linear_vel * math.sin(self.system_state['orientation'])

    def update_orientation_from_imu(self, imu_msg):
        """Update orientation from IMU data"""
        # Convert quaternion to euler angle (simplified)
        # In practice, would use proper quaternion math
        self.system_state['orientation'] = math.atan2(
            2 * (imu_msg.orientation.w * imu_msg.orientation.z +
                 imu_msg.orientation.x * imu_msg.orientation.y),
            1 - 2 * (imu_msg.orientation.y**2 + imu_msg.orientation.z**2)
        )

class PerceptionProcessor:
    """Component for processing perceptual data"""
    def __init__(self):
        self.feature_extractors = {}
        self.scene_understanding = None

    def process_sensor_data(self, sensor_data):
        """Process raw sensor data into meaningful perceptions"""
        processed_data = {}

        # Extract features from different sensors
        if 'image' in sensor_data:
            processed_data['visual_features'] = self.extract_visual_features(sensor_data['image'])

        if 'laser' in sensor_data:
            processed_data['spatial_features'] = self.extract_spatial_features(sensor_data['laser'])

        if 'imu' in sensor_data:
            processed_data['inertial_features'] = self.extract_inertial_features(sensor_data['imu'])

        return processed_data

    def extract_visual_features(self, image):
        """Extract visual features from image"""
        # Placeholder for actual feature extraction
        return {'features': [], 'objects': []}

    def extract_spatial_features(self, laser_data):
        """Extract spatial features from laser data"""
        # Placeholder for actual feature extraction
        return {'obstacles': [], 'free_space': []}

    def extract_inertial_features(self, imu_data):
        """Extract inertial features from IMU data"""
        # Placeholder for actual feature extraction
        return {'orientation': 0.0, 'acceleration': [0, 0, 0]}

class ActionSelector:
    """Component for selecting actions based on state"""
    def __init__(self):
        self.policy_network = None
        self.value_function = None

    def select_action(self, state, policy_type='reactive'):
        """Select action based on current state"""
        if policy_type == 'reactive':
            return self.reactive_policy(state)
        elif policy_type == 'planning':
            return self.planning_policy(state)
        else:
            return self.default_policy(state)

    def reactive_policy(self, state):
        """Simple reactive policy"""
        # Placeholder for reactive action selection
        return {'linear': 0.0, 'angular': 0.0}

    def planning_policy(self, state):
        """Planning-based policy"""
        # Placeholder for planning-based action selection
        return {'linear': 0.0, 'angular': 0.0}

    def default_policy(self, state):
        """Default fallback policy"""
        return {'linear': 0.0, 'angular': 0.0}

class PredictiveModel:
    """Component for predicting environmental changes"""
    def __init__(self):
        self.dynamics_model = None
        self.uncertainty_model = None

    def predict_next_state(self, current_state, action):
        """Predict next environmental state given action"""
        # Placeholder for prediction
        return current_state

    def predict_sensor_observations(self, predicted_state):
        """Predict what sensors will observe in predicted state"""
        # Placeholder for sensor prediction
        return {}

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionActionLoopNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Emergent Behaviors from Perception-Action Loops

Complex behaviors can emerge from simple perception-action rules:

```python
# Example: Emergent flocking behavior from simple rules
class FlockingAgent:
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.max_speed = 2.0
        self.perception_radius = 5.0

    def update(self, neighbors, dt):
        """Update agent based on neighbors and environment"""
        # Apply flocking rules
        alignment = self.align_with_neighbors(neighbors)
        cohesion = self.move_toward_center(neighbors)
        separation = self.avoid_crowding(neighbors)

        # Combine behaviors
        acceleration = 0.5 * alignment + 0.3 * cohesion + 0.7 * separation

        # Update velocity and position
        self.velocity += acceleration * dt
        if np.linalg.norm(self.velocity) > self.max_speed:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.max_speed

        self.position += self.velocity * dt

    def align_with_neighbors(self, neighbors):
        """Steer to align with neighbors' average heading"""
        if not neighbors:
            return np.zeros(2)

        avg_heading = np.mean([n.velocity for n in neighbors], axis=0)
        return (avg_heading - self.velocity) * 0.05

    def move_toward_center(self, neighbors):
        """Steer toward average position of neighbors"""
        if not neighbors:
            return np.zeros(2)

        center = np.mean([n.position for n in neighbors], axis=0)
        direction = center - self.position
        return (direction - self.velocity) * 0.02

    def avoid_crowding(self, neighbors):
        """Steer to avoid crowding local flockmates"""
        if not neighbors:
            return np.zeros(2)

        repulsion = np.zeros(2)
        for neighbor in neighbors:
            distance = np.linalg.norm(neighbor.position - self.position)
            if distance < 2.0:  # Too close
                repulsion += (self.position - neighbor.position) / (distance + 0.01)

        return repulsion * 0.1

# Example: Foraging behavior
class ForagingAgent:
    def __init__(self):
        self.state = 'searching'  # searching, approaching, collecting, returning
        self.cargo = 0
        self.cargo_capacity = 10
        self.goal_location = None

    def perception_action_cycle(self, sensor_data):
        """Perception-action cycle for foraging"""
        # Perception
        food_detected = sensor_data.get('food_nearby', False)
        nest_detected = sensor_data.get('nest_nearby', False)
        obstacle_ahead = sensor_data.get('obstacle_ahead', False)

        # State transitions based on perception
        if self.state == 'searching':
            if food_detected:
                self.state = 'approaching'
                self.goal_location = sensor_data['food_location']
        elif self.state == 'approaching':
            if sensor_data['at_food_location']:
                self.state = 'collecting'
        elif self.state == 'collecting':
            if self.cargo >= self.cargo_capacity or not food_detected:
                self.state = 'returning'
                self.goal_location = sensor_data['nest_location']
        elif self.state == 'returning':
            if nest_detected:
                self.state = 'searching'
                self.cargo = 0  # Drop off cargo

        # Action selection based on state
        return self.select_action(obstacle_ahead)

    def select_action(self, obstacle_ahead):
        """Select action based on current state"""
        action = {'linear': 0.0, 'angular': 0.0}

        if obstacle_ahead:
            action['angular'] = 0.5  # Turn to avoid
        elif self.state == 'searching':
            action['linear'] = 0.3
            action['angular'] = 0.1  # Spiral search pattern
        elif self.state == 'approaching':
            action['linear'] = 0.2  # Careful approach
        elif self.state == 'collecting':
            action['linear'] = 0.0  # Stay still to collect
        elif self.state == 'returning':
            action['linear'] = 0.25  # Steady return

        return action
```

## Lab: Implementing Perception-Action Loops

In this lab, you'll implement a simple perception-action loop:

```python
# lab_perception_action_loop.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import numpy as np
import cv2

class PerceptionActionLabNode(Node):
    def __init__(self):
        super().__init__('perception_action_lab')

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/lab_status', 10)
        self.emergency_pub = self.create_publisher(Bool, '/emergency_stop', 10)

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        # Data storage
        self.scan_data = None
        self.image_data = None
        self.cv_bridge = CvBridge()

        # Lab parameters
        self.loop_state = 'exploration'  # exploration, obstacle_avoidance, object_interaction
        self.object_detected = False
        self.target_color = [0, 255, 0]  # Green object to follow

        # Control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

    def scan_callback(self, msg):
        """Handle laser scan data"""
        self.scan_data = msg

    def image_callback(self, msg):
        """Handle camera image data"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_data = cv_image
        except Exception as e:
            self.get_logger().error(f'Image callback error: {e}')

    def control_loop(self):
        """Main perception-action control loop"""
        if not self.scan_data and not self.image_data:
            return

        # 1. PERCEPTION PHASE
        perceptions = self.process_perceptions()

        # 2. STATE EVALUATION PHASE
        self.evaluate_state(perceptions)

        # 3. ACTION SELECTION PHASE
        action = self.select_action(perceptions)

        # 4. ACTION EXECUTION PHASE
        self.execute_action(action)

        # 5. STATUS UPDATE
        self.status_pub.publish(
            String(data=f"State: {self.loop_state}, Objects: {int(self.object_detected)}")
        )

    def process_perceptions(self):
        """Process sensor data to extract meaningful perceptions"""
        perceptions = {}

        # Process laser data
        if self.scan_data:
            perceptions['obstacles'] = self.analyze_obstacles()
            perceptions['clear_front'] = self.is_front_clear()

        # Process image data
        if self.image_data is not None:
            perceptions['objects'] = self.detect_target_objects()
            perceptions['object_direction'] = self.get_object_direction()

        return perceptions

    def analyze_obstacles(self):
        """Analyze laser data for obstacles"""
        ranges = np.array(self.scan_data.ranges)
        valid_ranges = ranges[(ranges > 0) & (ranges < self.scan_data.range_max)]

        if len(valid_ranges) > 0:
            return {
                'closest': min(valid_ranges),
                'front_clear': all(r > 1.0 for r in self.scan_data.ranges[300:600] if r > 0),
                'left_clear': all(r > 0.8 for r in self.scan_data.ranges[0:180] if r > 0),
                'right_clear': all(r > 0.8 for r in self.scan_data.ranges[540:720] if r > 0)
            }
        return {'closest': float('inf'), 'front_clear': True, 'left_clear': True, 'right_clear': True}

    def is_front_clear(self):
        """Check if front path is clear"""
        if self.scan_data:
            front_ranges = self.scan_data.ranges[300:600]
            return all(r > 1.0 for r in front_ranges if r > 0)
        return True

    def detect_target_objects(self):
        """Detect objects of target color in image"""
        if self.image_data is not None:
            # Convert BGR to HSV for better color detection
            hsv = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2HSV)

            # Define range for target color (green)
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])

            mask = cv2.inRange(hsv, lower_green, upper_green)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            objects = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area threshold
                    # Calculate centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        objects.append({
                            'x': cx,
                            'y': cy,
                            'area': area,
                            'contour': contour
                        })

            self.object_detected = len(objects) > 0
            return objects

        self.object_detected = False
        return []

    def get_object_direction(self):
        """Get direction to detected object"""
        if self.image_data is not None and self.object_detected:
            objects = self.detect_target_objects()
            if objects:
                # Use the largest object
                largest_obj = max(objects, key=lambda x: x['area'])
                image_center_x = self.image_data.shape[1] / 2
                object_x = largest_obj['x']

                # Calculate direction (-1 for left, 1 for right)
                direction = (object_x - image_center_x) / image_center_x
                return max(-1.0, min(1.0, direction))  # Clamp to [-1, 1]

        return 0.0  # No object detected

    def evaluate_state(self, perceptions):
        """Evaluate current state based on perceptions"""
        # Update state based on what we perceive
        if self.object_detected and perceptions['objects']:
            if self.loop_state != 'object_following':
                self.loop_state = 'object_following'
                self.get_logger().info('Switching to object following mode')
        elif perceptions['obstacles']['closest'] < 0.5:
            if self.loop_state != 'obstacle_avoidance':
                self.loop_state = 'obstacle_avoidance'
                self.get_logger().info('Switching to obstacle avoidance mode')
        else:
            if self.loop_state != 'exploration':
                self.loop_state = 'exploration'
                self.get_logger().info('Switching to exploration mode')

    def select_action(self, perceptions):
        """Select action based on current state and perceptions"""
        cmd = Twist()

        if self.loop_state == 'object_following':
            cmd = self.follow_object_action(perceptions)
        elif self.loop_state == 'obstacle_avoidance':
            cmd = self.avoid_obstacle_action(perceptions)
        elif self.loop_state == 'exploration':
            cmd = self.explore_action(perceptions)
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        return cmd

    def follow_object_action(self, perceptions):
        """Action for following detected object"""
        cmd = Twist()

        if self.object_detected:
            object_direction = perceptions['object_direction']

            # Move toward object if it's far enough
            if perceptions['obstacles']['closest'] > 0.8:
                cmd.linear.x = 0.2
            else:
                cmd.linear.x = 0.0  # Stop if too close to obstacles

            # Turn toward object
            cmd.angular.z = -0.8 * object_direction  # Negative for correct direction
        else:
            # If no object detected, search for it
            cmd.linear.x = 0.0
            cmd.angular.z = 0.3  # Turn to find object

        return cmd

    def avoid_obstacle_action(self, perceptions):
        """Action for avoiding obstacles"""
        cmd = Twist()

        obstacles = perceptions['obstacles']

        if obstacles['closest'] < 0.5:
            # Emergency stop or turn sharply
            cmd.linear.x = 0.0
            if obstacles['left_clear'] and not obstacles['right_clear']:
                cmd.angular.z = 0.5  # Turn left
            elif obstacles['right_clear'] and not obstacles['left_clear']:
                cmd.angular.z = -0.5  # Turn right
            else:
                cmd.angular.z = 0.3  # Turn randomly
        elif obstacles['closest'] < 1.0:
            # Slow down and prepare to turn
            cmd.linear.x = 0.1
            if not obstacles['front_clear']:
                cmd.angular.z = 0.2  # Gentle turn
        else:
            # Path is clear, move forward
            cmd.linear.x = 0.3

        return cmd

    def explore_action(self, perceptions):
        """Action for exploration"""
        cmd = Twist()

        # Simple exploration pattern
        cmd.linear.x = 0.2  # Move forward
        cmd.angular.z = 0.1  # Gentle turn for exploration

        return cmd

    def execute_action(self, action):
        """Execute the selected action"""
        self.cmd_pub.publish(action)

def main(args=None):
    rclpy.init(args=args)
    lab_node = PerceptionActionLabNode()

    try:
        rclpy.spin(lab_node)
    except KeyboardInterrupt:
        pass
    finally:
        lab_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercise: Design Your Own Perception-Action Loop

Consider a specific task and design a perception-action loop for it:

1. What sensors would be most relevant for your task?
2. What would the perception phase involve?
3. How would you process the perceptual information?
4. What actions would be available?
5. How would you select between different actions?
6. What emergent behaviors might arise from your design?

## Summary

Perception-action loops are fundamental to embodied intelligence, enabling robots to interact with their environment through continuous cycles of sensing, processing, and acting. Key concepts include:

- **Continuous Processing**: Unlike discrete AI systems, embodied agents operate in real-time
- **Multiple Loop Types**: Reactive, predictive, and hierarchical loops serve different purposes
- **Emergent Behaviors**: Complex behaviors arise from simple perception-action rules
- **Real-time Constraints**: Systems must respond within strict timing requirements
- **Environmental Adaptation**: Behavior adapts continuously based on environmental feedback

The implementation of perception-action loops using ROS2 enables the development of sophisticated embodied systems that can operate effectively in dynamic environments. Understanding these loops is crucial for developing robots that can learn and adapt through interaction with their environment.

In the next lesson, we'll explore examples of successful humanoid robots and analyze how they implement perception-action loops in practice.