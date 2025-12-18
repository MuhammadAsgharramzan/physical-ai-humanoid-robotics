---
sidebar_position: 4
---

# Integration of AI Techniques in Robotics

## Introduction

The true power of Physical AI emerges when multiple AI techniques are integrated into a cohesive system. This lesson explores how computer vision, machine learning, path planning, and other AI components work together to create intelligent robotic systems. We'll examine integration patterns, system architectures, and practical implementation strategies.

## System Architecture for Integrated AI

### Hierarchical Integration Architecture

A well-designed robotic AI system typically follows a hierarchical architecture:

```
High-Level Planning
    ↓
Behavior Selection
    ↓
Path Planning & Navigation
    ↓
Perception & State Estimation
    ↓
Action Execution & Control
```

Each level operates at different temporal and spatial scales, with information flowing both up and down the hierarchy.

```python
# Example: Hierarchical AI integration architecture
class IntegratedAISystem:
    def __init__(self):
        # High-level components
        self.task_planner = TaskPlanner()
        self.behavior_selector = BehaviorSelector()

        # Mid-level components
        self.navigation_system = IntegratedNavigationSystem()
        self.perception_system = ComputerVisionRobotics()

        # Low-level components
        self.control_system = MLRobotControl()
        self.action_executor = ActionExecutor()

    def execute_task(self, task_description):
        """Execute a high-level task through integrated AI components"""
        # 1. High-level planning
        task_plan = self.task_planner.plan_task(task_description)

        # 2. Behavior selection based on plan
        behaviors = self.behavior_selector.select_behaviors(task_plan)

        # 3. Execute each behavior using integrated components
        for behavior in behaviors:
            self.execute_behavior(behavior)

    def execute_behavior(self, behavior):
        """Execute a specific behavior using appropriate AI components"""
        # Perception
        perceptual_state = self.perception_system.get_current_state()

        # Planning
        action_plan = self.navigation_system.plan_action(behavior, perceptual_state)

        # Control
        control_commands = self.control_system.generate_commands(action_plan)

        # Execution
        self.action_executor.execute(control_commands)
```

### Parallel Processing Architecture

For real-time performance, many AI components run in parallel:

```python
# Example: Parallel AI processing system
import threading
import queue
import time

class ParallelAISystem:
    def __init__(self):
        # Component queues
        self.perception_queue = queue.Queue(maxsize=10)
        self.planning_queue = queue.Queue(maxsize=10)
        self.control_queue = queue.Queue(maxsize=10)

        # Components
        self.perception_component = ComputerVisionRobotics()
        self.planning_component = IntegratedNavigationSystem()
        self.control_component = MLRobotControl()

        # Threads for parallel processing
        self.perception_thread = threading.Thread(target=self.run_perception, daemon=True)
        self.planning_thread = threading.Thread(target=self.run_planning, daemon=True)
        self.control_thread = threading.Thread(target=self.run_control, daemon=True)

        # Shared state
        self.shared_state = {
            'current_pose': None,
            'detected_objects': [],
            'planned_path': [],
            'control_commands': None
        }

        # Start threads
        self.perception_thread.start()
        self.planning_thread.start()
        self.control_thread.start()

    def run_perception(self):
        """Run perception component in parallel"""
        while True:
            # Process sensor data
            perceptual_data = self.perception_component.process_frame()

            # Update shared state
            self.shared_state['detected_objects'] = perceptual_data.get('objects', [])
            self.shared_state['current_pose'] = perceptual_data.get('pose', None)

            time.sleep(0.05)  # 20 Hz processing

    def run_planning(self):
        """Run planning component in parallel"""
        while True:
            if self.shared_state['current_pose'] and self.shared_state['detected_objects']:
                # Plan based on current state
                planned_path = self.planning_component.plan_path(
                    self.shared_state['current_pose'],
                    self.shared_state['detected_objects']
                )

                # Update shared state
                self.shared_state['planned_path'] = planned_path

            time.sleep(0.1)  # 10 Hz planning

    def run_control(self):
        """Run control component in parallel"""
        while True:
            if self.shared_state['planned_path']:
                # Generate control commands
                control_commands = self.control_component.execute_path(
                    self.shared_state['planned_path']
                )

                # Update shared state
                self.shared_state['control_commands'] = control_commands

                # Execute commands
                self.execute_commands(control_commands)

            time.sleep(0.02)  # 50 Hz control

    def execute_commands(self, commands):
        """Execute the generated commands"""
        # In practice, this would send commands to robot actuators
        pass
```

## Integration Patterns

### 1. Sensor Fusion Integration

Combining data from multiple sensors for better perception:

```python
# Example: Multi-sensor fusion for state estimation
class SensorFusion:
    def __init__(self):
        self.camera_weight = 0.4
        self.lidar_weight = 0.4
        self.imu_weight = 0.2
        self.kalman_filter = self.initialize_kalman_filter()

    def initialize_kalman_filter(self):
        """Initialize Kalman filter for state estimation"""
        # State: [x, y, vx, vy]
        dt = 0.05  # Time step
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])  # State transition matrix

        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])  # Measurement matrix

        Q = np.eye(4) * 0.1  # Process noise
        R = np.eye(2) * 0.5  # Measurement noise
        P = np.eye(4)        # Error covariance

        return {
            'F': F, 'H': H, 'Q': Q, 'R': R, 'P': P,
            'state': np.zeros(4)  # [x, y, vx, vy]
        }

    def fuse_sensor_data(self, camera_data, lidar_data, imu_data):
        """Fuse data from multiple sensors"""
        # Extract measurements
        camera_measurement = self.extract_camera_measurement(camera_data)
        lidar_measurement = self.extract_lidar_measurement(lidar_data)
        imu_measurement = self.extract_imu_measurement(imu_data)

        # Weighted fusion
        fused_measurement = (
            self.camera_weight * camera_measurement +
            self.lidar_weight * lidar_measurement +
            self.imu_weight * imu_measurement
        )

        # Update Kalman filter
        self.update_kalman_filter(fused_measurement)

        return self.kalman_filter['state']

    def extract_camera_measurement(self, camera_data):
        """Extract position measurement from camera"""
        # Use visual landmarks or object detection
        # Return [x, y] position estimate
        return np.array([0.0, 0.0])  # Placeholder

    def extract_lidar_measurement(self, lidar_data):
        """Extract position measurement from LIDAR"""
        # Use LIDAR landmarks or scan matching
        # Return [x, y] position estimate
        return np.array([0.0, 0.0])  # Placeholder

    def extract_imu_measurement(self, imu_data):
        """Extract velocity measurement from IMU"""
        # Use IMU for velocity estimation
        # Return [vx, vy] velocity estimate
        return np.array([0.0, 0.0])  # Placeholder

    def update_kalman_filter(self, measurement):
        """Update Kalman filter with new measurement"""
        # Prediction step
        kf = self.kalman_filter
        kf['state'] = kf['F'] @ kf['state']
        kf['P'] = kf['F'] @ kf['P'] @ kf['F'].T + kf['Q']

        # Update step
        y = measurement - kf['H'] @ kf['state']  # Innovation
        S = kf['H'] @ kf['P'] @ kf['H'].T + kf['R']  # Innovation covariance
        K = kf['P'] @ kf['H'].T @ np.linalg.inv(S)  # Kalman gain

        kf['state'] = kf['state'] + K @ y
        kf['P'] = (np.eye(len(kf['P'])) - K @ kf['H']) @ kf['P']
```

### 2. Learning-Enhanced Planning

Using machine learning to improve traditional planning algorithms:

```python
# Example: Learning-enhanced path planning
import torch
import torch.nn as nn

class LearningEnhancedPlanner:
    def __init__(self):
        self.traditional_planner = AStarPlanner()
        self.learning_model = self.create_learning_model()
        self.obstacle_predictor = self.create_obstacle_predictor()
        self.experience_buffer = []

    def create_learning_model(self):
        """Create neural network to improve planning"""
        return nn.Sequential(
            nn.Linear(24, 64),  # Input: current state features
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),  # Output: planning improvement factors
        )

    def create_obstacle_predictor(self):
        """Create model to predict dynamic obstacles"""
        return nn.Sequential(
            nn.Linear(12, 32),  # Input: past obstacle positions
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),   # Output: predicted next position
        )

    def plan_path_with_learning(self, start, goal, occupancy_grid, dynamic_obstacles):
        """Plan path using both traditional and learned methods"""
        # Traditional planning
        base_path = self.traditional_planner.plan_path(start, goal, occupancy_grid)

        if not base_path:
            return []

        # Predict future obstacles
        predicted_obstacles = self.predict_obstacles(dynamic_obstacles)

        # Enhance path with learned improvements
        enhanced_path = self.enhance_path(base_path, predicted_obstacles)

        return enhanced_path

    def predict_obstacles(self, dynamic_obstacles):
        """Predict future positions of dynamic obstacles"""
        if len(dynamic_obstacles) < 2:
            return dynamic_obstacles

        # Prepare input for prediction
        recent_positions = torch.FloatTensor(dynamic_obstacles[-2:]).flatten()

        # Predict next position
        with torch.no_grad():
            prediction = self.obstacle_predictor(recent_positions.unsqueeze(0))

        predicted_position = prediction.squeeze(0).numpy()
        return dynamic_obstacles + [predicted_position.tolist()]

    def enhance_path(self, base_path, predicted_obstacles):
        """Enhance path using learned improvements"""
        # Prepare path features
        path_features = self.extract_path_features(base_path, predicted_obstacles)

        # Get improvement factors from learning model
        with torch.no_grad():
            improvements = self.learning_model(torch.FloatTensor(path_features).unsqueeze(0))

        # Apply improvements to path
        enhanced_path = self.apply_improvements(base_path, improvements.squeeze(0).numpy())

        return enhanced_path

    def extract_path_features(self, path, obstacles):
        """Extract features from path and obstacles for learning model"""
        features = []

        # Path length
        features.append(len(path))

        # Average curvature
        if len(path) > 2:
            total_curvature = 0
            for i in range(1, len(path) - 1):
                p1, p2, p3 = path[i-1], path[i], path[i+1]
                # Simplified curvature calculation
                angle = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
                total_curvature += abs(angle)
            features.append(total_curvature / (len(path) - 2))
        else:
            features.append(0)

        # Obstacle proximity
        if obstacles:
            min_distances = []
            for point in path:
                min_dist = min([np.sqrt((point[0]-obs[0])**2 + (point[1]-obs[1])**2) for obs in obstacles])
                min_distances.append(min_dist)
            features.append(sum(min_distances) / len(min_distances))
        else:
            features.append(10)  # Large distance if no obstacles

        # Pad features to fixed size
        while len(features) < 24:
            features.append(0)
        return features[:24]

    def apply_improvements(self, base_path, improvements):
        """Apply learned improvements to the path"""
        # In practice, this would involve more sophisticated path modification
        # For this example, we'll just return the base path
        return base_path
```

### 3. Perception-Action Integration

Tightly coupling perception with action selection:

```python
# Example: Perception-action integration
class PerceptionActionIntegrator:
    def __init__(self):
        self.perception_system = ComputerVisionRobotics()
        self.action_selector = MLRobotControl()
        self.behavior_tree = self.create_behavior_tree()

    def create_behavior_tree(self):
        """Create behavior tree for perception-action integration"""
        return {
            'root': 'sequence',
            'children': [
                {'type': 'condition', 'name': 'object_detected', 'func': self.is_object_detected},
                {'type': 'action', 'name': 'approach_object', 'func': self.approach_object},
                {'type': 'action', 'name': 'grasp_object', 'func': self.grasp_object}
            ]
        }

    def is_object_detected(self):
        """Check if target object is detected"""
        detections = self.perception_system.get_current_detections()
        return len(detections) > 0

    def approach_object(self):
        """Generate commands to approach detected object"""
        detections = self.perception_system.get_current_detections()

        if detections:
            target = detections[0]  # Use first detection
            target_position = target['center']

            # Generate approach command based on target position
            command = self.generate_approach_command(target_position)
            return command

        return None

    def generate_approach_command(self, target_position):
        """Generate command to approach target position"""
        # Calculate direction to target
        image_center = [320, 240]  # Assuming 640x480 image
        direction = np.array(target_position) - np.array(image_center)

        # Generate velocity command
        linear_vel = 0.2  # Move forward
        angular_vel = -direction[0] * 0.001  # Turn toward object

        return {'linear': linear_vel, 'angular': angular_vel}

    def grasp_object(self):
        """Generate commands to grasp object"""
        # This would involve more complex manipulation planning
        return {'action': 'grasp', 'parameters': {}}

    def execute_perception_action_cycle(self):
        """Execute one cycle of perception-action integration"""
        # 1. Perception
        self.perception_system.process_frame()

        # 2. Action selection based on perception
        action = self.select_action_from_perception()

        # 3. Action execution
        self.action_selector.execute_action(action)

        # 4. Feedback for learning
        self.update_perception_action_model()

    def select_action_from_perception(self):
        """Select action based on current perception"""
        # Evaluate behavior tree
        result = self.evaluate_behavior_tree(self.behavior_tree)
        return result

    def evaluate_behavior_tree(self, tree):
        """Evaluate behavior tree and return selected action"""
        if tree['type'] == 'sequence':
            for child in tree['children']:
                result = self.evaluate_behavior_tree(child)
                if result is None:
                    return None
            return result
        elif tree['type'] == 'condition':
            return tree['func']()
        elif tree['type'] == 'action':
            return tree['func']()

    def update_perception_action_model(self):
        """Update learning model based on perception-action outcomes"""
        # This would update the integrated perception-action system
        # based on success/failure of actions
        pass
```

## ROS2 Implementation: Integrated AI System

Here's a comprehensive ROS2 implementation that integrates all AI techniques:

```python
# integrated_ai_system.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, JointState, Imu
from geometry_msgs.msg import Twist, Pose, Point
from nav_msgs.msg import Path
from std_msgs.msg import String, Bool, Float32
from cv_bridge import CvBridge
from tf2_ros import TransformListener, Buffer
import numpy as np
import torch
import torch.nn as nn
from collections import deque

class IntegratedAISystemNode(Node):
    def __init__(self):
        super().__init__('integrated_ai_system')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/integrated_plan', 10)
        self.status_pub = self.create_publisher(String, '/ai_system_status', 10)
        self.detection_pub = self.create_publisher(String, '/object_detections', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )

        # TF listener for localization
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # System components
        self.cv_bridge = CvBridge()
        self.sensor_fusion = SensorFusion()
        self.learning_planner = LearningEnhancedPlanner()
        self.perception_action_integrator = PerceptionActionIntegrator()

        # Data storage
        self.image_data = None
        self.scan_data = None
        self.joint_data = None
        self.imu_data = None

        # AI system state
        self.robot_pose = None
        self.detected_objects = []
        self.current_goal = None
        self.system_state = 'idle'  # idle, perceiving, planning, executing, learning

        # Learning components
        self.learning_enabled = True
        self.experience_buffer = deque(maxlen=1000)
        self.performance_metrics = {
            'navigation_success': 0,
            'object_detection_accuracy': 0,
            'task_completion_rate': 0
        }

        # Control parameters
        self.control_frequency = 20.0  # Hz
        self.perception_frequency = 10.0  # Hz
        self.planning_frequency = 2.0  # Hz

        # Timers
        self.control_timer = self.create_timer(1.0/self.control_frequency, self.control_loop)
        self.perception_timer = self.create_timer(1.0/self.perception_frequency, self.perception_loop)
        self.planning_timer = self.create_timer(1.0/self.planning_frequency, self.planning_loop)

        # Integration parameters
        self.integration_weights = {
            'vision': 0.4,
            'lidar': 0.3,
            'learning': 0.3
        }

    def image_callback(self, msg):
        """Handle camera image data"""
        try:
            self.image_data = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Image callback error: {e}')

    def scan_callback(self, msg):
        """Handle laser scan data"""
        self.scan_data = msg

    def joint_callback(self, msg):
        """Handle joint state data"""
        self.joint_data = msg

    def imu_callback(self, msg):
        """Handle IMU data"""
        self.imu_data = msg

    def control_loop(self):
        """Main control loop for integrated AI system"""
        # Get current robot pose
        self.robot_pose = self.get_robot_pose()

        # Execute based on system state
        if self.system_state == 'idle':
            self.execute_idle_behavior()
        elif self.system_state == 'perceiving':
            self.execute_perception_behavior()
        elif self.system_state == 'planning':
            self.execute_planning_behavior()
        elif self.system_state == 'executing':
            self.execute_action_behavior()
        elif self.system_state == 'learning':
            self.execute_learning_behavior()

        # Publish system status
        self.publish_system_status()

    def perception_loop(self):
        """Perception processing loop"""
        if self.image_data is not None and self.scan_data is not None:
            # Process visual data
            visual_detections = self.process_visual_data()

            # Process LIDAR data
            lidar_detections = self.process_lidar_data()

            # Fuse detections
            self.detected_objects = self.fuse_detections(visual_detections, lidar_detections)

            # Publish detections
            self.publish_detections(self.detected_objects)

    def planning_loop(self):
        """Planning processing loop"""
        if (self.robot_pose is not None and
            self.current_goal is not None and
            self.scan_data is not None):

            # Plan path using integrated approach
            planned_path = self.learning_planner.plan_path_with_learning(
                [self.robot_pose.position.x, self.robot_pose.position.y],
                [self.current_goal.position.x, self.current_goal.position.y],
                self.get_occupancy_grid(),
                self.get_dynamic_obstacles()
            )

            # Publish planned path
            if planned_path:
                ros_path = self.path_to_ros_path(planned_path)
                self.path_pub.publish(ros_path)

    def execute_idle_behavior(self):
        """Execute idle behavior"""
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

    def execute_perception_behavior(self):
        """Execute perception-focused behavior"""
        # This could involve active sensing behaviors
        cmd = Twist()
        cmd.angular.z = 0.2  # Slow rotation for 360-degree perception
        self.cmd_vel_pub.publish(cmd)

    def execute_planning_behavior(self):
        """Execute planning-focused behavior"""
        # Stop robot while planning
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

    def execute_action_behavior(self):
        """Execute action behavior based on integrated AI"""
        # Use perception-action integration
        action = self.perception_action_integrator.select_action_from_perception()

        if action and 'linear' in action and 'angular' in action:
            cmd = Twist()
            cmd.linear.x = action['linear']
            cmd.angular.z = action['angular']
            self.cmd_vel_pub.publish(cmd)

    def execute_learning_behavior(self):
        """Execute learning-focused behavior"""
        # Update learning models based on recent experiences
        if self.learning_enabled:
            self.update_learning_models()

    def process_visual_data(self):
        """Process visual data for object detection"""
        # This would use the computer vision components
        # For this example, return empty list
        return []

    def process_lidar_data(self):
        """Process LIDAR data for obstacle detection"""
        if self.scan_data:
            obstacles = []
            for i, range_val in enumerate(self.scan_data.ranges):
                if 0 < range_val < self.scan_data.range_max:
                    angle = self.scan_data.angle_min + i * self.scan_data.angle_increment
                    x = range_val * np.cos(angle)
                    y = range_val * np.sin(angle)
                    obstacles.append([x, y, range_val])
            return obstacles
        return []

    def fuse_detections(self, visual_detections, lidar_detections):
        """Fuse detections from different sensors"""
        # Weighted fusion based on sensor reliability
        fused_detections = []

        # Combine detections with weights
        for detection in visual_detections:
            detection['confidence'] *= self.integration_weights['vision']
            fused_detections.append(detection)

        for detection in lidar_detections:
            # LIDAR detections have different format, adjust confidence accordingly
            detection_confidence = min(1.0, len(detection) / 10.0)  # Simplified
            detection.append(detection_confidence * self.integration_weights['lidar'])
            fused_detections.append(detection)

        return fused_detections

    def publish_detections(self, detections):
        """Publish object detections"""
        detection_str = f"Objects detected: {len(detections)}"
        self.detection_pub.publish(String(data=detection_str))

    def get_robot_pose(self):
        """Get current robot pose from TF"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time()
            )

            pose = Pose()
            pose.position.x = transform.transform.translation.x
            pose.position.y = transform.transform.translation.y
            pose.position.z = transform.transform.translation.z
            pose.orientation = transform.transform.rotation

            return pose
        except Exception as e:
            self.get_logger().warn(f'Could not get robot pose: {e}')
            return None

    def get_occupancy_grid(self):
        """Get current occupancy grid (simplified)"""
        # In a real system, this would come from a SLAM or mapping node
        # For this example, return a simple grid
        return np.zeros((100, 100))

    def get_dynamic_obstacles(self):
        """Get dynamic obstacles from tracking"""
        # This would use object tracking to identify moving obstacles
        return []

    def path_to_ros_path(self, path_points):
        """Convert path points to ROS Path message"""
        ros_path = Path()
        ros_path.header.frame_id = 'map'

        for point in path_points:
            pose = Pose()
            pose.position.x = point[0]
            pose.position.y = point[1]
            pose.position.z = 0.0
            ros_path.poses.append(pose)

        return ros_path

    def set_goal(self, goal_pose):
        """Set navigation goal"""
        self.current_goal = goal_pose
        self.system_state = 'planning'
        self.get_logger().info(f'Set goal to {goal_pose}')

    def update_learning_models(self):
        """Update learning models based on experiences"""
        # This would update various learning components
        # based on the experience buffer
        pass

    def publish_system_status(self):
        """Publish integrated AI system status"""
        status_msg = String()
        status_msg.data = (
            f"State: {self.system_state}, "
            f"Objects: {len(self.detected_objects)}, "
            f"Learning: {self.learning_enabled}"
        )
        self.status_pub.publish(status_msg)

class MultiModalPerceptionFusion:
    """Fuse multiple perception modalities"""
    def __init__(self):
        self.camera_processor = self.initialize_camera_processor()
        self.lidar_processor = self.initialize_lidar_processor()
        self.fusion_weights = {
            'camera': 0.5,
            'lidar': 0.4,
            'imu': 0.1
        }

    def initialize_camera_processor(self):
        """Initialize camera-based perception"""
        # This would initialize deep learning models for vision
        return lambda img: []  # Placeholder

    def initialize_lidar_processor(self):
        """Initialize LIDAR-based perception"""
        # This would initialize LIDAR processing algorithms
        return lambda scan: []  # Placeholder

    def fuse_perceptions(self, camera_data, lidar_data, imu_data):
        """Fuse perceptions from multiple modalities"""
        # Process each modality
        camera_features = self.camera_processor(camera_data)
        lidar_features = self.lidar_processor(lidar_data)

        # Weighted fusion
        fused_features = (
            self.fusion_weights['camera'] * np.array(camera_features) +
            self.fusion_weights['lidar'] * np.array(lidar_features) +
            self.fusion_weights['imu'] * np.array(imu_data) if imu_data else np.zeros_like(camera_features)
        )

        return fused_features

class AdaptiveLearningSystem:
    """Adapt learning based on performance"""
    def __init__(self):
        self.performance_history = deque(maxlen=100)
        self.learning_rate = 0.001
        self.adaptation_threshold = 0.7

    def update_performance(self, task_success):
        """Update performance metrics"""
        self.performance_history.append(task_success)

    def adapt_learning_rate(self):
        """Adapt learning rate based on performance"""
        if len(self.performance_history) >= 10:
            recent_performance = np.mean(list(self.performance_history)[-10:])

            if recent_performance < self.adaptation_threshold:
                # Performance is poor, increase learning rate
                self.learning_rate = min(0.01, self.learning_rate * 1.1)
            else:
                # Performance is good, decrease learning rate
                self.learning_rate = max(0.0001, self.learning_rate * 0.95)

    def get_adapted_parameters(self):
        """Get adapted learning parameters"""
        return {
            'learning_rate': self.learning_rate,
            'performance': np.mean(self.performance_history) if self.performance_history else 0
        }

def main(args=None):
    rclpy.init(args=args)
    ai_system = IntegratedAISystemNode()

    # Example: Set a goal after a delay
    def set_example_goal():
        goal = Pose()
        goal.position.x = 5.0
        goal.position.y = 5.0
        goal.orientation.w = 1.0
        ai_system.set_goal(goal)

    # Set goal after 5 seconds
    timer = ai_system.create_timer(5.0, set_example_goal)

    try:
        rclpy.spin(ai_system)
    except KeyboardInterrupt:
        pass
    finally:
        ai_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Integration Techniques

### Cognitive Architecture for Robotics

A cognitive architecture provides a framework for integrating multiple AI components:

```python
# Example: Cognitive architecture for integrated AI
class CognitiveArchitecture:
    def __init__(self):
        # Perception module
        self.perception_module = {
            'vision': ComputerVisionRobotics(),
            'lidar': IntegratedNavigationSystem(),
            'fusion': SensorFusion()
        }

        # Memory module
        self.memory_module = {
            'working_memory': {},
            'long_term_memory': {},
            'episodic_memory': deque(maxlen=1000)
        }

        # Reasoning module
        self.reasoning_module = {
            'planning': LearningEnhancedPlanner(),
            'decision_making': self.make_decision,
            'learning': AdaptiveLearningSystem()
        }

        # Action module
        self.action_module = {
            'navigation': IntegratedNavigationSystem(),
            'manipulation': MLRobotControl(),
            'communication': self.generate_speech
        }

        # Control cycle
        self.control_cycle = 'sense-think-act'

    def sense_think_act_cycle(self):
        """Execute the main cognitive cycle"""
        # Sense (perception)
        perceptual_state = self.perceive_environment()

        # Think (reasoning)
        decision = self.reason_about_state(perceptual_state)

        # Act (action selection and execution)
        self.execute_decision(decision, perceptual_state)

    def perceive_environment(self):
        """Perceive the environment using all sensors"""
        vision_data = self.perception_module['vision'].get_current_state()
        lidar_data = self.perception_module['lidar'].get_current_state()

        # Fuse perceptions
        fused_state = self.perception_module['fusion'].fuse_sensor_data(
            vision_data, lidar_data, None  # IMU data
        )

        # Store in working memory
        self.memory_module['working_memory']['perceptual_state'] = fused_state

        return fused_state

    def reason_about_state(self, perceptual_state):
        """Reason about current state and make decisions"""
        # Update long-term memory with current experience
        self.memory_module['episodic_memory'].append(perceptual_state)

        # Plan actions
        plan = self.reasoning_module['planning'].plan_path_with_learning(
            perceptual_state['current_pose'],
            perceptual_state['goal_pose'],
            perceptual_state.get('map', np.zeros((100, 100))),
            perceptual_state.get('obstacles', [])
        )

        # Make decision based on plan and current situation
        decision = self.make_decision(plan, perceptual_state)

        return decision

    def make_decision(self, plan, perceptual_state):
        """Make high-level decisions"""
        if not plan:
            return {'action': 'wait', 'reason': 'no_path_found'}

        # Check for safety
        if self.is_safe_to_proceed(plan, perceptual_state):
            return {'action': 'execute', 'plan': plan}
        else:
            return {'action': 'replan', 'reason': 'unsafe_condition'}

    def is_safe_to_proceed(self, plan, perceptual_state):
        """Check if it's safe to proceed with the plan"""
        # Check for obstacles on path
        obstacles = perceptual_state.get('obstacles', [])
        path = plan  # Simplified

        # Simple safety check
        for point in path[:10]:  # Check first 10 points
            for obs in obstacles:
                distance = np.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2)
                if distance < 0.5:  # 0.5m safety margin
                    return False

        return True

    def execute_decision(self, decision, perceptual_state):
        """Execute the decision"""
        if decision['action'] == 'execute':
            plan = decision['plan']
            # Execute navigation plan
            self.action_module['navigation'].execute_path(plan)
        elif decision['action'] == 'replan':
            # Trigger replanning
            self.reasoning_module['planning'].trigger_replanning()
        elif decision['action'] == 'wait':
            # Stop robot
            cmd = Twist()
            # Publish stop command to navigation system

    def generate_speech(self, message):
        """Generate speech output"""
        # This would interface with text-to-speech
        return f"Speaking: {message}"
```

### Emergent Behavior Through Integration

Complex behaviors emerge when multiple AI components work together:

```python
# Example: Emergent behaviors from AI integration
class EmergentBehaviorSystem:
    def __init__(self):
        self.components = {
            'planning': LearningEnhancedPlanner(),
            'perception': ComputerVisionRobotics(),
            'control': MLRobotControl(),
            'navigation': IntegratedNavigationSystem()
        }
        self.emergent_behaviors = {}

    def develop_emergent_behaviors(self):
        """Develop emergent behaviors through component interaction"""
        # Curiosity-driven exploration
        self.emergent_behaviors['curiosity_exploration'] = self.curiosity_driven_exploration

        # Social interaction through perception and learning
        self.emergent_behaviors['social_interaction'] = self.social_interaction_behavior

        # Adaptive problem solving
        self.emergent_behaviors['adaptive_problem_solving'] = self.adaptive_problem_solving

    def curiosity_driven_exploration(self):
        """Emergent exploration behavior driven by curiosity"""
        # Components work together to drive exploration
        current_state = self.get_integrated_state()

        # Identify novel areas to explore
        novel_areas = self.find_novel_areas(current_state)

        if novel_areas:
            # Plan path to most interesting area
            target = self.select_most_interesting_area(novel_areas)
            plan = self.components['planning'].plan_path_with_learning(
                current_state['pose'], target, current_state['map'], []
            )

            # Execute exploration
            self.components['navigation'].execute_path(plan)

    def find_novel_areas(self, state):
        """Find areas that are novel or interesting"""
        # Use perception to identify interesting features
        # Use mapping to identify unexplored areas
        # Return list of interesting locations
        return []

    def select_most_interesting_area(self, areas):
        """Select the most interesting area to explore"""
        # Use learning to rank areas by interest value
        return areas[0] if areas else None

    def social_interaction_behavior(self):
        """Emergent social interaction behavior"""
        # Detect humans through vision
        humans = self.components['perception'].detect_humans()

        if humans:
            # Approach and interact
            closest_human = min(humans, key=lambda h: h['distance'])
            self.approach_and_interact(closest_human)

    def adaptive_problem_solving(self):
        """Emergent adaptive problem solving"""
        # When faced with an obstacle, try different approaches
        # combining planning, learning, and perception
        pass

    def get_integrated_state(self):
        """Get state from all integrated components"""
        return {
            'pose': self.get_current_pose(),
            'map': self.get_current_map(),
            'objects': self.get_detected_objects(),
            'obstacles': self.get_detected_obstacles()
        }
```

## Lab: Implementing Integrated AI System

In this lab, you'll implement an integrated AI system:

```python
# lab_integrated_ai_system.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import numpy as np

class IntegratedAILab(Node):
    def __init__(self):
        super().__init__('integrated_ai_lab')

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/ai_status', 10)
        self.behavior_pub = self.create_publisher(String, '/selected_behavior', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        # System components
        self.cv_bridge = CvBridge()
        self.image_data = None
        self.scan_data = None

        # AI integration components
        self.perception_state = {
            'objects_detected': 0,
            'obstacle_distance': float('inf'),
            'navigation_clear': True
        }

        self.learning_state = {
            'exploration_rate': 0.3,
            'success_rate': 0.0,
            'experience_count': 0
        }

        # Behavior selection
        self.available_behaviors = [
            'explore', 'avoid_obstacles', 'approach_object', 'idle'
        ]
        self.current_behavior = 'explore'

        # Control loop
        self.control_timer = self.create_timer(0.05, self.integrated_control_loop)

        # Integration state
        self.integration_level = 0  # 0 = basic, 1 = moderate, 2 = advanced

    def image_callback(self, msg):
        """Handle camera image data"""
        try:
            self.image_data = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Image callback error: {e}')

    def scan_callback(self, msg):
        """Handle laser scan data"""
        self.scan_data = msg

    def integrated_control_loop(self):
        """Main integrated AI control loop"""
        # 1. PERCEPTION INTEGRATION
        self.update_perception_state()

        # 2. LEARNING INTEGRATION
        self.update_learning_state()

        # 3. BEHAVIOR SELECTION INTEGRATION
        self.select_behavior()

        # 4. ACTION EXECUTION INTEGRATION
        action = self.execute_behavior(self.current_behavior)
        self.cmd_pub.publish(action)

        # 5. FEEDBACK INTEGRATION
        self.update_integration_feedback()

        # 6. STATUS REPORTING
        self.publish_status()

    def update_perception_state(self):
        """Update perception state by integrating multiple sensors"""
        if self.scan_data:
            # Process LIDAR data
            valid_ranges = [r for r in self.scan_data.ranges if 0 < r < self.scan_data.range_max]
            if valid_ranges:
                self.perception_state['obstacle_distance'] = min(valid_ranges)
                self.perception_state['navigation_clear'] = min(valid_ranges) > 1.0

        if self.image_data is not None:
            # Process visual data (simplified)
            # Count significant features in image
            gray = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            feature_count = np.sum(edges > 0)
            self.perception_state['objects_detected'] = feature_count / 1000  # Normalize

    def update_learning_state(self):
        """Update learning state based on experiences"""
        self.learning_state['experience_count'] += 1

        # Simulate success rate improvement over time
        if self.learning_state['experience_count'] % 50 == 0:
            self.learning_state['success_rate'] = min(
                1.0,
                self.learning_state['success_rate'] + 0.05
            )

        # Adjust exploration rate based on success
        if self.learning_state['success_rate'] > 0.7:
            self.learning_state['exploration_rate'] = max(0.1, self.learning_state['exploration_rate'] * 0.99)
        else:
            self.learning_state['exploration_rate'] = min(0.8, self.learning_state['exploration_rate'] * 1.01)

    def select_behavior(self):
        """Select behavior based on integrated perception and learning"""
        # Behavior selection based on perception and learning state
        if self.perception_state['obstacle_distance'] < 0.5:
            self.current_behavior = 'avoid_obstacles'
        elif self.perception_state['objects_detected'] > 10:  # Arbitrary threshold
            if np.random.random() < 0.7:  # 70% chance to approach if object detected
                self.current_behavior = 'approach_object'
            else:
                self.current_behavior = 'explore'
        elif np.random.random() < self.learning_state['exploration_rate']:
            self.current_behavior = 'explore'
        else:
            self.current_behavior = 'idle'

    def execute_behavior(self, behavior):
        """Execute the selected behavior"""
        cmd = Twist()

        if behavior == 'explore':
            cmd.linear.x = 0.3
            cmd.angular.z = 0.1 * np.sin(self.get_clock().now().nanoseconds / 1e9)  # Gentle random turning
        elif behavior == 'avoid_obstacles':
            if self.perception_state['navigation_clear']:
                cmd.linear.x = 0.2
            else:
                cmd.angular.z = 0.5  # Turn to avoid
        elif behavior == 'approach_object':
            cmd.linear.x = 0.1  # Slow approach
            cmd.angular.z = 0.0
        elif behavior == 'idle':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        return cmd

    def update_integration_feedback(self):
        """Update system based on integration feedback"""
        # Increase integration level over time as system learns
        if self.learning_state['experience_count'] % 100 == 0:
            self.integration_level = min(2, self.integration_level + 1)

    def publish_status(self):
        """Publish integrated AI system status"""
        status_msg = String()
        status_msg.data = (
            f"Behavior: {self.current_behavior}, "
            f"Objects: {self.perception_state['objects_detected']:.1f}, "
            f"Obstacle Dist: {self.perception_state['obstacle_distance']:.2f}, "
            f"Success: {self.learning_state['success_rate']:.2f}, "
            f"Integration: {self.integration_level}"
        )
        self.status_pub.publish(status_msg)

        behavior_msg = String()
        behavior_msg.data = self.current_behavior
        self.behavior_pub.publish(behavior_msg)

def main(args=None):
    rclpy.init(args=args)
    lab = IntegratedAILab()

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

## Exercise: Design Your Own Integrated AI System

Consider the following design challenge:

1. What specific robotic task requires integration of multiple AI techniques?
2. Which AI components (perception, planning, learning, control) are most important?
3. How will these components communicate and share information?
4. What emergent behaviors do you expect to see?
5. How will the system adapt and improve over time?
6. What metrics will you use to evaluate integration effectiveness?
7. How will you handle conflicts between different AI components?

## Summary

Integration of AI techniques in robotics creates powerful systems that are greater than the sum of their parts:

- **Hierarchical Integration**: Organizing AI components at different temporal and spatial scales
- **Parallel Processing**: Running multiple AI components simultaneously for real-time performance
- **Sensor Fusion**: Combining data from multiple sensors for robust perception
- **Learning-Enhanced Planning**: Using machine learning to improve traditional algorithms
- **Perception-Action Integration**: Tightly coupling perception with action selection
- **Cognitive Architectures**: Frameworks for organizing integrated AI systems
- **Emergent Behaviors**: Complex behaviors arising from component interaction

The key to successful integration is designing clear interfaces between components, managing information flow effectively, and ensuring that the combined system performs better than individual components. Understanding these integration principles is essential for developing sophisticated Physical AI systems.

In the next module, we'll explore human-robot interaction, including intuitive interfaces, natural language processing, and social robotics principles.