---
sidebar_position: 3
---

# Path Planning and Navigation in Physical AI

## Introduction

Path planning and navigation are fundamental capabilities for mobile robots, enabling them to move autonomously from one location to another while avoiding obstacles. In Physical AI systems, these capabilities must be integrated with real-time perception and control, making them more complex than traditional planning algorithms. This lesson explores various path planning and navigation techniques specifically designed for embodied agents.

## Fundamentals of Robot Navigation

### The Navigation Stack

Modern robot navigation typically uses a layered approach:

```
High-level Planning → Global Path Planning → Local Path Planning → Motion Control
```

Each layer operates at different temporal and spatial scales:

- **High-level Planning**: Long-term goal planning and route selection
- **Global Path Planning**: Static obstacle-aware path computation
- **Local Path Planning**: Dynamic obstacle avoidance and path following
- **Motion Control**: Low-level motor control for execution

### Navigation System Components

```python
# Example: Navigation system architecture
class NavigationSystem:
    def __init__(self):
        self.map_manager = MapManager()
        self.global_planner = GlobalPlanner()
        self.local_planner = LocalPlanner()
        self.motion_controller = MotionController()
        self.obstacle_detector = ObstacleDetector()

    def navigate_to_goal(self, start_pose, goal_pose):
        """Navigate from start to goal through the navigation stack"""
        # 1. Global path planning
        global_path = self.global_planner.plan_path(
            start_pose, goal_pose, self.map_manager.get_map()
        )

        # 2. Local path planning and execution
        current_pose = start_pose
        path_following_success = True

        for waypoint in global_path:
            # 3. Local planning to reach waypoint
            local_path = self.local_planner.plan_local_path(
                current_pose, waypoint, self.obstacle_detector.get_obstacles()
            )

            # 4. Execute local path with motion control
            success = self.motion_controller.follow_path(local_path)
            if not success:
                path_following_success = False
                break

            # 5. Update current pose
            current_pose = self.get_current_pose()

        return path_following_success
```

## Global Path Planning

### A* Algorithm for Static Environments

A* is a popular algorithm for finding optimal paths in static environments:

```python
# Example: A* path planning implementation
import heapq
import numpy as np

class AStarPlanner:
    def __init__(self, grid_resolution=0.1):
        self.grid_resolution = grid_resolution

    def plan_path(self, start, goal, occupancy_grid):
        """Plan path using A* algorithm"""
        # Convert continuous coordinates to grid coordinates
        start_grid = self.world_to_grid(start, occupancy_grid)
        goal_grid = self.world_to_grid(goal, occupancy_grid)

        # Initialize open and closed sets
        open_set = [(0, start_grid)]  # (f_score, position)
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal_grid:
                # Reconstruct path
                return self.reconstruct_path(came_from, current)

            # Explore neighbors
            for neighbor in self.get_neighbors(current, occupancy_grid):
                tentative_g_score = g_score[current] + self.distance(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_grid)

                    # Add to open set if not already there
                    if not any(neighbor == item[1] for item in open_set):
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # No path found

    def heuristic(self, pos1, pos2):
        """Heuristic function (Euclidean distance)"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def get_neighbors(self, pos, occupancy_grid):
        """Get valid neighboring cells"""
        neighbors = []
        rows, cols = occupancy_grid.shape

        # 8-connected neighborhood
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                new_x, new_y = pos[0] + dx, pos[1] + dy

                # Check bounds
                if 0 <= new_x < rows and 0 <= new_y < cols:
                    # Check if cell is free (assuming 0 = free, 100 = occupied)
                    if occupancy_grid[new_x, new_y] < 50:  # Threshold for free space
                        neighbors.append((new_x, new_y))

        return neighbors

    def world_to_grid(self, world_pos, occupancy_grid):
        """Convert world coordinates to grid coordinates"""
        # Simplified conversion - in practice, this would use proper transforms
        grid_x = int(world_pos[0] / self.grid_resolution)
        grid_y = int(world_pos[1] / self.grid_resolution)
        return (min(grid_x, occupancy_grid.shape[0] - 1), min(grid_y, occupancy_grid.shape[1] - 1))

    def reconstruct_path(self, came_from, current):
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # Reverse to get start-to-goal path
```

### Dijkstra's Algorithm

Dijkstra's algorithm guarantees optimal paths but can be slower than A*:

```python
# Example: Dijkstra's algorithm for path planning
class DijkstraPlanner:
    def __init__(self):
        pass

    def plan_path(self, start, goal, occupancy_grid):
        """Plan path using Dijkstra's algorithm"""
        # Convert to grid coordinates
        start_grid = self.world_to_grid(start, occupancy_grid)
        goal_grid = self.world_to_grid(goal, occupancy_grid)

        # Initialize distances and priority queue
        distances = {start_grid: 0}
        previous = {}
        pq = [(0, start_grid)]
        visited = set()

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            if current == goal_grid:
                return self.reconstruct_path(previous, start_grid, goal_grid)

            # Explore neighbors
            for neighbor in self.get_neighbors(current, occupancy_grid):
                if neighbor in visited:
                    continue

                # Calculate distance (considering terrain cost)
                new_dist = current_dist + self.calculate_edge_cost(current, neighbor, occupancy_grid)

                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))

        return []  # No path found

    def calculate_edge_cost(self, pos1, pos2, occupancy_grid):
        """Calculate cost of traversing from pos1 to pos2"""
        # Base cost is Euclidean distance
        base_cost = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

        # Add terrain cost based on occupancy grid value
        terrain_cost = occupancy_grid[pos2] / 100.0  # Normalize occupancy value

        return base_cost + terrain_cost

    def world_to_grid(self, world_pos, occupancy_grid):
        """Convert world coordinates to grid coordinates"""
        # Simplified - in practice, use proper coordinate transforms
        grid_x = int(world_pos[0])
        grid_y = int(world_pos[1])
        return (min(grid_x, occupancy_grid.shape[0] - 1), min(grid_y, occupancy_grid.shape[1] - 1))

    def get_neighbors(self, pos, occupancy_grid):
        """Get valid neighboring cells"""
        neighbors = []
        rows, cols = occupancy_grid.shape

        # 4-connected neighborhood for Dijkstra
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_x, new_y = pos[0] + dx, pos[1] + dy

            if 0 <= new_x < rows and 0 <= new_y < cols:
                if occupancy_grid[new_x, new_y] < 90:  # Not heavily occupied
                    neighbors.append((new_x, new_y))

        return neighbors

    def reconstruct_path(self, previous, start, goal):
        """Reconstruct path from previous dictionary"""
        path = []
        current = goal
        while current != start:
            path.append(current)
            current = previous.get(current)
            if current is None:
                return []  # No path found
        path.append(start)
        return path[::-1]
```

## Local Path Planning and Dynamic Obstacle Avoidance

### Dynamic Window Approach (DWA)

DWA is excellent for local path planning with dynamic obstacle avoidance:

```python
# Example: Dynamic Window Approach for local planning
class DWAPlanner:
    def __init__(self):
        # Robot parameters
        self.max_vel_x = 0.5  # m/s
        self.min_vel_x = 0.0  # m/s
        self.max_vel_theta = 1.0  # rad/s
        self.min_vel_theta = -1.0  # rad/s

        # Acceleration limits
        self.max_acc_x = 2.0  # m/s^2
        self.max_acc_theta = 3.0  # rad/s^2

        # Time horizon
        self.dt = 0.1  # Time step
        self.predict_time = 2.0  # Prediction horizon
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0

    def plan_local_path(self, robot_pose, goal_pose, obstacles, current_vel):
        """Plan local path using Dynamic Window Approach"""
        # Calculate dynamic window
        vs = self.calculate_dynamic_window(current_vel)

        # Evaluate trajectories
        best_trajectory = None
        min_cost = float('inf')

        for vel_x in np.arange(vs[0], vs[1], 0.1):  # Linear velocity
            for vel_theta in np.arange(vs[2], vs[3], 0.1):  # Angular velocity
                trajectory = self.predict_trajectory(robot_pose, [vel_x, vel_theta])

                # Calculate costs
                to_goal_cost = self.calculate_to_goal_cost(trajectory, goal_pose)
                speed_cost = self.calculate_speed_cost([vel_x, vel_theta])
                obstacle_cost = self.calculate_obstacle_cost(trajectory, obstacles)

                # Total cost
                total_cost = (self.to_goal_cost_gain * to_goal_cost +
                             self.speed_cost_gain * speed_cost +
                             self.obstacle_cost_gain * obstacle_cost)

                if total_cost < min_cost:
                    min_cost = total_cost
                    best_trajectory = trajectory

        if best_trajectory is not None:
            # Return the first control command from the best trajectory
            return [best_trajectory[0][3], best_trajectory[0][4]]  # [vel_x, vel_theta]
        else:
            return [0.0, 0.0]  # Stop if no valid trajectory found

    def calculate_dynamic_window(self, current_vel):
        """Calculate dynamic window based on current velocity and constraints"""
        # Velocity limits based on acceleration constraints
        max_vel_x = min(self.max_vel_x,
                       current_vel[0] + self.max_acc_x * self.dt)
        min_vel_x = max(self.min_vel_x,
                       current_vel[0] - self.max_acc_x * self.dt)

        max_vel_theta = min(self.max_vel_theta,
                           current_vel[1] + self.max_acc_theta * self.dt)
        min_vel_theta = max(self.min_vel_theta,
                           current_vel[1] - self.max_acc_theta * self.dt)

        return [min_vel_x, max_vel_x, min_vel_theta, max_vel_theta]

    def predict_trajectory(self, start_pose, velocity):
        """Predict trajectory given initial pose and constant velocity"""
        x, y, theta = start_pose
        vel_x, vel_theta = velocity

        trajectory = []
        time = 0

        while time <= self.predict_time:
            # Simple motion model
            new_x = x + vel_x * np.cos(theta) * self.dt
            new_y = y + vel_x * np.sin(theta) * self.dt
            new_theta = theta + vel_theta * self.dt

            trajectory.append([new_x, new_y, new_theta, vel_x, vel_theta])

            x, y, theta = new_x, new_y, new_theta
            time += self.dt

        return trajectory

    def calculate_to_goal_cost(self, trajectory, goal_pose):
        """Calculate cost to goal for trajectory"""
        if len(trajectory) == 0:
            return float('inf')

        last_pos = trajectory[-1][:2]  # x, y
        goal_pos = goal_pose[:2]

        # Euclidean distance to goal
        return np.sqrt((last_pos[0] - goal_pos[0])**2 + (last_pos[1] - goal_pos[1])**2)

    def calculate_speed_cost(self, velocity):
        """Calculate cost based on speed (prefer higher speeds)"""
        return self.max_vel_x - velocity[0]

    def calculate_obstacle_cost(self, trajectory, obstacles):
        """Calculate cost based on proximity to obstacles"""
        if len(trajectory) == 0:
            return float('inf')

        min_dist_to_obstacle = float('inf')

        for point in trajectory:
            for obs in obstacles:
                dist = np.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2)
                min_dist_to_obstacle = min(min_dist_to_obstacle, dist)

        # Return inverse of distance (higher cost for closer obstacles)
        return 1.0 / min_dist_to_obstacle if min_dist_to_obstacle > 0 else float('inf')
```

### Vector Field Histogram (VFH)

VFH is another approach for local navigation:

```python
# Example: Vector Field Histogram for obstacle avoidance
class VFHPlanner:
    def __init__(self):
        self.sector_count = 72  # 5-degree sectors
        self.safe_distance = 0.5  # meters
        self.max_range = 3.0  # meters

    def plan_with_vfh(self, robot_pose, goal_pose, laser_data):
        """Plan motion using Vector Field Histogram"""
        # Create polar histogram from laser data
        histogram = self.create_polar_histogram(laser_data)

        # Create navigation field combining goal direction and obstacle avoidance
        navigation_field = self.create_navigation_field(histogram, goal_pose, robot_pose)

        # Select best direction
        best_direction = self.select_best_direction(navigation_field)

        # Convert to velocity command
        return self.direction_to_command(best_direction)

    def create_polar_histogram(self, laser_data):
        """Create polar histogram from laser scan data"""
        histogram = np.zeros(self.sector_count)

        # Convert laser ranges to histogram
        angle_increment = laser_data.angle_increment
        start_angle = laser_data.angle_min

        for i, range_val in enumerate(laser_data.ranges):
            if 0 < range_val < self.max_range:
                # Calculate which sector this range belongs to
                angle = start_angle + i * angle_increment
                sector = int((angle + np.pi) / (2 * np.pi) * self.sector_count) % self.sector_count

                # Mark sector as occupied if obstacle is too close
                if range_val < self.safe_distance:
                    histogram[sector] = 1  # Occupied

        return histogram

    def create_navigation_field(self, histogram, goal_pose, robot_pose):
        """Create navigation field combining goal and obstacle information"""
        # Calculate goal direction
        dx = goal_pose[0] - robot_pose[0]
        dy = goal_pose[1] - robot_pose[1]
        goal_angle = np.arctan2(dy, dx)

        # Convert to robot's local frame
        robot_angle = robot_pose[2]  # Robot's orientation
        relative_goal_angle = goal_angle - robot_angle

        # Normalize to [-π, π]
        while relative_goal_angle > np.pi:
            relative_goal_angle -= 2 * np.pi
        while relative_goal_angle < -np.pi:
            relative_goal_angle += 2 * np.pi

        # Create navigation field
        field = np.zeros(self.sector_count)

        # Add goal attraction
        goal_sector = int((relative_goal_angle + np.pi) / (2 * np.pi) * self.sector_count) % self.sector_count
        field[goal_sector] += 10  # Strong attraction to goal direction

        # Add obstacle repulsion (inverse of histogram)
        for i in range(self.sector_count):
            if histogram[i] > 0:  # Obstacle detected
                # Repel from obstacle directions
                for j in range(max(0, i-2), min(self.sector_count, i+3)):
                    field[j] -= 5  # Repulsion

        return field

    def select_best_direction(self, navigation_field):
        """Select best direction from navigation field"""
        # Find the direction with highest value
        best_sector = np.argmax(navigation_field)

        # Convert sector back to angle
        angle = (best_sector / self.sector_count) * 2 * np.pi - np.pi

        return angle

    def direction_to_command(self, direction):
        """Convert direction to velocity command"""
        # Simple proportional control
        linear_vel = 0.3  # Base forward speed
        angular_vel = direction * 0.5  # Proportional to desired direction

        # Limit angular velocity
        angular_vel = max(-1.0, min(1.0, angular_vel))

        return [linear_vel, angular_vel]
```

## ROS2 Implementation: Integrated Navigation System

Here's a comprehensive ROS2 implementation of the navigation system:

```python
# integrated_navigation_system.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, OccupancyGrid
from geometry_msgs.msg import Twist, Pose, Point
from nav_msgs.msg import Path
from std_msgs.msg import String, Bool
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs
import numpy as np
from collections import deque

class IntegratedNavigationSystem(Node):
    def __init__(self):
        super().__init__('integrated_navigation')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.global_path_pub = self.create_publisher(Path, '/global_plan', 10)
        self.local_path_pub = self.create_publisher(Path, '/local_plan', 10)
        self.status_pub = self.create_publisher(String, '/navigation_status', 10)
        self.goal_reached_pub = self.create_publisher(Bool, '/goal_reached', 10)

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )

        # TF listener for robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Navigation components
        self.global_planner = AStarPlanner(grid_resolution=0.05)
        self.local_planner = DWAPlanner()
        self.vfh_planner = VFHPlanner()

        # Data storage
        self.laser_data = None
        self.occupancy_map = None
        self.current_pose = None
        self.goal_pose = None
        self.global_path = []
        self.local_path = []

        # Navigation state
        self.navigation_state = 'idle'  # idle, planning, executing, paused, error
        self.current_waypoint_idx = 0
        self.arrival_threshold = 0.5  # meters

        # Control parameters
        self.control_frequency = 20.0  # Hz
        self.planning_frequency = 1.0  # Hz
        self.replan_distance = 1.0  # Replan if goal is this far away

        # Timers
        self.control_timer = self.create_timer(1.0/self.control_frequency, self.control_loop)
        self.planning_timer = self.create_timer(1.0/self.planning_frequency, self.planning_loop)

        # Navigation history
        self.navigation_history = deque(maxlen=1000)

    def scan_callback(self, msg):
        """Handle laser scan data"""
        self.laser_data = msg

    def map_callback(self, msg):
        """Handle occupancy grid map"""
        self.occupancy_map = msg

    def set_goal(self, goal_pose):
        """Set navigation goal"""
        self.goal_pose = goal_pose
        self.navigation_state = 'planning'
        self.current_waypoint_idx = 0
        self.get_logger().info(f'Set navigation goal to {goal_pose}')

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

    def control_loop(self):
        """Main control loop for navigation"""
        # Get current robot pose
        self.current_pose = self.get_robot_pose()
        if self.current_pose is None:
            return

        # Execute navigation based on state
        if self.navigation_state == 'idle':
            # Stop the robot
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)

        elif self.navigation_state == 'planning':
            # Wait for planning to complete
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)

        elif self.navigation_state == 'executing':
            # Execute planned path
            cmd = self.execute_path()
            self.cmd_vel_pub.publish(cmd)

            # Check if goal is reached
            if self.is_goal_reached():
                self.navigation_state = 'idle'
                self.goal_reached_pub.publish(Bool(data=True))
                self.get_logger().info('Goal reached!')

        elif self.navigation_state == 'paused':
            # Stop robot temporarily
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)

        elif self.navigation_state == 'error':
            # Emergency stop
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)

        # Update navigation status
        self.publish_navigation_status()

    def planning_loop(self):
        """Periodic planning loop"""
        if (self.navigation_state == 'planning' and
            self.current_pose is not None and
            self.goal_pose is not None and
            self.occupancy_map is not None):

            # Plan global path
            start = [self.current_pose.position.x, self.current_pose.position.y]
            goal = [self.goal_pose.position.x, self.goal_pose.position.y]

            # Convert occupancy grid to numpy array
            map_array = np.array(self.occupancy_map.data).reshape(
                self.occupancy_map.info.height,
                self.occupancy_map.info.width
            )

            # Plan path
            path = self.global_planner.plan_path(start, goal, map_array)

            if path:
                # Convert path to ROS Path message
                self.global_path = self.path_to_ros_path(path, self.occupancy_map.info)
                self.global_path_pub.publish(self.global_path)

                self.navigation_state = 'executing'
                self.get_logger().info('Global path planned successfully')
            else:
                self.navigation_state = 'error'
                self.get_logger().error('Could not find global path to goal')

    def execute_path(self):
        """Execute the planned path with local obstacle avoidance"""
        if not self.global_path.poses:
            return Twist()  # Stop if no path

        # Get current robot position
        robot_pos = np.array([
            self.current_pose.position.x,
            self.current_pose.position.y
        ])

        # Find closest waypoint
        closest_idx = self.find_closest_waypoint(robot_pos)
        self.current_waypoint_idx = closest_idx

        # Get target waypoint (look ahead)
        target_idx = min(closest_idx + 5, len(self.global_path.poses) - 1)
        target_pos = np.array([
            self.global_path.poses[target_idx].pose.position.x,
            self.global_path.poses[target_idx].pose.position.y
        ])

        # Local planning with obstacle avoidance
        if self.laser_data is not None:
            # Use VFH for local obstacle avoidance
            cmd_vel = self.vfh_planner.plan_with_vfh(
                [robot_pos[0], robot_pos[1], self.get_robot_yaw()],
                [target_pos[0], target_pos[1], 0],
                self.laser_data
            )

            twist = Twist()
            twist.linear.x = cmd_vel[0]
            twist.angular.z = cmd_vel[1]
        else:
            # Simple proportional control to target
            direction = target_pos - robot_pos
            distance = np.linalg.norm(direction)

            twist = Twist()
            twist.linear.x = min(0.3, distance * 0.5)  # Proportional to distance

            if distance > 0.1:  # Avoid division by zero
                target_angle = np.arctan2(direction[1], direction[0])
                current_yaw = self.get_robot_yaw()

                angle_error = target_angle - current_yaw
                # Normalize angle to [-π, π]
                while angle_error > np.pi:
                    angle_error -= 2 * np.pi
                while angle_error < -np.pi:
                    angle_error += 2 * np.pi

                twist.angular.z = angle_error * 1.0  # Proportional control

        return twist

    def find_closest_waypoint(self, robot_pos):
        """Find the closest waypoint on the path"""
        if not self.global_path.poses:
            return 0

        min_distance = float('inf')
        closest_idx = 0

        for i, pose in enumerate(self.global_path.poses):
            path_pos = np.array([pose.pose.position.x, pose.pose.position.y])
            distance = np.linalg.norm(robot_pos - path_pos)

            if distance < min_distance:
                min_distance = distance
                closest_idx = i

        return closest_idx

    def is_goal_reached(self):
        """Check if robot has reached the goal"""
        if (self.current_pose is None or
            self.goal_pose is None):
            return False

        distance = np.sqrt(
            (self.current_pose.position.x - self.goal_pose.position.x)**2 +
            (self.current_pose.position.y - self.goal_pose.position.y)**2
        )

        return distance < self.arrival_threshold

    def get_robot_yaw(self):
        """Get robot's yaw angle from orientation quaternion"""
        if self.current_pose is not None:
            # Convert quaternion to yaw (simplified)
            q = self.current_pose.orientation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            return np.arctan2(siny_cosp, cosy_cosp)

        return 0.0

    def path_to_ros_path(self, path_grid, map_info):
        """Convert grid path to ROS Path message"""
        ros_path = Path()
        ros_path.header.frame_id = 'map'

        for grid_pos in path_grid:
            pose = Pose()
            # Convert grid coordinates back to world coordinates
            pose.position.x = grid_pos[0] * map_info.resolution + map_info.origin.position.x
            pose.position.y = grid_pos[1] * map_info.resolution + map_info.origin.position.y
            pose.position.z = map_info.origin.position.z

            ros_path.poses.append(pose)

        return ros_path

    def publish_navigation_status(self):
        """Publish navigation status"""
        status_msg = String()
        status_msg.data = f"State: {self.navigation_state}, Waypoint: {self.current_waypoint_idx}, Goal: {self.goal_pose is not None}"
        self.status_pub.publish(status_msg)

class MapManager:
    """Manage map-related functionality"""
    def __init__(self):
        self.map = None
        self.resolution = 0.05  # meters per cell
        self.origin = [0, 0]  # map origin in world coordinates

    def update_map(self, occupancy_grid):
        """Update internal map representation"""
        self.map = np.array(occupancy_grid.data).reshape(
            occupancy_grid.info.height,
            occupancy_grid.info.width
        )
        self.resolution = occupancy_grid.info.resolution
        self.origin = [
            occupancy_grid.info.origin.position.x,
            occupancy_grid.info.origin.position.y
        ]

    def world_to_map(self, world_x, world_y):
        """Convert world coordinates to map coordinates"""
        map_x = int((world_x - self.origin[0]) / self.resolution)
        map_y = int((world_y - self.origin[1]) / self.resolution)
        return map_x, map_y

    def map_to_world(self, map_x, map_y):
        """Convert map coordinates to world coordinates"""
        world_x = map_x * self.resolution + self.origin[0]
        world_y = map_y * self.resolution + self.origin[1]
        return world_x, world_y

class ObstacleDetector:
    """Detect obstacles from sensor data"""
    def __init__(self):
        self.obstacles = []
        self.min_obstacle_distance = 0.3  # meters
        self.max_detection_range = 3.0    # meters

    def detect_obstacles(self, laser_scan):
        """Detect obstacles from laser scan data"""
        obstacles = []

        for i, range_val in enumerate(laser_scan.ranges):
            if self.min_obstacle_distance < range_val < self.max_detection_range:
                # Calculate obstacle position in robot frame
                angle = laser_scan.angle_min + i * laser_scan.angle_increment
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)

                obstacles.append([x, y, range_val])  # x, y, distance

        self.obstacles = obstacles
        return obstacles

    def get_dynamic_obstacles(self):
        """Get obstacles that are moving (if available)"""
        # In a real implementation, this would use multiple scans over time
        # to detect moving obstacles
        return []

def main(args=None):
    rclpy.init(args=args)
    navigation_system = IntegratedNavigationSystem()

    # Example: Set a goal after a delay
    def set_example_goal():
        goal = Pose()
        goal.position.x = 5.0
        goal.position.y = 5.0
        goal.orientation.w = 1.0
        navigation_system.set_goal(goal)

    # Set goal after 5 seconds
    timer = navigation_system.create_timer(5.0, set_example_goal)

    try:
        rclpy.spin(navigation_system)
    except KeyboardInterrupt:
        pass
    finally:
        navigation_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Navigation Techniques

### Machine Learning-Based Navigation

Modern navigation systems increasingly incorporate machine learning:

```python
# Example: Learning-based navigation
import torch
import torch.nn as nn

class LearningBasedNavigator(nn.Module):
    def __init__(self, state_size=50, action_size=2):
        super(LearningBasedNavigator, self).__init__()

        # Sensor processing network
        self.sensor_encoder = nn.Sequential(
            nn.Linear(360, 128),  # Assuming 360 laser readings
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Goal processing network
        self.goal_encoder = nn.Sequential(
            nn.Linear(2, 16),  # Goal x, y
            nn.ReLU()
        )

        # Pose processing network
        self.pose_encoder = nn.Sequential(
            nn.Linear(3, 16),  # x, y, theta
            nn.ReLU()
        )

        # Combined processing network
        self.combined_processor = nn.Sequential(
            nn.Linear(64 + 16 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)  # linear_vel, angular_vel
        )

    def forward(self, laser_scan, goal_pos, current_pose):
        """Forward pass to compute navigation command"""
        # Process sensor data
        sensor_features = self.sensor_encoder(laser_scan)

        # Process goal
        goal_features = self.goal_encoder(goal_pos)

        # Process current pose
        pose_features = self.pose_encoder(current_pose)

        # Combine all features
        combined = torch.cat([sensor_features, goal_features, pose_features], dim=-1)

        # Compute action
        action = self.combined_processor(combined)

        return action

class LearningBasedNavigationSystem:
    def __init__(self):
        self.navigator = LearningBasedNavigator()
        self.load_trained_model()

    def load_trained_model(self):
        """Load pre-trained navigation model"""
        # In practice, you would load a saved model
        pass

    def navigate(self, laser_scan, goal_pos, current_pose):
        """Navigate using learned policy"""
        # Convert inputs to tensors
        laser_tensor = torch.FloatTensor(laser_scan).unsqueeze(0)
        goal_tensor = torch.FloatTensor(goal_pos).unsqueeze(0)
        pose_tensor = torch.FloatTensor(current_pose).unsqueeze(0)

        # Get action from network
        with torch.no_grad():
            action = self.navigator(laser_tensor, goal_tensor, pose_tensor)

        return action.squeeze(0).numpy()
```

### Multi-Robot Navigation

For coordinating multiple robots:

```python
# Example: Multi-robot navigation coordination
class MultiRobotNavigator:
    def __init__(self, robot_id, total_robots):
        self.robot_id = robot_id
        self.total_robots = total_robots
        self.robot_positions = {}
        self.communication_range = 5.0  # meters

    def update_robot_positions(self, positions_dict):
        """Update positions of all robots"""
        self.robot_positions = positions_dict

    def coordinate_navigation(self, my_position, goal_position, all_robots_positions):
        """Coordinate navigation to avoid conflicts with other robots"""
        # Calculate potential conflicts with other robots
        conflicts = self.detect_conflicts(my_position, all_robots_positions)

        if conflicts:
            # Adjust navigation to avoid conflicts
            return self.resolve_conflicts(my_position, goal_position, conflicts)
        else:
            # Navigate normally
            return self.plan_path(my_position, goal_position)

    def detect_conflicts(self, my_position, all_positions):
        """Detect potential conflicts with other robots"""
        conflicts = []

        for robot_id, position in all_positions.items():
            if robot_id != self.robot_id:
                distance = np.sqrt(
                    (my_position[0] - position[0])**2 +
                    (my_position[1] - position[1])**2
                )

                if distance < self.communication_range:
                    conflicts.append({
                        'robot_id': robot_id,
                        'position': position,
                        'distance': distance
                    })

        return conflicts

    def resolve_conflicts(self, my_position, goal_position, conflicts):
        """Resolve conflicts with other robots"""
        # Simple round-robin approach: even robots go first
        if self.robot_id % 2 == 0:
            # Priority robot, navigate normally
            return self.plan_path(my_position, goal_position)
        else:
            # Wait or take detour
            return self.plan_detour(my_position, goal_position, conflicts)

    def plan_detour(self, my_position, goal_position, conflicts):
        """Plan a detour to avoid other robots"""
        # Calculate a temporary goal that avoids conflicts
        temp_goal = self.calculate_temp_goal(my_position, goal_position, conflicts)
        return self.plan_path(my_position, temp_goal)

    def calculate_temp_goal(self, my_position, goal_position, conflicts):
        """Calculate temporary goal to avoid conflicts"""
        # Simple approach: offset from goal direction
        goal_direction = np.array(goal_position) - np.array(my_position)
        goal_distance = np.linalg.norm(goal_direction)

        if goal_distance > 0:
            goal_direction = goal_direction / goal_distance

        # Add small offset to avoid conflicts
        offset = np.array([-goal_direction[1], goal_direction[0]]) * 1.0  # Perpendicular offset
        temp_goal = np.array(my_position) + goal_direction * 2.0 + offset  # 2m ahead with offset

        return temp_goal.tolist()
```

## Lab: Implementing Navigation System

In this lab, you'll implement a complete navigation system:

```python
# lab_navigation_system.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Path
from std_msgs.msg import String, Bool
import numpy as np
from collections import deque

class NavigationLab(Node):
    def __init__(self):
        super().__init__('navigation_lab')

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.status_pub = self.create_publisher(String, '/navigation_status', 10)
        self.arrival_pub = self.create_publisher(Bool, '/goal_arrived', 10)

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        # Data storage
        self.scan_data = None
        self.robot_pose = [0.0, 0.0, 0.0]  # x, y, theta
        self.goal_pose = [5.0, 5.0]  # x, y
        self.current_path = []
        self.path_index = 0

        # Navigation parameters
        self.linear_speed = 0.3
        self.angular_speed = 0.5
        self.arrival_threshold = 0.5
        self.rotation_threshold = 0.1  # radians

        # Control loop
        self.control_timer = self.create_timer(0.05, self.navigation_control_loop)

        # Navigation state
        self.navigation_state = 'rotating_to_goal'  # rotating_to_goal, moving_to_goal, arrived
        self.last_command_time = self.get_clock().now()

    def scan_callback(self, msg):
        """Handle laser scan data"""
        self.scan_data = msg

    def navigation_control_loop(self):
        """Main navigation control loop"""
        if self.scan_data is None:
            return

        # Update robot position estimate (simplified)
        self.update_robot_position()

        # Navigation state machine
        if self.navigation_state == 'rotating_to_goal':
            self.rotate_to_goal_direction()
        elif self.navigation_state == 'moving_to_goal':
            self.move_towards_goal()
        elif self.navigation_state == 'arrived':
            self.stop_robot()

        # Check for obstacles and adjust behavior
        if self.is_obstacle_ahead():
            self.handle_obstacle()

        # Check if goal is reached
        if self.is_goal_reached():
            self.navigation_state = 'arrived'
            self.arrival_pub.publish(Bool(data=True))
            self.get_logger().info('Goal reached!')

    def update_robot_position(self):
        """Update robot position estimate (simplified)"""
        # In a real system, this would come from odometry or localization
        # For this lab, we'll just keep track of the current pose
        pass

    def rotate_to_goal_direction(self):
        """Rotate robot to face the goal direction"""
        # Calculate direction to goal
        dx = self.goal_pose[0] - self.robot_pose[0]
        dy = self.goal_pose[1] - self.robot_pose[1]
        goal_angle = np.arctan2(dy, dx)

        # Current robot angle
        current_angle = self.robot_pose[2]

        # Calculate angle difference
        angle_diff = goal_angle - current_angle

        # Normalize angle to [-π, π]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        # Create command
        cmd = Twist()

        if abs(angle_diff) > self.rotation_threshold:
            # Rotate towards goal
            cmd.angular.z = np.clip(angle_diff * 1.0, -self.angular_speed, self.angular_speed)
        else:
            # Facing correct direction, start moving forward
            self.navigation_state = 'moving_to_goal'
            cmd.linear.x = self.linear_speed

        self.cmd_pub.publish(cmd)

    def move_towards_goal(self):
        """Move robot towards the goal"""
        # Calculate distance to goal
        dx = self.goal_pose[0] - self.robot_pose[0]
        dy = self.goal_pose[1] - self.robot_pose[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Calculate direction to goal
        goal_angle = np.arctan2(dy, dx)
        current_angle = self.robot_pose[2]

        # Calculate angle difference
        angle_diff = goal_angle - current_angle

        # Normalize angle to [-π, π]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        cmd = Twist()

        if distance > self.arrival_threshold:
            # Move forward
            cmd.linear.x = min(self.linear_speed, distance * 0.5)  # Proportional to distance

            # Correct orientation
            cmd.angular.z = np.clip(angle_diff * 0.5, -self.angular_speed, self.angular_speed)
        else:
            # Close enough to goal
            self.navigation_state = 'arrived'
            self.arrival_pub.publish(Bool(data=True))
            self.get_logger().info('Goal reached!')

        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        """Stop the robot"""
        cmd = Twist()
        self.cmd_pub.publish(cmd)

    def is_obstacle_ahead(self):
        """Check if there's an obstacle directly ahead"""
        if self.scan_data is None:
            return False

        # Check the front 30 degrees of the scan
        front_ranges = self.scan_data.ranges[165:255]  # Assuming 360-degree scan
        valid_ranges = [r for r in front_ranges if 0 < r < self.scan_data.range_max]

        if valid_ranges:
            min_distance = min(valid_ranges)
            return min_distance < 0.8  # Obstacle within 0.8m

        return False

    def handle_obstacle(self):
        """Handle obstacle detection"""
        if self.navigation_state == 'moving_to_goal':
            # Simple obstacle avoidance: turn right
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = -0.5  # Turn right
            self.cmd_pub.publish(cmd)

            # After a short time, resume navigation
            # In a real implementation, you might use more sophisticated obstacle avoidance
            self.get_logger().warn('Obstacle detected!')

    def is_goal_reached(self):
        """Check if the robot has reached the goal"""
        dx = self.goal_pose[0] - self.robot_pose[0]
        dy = self.goal_pose[1] - self.robot_pose[1]
        distance = np.sqrt(dx**2 + dy**2)
        return distance < self.arrival_threshold

def main(args=None):
    rclpy.init(args=args)
    lab = NavigationLab()

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

## Exercise: Design Your Own Navigation System

Consider the following design challenge:

1. What type of environment will your robot navigate (indoor, outdoor, dynamic, static)?
2. What sensors will you use for navigation (laser, camera, IMU, etc.)?
3. What path planning algorithm is most appropriate for your scenario?
4. How will you handle dynamic obstacles?
5. What safety measures will you implement?
6. How will you handle localization and mapping?
7. What performance metrics will you use to evaluate your navigation system?

## Summary

Path planning and navigation are critical capabilities for autonomous robots, requiring integration of multiple components:

- **Global Planning**: Finding optimal paths in static environments using algorithms like A* or Dijkstra
- **Local Planning**: Handling dynamic obstacles and real-time adjustments using techniques like DWA or VFH
- **Motion Control**: Executing planned paths with precise motor control
- **Perception Integration**: Using sensor data for obstacle detection and localization
- **Learning-Based Approaches**: Using machine learning for adaptive navigation

The integration of these components in ROS2 enables the development of robust navigation systems for physical AI applications. Understanding these concepts is essential for developing robots that can operate autonomously in complex environments.

In the next lesson, we'll explore how to integrate all AI techniques into a cohesive system for robotics applications.