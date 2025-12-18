---
sidebar_position: 1
---

# Advanced Control Systems for Robotics

## Introduction

Advanced control systems are essential for creating sophisticated Physical AI systems that can operate effectively in dynamic environments. These systems go beyond basic position or velocity control to enable complex behaviors, adaptive responses, and robust performance in the face of uncertainty. This lesson explores various advanced control techniques specifically designed for robotics applications.

## Classical Control Theory in Robotics

### PID Control and Its Variants

PID (Proportional-Integral-Derivative) control remains fundamental in robotics:

```python
# Example: Advanced PID controller for robotics
class AdvancedPIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, dt=0.01):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.dt = dt  # Time step

        # Internal state
        self.error_sum = 0.0
        self.error_prev = 0.0
        self.derivative_filtered = 0.0
        self.integral_limit = 10.0  # Anti-windup limit

        # Derivative filtering parameters
        self.alpha = 0.1  # Filter coefficient for derivative term
        self.setpoint = 0.0
        self.measurement = 0.0

    def update(self, setpoint, measurement):
        """Update PID controller with new setpoint and measurement"""
        self.setpoint = setpoint
        self.measurement = measurement

        # Calculate error
        error = setpoint - measurement

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup
        self.error_sum += error * self.dt
        # Anti-windup: limit integral term
        self.error_sum = max(-self.integral_limit, min(self.integral_limit, self.error_sum))
        i_term = self.ki * self.error_sum

        # Derivative term with filtering (noise reduction)
        raw_derivative = (error - self.error_prev) / self.dt
        # First-order low-pass filter for derivative
        self.derivative_filtered = (
            self.alpha * raw_derivative +
            (1 - self.alpha) * self.derivative_filtered
        )
        d_term = self.kd * self.derivative_filtered

        # Calculate output
        output = p_term + i_term + d_term

        # Store error for next iteration
        self.error_prev = error

        return output

    def tune_pid(self, method='ziegler_nichols'):
        """Tune PID parameters using different methods"""
        if method == 'ziegler_nichols':
            # Ziegler-Nichols tuning method (simplified)
            ku = 2.0  # Ultimate gain (would be determined experimentally)
            tu = 1.0  # Oscillation period (would be determined experimentally)

            self.kp = 0.6 * ku
            self.ki = 1.2 * ku / tu
            self.kd = 0.075 * ku * tu

        elif method == 'cruise_control':
            # Cruise control specific tuning
            self.kp = 0.8
            self.ki = 0.1
            self.kd = 0.05

    def reset(self):
        """Reset internal state"""
        self.error_sum = 0.0
        self.error_prev = 0.0
        self.derivative_filtered = 0.0
```

### Cascade Control Systems

For complex robotic systems, cascade control provides superior performance:

```python
# Example: Cascade control for robot joint control
class CascadeController:
    def __init__(self):
        # Outer loop: position control
        self.position_controller = AdvancedPIDController(kp=2.0, ki=0.1, kd=0.05)

        # Inner loop: velocity control
        self.velocity_controller = AdvancedPIDController(kp=1.5, ki=0.2, kd=0.02)

        # Innermost loop: current/torque control
        self.current_controller = AdvancedPIDController(kp=1.0, ki=0.3, kd=0.01)

        self.dt = 0.01  # Control loop time step

    def compute_control(self, desired_position, current_position,
                       current_velocity, current_current):
        """Compute control through cascade structure"""

        # Position loop: generate velocity command
        velocity_command = self.position_controller.update(
            desired_position, current_position
        )

        # Velocity loop: generate current/torque command
        current_command = self.velocity_controller.update(
            velocity_command, current_velocity
        )

        # Current loop: generate final control output
        control_output = self.current_controller.update(
            current_command, current_current
        )

        return {
            'position_error': desired_position - current_position,
            'velocity_command': velocity_command,
            'current_command': current_command,
            'final_output': control_output
        }

    def update_gains_adaptively(self, performance_metrics):
        """Adaptively update gains based on performance"""
        # Example: Adjust gains based on tracking error
        position_error = performance_metrics.get('position_error', 0.0)
        velocity_error = performance_metrics.get('velocity_error', 0.0)

        # Increase gains if error is large
        if abs(position_error) > 0.1:  # Threshold
            self.position_controller.kp *= 1.05
            self.position_controller.ki *= 1.05
        else:
            # Decrease gains if error is small (reduce oscillation)
            self.position_controller.kp *= 0.99
            self.position_controller.ki *= 0.99

        # Similarly for velocity controller
        if abs(velocity_error) > 0.05:
            self.velocity_controller.kp *= 1.05
            self.velocity_controller.ki *= 1.05
```

## Modern Control Techniques

### Model Predictive Control (MPC)

MPC is particularly powerful for systems with constraints and multiple objectives:

```python
# Example: Model Predictive Control for robotics
import numpy as np
from scipy.optimize import minimize
import cvxpy as cp

class ModelPredictiveController:
    def __init__(self, state_dim, control_dim, prediction_horizon=10):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.N = prediction_horizon  # Prediction horizon

        # System matrices (would be identified for specific robot)
        self.A = np.eye(state_dim)  # State transition matrix
        self.B = np.zeros((state_dim, control_dim))  # Control input matrix
        self.C = np.eye(state_dim)  # Output matrix

        # Cost matrices
        self.Q = np.eye(state_dim)  # State cost matrix
        self.R = np.eye(control_dim)  # Control cost matrix
        self.Qf = np.eye(state_dim)  # Terminal cost matrix

        # Constraints
        self.u_min = -1.0  # Minimum control input
        self.u_max = 1.0   # Maximum control input
        self.x_min = -np.inf  # State constraints
        self.x_max = np.inf

    def setup_optimization_problem(self, x0, x_ref):
        """Set up MPC optimization problem"""
        # Decision variables: control inputs over prediction horizon
        U = cp.Variable((self.N, self.control_dim))

        # State variables over prediction horizon
        X = cp.Variable((self.N + 1, self.state_dim))

        # Objective function: minimize state and control costs
        cost = 0

        # Running costs
        for k in range(self.N):
            cost += cp.quad_form(X[k] - x_ref, self.Q)
            cost += cp.quad_form(U[k], self.R)

        # Terminal cost
        cost += cp.quad_form(X[self.N] - x_ref, self.Qf)

        # Constraints
        constraints = []

        # Initial state constraint
        constraints.append(X[0] == x0)

        # System dynamics constraints
        for k in range(self.N):
            constraints.append(X[k+1] == self.A @ X[k] + self.B @ U[k])

        # Control input constraints
        for k in range(self.N):
            constraints.append(U[k] >= self.u_min)
            constraints.append(U[k] <= self.u_max)

        # State constraints (if applicable)
        for k in range(self.N + 1):
            constraints.append(X[k] >= self.x_min)
            constraints.append(X[k] <= self.x_max)

        # Formulate optimization problem
        problem = cp.Problem(cp.Minimize(cost), constraints)

        return problem, U, X

    def compute_control(self, current_state, reference_trajectory):
        """Compute optimal control using MPC"""
        # For simplicity, use single reference point
        # In practice, use trajectory
        x_ref = reference_trajectory[0] if len(reference_trajectory) > 0 else np.zeros(self.state_dim)

        # Set up optimization
        problem, U, X = self.setup_optimization_problem(current_state, x_ref)

        # Solve optimization problem
        try:
            problem.solve(solver=cp.ECOS, verbose=False)

            if problem.status == cp.OPTIMAL:
                # Return first control input (receding horizon)
                optimal_control = U.value[0] if U.value is not None else np.zeros(self.control_dim)
                return optimal_control, True
            else:
                # Fallback to simple control if optimization fails
                return self.fallback_control(current_state, x_ref), False

        except Exception as e:
            self.get_logger().error(f'MPC optimization failed: {e}')
            return self.fallback_control(current_state, x_ref), False

    def fallback_control(self, current_state, reference_state):
        """Fallback control law if MPC fails"""
        # Simple state feedback control
        K = np.eye(self.control_dim)  # Would be designed properly
        error = reference_state - current_state[:self.control_dim]  # Simplified
        return K @ error

    def update_model_matrices(self, new_A, new_B):
        """Update system matrices for adaptive MPC"""
        self.A = new_A
        self.B = new_B
```

### Adaptive Control Systems

Adaptive control adjusts parameters based on system behavior:

```python
# Example: Model Reference Adaptive Control (MRAC)
class ModelReferenceAdaptiveController:
    def __init__(self, reference_model_params, initial_controller_params):
        self.reference_model = self.initialize_reference_model(reference_model_params)
        self.controller_params = initial_controller_params.copy()
        self.adaptation_rate = 0.01  # Learning rate
        self.param_bounds = {'min': -10.0, 'max': 10.0}  # Parameter bounds

        # Adaptation state
        self.error_history = []
        self.parameter_history = []

    def initialize_reference_model(self, params):
        """Initialize reference model for desired behavior"""
        # Reference model: desired system dynamics
        # Example: second-order system
        wn = params.get('natural_frequency', 1.0)
        zeta = params.get('damping_ratio', 0.7)

        # Second-order system: s^2 + 2*zeta*wn*s + wn^2
        return {
            'omega_n': wn,
            'zeta': zeta,
            'denominator': [1.0, 2*zeta*wn, wn**2]
        }

    def compute_control(self, reference_input, actual_output, reference_output):
        """Compute control with adaptation"""
        # Calculate tracking error
        tracking_error = reference_output - actual_output

        # Update parameter estimates based on error
        self.update_parameters(tracking_error, actual_output)

        # Compute control using updated parameters
        control_signal = self.compute_adaptive_control(
            reference_input, actual_output, tracking_error
        )

        return control_signal

    def update_parameters(self, error, output):
        """Update controller parameters based on error"""
        # Gradient descent adaptation law
        # Simplified example: adapt proportional gain
        if len(self.error_history) > 1:
            # Use gradient of error with respect to parameters
            param_gradient = self.compute_param_gradient(error, output)

            # Update parameters
            for param_name, current_val in self.controller_params.items():
                gradient_val = param_gradient.get(param_name, 0.0)
                new_val = current_val - self.adaptation_rate * gradient_val

                # Apply bounds
                new_val = max(self.param_bounds['min'],
                             min(self.param_bounds['max'], new_val))

                self.controller_params[param_name] = new_val

        # Store for history
        self.error_history.append(error)
        self.parameter_history.append(self.controller_params.copy())

    def compute_param_gradient(self, error, output):
        """Compute gradient of error with respect to parameters"""
        # This would be computed based on system model
        # For this example, use simplified gradients
        gradients = {}

        # Example gradients (would be derived from system model)
        if 'kp' in self.controller_params:
            gradients['kp'] = -error * output  # Simplified gradient

        if 'ki' in self.controller_params:
            gradients['ki'] = -error  # Simplified gradient

        return gradients

    def compute_adaptive_control(self, reference, output, error):
        """Compute control using adaptive parameters"""
        # Use current adaptive parameters
        kp = self.controller_params.get('kp', 1.0)
        ki = self.controller_params.get('ki', 0.0)

        # PID-like control with adaptive gains
        control = kp * error

        # Add integral term if available
        if len(self.error_history) > 0:
            integral_error = sum(self.error_history) * 0.01  # dt approximation
            control += ki * integral_error

        return control

    def reset_adaptation(self):
        """Reset adaptation to initial parameters"""
        self.error_history.clear()
        self.parameter_history.clear()
```

## Robust Control Systems

### H-infinity Control

H-infinity control provides robustness to model uncertainty:

```python
# Example: H-infinity control framework
class HInfinityController:
    def __init__(self, system_order, uncertainty_bound=0.1):
        self.system_order = system_order
        self.uncertainty_bound = uncertainty_bound

        # Controller matrices (would be synthesized)
        self.K = np.zeros((system_order, system_order))  # State feedback gain
        self.L = np.zeros((system_order, system_order))  # Observer gain
        self.P = np.eye(system_order)  # Solution to Riccati equation

        # Performance weighting matrices
        self.W_performance = np.eye(system_order)
        self.W_control = 0.1 * np.eye(system_order)
        self.W_uncertainty = np.eye(system_order)

        self.gamma = 1.0  # Performance bound

    def synthesize_controller(self, nominal_system):
        """Synthesize H-infinity controller for nominal system"""
        # This would involve solving H-infinity synthesis problem
        # For this example, return simplified result
        A, B, C, D = nominal_system

        # Simplified design: pole placement for robustness
        desired_poles = self.compute_robust_poles(A, self.gamma)

        # Design state feedback gain K
        self.K = self.place_poles(A, B, desired_poles)

        # Design observer gain L
        self.L = self.place_poles(A.T, C.T, desired_poles).T

        return self.K, self.L

    def compute_robust_poles(self, A, gamma):
        """Compute poles for robust performance"""
        # This would involve H-infinity optimization
        # For this example, use simple robust pole placement
        eigenvals = np.linalg.eigvals(A)

        # Shift poles for robustness
        robust_poles = []
        for ev in eigenvals:
            if np.real(ev) >= 0:  # Unstable poles
                robust_poles.append(-abs(ev) - 0.5)  # Stabilize with margin
            else:  # Stable poles
                robust_poles.append(ev * 0.8)  # Make more stable

        return robust_poles

    def place_poles(self, A, B, poles):
        """Place poles at desired locations"""
        # Use Ackermann's formula for single-input case
        # For multi-input, would use more sophisticated method
        n = A.shape[0]

        # Controllability matrix
        C = np.zeros((n, n))
        C[:, 0] = B.flatten()
        for i in range(1, n):
            C[:, i] = (A @ C[:, i-1]).flatten()

        # Check controllability
        if np.linalg.matrix_rank(C) < n:
            raise ValueError("System is not controllable")

        # Characteristic polynomial coefficients
        poly_coeffs = np.poly(poles)

        # Ackermann's formula
        K = np.zeros(n)
        K[-1] = 1.0
        for i in range(n-1):
            K[i] = -poly_coeffs[n-i]

        # Transform to original coordinates
        K = K @ np.linalg.inv(C)

        return K.reshape(1, -1)

    def compute_robust_control(self, state_estimate, reference, disturbance_estimate=0):
        """Compute robust control action"""
        # State feedback with reference tracking
        error = reference - state_estimate
        control = -self.K @ state_estimate + self.K @ reference

        # Add disturbance rejection
        control -= self.K @ disturbance_estimate

        return control.flatten()

    def update_for_uncertainty(self, measured_uncertainty):
        """Update controller for measured uncertainty"""
        # Adjust performance bound based on measured uncertainty
        if measured_uncertainty > self.uncertainty_bound:
            # Increase robustness by reducing performance
            self.gamma *= 1.1
        else:
            # Can afford better performance
            self.gamma *= 0.95

        # Re-synthesize controller
        # In practice, would update matrices incrementally
```

## ROS2 Implementation: Advanced Control System

Here's a comprehensive ROS2 implementation of advanced control systems:

```python
# advanced_control_systems.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import String, Float32
from builtin_interfaces.msg import Time
import numpy as np
from scipy import linalg
import control  # Python Control Systems Library

class AdvancedControlSystem(Node):
    def __init__(self):
        super().__init__('advanced_control_system')

        # Publishers
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.control_status_pub = self.create_publisher(String, '/control_status', 10)
        self.performance_pub = self.create_publisher(Float32, '/control_performance', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )

        # Control system components
        self.pid_controllers = {}
        self.mpc_controller = ModelPredictiveController(state_dim=6, control_dim=2)
        self.adaptive_controller = ModelReferenceAdaptiveController(
            {'natural_frequency': 1.0, 'damping_ratio': 0.7},
            {'kp': 1.0, 'ki': 0.1}
        )
        self.h_inf_controller = HInfinityController(system_order=4)

        # Robot state
        self.joint_states = None
        self.imu_data = None
        self.current_pose = None
        self.desired_trajectory = []

        # Control parameters
        self.control_frequency = 100.0  # Hz
        self.dt = 1.0 / self.control_frequency
        self.control_mode = 'pid'  # pid, mpc, adaptive, robust

        # Joint names and initial setup
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.initialize_controllers()

        # Performance monitoring
        self.performance_metrics = {
            'tracking_error': 0.0,
            'control_effort': 0.0,
            'stability_margin': 0.0
        }

        # Control timer
        self.control_timer = self.create_timer(1.0/self.control_frequency, self.control_loop)

    def initialize_controllers(self):
        """Initialize PID controllers for each joint"""
        for joint_name in self.joint_names:
            self.pid_controllers[joint_name] = AdvancedPIDController(
                kp=2.0, ki=0.1, kd=0.05, dt=self.dt
            )

    def joint_state_callback(self, msg):
        """Handle joint state updates"""
        self.joint_states = msg

    def imu_callback(self, msg):
        """Handle IMU data for attitude control"""
        self.imu_data = msg
        # Update current pose from IMU data
        self.current_pose = self.imu_to_pose(msg)

    def imu_to_pose(self, imu_msg):
        """Convert IMU data to pose estimate"""
        pose = Pose()
        pose.orientation = imu_msg.orientation
        # Position would come from other sources (odometry, etc.)
        return pose

    def control_loop(self):
        """Main control loop with multiple control strategies"""
        if not self.joint_states:
            return

        # Get current state
        current_positions = dict(zip(self.joint_states.name, self.joint_states.position))
        current_velocities = dict(zip(self.joint_states.name, self.joint_states.velocity))

        # Select control strategy
        if self.control_mode == 'pid':
            commands = self.pid_control(current_positions, current_velocities)
        elif self.control_mode == 'mpc':
            commands = self.mpc_control(current_positions, current_velocities)
        elif self.control_mode == 'adaptive':
            commands = self.adaptive_control(current_positions, current_velocities)
        elif self.control_mode == 'robust':
            commands = self.robust_control(current_positions, current_velocities)
        else:
            commands = self.fallback_control(current_positions)

        # Execute commands
        self.execute_commands(commands)

        # Monitor performance
        self.monitor_performance(current_positions, commands)

        # Publish status
        self.publish_control_status()

    def pid_control(self, current_positions, current_velocities):
        """PID-based joint control"""
        commands = JointState()
        commands.name = self.joint_names
        commands.position = []
        commands.velocity = []
        commands.effort = []

        # Simple trajectory following
        desired_positions = self.get_desired_positions()

        for joint_name in self.joint_names:
            if joint_name in current_positions and joint_name in desired_positions:
                # Update PID controller
                control_output = self.pid_controllers[joint_name].update(
                    desired_positions[joint_name],
                    current_positions[joint_name]
                )

                commands.position.append(desired_positions[joint_name])
                commands.velocity.append(control_output)
                commands.effort.append(control_output)  # Simplified mapping
            else:
                commands.position.append(0.0)
                commands.velocity.append(0.0)
                commands.effort.append(0.0)

        return commands

    def mpc_control(self, current_positions, current_velocities):
        """MPC-based control"""
        # Prepare state vector
        state = self.pack_state_vector(current_positions, current_velocities)

        # Get reference trajectory
        reference_trajectory = self.get_reference_trajectory()

        # Compute optimal control
        optimal_control, success = self.mpc_controller.compute_control(
            state, reference_trajectory
        )

        # Convert to joint commands
        commands = self.convert_control_to_commands(optimal_control)
        return commands

    def adaptive_control(self, current_positions, current_velocities):
        """Adaptive control based on system identification"""
        # Get current state
        current_state = self.pack_state_vector(current_positions, current_velocities)

        # Get reference state
        reference_state = self.get_reference_state()

        # Compute adaptive control
        control_signal = self.adaptive_controller.compute_control(
            reference_state, current_state, reference_state
        )

        # Convert to commands
        commands = self.convert_control_to_commands(control_signal)
        return commands

    def robust_control(self, current_positions, current_velocities):
        """Robust H-infinity control"""
        # Pack state vector
        state = self.pack_state_vector(current_positions, current_velocities)

        # Get reference
        reference = self.get_reference_state()

        # Compute robust control
        control_signal = self.h_inf_controller.compute_robust_control(
            state, reference
        )

        # Convert to commands
        commands = self.convert_control_to_commands(control_signal)
        return commands

    def fallback_control(self, current_positions):
        """Fallback control if advanced methods fail"""
        commands = JointState()
        commands.name = list(current_positions.keys())
        commands.position = list(current_positions.values())
        commands.velocity = [0.0] * len(current_positions)
        commands.effort = [0.0] * len(current_positions)
        return commands

    def pack_state_vector(self, positions, velocities):
        """Pack positions and velocities into state vector"""
        state = np.zeros(2 * len(self.joint_names))

        for i, joint_name in enumerate(self.joint_names):
            state[i] = positions.get(joint_name, 0.0)  # Positions in first half
            state[i + len(self.joint_names)] = velocities.get(joint_name, 0.0)  # Velocities in second half

        return state

    def get_desired_positions(self):
        """Get desired joint positions (from trajectory planner)"""
        # This would interface with trajectory planner
        # For now, return simple sinusoidal trajectory
        import time
        t = time.time()
        desired = {}
        for i, joint_name in enumerate(self.joint_names):
            desired[joint_name] = 0.5 * np.sin(0.5 * t + i * np.pi / 3)
        return desired

    def get_reference_trajectory(self):
        """Get reference trajectory for MPC"""
        # Generate simple reference trajectory
        reference_trajectory = []
        for k in range(self.mpc_controller.N):
            state = np.zeros(self.mpc_controller.state_dim)
            # Simple circular trajectory
            state[0] = 0.5 * np.cos(0.1 * k)  # x position
            state[1] = 0.5 * np.sin(0.1 * k)  # y position
            reference_trajectory.append(state)
        return reference_trajectory

    def get_reference_state(self):
        """Get current reference state"""
        return np.zeros(2 * len(self.joint_names))  # Placeholder

    def convert_control_to_commands(self, control_signal):
        """Convert control signal to ROS command"""
        commands = JointState()
        commands.name = self.joint_names

        # Map control signal to joint commands
        # This would depend on the specific robot kinematics
        for i, joint_name in enumerate(self.joint_names):
            if i < len(control_signal):
                commands.position.append(control_signal[i])
                commands.velocity.append(control_signal[i])
                commands.effort.append(control_signal[i])
            else:
                commands.position.append(0.0)
                commands.velocity.append(0.0)
                commands.effort.append(0.0)

        return commands

    def execute_commands(self, commands):
        """Execute the computed control commands"""
        self.joint_cmd_pub.publish(commands)

    def monitor_performance(self, current_positions, commands):
        """Monitor control performance metrics"""
        # Calculate tracking error
        desired_positions = self.get_desired_positions()
        tracking_errors = []
        for joint_name in self.joint_names:
            if joint_name in current_positions and joint_name in desired_positions:
                error = abs(desired_positions[joint_name] - current_positions[joint_name])
                tracking_errors.append(error)

        self.performance_metrics['tracking_error'] = np.mean(tracking_errors) if tracking_errors else 0.0

        # Calculate control effort
        control_efforts = [abs(effort) for effort in commands.effort]
        self.performance_metrics['control_effort'] = np.mean(control_efforts) if control_efforts else 0.0

        # Publish performance metrics
        perf_msg = Float32()
        perf_msg.data = self.performance_metrics['tracking_error']
        self.performance_pub.publish(perf_msg)

    def publish_control_status(self):
        """Publish control system status"""
        status_msg = String()
        status_msg.data = (
            f"Mode: {self.control_mode}, "
            f"Error: {self.performance_metrics['tracking_error']:.3f}, "
            f"Effort: {self.performance_metrics['control_effort']:.3f}, "
            f"Joints: {len(self.joint_names)}"
        )
        self.control_status_pub.publish(status_msg)

    def switch_control_mode(self, new_mode):
        """Switch between different control modes"""
        if new_mode in ['pid', 'mpc', 'adaptive', 'robust']:
            self.control_mode = new_mode
            self.get_logger().info(f'Switched to {new_mode} control mode')

class SlidingModeController:
    """Sliding mode control for robust performance"""
    def __init__(self, state_dim, control_dim, sliding_surface_params=None):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.lambda_ = 1.0  # Sliding surface parameter
        self.kappa = 0.1   # Boundary layer thickness
        self.rho = 10.0    # Switching gain

        if sliding_surface_params:
            self.surface_params = sliding_surface_params
        else:
            # Default sliding surface: s = λe + ė
            self.surface_params = {'lambda': self.lambda_}

    def compute_control(self, state, reference, dt):
        """Compute sliding mode control"""
        # Tracking error
        error = reference - state[:self.state_dim//2]  # Assuming state = [positions, velocities]
        error_derivative = state[self.state_dim//2:]   # Velocities

        # Sliding surface
        s = self.surface_params['lambda'] * error + error_derivative

        # Equivalent control (nominal system)
        u_eq = self.compute_equivalent_control(state, reference)

        # Switching control
        u_sw = self.compute_switching_control(s)

        # Total control
        u_total = u_eq + u_sw

        return u_total

    def compute_equivalent_control(self, state, reference):
        """Compute equivalent control (nominal system)"""
        # For this example, use simple PD control
        error = reference - state[:self.state_dim//2]
        error_derivative = state[self.state_dim//2:]
        return 2.0 * error + 1.0 * error_derivative

    def compute_switching_control(self, s):
        """Compute switching control"""
        # Boundary layer to reduce chattering
        saturation = np.tanh(s / self.kappa)
        return -self.rho * saturation

class FuzzyLogicController:
    """Fuzzy logic controller for nonlinear systems"""
    def __init__(self):
        # Define fuzzy sets and rules
        self.fuzzy_rules = [
            {'input': {'error': 'negative', 'derivative': 'negative'}, 'output': 'negative_high'},
            {'input': {'error': 'negative', 'derivative': 'zero'}, 'output': 'negative_medium'},
            {'input': {'error': 'negative', 'derivative': 'positive'}, 'output': 'negative_low'},
            {'input': {'error': 'zero', 'derivative': 'negative'}, 'output': 'negative_medium'},
            {'input': {'error': 'zero', 'derivative': 'zero'}, 'output': 'zero'},
            {'input': {'error': 'zero', 'derivative': 'positive'}, 'output': 'positive_medium'},
            {'input': {'error': 'positive', 'derivative': 'negative'}, 'output': 'positive_low'},
            {'input': {'error': 'positive', 'derivative': 'zero'}, 'output': 'positive_medium'},
            {'input': {'error': 'positive', 'derivative': 'positive'}, 'output': 'positive_high'}
        ]

    def fuzzify(self, error, error_derivative):
        """Convert crisp inputs to fuzzy values"""
        membership = {
            'error': self.get_membership(error, 'error'),
            'derivative': self.get_membership(error_derivative, 'derivative')
        }
        return membership

    def get_membership(self, value, var_type):
        """Get membership values for a variable"""
        if var_type == 'error':
            return {
                'negative': self.triangle_membership(value, -2, -1, 0),
                'zero': self.triangle_membership(value, -1, 0, 1),
                'positive': self.triangle_membership(value, 0, 1, 2)
            }
        elif var_type == 'derivative':
            return {
                'negative': self.triangle_membership(value, -1, -0.5, 0),
                'zero': self.triangle_membership(value, -0.5, 0, 0.5),
                'positive': self.triangle_membership(value, 0, 0.5, 1)
            }

    def triangle_membership(self, x, a, b, c):
        """Triangular membership function"""
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:  # b < x < c
            return (c - x) / (c - b)

    def infer(self, memberships):
        """Apply fuzzy rules to get fuzzy output"""
        output_membership = {}
        for rule in self.fuzzy_rules:
            # Get firing strength
            error_fuzz = memberships['error'][rule['input']['error']]
            deriv_fuzz = memberships['derivative'][rule['input']['derivative']]
            firing_strength = min(error_fuzz, deriv_fuzz)

            # Apply to output
            output_type = rule['output']
            if output_type not in output_membership:
                output_membership[output_type] = 0.0
            output_membership[output_type] = max(output_membership[output_type], firing_strength)

        return output_membership

    def defuzzify(self, output_membership):
        """Convert fuzzy output to crisp control value"""
        # Center of gravity method
        numerator = 0.0
        denominator = 0.0

        # Define output membership functions
        output_centers = {
            'negative_high': -3.0,
            'negative_medium': -2.0,
            'negative_low': -1.0,
            'zero': 0.0,
            'positive_low': 1.0,
            'positive_medium': 2.0,
            'positive_high': 3.0
        }

        for output_type, membership_value in output_membership.items():
            if output_type in output_centers:
                numerator += output_centers[output_type] * membership_value
                denominator += membership_value

        return numerator / denominator if denominator != 0.0 else 0.0

    def compute_control(self, error, error_derivative):
        """Compute fuzzy control output"""
        memberships = self.fuzzify(error, error_derivative)
        fuzzy_output = self.infer(memberships)
        crisp_output = self.defuzzify(fuzzy_output)
        return crisp_output

def main(args=None):
    rclpy.init(args=args)
    control_system = AdvancedControlSystem()

    try:
        rclpy.spin(control_system)
    except KeyboardInterrupt:
        pass
    finally:
        control_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Nonlinear Control Techniques

### Feedback Linearization

For highly nonlinear robotic systems:

```python
# Example: Feedback linearization for robot manipulator
class FeedbackLinearizationController:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.nominal_controller = AdvancedPIDController()

    def feedback_linearize(self, q, dq, ddq_desired):
        """Apply feedback linearization to robot dynamics"""
        # Robot dynamics: M(q)ddq + C(q,dq)dq + g(q) = τ
        M = self.robot_model.mass_matrix(q)
        C = self.robot_model.coriolis_matrix(q, dq)
        g = self.robot_model.gravity_vector(q)

        # Desired torque to achieve desired acceleration
        tau = M @ ddq_desired + C @ dq + g

        return tau

    def compute_control(self, q, dq, q_desired, dq_desired, ddq_desired):
        """Compute control using feedback linearization"""
        # First, compute error in original coordinates
        q_error = q_desired - q
        dq_error = dq_desired - dq

        # Apply nominal controller to get desired acceleration
        ddq_command = self.nominal_controller.update(q_desired, q)

        # Apply feedback linearization
        tau = self.feedback_linearize(q, dq, ddq_command)

        return tau
```

### Backstepping Control

Systematic design for cascade nonlinear systems:

```python
# Example: Backstepping controller
class BacksteppingController:
    def __init__(self):
        self.stabilizing_functions = []
        self.control_gains = []

    def design_for_system(self, system_order):
        """Design backstepping controller for system of given order"""
        for i in range(system_order):
            # Design stabilizing function for i-th subsystem
            self.stabilizing_functions.append(self.design_stabilizing_function(i))
            self.control_gains.append(1.0)  # Initial gains

    def design_stabilizing_function(self, step):
        """Design stabilizing function for backstepping step"""
        # This would implement the systematic backstepping design
        # For each step, design a virtual control that stabilizes the subsystem
        def stabilizing_func(states_up_to_i, desired_states_up_to_i):
            # Virtual control for step i
            error = desired_states_up_to_i[-1] - states_up_to_i[-1]
            virtual_control = -self.control_gains[step] * error
            return virtual_control
        return stabilizing_func

    def compute_control(self, full_state, desired_trajectory):
        """Compute control using backstepping design"""
        # Implement the recursive backstepping algorithm
        # This would involve computing virtual controls for each subsystem
        # and finally the actual control input
        control = 0.0

        # Simplified implementation
        for i in range(len(full_state)):
            if i < len(self.stabilizing_functions):
                # Compute virtual control for this step
                current_states = full_state[:i+1]
                desired_states = desired_trajectory[:i+1]
                virtual_control = self.stabilizing_functions[i](
                    current_states, desired_states
                )
                control = virtual_control

        return control
```

## Lab: Implementing Advanced Control Systems

In this lab, you'll implement an advanced control system:

```python
# lab_advanced_control.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32
import numpy as np

class AdvancedControlLab(Node):
    def __init__(self):
        super().__init__('advanced_control_lab')

        # Publishers
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.performance_pub = self.create_publisher(Float32, '/control_performance', 10)
        self.status_pub = self.create_publisher(String, '/control_status', 10)

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10
        )

        # Control components
        self.cascade_controller = CascadeController()
        self.sliding_controller = SlidingModeController(state_dim=4, control_dim=2)
        self.fuzzy_controller = FuzzyLogicController()

        # Data storage
        self.joint_data = None
        self.control_mode = 'cascade'  # cascade, sliding, fuzzy

        # Control parameters
        self.control_frequency = 50.0
        self.performance_history = []

        # Control timer
        self.control_timer = self.create_timer(1.0/self.control_frequency, self.control_loop)

    def joint_callback(self, msg):
        """Handle joint state data"""
        self.joint_data = msg

    def control_loop(self):
        """Main control loop"""
        if self.joint_data is None:
            return

        # Get current state
        current_state = np.array(self.joint_data.position + self.joint_data.velocity)

        # Get desired state
        desired_state = self.get_desired_state()

        # Compute control based on mode
        if self.control_mode == 'cascade':
            control_output = self.cascade_control(current_state, desired_state)
        elif self.control_mode == 'sliding':
            control_output = self.sliding_control(current_state, desired_state)
        elif self.control_mode == 'fuzzy':
            control_output = self.fuzzy_control(current_state, desired_state)
        else:
            control_output = np.zeros(len(current_state)//2)  # Zero control

        # Execute control
        self.execute_control(control_output)

        # Monitor performance
        self.monitor_performance(current_state, desired_state, control_output)

        # Publish status
        self.publish_status()

    def get_desired_state(self):
        """Get desired state trajectory"""
        import time
        t = time.time()
        # Simple sinusoidal trajectory
        desired_positions = [0.5 * np.sin(0.5 * t + i * np.pi/4) for i in range(6)]
        desired_velocities = [0.5 * 0.5 * np.cos(0.5 * t + i * np.pi/4) for i in range(6)]
        return np.array(desired_positions + desired_velocities)

    def cascade_control(self, current_state, desired_state):
        """Cascade control implementation"""
        # Simplified cascade control
        position_error = desired_state[:6] - current_state[:6]
        velocity_error = desired_state[6:] - current_state[6:]

        # Position controller
        velocity_command = position_error * 2.0  # Simple proportional

        # Velocity controller
        control_output = (velocity_command - current_state[6:]) * 1.5  # Simple proportional

        return control_output

    def sliding_control(self, current_state, desired_state):
        """Sliding mode control implementation"""
        # Calculate error
        error = desired_state[:6] - current_state[:6]
        error_deriv = desired_state[6:] - current_state[6:]

        # Sliding surface
        s = error + error_deriv

        # Control law
        control_output = -5.0 * np.sign(s)  # Simplified sliding control

        return control_output

    def fuzzy_control(self, current_state, desired_state):
        """Fuzzy logic control implementation"""
        # Calculate errors for first joint as example
        error = desired_state[0] - current_state[0]
        error_deriv = desired_state[6] - current_state[6]

        # Fuzzy control for first joint
        fuzzy_output = self.fuzzy_controller.compute_control(error, error_deriv)

        # Apply to all joints (simplified)
        control_output = np.array([fuzzy_output] * 6)

        return control_output

    def execute_control(self, control_output):
        """Execute the computed control"""
        # Create joint command
        joint_cmd = JointState()
        joint_cmd.name = [f'joint_{i}' for i in range(len(control_output))]
        joint_cmd.effort = control_output.tolist()
        joint_cmd.position = [0.0] * len(control_output)  # Will be updated by position control
        joint_cmd.velocity = control_output.tolist()

        self.joint_cmd_pub.publish(joint_cmd)

    def monitor_performance(self, current_state, desired_state, control_output):
        """Monitor control performance"""
        # Calculate tracking error
        position_error = desired_state[:6] - current_state[:6]
        tracking_error = np.mean(np.abs(position_error))

        # Calculate control effort
        control_effort = np.mean(np.abs(control_output))

        # Store in history
        self.performance_history.append({
            'error': tracking_error,
            'effort': control_effort,
            'timestamp': self.get_clock().now()
        })

        # Keep history manageable
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

        # Publish performance metric
        perf_msg = Float32()
        perf_msg.data = tracking_error
        self.performance_pub.publish(perf_msg)

    def publish_status(self):
        """Publish control status"""
        status_msg = String()
        if self.performance_history:
            avg_error = np.mean([p['error'] for p in self.performance_history[-10:]])
        else:
            avg_error = 0.0

        status_msg.data = (
            f"Mode: {self.control_mode}, "
            f"Avg Error: {avg_error:.3f}, "
            f"Joints: {len(self.joint_data.position) if self.joint_data else 0}"
        )
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    lab = AdvancedControlLab()

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

## Exercise: Design Your Own Advanced Control System

Consider the following design challenge:

1. What specific robot system are you controlling (mobile robot, manipulator, humanoid)?
2. What are the key dynamic characteristics of your system?
3. Which advanced control technique is most appropriate (PID variants, MPC, adaptive, robust, nonlinear)?
4. What constraints must your controller handle?
5. How will you handle model uncertainty?
6. What performance metrics are most important?
7. How will you ensure stability and safety?
8. What experimental validation will you perform?

## Summary

Advanced control systems are essential for sophisticated robotics applications, enabling robots to operate effectively in complex, dynamic environments. Key concepts include:

- **PID and Cascade Control**: Foundation for many robotic control systems
- **Model Predictive Control**: Handles constraints and multi-objective optimization
- **Adaptive Control**: Adjusts parameters based on system behavior
- **Robust Control**: Maintains performance despite uncertainty
- **Nonlinear Control**: Handles inherent nonlinearities in robotic systems
- **Fuzzy Logic Control**: Deals with imprecise or uncertain information

The integration of these advanced control techniques in ROS2 enables the development of sophisticated robotic systems that can handle complex tasks with high performance and reliability. Understanding these concepts is crucial for developing robots that can operate effectively in real-world scenarios.

In the next lesson, we'll explore dynamic interaction patterns and how robots can adapt their behavior based on changing environmental conditions and user needs.