---
sidebar_position: 2
---

# Validation and Testing Methods for Physical AI Systems

## Introduction

Validation and testing are critical for ensuring the safety, reliability, and effectiveness of Physical AI systems in real-world deployments. Unlike traditional software systems, Physical AI systems must be tested across multiple dimensions: functional correctness, safety, human interaction, environmental adaptability, and long-term reliability. This lesson explores comprehensive validation and testing methodologies specifically designed for embodied AI systems.

## Testing Methodologies for Physical AI

### 1. Unit Testing for Physical Components

Testing individual physical AI components in isolation:

```python
# Example: Unit testing for physical AI components
import unittest
import numpy as np
from unittest.mock import Mock, patch

class TestPhysicalAIComponents(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_robot = Mock()
        self.test_robot.joint_states = Mock()
        self.test_robot.joint_states.position = [0.0, 0.0, 0.0]
        self.test_robot.joint_states.velocity = [0.0, 0.0, 0.0]
        self.test_robot.joint_states.effort = [0.0, 0.0, 0.0]

    def test_kinematic_solver_basic_functionality(self):
        """Test basic functionality of kinematic solver"""
        from kinematics import KinematicSolver

        solver = KinematicSolver()

        # Test forward kinematics
        joint_angles = [0.0, 0.0, 0.0]
        expected_position = [0.5, 0.0, 0.5]  # Expected for zero configuration

        result = solver.forward_kinematics(joint_angles)

        # Check if result is approximately equal to expected
        np.testing.assert_array_almost_equal(result, expected_position, decimal=2)

    def test_inverse_kinematics_solution(self):
        """Test inverse kinematics solution"""
        from kinematics import InverseKinematicSolver

        solver = InverseKinematicSolver()

        # Test reaching a reachable position
        target_position = [0.3, 0.2, 0.4]
        initial_guess = [0.0, 0.0, 0.0]

        solution = solver.solve(target_position, initial_guess)

        # Verify solution is not None
        self.assertIsNotNone(solution)

        # Verify solution has reasonable joint angles
        self.assertEqual(len(solution), len(initial_guess))
        for angle in solution:
            self.assertTrue(-np.pi <= angle <= np.pi, f"Angle {angle} is outside valid range")

    def test_collision_detection_basic(self):
        """Test basic collision detection functionality"""
        from collision_detection import CollisionDetector

        detector = CollisionDetector()

        # Test with no obstacles
        robot_position = [0.0, 0.0, 0.0]
        obstacles = []

        collision_result = detector.check_collision(robot_position, obstacles)

        self.assertFalse(collision_result['collision_detected'])
        self.assertEqual(len(collision_result['colliding_objects']), 0)

    def test_collision_detection_with_obstacle(self):
        """Test collision detection with nearby obstacle"""
        from collision_detection import CollisionDetector

        detector = CollisionDetector()

        # Test with nearby obstacle
        robot_position = [0.0, 0.0, 0.0]
        obstacles = [{'position': [0.1, 0.0, 0.0], 'radius': 0.2}]  # Within collision distance

        collision_result = detector.check_collision(robot_position, obstacles)

        self.assertTrue(collision_result['collision_detected'])
        self.assertEqual(len(collision_result['colliding_objects']), 1)

    def test_path_planner_basic_functionality(self):
        """Test basic path planning functionality"""
        from path_planning import PathPlanner

        planner = PathPlanner()

        # Test with simple start and goal
        start = [0.0, 0.0]
        goal = [1.0, 1.0]
        obstacles = []

        path = planner.plan_path(start, goal, obstacles)

        # Path should not be empty
        self.assertGreater(len(path), 0)

        # First point should be near start
        np.testing.assert_array_almost_equal(path[0], start, decimal=1)

        # Last point should be near goal
        np.testing.assert_array_almost_equal(path[-1], goal, decimal=1)

    def test_trajectory_generator_basic(self):
        """Test basic trajectory generation"""
        from trajectory_generation import TrajectoryGenerator

        generator = TrajectoryGenerator()

        # Test with simple path
        path = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
        max_velocity = 0.5
        max_acceleration = 1.0

        trajectory = generator.generate_trajectory(path, max_velocity, max_acceleration)

        # Should have more points than original path
        self.assertGreater(len(trajectory), len(path))

        # Each point should have time and position
        for point in trajectory:
            self.assertIn('position', point)
            self.assertIn('time', point)
            self.assertIn('velocity', point)

    def test_sensor_fusion_basic(self):
        """Test basic sensor fusion functionality"""
        from sensor_fusion import SensorFusion

        fusion = SensorFusion()

        # Test with mock sensor data
        sensor_data = {
            'camera': {'position': [1.0, 0.0, 0.0], 'confidence': 0.8},
            'lidar': {'position': [1.1, 0.05, 0.0], 'confidence': 0.9},
            'imu': {'position': [0.95, -0.02, 0.0], 'confidence': 0.7}
        }

        fused_result = fusion.fuse_sensors(sensor_data)

        # Check that fused result is reasonable
        self.assertIn('position', fused_result)
        self.assertIn('confidence', fused_result)

        # Fused position should be close to sensor measurements
        fused_pos = fused_result['position']
        self.assertAlmostEqual(fused_pos[0], 1.0, places=1)  # Should be around 1.0

    def test_control_system_stability(self):
        """Test control system stability"""
        from control_system import PIDController

        controller = PIDController(kp=1.0, ki=0.1, kd=0.05)

        # Test with simple error
        error = 0.1
        dt = 0.01

        control_output = controller.update(error, dt)

        # Control output should be finite
        self.assertTrue(np.isfinite(control_output))

        # Control output should be reasonable magnitude
        self.assertLess(abs(control_output), 10.0)

    def test_perception_pipeline(self):
        """Test perception pipeline functionality"""
        from perception import PerceptionPipeline

        pipeline = PerceptionPipeline()

        # Mock sensor input (in practice, this would be actual sensor data)
        mock_image = np.random.rand(480, 640, 3).astype(np.uint8)
        mock_lidar = [1.0, 1.2, 1.5, 2.0] * 90  # Simulated laser scan

        # Process perception
        perception_result = pipeline.process(mock_image, mock_lidar)

        # Check that result has expected components
        self.assertIn('objects', perception_result)
        self.assertIn('obstacles', perception_result)
        self.assertIn('features', perception_result)

    def test_human_interaction_classifier(self):
        """Test human interaction intent classifier"""
        from interaction import HumanInteractionClassifier

        classifier = HumanInteractionClassifier()

        # Test with various interaction types
        test_inputs = [
            {'type': 'greeting', 'content': 'hello robot'},
            {'type': 'command', 'content': 'move forward'},
            {'type': 'question', 'content': 'what can you do?'}
        ]

        for test_input in test_inputs:
            intent = classifier.classify_intent(test_input['content'])
            self.assertIsNotNone(intent)
            self.assertIsInstance(intent, str)

class TestKinematicSolver(unittest.TestCase):
    def test_forward_kinematics(self):
        """Test forward kinematics calculations"""
        solver = KinematicSolver(robot_config={'link_lengths': [0.5, 0.4, 0.3]})

        # Test with known angles
        joint_angles = [np.pi/4, -np.pi/6, np.pi/3]
        expected_position = [0.354, 0.146, 0.950]  # Calculated manually

        result = solver.forward_kinematics(joint_angles)

        np.testing.assert_array_almost_equal(result[:3], expected_position, decimal=3)

    def test_inverse_kinematics_accuracy(self):
        """Test accuracy of inverse kinematics"""
        solver = InverseKinematicSolver(robot_config={'link_lengths': [0.5, 0.4, 0.3]})

        # Test target that should be achievable
        target = [0.4, 0.2, 0.3]

        solution = solver.solve(target, [0, 0, 0])
        self.assertIsNotNone(solution)

        # Verify solution reaches target
        calculated_pos = solver.forward_kinematics(solution)
        distance_error = np.linalg.norm(np.array(calculated_pos[:3]) - np.array(target))

        self.assertLess(distance_error, 0.05)  # Less than 5cm error

class TestCollisionDetection(unittest.TestCase):
    def test_sphere_collision(self):
        """Test sphere-based collision detection"""
        detector = SphereCollisionDetector()

        # Two spheres that should collide
        sphere1 = {'position': [0, 0, 0], 'radius': 0.5}
        sphere2 = {'position': [0.6, 0, 0], 'radius': 0.5}

        result = detector.check_collision(sphere1, sphere2)
        self.assertTrue(result)

    def test_sphere_no_collision(self):
        """Test sphere-based collision detection with no collision"""
        detector = SphereCollisionDetector()

        # Two spheres that should not collide
        sphere1 = {'position': [0, 0, 0], 'radius': 0.5}
        sphere2 = {'position': [1.1, 0, 0], 'radius': 0.5}

        result = detector.check_collision(sphere1, sphere2)
        self.assertFalse(result)

    def test_aabb_collision(self):
        """Test axis-aligned bounding box collision detection"""
        detector = AABBCollisionDetector()

        # Two boxes that should collide
        box1 = {'min': [0, 0, 0], 'max': [1, 1, 1]}
        box2 = {'min': [0.5, 0.5, 0.5], 'max': [1.5, 1.5, 1.5]}

        result = detector.check_collision(box1, box2)
        self.assertTrue(result)

    def test_aabb_no_collision(self):
        """Test AABB collision detection with no collision"""
        detector = AABBCollisionDetector()

        # Two boxes that should not collide
        box1 = {'min': [0, 0, 0], 'max': [1, 1, 1]}
        box2 = {'min': [2, 2, 2], 'max': [3, 3, 3]}

        result = detector.check_collision(box1, box2)
        self.assertFalse(result)

class TestPathPlanning(unittest.TestCase):
    def test_astar_basic(self):
        """Test basic A* pathfinding"""
        planner = AStarPlanner(grid_resolution=0.1)

        # Simple grid with no obstacles
        grid = np.zeros((20, 20))  # Free space
        start = (1, 1)
        goal = (18, 18)

        path = planner.plan_path(grid, start, goal)

        self.assertIsNotNone(path)
        self.assertGreater(len(path), 0)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)

    def test_astar_with_obstacles(self):
        """Test A* pathfinding with obstacles"""
        planner = AStarPlanner(grid_resolution=0.1)

        # Grid with obstacle in the middle
        grid = np.zeros((20, 20))
        grid[10:12, :] = 1  # Horizontal wall
        start = (1, 10)
        goal = (18, 10)

        path = planner.plan_path(grid, start, goal)

        # Should find path around obstacle
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 10)  # Should be longer than direct path

    def test_rrt_basic(self):
        """Test basic RRT pathfinding"""
        planner = RRTPlanner()

        # Simple 2D space with no obstacles
        start = [0, 0]
        goal = [5, 5]
        obstacles = []

        path = planner.plan_path(start, goal, obstacles)

        if path:
            self.assertGreater(len(path), 0)
            # Check that path starts and ends at correct positions
            np.testing.assert_array_almost_equal(path[0], start, decimal=1)
            np.testing.assert_array_almost_equal(path[-1], goal, decimal=1)

class TestControlSystems(unittest.TestCase):
    def test_pid_stability(self):
        """Test PID controller stability"""
        controller = PIDController(kp=1.0, ki=0.1, kd=0.05)

        # Test with oscillating error to check stability
        errors = [0.1, -0.05, 0.02, -0.01, 0.005]
        dt = 0.01

        outputs = []
        for error in errors:
            output = controller.update(error, dt)
            outputs.append(output)

        # Check that outputs don't grow unbounded
        self.assertLess(max(abs(o) for o in outputs), 100.0)

    def test_pid_integral_windup(self):
        """Test PID integral windup protection"""
        controller = PIDController(kp=1.0, ki=0.1, kd=0.05, integral_limit=10.0)

        # Apply large error for many iterations
        dt = 0.01
        for i in range(1000):
            output = controller.update(10.0, dt)  # Large constant error

        # Check that integral term is limited
        self.assertLessEqual(abs(controller.integral_error), 10.0)

    def test_trajectory_tracking(self):
        """Test trajectory tracking controller"""
        controller = TrajectoryTrackingController()

        # Simple trajectory: straight line
        trajectory = [
            {'position': [0, 0], 'time': 0},
            {'position': [1, 0], 'time': 1},
            {'position': [2, 0], 'time': 2}
        ]

        # Current state
        current_state = {'position': [0.1, 0.0], 'velocity': [0.9, 0.0], 'time': 0.1}

        control_command = controller.track_trajectory(current_state, trajectory)

        # Should return reasonable control command
        self.assertIn('linear_velocity', control_command)
        self.assertIn('angular_velocity', control_command)
        self.assertTrue(isinstance(control_command['linear_velocity'], (int, float)))
        self.assertTrue(isinstance(control_command['angular_velocity'], (int, float)))

class TestPerception(unittest.TestCase):
    def test_object_detection_accuracy(self):
        """Test object detection accuracy"""
        detector = ObjectDetector(model_path='mock_model')

        # Mock image with known objects
        mock_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        # Add a red rectangle to represent an object
        cv2.rectangle(mock_image, (100, 100), (200, 200), (0, 0, 255), -1)

        detections = detector.detect_objects(mock_image)

        # Should detect the red rectangle
        self.assertGreater(len(detections), 0)

        # Check that at least one detection is in the expected location
        found_expected_detection = False
        for detection in detections:
            x, y, w, h = detection['bbox']
            if (100 <= x <= 110 and 100 <= y <= 110 and
                90 <= w <= 110 and 90 <= h <= 110):
                found_expected_detection = True
                break

        self.assertTrue(found_expected_detection)

    def test_feature_matching(self):
        """Test feature matching functionality"""
        matcher = FeatureMatcher()

        # Create mock feature descriptors
        desc1 = np.random.rand(50, 128).astype(np.float32)
        desc2 = np.random.rand(40, 128).astype(np.float32)

        matches = matcher.match_features(desc1, desc2)

        # Should return list of matches
        self.assertIsInstance(matches, list)
        # Matches should not exceed the smaller descriptor count
        self.assertLessEqual(len(matches), min(50, 40))

    def test_optical_flow(self):
        """Test optical flow computation"""
        flow_calculator = OpticalFlowCalculator()

        # Create mock consecutive frames
        frame1 = np.random.rand(480, 640).astype(np.uint8) * 255
        frame2 = np.random.rand(480, 640).astype(np.uint8) * 255

        flow = flow_calculator.calculate_flow(frame1, frame2)

        # Should return flow field with same dimensions as input
        self.assertEqual(flow.shape, (480, 640, 2))  # x and y components

class TestHumanInteraction(unittest.TestCase):
    def test_intent_classification(self):
        """Test human intent classification accuracy"""
        classifier = IntentClassifier()

        test_cases = [
            ("Please move forward", "navigation"),
            ("Can you help me?", "assistance_request"),
            ("Hello robot", "greeting"),
            ("Stop what you're doing", "command_stop")
        ]

        for text, expected_intent in test_cases:
            predicted_intent = classifier.classify_intent(text)
            self.assertEqual(predicted_intent, expected_intent)

    def test_speech_recognition(self):
        """Test speech recognition functionality"""
        recognizer = SpeechRecognizer()

        # This would typically use actual audio processing
        # For testing, we'll mock the recognition
        mock_audio = np.zeros(16000)  # 1 second of mock audio

        with patch.object(recognizer, 'recognize_from_audio') as mock_recognize:
            mock_recognize.return_value = "test command"
            result = recognizer.recognize_speech(mock_audio)

        self.assertEqual(result, "test command")

    def test_gesture_recognition(self):
        """Test gesture recognition functionality"""
        recognizer = GestureRecognizer()

        # Mock gesture data (simplified)
        mock_gesture = {
            'hand_positions': [[0, 0, 0], [0.1, 0.1, 0]],
            'hand_velocities': [[0.1, 0.1, 0], [0.2, 0.2, 0]]
        }

        gesture_type = recognizer.recognize_gesture(mock_gesture)

        # Should return some gesture type
        self.assertIsInstance(gesture_type, str)

class TestSafetySystems(unittest.TestCase):
    def test_emergency_stop(self):
        """Test emergency stop functionality"""
        safety_system = SafetySystem()

        # Initially safe
        self.assertFalse(safety_system.emergency_active)

        # Trigger emergency
        safety_system.trigger_emergency_stop("test emergency")

        # Should be in emergency state
        self.assertTrue(safety_system.emergency_active)
        self.assertEqual(len(safety_system.emergency_reasons), 1)

        # Reset emergency
        safety_system.reset_emergency()

        # Should be back to normal
        self.assertFalse(safety_system.emergency_active)

    def test_collision_avoidance(self):
        """Test collision avoidance system"""
        avoidance_system = CollisionAvoidanceSystem()

        # Mock robot state
        robot_state = {
            'position': [0, 0, 0],
            'velocity': [0.5, 0, 0],  # Moving forward
            'dimensions': [0.5, 0.5, 0.8]  # width, depth, height
        }

        # Mock environment with obstacle ahead
        environment_state = {
            'obstacles': [
                {'position': [0.8, 0, 0], 'dimensions': [0.2, 0.2, 0.2]}
            ]
        }

        avoidance_command = avoidance_system.compute_avoidance_command(
            robot_state, environment_state
        )

        # Should return avoidance command
        self.assertIsNotNone(avoidance_command)
        self.assertIn('linear_velocity', avoidance_command)
        self.assertIn('angular_velocity', avoidance_command)

    def test_human_safety_zone(self):
        """Test human safety zone enforcement"""
        safety_system = HumanSafetySystem(safety_radius=1.0)

        # Robot position
        robot_pos = [0, 0, 0]

        # Human close to robot (within safety zone)
        human_pos = [0.5, 0, 0]

        safety_violation = safety_system.check_human_safety(robot_pos, [human_pos])

        # Should detect safety violation
        self.assertTrue(safety_violation['violation_detected'])
        self.assertLess(safety_violation['min_distance'], 1.0)

class TestLearningSystems(unittest.TestCase):
    def test_reinforcement_learning_environment(self):
        """Test reinforcement learning environment"""
        env = PhysicalAIEnvironment()

        # Test reset
        state = env.reset()
        self.assertIsNotNone(state)

        # Test step
        action = np.random.uniform(-1, 1, size=2)  # Random action
        next_state, reward, done, info = env.step(action)

        self.assertIsNotNone(next_state)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_neural_network_training(self):
        """Test neural network training"""
        network = NeuralNetwork(input_size=10, hidden_size=20, output_size=4)

        # Mock training data
        X = np.random.rand(100, 10)
        y = np.random.rand(100, 4)

        initial_loss = network.compute_loss(X[0], y[0])

        # Train for one epoch
        network.train_batch(X, y, epochs=1)

        final_loss = network.compute_loss(X[0], y[0])

        # Loss should be reduced after training
        self.assertLess(final_loss, initial_loss)

    def test_online_learning(self):
        """Test online learning capability"""
        learner = OnlineLearner()

        # Initial prediction
        initial_prediction = learner.predict([1, 2, 3])

        # Update with new experience
        experience = {'input': [1, 2, 3], 'output': [4, 5, 6]}
        learner.update(experience)

        # New prediction should be different (due to learning)
        updated_prediction = learner.predict([1, 2, 3])

        # In a real system, predictions would change after learning
        # For this test, we just verify the system can update
        self.assertIsNotNone(updated_prediction)

# Additional test suites for specific components
class TestROSIntegration(unittest.TestCase):
    def test_message_publishing(self):
        """Test ROS message publishing functionality"""
        # Mock ROS node
        mock_node = Mock()
        mock_publisher = Mock()
        mock_node.create_publisher.return_value = mock_publisher

        # Test publisher creation and usage
        publisher = mock_node.create_publisher(String, '/test_topic', 10)

        # Publish a message
        test_msg = String()
        test_msg.data = "test message"
        publisher.publish(test_msg)

        # Verify publish was called
        mock_publisher.publish.assert_called_once()

    def test_service_call(self):
        """Test ROS service calling functionality"""
        # Mock service client
        mock_client = Mock()
        mock_client.call_async.return_value = "mock_response"

        # Call service
        request = Mock()
        future = mock_client.call_async(request)

        # Should return a future object
        self.assertIsNotNone(future)

    def test_action_server(self):
        """Test ROS action server functionality"""
        # This would test action server implementation
        # For now, just verify structure
        pass

class TestSimulation(unittest.TestCase):
    def test_physics_simulation_accuracy(self):
        """Test physics simulation accuracy"""
        simulator = PhysicsSimulator()

        # Test simple physics: falling object
        initial_state = {
            'position': [0, 0, 1],  # 1m above ground
            'velocity': [0, 0, 0],
            'mass': 1.0
        }

        # Simulate for 1 second (should fall ~4.9m due to gravity)
        simulated_state = simulator.simulate(initial_state, dt=1.0)

        # Check that object has fallen
        self.assertLess(simulated_state['position'][2], 1.0)
        # Should be close to ground (z=0) after 1 second
        self.assertLessEqual(simulated_state['position'][2], 0.1)

    def test_collision_simulation(self):
        """Test collision simulation"""
        simulator = PhysicsSimulator()

        # Two objects approaching each other
        object1 = {
            'position': [0, 0, 0],
            'velocity': [1, 0, 0],
            'mass': 1.0,
            'radius': 0.1
        }
        object2 = {
            'position': [2, 0, 0],
            'velocity': [-1, 0, 0],
            'mass': 1.0,
            'radius': 0.1
        }

        # Simulate collision
        result = simulator.simulate_collision(object1, object2)

        # After elastic collision, velocities should swap
        # (simplified test - in reality would be more complex)
        self.assertIsNotNone(result)

    def test_sensor_simulation(self):
        """Test sensor simulation accuracy"""
        simulator = SensorSimulator()

        # Test camera simulation
        camera_config = {
            'fov': 60,
            'resolution': [640, 480],
            'position': [0, 0, 1]
        }

        scene = {
            'objects': [
                {'position': [1, 0, 0], 'color': [255, 0, 0], 'size': 0.2}
            ]
        }

        simulated_image = simulator.simulate_camera(camera_config, scene)

        # Should return image-like array
        self.assertEqual(simulated_image.shape, (480, 640, 3))

def run_unit_tests():
    """Run all unit tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()

if __name__ == '__main__':
    # Run the tests
    success = run_unit_tests()
    sys.exit(0 if success else 1)
```

## Integration Testing

Testing how components work together in the Physical AI system:

```python
# Example: Integration testing for Physical AI system
class IntegrationTestSuite:
    def __init__(self):
        self.test_scenarios = [
            self.test_perception_control_integration,
            self.test_navigation_with_obstacle_avoidance,
            self.test_human_robot_interaction,
            self.test_multi_sensor_fusion,
            self.test_safety_intervention,
            self.test_learning_adaptation
        ]
        self.test_results = {}

    def run_all_integration_tests(self):
        """Run all integration tests"""
        results = {}

        for test_func in self.test_scenarios:
            test_name = test_func.__name__
            print(f"Running integration test: {test_name}")

            try:
                result = test_func()
                results[test_name] = result
                print(f"  Result: {'PASS' if result['success'] else 'FAIL'}")
                if not result['success']:
                    print(f"  Error: {result.get('error', 'Unknown error')}")
            except Exception as e:
                results[test_name] = {
                    'success': False,
                    'error': str(e),
                    'details': f"Exception during test: {e}"
                }
                print(f"  Exception: {e}")

        self.test_results = results
        return results

    def test_perception_control_integration(self):
        """Test integration between perception and control systems"""
        try:
            # Set up mock environment
            robot = MockRobot()
            perception_system = PerceptionSystem()
            control_system = ControlSystem()

            # Simulate sensor data
            mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            mock_scan = [float(i) for i in range(360)]  # Mock laser scan

            # Process perception
            perception_result = perception_system.process_sensor_data({
                'camera': mock_image,
                'lidar': mock_scan
            })

            # Use perception result to inform control
            control_command = control_system.generate_command_from_perception(
                perception_result
            )

            # Verify integration worked
            assert control_command is not None
            assert isinstance(control_command, dict)
            assert 'linear_velocity' in control_command
            assert 'angular_velocity' in control_command

            return {
                'success': True,
                'details': 'Perception-control integration successful',
                'perception_objects': len(perception_result.get('objects', [])),
                'control_generated': control_command is not None
            }

        except AssertionError as e:
            return {
                'success': False,
                'error': f'Integration assertion failed: {e}',
                'details': str(e)
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Integration test failed: {e}',
                'details': str(e)
            }

    def test_navigation_with_obstacle_avoidance(self):
        """Test navigation system with obstacle avoidance integration"""
        try:
            # Set up navigation and obstacle avoidance systems
            navigation_system = NavigationSystem()
            obstacle_detector = ObstacleDetectionSystem()
            path_planner = PathPlanner()

            # Define start and goal
            start = [0.0, 0.0]
            goal = [5.0, 5.0]

            # Add obstacles in environment
            obstacles = [
                {'position': [2.0, 2.0], 'radius': 0.5},
                {'position': [3.0, 3.0], 'radius': 0.3}
            ]

            # Plan path initially
            initial_path = path_planner.plan_path(start, goal, obstacles)

            # Simulate robot moving along path while detecting new obstacles
            robot_position = start.copy()
            path_index = 0

            for step in range(100):  # Simulate 100 steps
                # Check for new obstacles (simulation)
                new_obstacles = self.simulate_new_obstacles(robot_position)

                if new_obstacles:
                    # Replan path considering new obstacles
                    updated_obstacles = obstacles + new_obstacles
                    updated_path = path_planner.plan_path(
                        robot_position, goal, updated_obstacles
                    )

                    if updated_path:
                        initial_path = updated_path
                        path_index = 0  # Reset path following

                # Move robot along path
                if path_index < len(initial_path):
                    robot_position = initial_path[path_index]
                    path_index += 1
                else:
                    # Reached end of path, continue toward goal
                    direction_to_goal = np.array(goal) - np.array(robot_position)
                    if np.linalg.norm(direction_to_goal) > 0.1:  # Still far from goal
                        robot_position = (
                            np.array(robot_position) +
                            0.1 * direction_to_goal / np.linalg.norm(direction_to_goal)
                        ).tolist()

                # Check if robot has reached goal (with tolerance)
                if np.linalg.norm(np.array(robot_position) - np.array(goal)) < 0.5:
                    break

            # Verify successful navigation
            final_distance = np.linalg.norm(np.array(robot_position) - np.array(goal))
            success = final_distance < 0.5  # Within 0.5m of goal

            return {
                'success': success,
                'final_distance': final_distance,
                'steps_taken': step + 1,
                'obstacles_encountered': len(obstacles) + len(new_obstacles if 'new_obstacles' in locals() else []),
                'path_replanned': True
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Navigation integration test failed: {e}',
                'details': str(e)
            }

    def test_human_robot_interaction(self):
        """Test human-robot interaction integration"""
        try:
            # Set up interaction components
            speech_recognizer = SpeechRecognizer()
            natural_language_processor = NaturalLanguageProcessor()
            behavior_selector = BehaviorSelector()
            motor_controller = MotorController()

            # Simulate human input
            human_input = "Please move forward slowly"

            # Process through pipeline
            intent = natural_language_processor.parse_intent(human_input)
            behavior = behavior_selector.select_behavior(intent)
            motor_commands = motor_controller.generate_commands(behavior)

            # Verify pipeline worked
            assert intent is not None
            assert behavior is not None
            assert motor_commands is not None

            # Check that appropriate commands were generated
            expected_behavior = 'move_forward_slowly'
            if expected_behavior in behavior.get('actions', []):
                command_generated = True
            else:
                # Check if similar behavior was generated
                command_generated = any('forward' in action for action in behavior.get('actions', []))

            return {
                'success': True,
                'details': 'Human-robot interaction pipeline successful',
                'intent_parsed': intent is not None,
                'behavior_selected': behavior is not None,
                'commands_generated': command_generated
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Human-robot interaction test failed: {e}',
                'details': str(e)
            }

    def test_multi_sensor_fusion(self):
        """Test multi-sensor fusion integration"""
        try:
            # Set up sensor fusion system
            fusion_system = MultiSensorFusionSystem()

            # Simulate data from multiple sensors
            sensor_data = {
                'camera': {
                    'objects': [{'type': 'person', 'position': [1.0, 0.0, 0.0], 'confidence': 0.9}],
                    'timestamp': time.time()
                },
                'lidar': {
                    'obstacles': [{'position': [1.1, 0.05, 0.0], 'distance': 1.1, 'confidence': 0.85}],
                    'timestamp': time.time()
                },
                'imu': {
                    'orientation': [0, 0, 0, 1],  # Quaternion
                    'linear_acceleration': [0.1, 0.0, 9.8],  # Gravity + small acceleration
                    'timestamp': time.time()
                },
                'gps': {
                    'position': [0.0, 0.0, 0.0],  # Global position
                    'accuracy': 2.0,  # meters
                    'timestamp': time.time()
                }
            }

            # Perform fusion
            fused_result = fusion_system.fuse_sensors(sensor_data)

            # Verify fusion produced meaningful result
            assert fused_result is not None
            assert 'position' in fused_result
            assert 'orientation' in fused_result
            assert 'confidence' in fused_result

            # Check that fused confidence is reasonable
            confidence = fused_result['confidence']
            assert 0.0 <= confidence <= 1.0

            return {
                'success': True,
                'details': 'Multi-sensor fusion successful',
                'fused_position': fused_result['position'],
                'fused_orientation': fused_result['orientation'],
                'fused_confidence': confidence,
                'sensor_inputs': len(sensor_data)
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Multi-sensor fusion test failed: {e}',
                'details': str(e)
            }

    def test_safety_intervention(self):
        """Test safety system intervention integration"""
        try:
            # Set up safety-critical scenario
            robot_controller = RobotController()
            safety_system = SafetySystem()
            environment_simulator = EnvironmentSimulator()

            # Initial safe state
            robot_state = {
                'position': [0.0, 0.0, 0.0],
                'velocity': [0.1, 0.0, 0.0],  # Moving slowly forward
                'orientation': [0, 0, 0, 1],
                'safety_status': 'normal'
            }

            # Simulate hazardous situation developing
            environment_state = {
                'obstacles': [],
                'humans': [{'position': [0.5, 0.1, 0.0], 'radius': 0.3}],  # Human approaching
                'hazards': []
            }

            # Run safety monitoring loop
            safety_interventions = 0
            test_duration = 50  # Simulate 50 steps

            for step in range(test_duration):
                # Update environment (human gets closer)
                environment_state['humans'][0]['position'][0] -= 0.01  # Human moves closer

                # Check safety
                safety_assessment = safety_system.assess_situation(
                    robot_state, environment_state
                )

                if safety_assessment['risk_level'] > 0.7:  # High risk detected
                    # Execute safety intervention
                    intervention = safety_system.execute_intervention(
                        safety_assessment, robot_state
                    )

                    if intervention['executed']:
                        safety_interventions += 1

                        # Apply intervention to robot state
                        robot_state = self.apply_safety_intervention(
                            robot_state, intervention
                        )

                        # Verify robot responded appropriately
                        if intervention['type'] == 'stop':
                            assert abs(robot_state['velocity'][0]) < 0.01
                        elif intervention['type'] == 'slow_down':
                            assert abs(robot_state['velocity'][0]) < 0.05
                        elif intervention['type'] == 'change_direction':
                            assert robot_state['velocity'][1] != 0  # Has lateral component

                    # Break if critical safety action taken
                    if safety_assessment['risk_level'] > 0.9:
                        break

            # Verify that safety system intervened appropriately
            success = safety_interventions > 0

            return {
                'success': success,
                'details': f'Safety intervention test completed with {safety_interventions} interventions',
                'interventions_performed': safety_interventions,
                'final_risk_level': safety_assessment.get('risk_level', 0.0),
                'robot_stopped_safely': robot_state['velocity'][0] < 0.01
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Safety intervention test failed: {e}',
                'details': str(e)
            }

    def test_learning_adaptation(self):
        """Test learning system adaptation integration"""
        try:
            # Set up learning system
            perception_system = PerceptionSystem()
            learning_system = LearningSystem()
            behavior_adaptor = BehaviorAdaptor()

            # Simulate initial interaction
            initial_environment = {
                'layout': 'open_space',
                'obstacles': [],
                'lighting': 'bright',
                'noise_level': 'low'
            }

            # Initial learning phase
            learning_episodes = 10
            adaptation_successes = 0

            for episode in range(learning_episodes):
                # Simulate environment interaction
                user_input = f"command_{episode}"
                environment_state = self.simulate_environment_change(initial_environment, episode)

                # Process through perception
                perceptual_state = perception_system.process_environment(environment_state)

                # Learn from interaction
                learning_result = learning_system.learn_from_interaction(
                    user_input, perceptual_state, environment_state
                )

                # Adapt behavior based on learning
                adapted_behavior = behavior_adaptor.adapt_to_environment(
                    perceptual_state, environment_state, learning_result
                )

                # Simulate execution and get feedback
                execution_result = self.simulate_behavior_execution(
                    adapted_behavior, environment_state
                )

                # Provide feedback for further learning
                feedback = self.generate_feedback(execution_result)

                # Update learning system with feedback
                learning_system.update_from_feedback(feedback)

                # Count successful adaptations
                if execution_result.get('success', False):
                    adaptation_successes += 1

            # Verify learning improved over time
            improvement_rate = adaptation_successes / learning_episodes
            success = improvement_rate > 0.7  # Require 70% success rate

            return {
                'success': success,
                'details': f'Learning adaptation test completed with {improvement_rate:.2%} success rate',
                'learning_episodes': learning_episodes,
                'successful_adaptations': adaptation_successes,
                'improvement_rate': improvement_rate,
                'final_performance': improvement_rate
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Learning adaptation test failed: {e}',
                'details': str(e)
            }

    def simulate_new_obstacles(self, robot_position):
        """Simulate detection of new obstacles"""
        # In simulation, sometimes add new obstacles near robot path
        if np.random.random() < 0.1:  # 10% chance of new obstacle
            # Add obstacle in front of robot
            direction_to_goal = np.array([5.0, 5.0]) - np.array(robot_position)
            obstacle_direction = direction_to_goal / np.linalg.norm(direction_to_goal)
            obstacle_position = (
                np.array(robot_position) +
                1.0 * obstacle_direction  # 1m ahead
            )

            return [{'position': obstacle_position.tolist(), 'radius': 0.4}]

        return []

    def apply_safety_intervention(self, robot_state, intervention):
        """Apply safety intervention to robot state"""
        new_state = robot_state.copy()

        if intervention['type'] == 'stop':
            new_state['velocity'] = [0.0, 0.0, 0.0]
        elif intervention['type'] == 'slow_down':
            new_state['velocity'] = [v * 0.3 for v in robot_state['velocity']]  # Reduce to 30%
        elif intervention['type'] == 'change_direction':
            # Add lateral velocity component
            new_state['velocity'][1] += 0.2  # Move sideways

        return new_state

    def simulate_environment_change(self, base_env, episode):
        """Simulate environment changes over episodes"""
        env = base_env.copy()

        # Introduce complexity gradually
        if episode > 3:  # After 3 episodes, add some obstacles
            env['obstacles'] = [{'position': [episode * 0.5, 0, 0], 'radius': 0.2}]

        if episode > 6:  # After 6 episodes, add lighting variation
            env['lighting'] = 'dim' if episode % 2 == 0 else 'bright'

        return env

    def simulate_behavior_execution(self, behavior, environment_state):
        """Simulate behavior execution in environment"""
        # Simplified execution simulation
        success_probability = 0.7  # Base success rate

        # Adjust based on environment complexity
        if environment_state.get('obstacles'):
            success_probability *= 0.8  # Harder with obstacles
        if environment_state.get('lighting') == 'dim':
            success_probability *= 0.9  # Slightly harder in dim light

        # Add some randomness
        success = np.random.random() < success_probability

        return {
            'success': success,
            'execution_time': np.random.uniform(1.0, 3.0),
            'energy_consumed': np.random.uniform(0.1, 0.5),
            'accuracy': np.random.uniform(0.7, 0.95) if success else np.random.uniform(0.1, 0.4)
        }

    def generate_feedback(self, execution_result):
        """Generate feedback from execution result"""
        if execution_result['success']:
            return {
                'type': 'positive',
                'rating': np.random.uniform(0.8, 1.0),
                'comments': 'Good execution'
            }
        else:
            return {
                'type': 'negative',
                'rating': np.random.uniform(0.0, 0.3),
                'comments': 'Poor execution'
            }

class SystemIntegrationTester:
    """Comprehensive system integration tester"""
    def __init__(self):
        self.test_suites = {
            'functional': FunctionalIntegrationSuite(),
            'safety': SafetyIntegrationSuite(),
            'performance': PerformanceIntegrationSuite(),
            'usability': UsabilityIntegrationSuite()
        }
        self.test_results = {}
        self.test_history = deque(maxlen=100)

    def run_comprehensive_test(self, test_scenario):
        """Run comprehensive integration test"""
        test_result = {
            'scenario': test_scenario,
            'timestamp': time.time(),
            'components_tested': [],
            'functional_tests_passed': 0,
            'functional_tests_total': 0,
            'safety_tests_passed': 0,
            'safety_tests_total': 0,
            'performance_metrics': {},
            'issues_identified': [],
            'recommendations': []
        }

        # Run each test suite
        for suite_name, suite in self.test_suites.items():
            suite_result = suite.run_tests(test_scenario)

            if suite_name == 'functional':
                test_result['functional_tests_passed'] = suite_result.get('passed', 0)
                test_result['functional_tests_total'] = suite_result.get('total', 0)
            elif suite_name == 'safety':
                test_result['safety_tests_passed'] = suite_result.get('passed', 0)
                test_result['safety_tests_total'] = suite_result.get('total', 0)
            elif suite_name == 'performance':
                test_result['performance_metrics'] = suite_result.get('metrics', {})
            elif suite_name == 'usability':
                test_result['usability_score'] = suite_result.get('score', 0.0)

            test_result['components_tested'].extend(suite_result.get('tested_components', []))
            test_result['issues_identified'].extend(suite_result.get('issues', []))
            test_result['recommendations'].extend(suite_result.get('recommendations', []))

        # Calculate overall success rate
        total_functional = test_result['functional_tests_total']
        total_safety = test_result['safety_tests_total']

        functional_success_rate = (
            test_result['functional_tests_passed'] / total_functional if total_functional > 0 else 0
        )
        safety_success_rate = (
            test_result['safety_tests_passed'] / total_safety if total_safety > 0 else 0
        )

        test_result['overall_success_rate'] = (
            0.6 * functional_success_rate + 0.4 * safety_success_rate
        )

        # Determine test outcome
        test_result['success'] = (
            functional_success_rate >= 0.8 and  # 80% functional tests passed
            safety_success_rate >= 0.95 and     # 95% safety tests passed
            test_result['overall_success_rate'] >= 0.85  # 85% overall success
        )

        # Store result
        self.test_results[test_scenario] = test_result
        self.test_history.append(test_result)

        return test_result

    def generate_test_report(self, scenario=None):
        """Generate comprehensive test report"""
        if scenario and scenario in self.test_results:
            results = [self.test_results[scenario]]
        else:
            results = list(self.test_results.values())

        if not results:
            return "No test results available"

        report = []
        report.append("=== PHYSICAL AI SYSTEM INTEGRATION TEST REPORT ===\n")
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"Total test scenarios: {len(results)}\n")

        for result in results:
            report.append(f"\n--- Scenario: {result['scenario']} ---")
            report.append(f"Success: {'PASS' if result['success'] else 'FAIL'}")
            report.append(f"Overall Success Rate: {result['overall_success_rate']:.2%}")
            report.append(f"Functional Tests: {result['functional_tests_passed']}/{result['functional_tests_total']}")
            report.append(f"Safety Tests: {result['safety_tests_passed']}/{result['safety_tests_total']}")

            if 'performance_metrics' in result:
                report.append("Performance Metrics:")
                for metric, value in result['performance_metrics'].items():
                    report.append(f"  {metric}: {value}")

            if result['issues_identified']:
                report.append("Issues Identified:")
                for issue in result['issues_identified']:
                    report.append(f"  - {issue}")

            if result['recommendations']:
                report.append("Recommendations:")
                for rec in result['recommendations']:
                    report.append(f"  - {rec}")

        return "\n".join(report)

    def run_regression_tests(self):
        """Run regression tests to ensure no functionality was broken"""
        # Compare current test results with previous baselines
        current_results = self.test_results
        baseline_results = self.load_test_baselines()

        regressions = []
        for scenario, current_result in current_results.items():
            if scenario in baseline_results:
                baseline_result = baseline_results[scenario]

                if current_result['overall_success_rate'] < baseline_result['overall_success_rate']:
                    regression = {
                        'scenario': scenario,
                        'baseline_rate': baseline_result['overall_success_rate'],
                        'current_rate': current_result['overall_success_rate'],
                        'difference': current_result['overall_success_rate'] - baseline_result['overall_success_rate']
                    }
                    regressions.append(regression)

        return regressions

    def load_test_baselines(self):
        """Load test baselines from storage"""
        # In practice, this would load from file/database
        # For this example, return empty dict
        return {}

class FunctionalIntegrationSuite:
    def run_tests(self, scenario):
        """Run functional integration tests"""
        # This would run tests specific to functionality
        # For this example, return placeholder results
        return {
            'passed': 8,
            'total': 10,
            'tested_components': ['navigation', 'perception', 'control', 'interaction'],
            'issues': ['Minor timing issue in perception-control loop'],
            'recommendations': ['Improve synchronization between perception and control']
        }

class SafetyIntegrationSuite:
    def run_tests(self, scenario):
        """Run safety integration tests"""
        # This would run tests specific to safety
        return {
            'passed': 19,
            'total': 20,
            'tested_components': ['emergency_stop', 'collision_avoidance', 'human_safety'],
            'issues': [],
            'recommendations': ['Continue monitoring safety system performance']
        }

class PerformanceIntegrationSuite:
    def run_tests(self, scenario):
        """Run performance integration tests"""
        # This would run tests specific to performance
        return {
            'metrics': {
                'average_response_time': 0.045,
                'cpu_utilization': 0.65,
                'memory_usage': 0.42,
                'success_rate': 0.92,
                'throughput': 22.3  # operations per second
            },
            'tested_components': ['processing_speed', 'resource_utilization', 'reliability'],
            'issues': [],
            'recommendations': ['Performance within acceptable limits']
        }

class UsabilityIntegrationSuite:
    def run_tests(self, scenario):
        """Run usability integration tests"""
        # This would run tests specific to usability
        return {
            'score': 0.85,  # 0-1 scale
            'tested_components': ['interface_design', 'interaction_naturalness', 'learnability'],
            'issues': ['Some users found voice commands inconsistent'],
            'recommendations': ['Improve voice command recognition and feedback']
        }
```

## Simulation-Based Testing

Testing in simulated environments before real-world deployment:

```python
# Example: Simulation-based testing framework
import gym
from gym import spaces
import numpy as np

class PhysicalAISimulationEnv(gym.Env):
    """Custom gym environment for Physical AI system testing"""
    def __init__(self):
        super(PhysicalAISimulationEnv, self).__init__()

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),  # linear vel, angular vel
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            'robot_state': spaces.Box(
                low=np.array([-10, -10, -np.pi, -5, -5, -10]),  # x, y, theta, vx, vy, omega
                high=np.array([10, 10, np.pi, 5, 5, 10]),
                dtype=np.float32
            ),
            'sensor_data': spaces.Box(
                low=np.zeros(360),  # 360 laser ranges
                high=np.full(360, 10.0),  # max range 10m
                dtype=np.float32
            ),
            'human_proximity': spaces.Box(
                low=np.array([0, 0, 0]),  # x, y, distance
                high=np.array([5, 5, 5]),
                dtype=np.float32
            ),
            'task_progress': spaces.Box(
                low=np.array([0.0]),
                high=np.array([1.0]),
                dtype=np.float32
            )
        })

        # Environment state
        self.robot_position = np.array([0.0, 0.0])
        self.robot_orientation = 0.0
        self.robot_velocity = np.array([0.0, 0.0])
        self.robot_angular_velocity = 0.0

        # Simulation parameters
        self.dt = 0.1  # Time step
        self.max_steps = 1000
        self.current_step = 0

        # Task parameters
        self.goal_position = np.array([5.0, 5.0])
        self.human_position = np.array([2.0, 2.0])

        # Environment obstacles
        self.obstacles = [
            {'position': [3.0, 3.0], 'radius': 0.5},
            {'position': [1.0, 4.0], 'radius': 0.3}
        ]

    def reset(self):
        """Reset environment to initial state"""
        self.robot_position = np.array([0.0, 0.0])
        self.robot_orientation = 0.0
        self.robot_velocity = np.array([0.0, 0.0])
        self.robot_angular_velocity = 0.0
        self.current_step = 0

        # Randomize human position occasionally
        if np.random.random() < 0.3:
            self.human_position = np.random.uniform([0.5, 0.5], [4.5, 4.5])

        return self.get_observation()

    def get_observation(self):
        """Get current observation"""
        # Simulate sensor data
        laser_scan = self.simulate_laser_scan()

        # Calculate human proximity
        human_distance = np.linalg.norm(self.robot_position - self.human_position)
        human_vector = self.human_position - self.robot_position
        human_angle = np.arctan2(human_vector[1], human_vector[0])

        # Calculate task progress
        distance_to_goal = np.linalg.norm(self.robot_position - self.goal_position)
        max_distance = np.linalg.norm(self.goal_position)  # Initial distance
        progress = max(0.0, 1.0 - distance_to_goal / max_distance)

        observation = {
            'robot_state': np.array([
                self.robot_position[0], self.robot_position[1], self.robot_orientation,
                self.robot_velocity[0], self.robot_velocity[1], self.robot_angular_velocity
            ]),
            'sensor_data': laser_scan,
            'human_proximity': np.array([
                self.human_position[0], self.human_position[1], human_distance
            ]),
            'task_progress': np.array([progress])
        }

        return observation

    def simulate_laser_scan(self):
        """Simulate laser scan data"""
        scan_data = np.full(360, 10.0)  # Default max range

        # Simulate obstacles
        for i in range(360):
            angle = np.radians(i)
            ray_direction = np.array([np.cos(angle), np.sin(angle)])

            # Check for obstacles along this ray
            min_distance = 10.0
            for obstacle in self.obstacles:
                distance = self.distance_ray_to_circle(
                    self.robot_position, ray_direction, obstacle['position'], obstacle['radius']
                )
                if distance > 0 and distance < min_distance:
                    min_distance = distance

            # Check for human
            human_distance = self.distance_ray_to_circle(
                self.robot_position, ray_direction, self.human_position, 0.3  # Human radius
            )
            if human_distance > 0 and human_distance < min_distance:
                min_distance = human_distance

            # Check for walls (simulate bounded environment)
            wall_distance = self.distance_to_walls(self.robot_position, angle)
            if wall_distance > 0 and wall_distance < min_distance:
                min_distance = wall_distance

            scan_data[i] = min_distance

        return scan_data

    def distance_ray_to_circle(self, ray_origin, ray_direction, circle_center, circle_radius):
        """Calculate distance from ray to circle"""
        # Vector from ray origin to circle center
        oc = circle_center - ray_origin

        # Quadratic equation coefficients
        a = np.dot(ray_direction, ray_direction)
        b = 2 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - circle_radius**2

        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            return -1  # No intersection
        else:
            sqrt_discriminant = np.sqrt(discriminant)
            t1 = (-b - sqrt_discriminant) / (2*a)
            t2 = (-b + sqrt_discriminant) / (2*a)

            # Return the closest positive intersection
            if t1 > 0:
                return t1
            elif t2 > 0:
                return t2
            else:
                return -1  # Intersection behind ray origin

    def distance_to_walls(self, position, angle):
        """Calculate distance to walls in bounded environment"""
        # Assume 10x10 environment with walls at x=-5, x=5, y=-5, y=5
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        distances = []

        # Wall at x = 5 (right)
        if cos_angle > 0:
            t = (5 - position[0]) / cos_angle
            if t > 0 and abs(position[1] + t*sin_angle) <= 5:
                distances.append(t)

        # Wall at x = -5 (left)
        if cos_angle < 0:
            t = (-5 - position[0]) / cos_angle
            if t > 0 and abs(position[1] + t*sin_angle) <= 5:
                distances.append(t)

        # Wall at y = 5 (top)
        if sin_angle > 0:
            t = (5 - position[1]) / sin_angle
            if t > 0 and abs(position[0] + t*cos_angle) <= 5:
                distances.append(t)

        # Wall at y = -5 (bottom)
        if sin_angle < 0:
            t = (-5 - position[1]) / sin_angle
            if t > 0 and abs(position[0] + t*cos_angle) <= 5:
                distances.append(t)

        return min(distances) if distances else 10.0

    def step(self, action):
        """Execute action and return new state"""
        # Unpack action
        linear_vel, angular_vel = action

        # Apply action with dynamics
        self.robot_angular_velocity = angular_vel
        self.robot_orientation += angular_vel * self.dt
        self.robot_velocity[0] = linear_vel * np.cos(self.robot_orientation)
        self.robot_velocity[1] = linear_vel * np.sin(self.robot_orientation)

        # Update position
        self.robot_position += self.robot_velocity * self.dt

        # Apply constraints (keep in bounds)
        self.robot_position = np.clip(self.robot_position, [-4.8, -4.8], [4.8, 4.8])

        # Calculate reward
        reward = self.calculate_reward()

        # Check termination conditions
        done = self.is_terminal()

        # Increment step counter
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        # Get new observation
        observation = self.get_observation()

        # Info dictionary
        info = {
            'distance_to_goal': np.linalg.norm(self.robot_position - self.goal_position),
            'distance_to_human': np.linalg.norm(self.robot_position - self.human_position),
            'collision_with_obstacle': self.check_collision_with_obstacles(),
            'human_safety_violation': self.check_human_safety_violation()
        }

        return observation, reward, done, info

    def calculate_reward(self):
        """Calculate reward based on current state"""
        reward = 0.0

        # Distance to goal reward (negative because closer is better)
        distance_to_goal = np.linalg.norm(self.robot_position - self.goal_position)
        reward -= distance_to_goal * 0.1  # Encourage moving toward goal

        # Goal reached bonus
        if distance_to_goal < 0.5:
            reward += 10.0

        # Human safety penalty
        distance_to_human = np.linalg.norm(self.robot_position - self.human_position)
        if distance_to_human < 0.5:  # Too close to human
            reward -= 5.0
        elif distance_to_human < 1.0:  # Close to human
            reward -= 1.0

        # Collision penalty
        if self.check_collision_with_obstacles():
            reward -= 10.0

        # Smooth movement reward
        speed = np.linalg.norm(self.robot_velocity)
        if speed < 0.8:  # Encourage reasonable speeds
            reward += 0.1

        return reward

    def is_terminal(self):
        """Check if episode is terminal"""
        distance_to_goal = np.linalg.norm(self.robot_position - self.goal_position)
        return distance_to_goal < 0.5  # Goal reached

    def check_collision_with_obstacles(self):
        """Check if robot collides with obstacles"""
        robot_radius = 0.3  # Robot collision radius

        for obstacle in self.obstacles:
            distance = np.linalg.norm(self.robot_position - obstacle['position'])
            if distance < (robot_radius + obstacle['radius']):
                return True

        return False

    def check_human_safety_violation(self):
        """Check if robot violates human safety zone"""
        safety_distance = 0.8  # Minimum safe distance from human
        distance_to_human = np.linalg.norm(self.robot_position - self.human_position)
        return distance_to_human < safety_distance

class SimulationTestRunner:
    """Runner for simulation-based tests"""
    def __init__(self, environment_class, num_episodes=100):
        self.env = environment_class()
        self.num_episodes = num_episodes
        self.results = {
            'success_rate': 0.0,
            'average_steps': 0.0,
            'average_reward': 0.0,
            'collision_rate': 0.0,
            'safety_violations': 0.0,
            'detailed_results': []
        }

    def run_tests(self, policy_function):
        """Run tests with given policy"""
        successful_episodes = 0
        total_steps = 0
        total_reward = 0.0
        total_collisions = 0
        total_safety_violations = 0

        for episode in range(self.num_episodes):
            obs = self.env.reset()
            episode_steps = 0
            episode_reward = 0.0
            episode_collisions = 0
            episode_safety_violations = 0

            done = False
            while not done:
                # Get action from policy
                action = policy_function(obs)

                # Take step
                obs, reward, done, info = self.env.step(action)

                # Track metrics
                episode_steps += 1
                episode_reward += reward

                if info.get('collision_with_obstacle', False):
                    episode_collisions += 1

                if info.get('human_safety_violation', False):
                    episode_safety_violations += 1

            # Update episode results
            if info.get('distance_to_goal', float('inf')) < 0.5:
                successful_episodes += 1

            total_steps += episode_steps
            total_reward += episode_reward
            total_collisions += episode_collisions
            total_safety_violations += episode_safety_violations

            # Store detailed result
            self.results['detailed_results'].append({
                'episode': episode,
                'success': info.get('distance_to_goal', float('inf')) < 0.5,
                'steps': episode_steps,
                'reward': episode_reward,
                'collisions': episode_collisions,
                'safety_violations': episode_safety_violations
            })

        # Calculate aggregate metrics
        self.results['success_rate'] = successful_episodes / self.num_episodes
        self.results['average_steps'] = total_steps / self.num_episodes
        self.results['average_reward'] = total_reward / self.num_episodes
        self.results['collision_rate'] = total_collisions / self.num_episodes
        self.results['safety_violations'] = total_safety_violations / self.num_episodes

        return self.results

    def run_comprehensive_simulation_test(self, ai_system):
        """Run comprehensive test of AI system in simulation"""
        # Wrap the AI system in a policy function
        def ai_policy(observation):
            # Convert observation to AI system input format
            ai_input = self.format_observation_for_ai(observation)

            # Get action from AI system
            action = ai_system.get_action(ai_input)

            # Convert to simulation format
            simulation_action = self.format_action_for_simulation(action)

            return simulation_action

        # Run tests
        results = self.run_tests(ai_policy)

        # Generate detailed report
        report = self.generate_simulation_report(results)

        return results, report

    def format_observation_for_ai(self, observation):
        """Format simulation observation for AI system"""
        # Convert from gym format to AI system format
        ai_input = {
            'position': observation['robot_state'][:2],
            'orientation': observation['robot_state'][2],
            'velocity': observation['robot_state'][3:5],
            'angular_velocity': observation['robot_state'][5],
            'laser_scan': observation['sensor_data'],
            'human_proximity': {
                'position': observation['human_proximity'][:2],
                'distance': observation['human_proximity'][2]
            },
            'task_progress': observation['task_progress'][0]
        }

        return ai_input

    def format_action_for_simulation(self, action):
        """Format AI system action for simulation"""
        # Convert from AI system format to gym format
        if isinstance(action, dict):
            return np.array([action.get('linear_velocity', 0.0), action.get('angular_velocity', 0.0)])
        elif isinstance(action, (list, tuple)):
            return np.array(action[:2])  # Take first two elements (linear, angular)
        else:
            # Assume it's already in correct format
            return np.array([0.0, 0.0])  # Default action

    def generate_simulation_report(self, results):
        """Generate detailed simulation test report"""
        report = []
        report.append("=== SIMULATION TEST RESULTS ===")
        report.append(f"Episodes run: {self.num_episodes}")
        report.append(f"Success rate: {results['success_rate']:.2%}")
        report.append(f"Average steps per episode: {results['average_steps']:.1f}")
        report.append(f"Average reward: {results['average_reward']:.2f}")
        report.append(f"Collision rate: {results['collision_rate']:.2%}")
        report.append(f"Safety violation rate: {results['safety_violations']:.2%}")

        # Success analysis
        if results['success_rate'] >= 0.8:
            report.append("\n✓ System demonstrates good navigation capability")
        else:
            report.append(f"\n⚠ Success rate below threshold (target: 80%, achieved: {results['success_rate']:.2%})")

        # Safety analysis
        if results['safety_violations'] < 0.05:  # Less than 5% safety violations
            report.append("✓ System maintains good safety standards")
        else:
            report.append(f"⚠ Safety violations detected (rate: {results['safety_violations']:.2%})")

        # Performance analysis
        if results['collision_rate'] < 0.1:  # Less than 10% collision rate
            report.append("✓ System demonstrates good obstacle avoidance")
        else:
            report.append(f"⚠ Collision rate elevated (rate: {results['collision_rate']:.2%})")

        return "\n".join(report)

class HardwareInLoopTester:
    """Test system components with actual hardware when possible"""
    def __init__(self):
        self.hardware_components = {}
        self.simulation_components = {}
        self.hybrid_test_scenarios = []

    def add_hardware_component(self, name, component_interface):
        """Add actual hardware component for testing"""
        self.hardware_components[name] = component_interface

    def add_simulation_component(self, name, simulation_model):
        """Add simulation model for component"""
        self.simulation_components[name] = simulation_model

    def run_hardware_in_loop_test(self, test_scenario):
        """Run test with mix of hardware and simulation"""
        test_result = {
            'scenario': test_scenario,
            'components_used': [],
            'results': {},
            'success': True,
            'issues': [],
            'recommendations': []
        }

        # Determine which components to use in hardware vs simulation
        for component_name, component_config in test_scenario.get('components', {}).items():
            if component_config.get('use_hardware', False) and component_name in self.hardware_components:
                # Use actual hardware
                component = self.hardware_components[component_name]
                test_result['components_used'].append(f"{component_name}_hardware")
            else:
                # Use simulation
                component = self.simulation_components.get(component_name)
                test_result['components_used'].append(f"{component_name}_simulation")

            # Run component-specific tests
            if component:
                try:
                    component_result = component.test_functionality()
                    test_result['results'][component_name] = component_result

                    if not component_result.get('success', True):
                        test_result['success'] = False
                        test_result['issues'].append(f"{component_name}: {component_result.get('error', 'Unknown error')}")
                except Exception as e:
                    test_result['success'] = False
                    test_result['issues'].append(f"{component_name}: Exception - {str(e)}")
            else:
                test_result['success'] = False
                test_result['issues'].append(f"{component_name}: Component not available")

        return test_result

    def create_hybrid_test_scenario(self, name, description, components):
        """Create hybrid test scenario mixing hardware and simulation"""
        scenario = {
            'name': name,
            'description': description,
            'components': components,
            'expected_outcomes': [],
            'safety_constraints': [],
            'success_criteria': []
        }

        self.hybrid_test_scenarios.append(scenario)
        return scenario

# Example usage of the testing framework
def main():
    # Example: Run unit tests
    print("Running unit tests...")
    unit_test_success = run_unit_tests()
    print(f"Unit tests: {'PASSED' if unit_test_success else 'FAILED'}\n")

    # Example: Run integration tests
    print("Running integration tests...")
    integration_tester = IntegrationTestSuite()
    integration_results = integration_tester.run_all_integration_tests()

    print("Integration test results:")
    for test_name, result in integration_results.items():
        status = "PASS" if result['success'] else "FAIL"
        print(f"  {test_name}: {status}")
    print()

    # Example: Run simulation tests
    print("Running simulation tests...")
    sim_runner = SimulationTestRunner(PhysicalAISimulationEnv, num_episodes=50)

    # Simple policy for testing (go toward goal)
    def simple_navigation_policy(obs):
        robot_pos = obs['robot_state'][:2]
        goal_pos = np.array([5.0, 5.0])

        direction = goal_pos - robot_pos
        distance = np.linalg.norm(direction)

        if distance > 0:
            direction_normalized = direction / distance
        else:
            direction_normalized = np.array([0.0, 1.0])

        # Simple proportional controller
        linear_vel = min(0.5, distance * 0.2)
        angular_vel = np.arctan2(direction_normalized[1], direction_normalized[0]) * 0.5

        return np.array([linear_vel, angular_vel])

    sim_results, sim_report = sim_runner.run_comprehensive_simulation_test(lambda obs: simple_navigation_policy(obs))
    print(sim_report)
    print()

    # Example: System integration test
    print("Running system integration test...")
    system_tester = SystemIntegrationTester()
    system_result = system_tester.run_comprehensive_test("navigation_with_interaction")

    print(f"System test: {'PASSED' if system_result['success'] else 'FAILED'}")
    print(f"Overall success rate: {system_result['overall_success_rate']:.2%}")
    print(f"Functional tests: {system_result['functional_tests_passed']}/{system_result['functional_tests_total']}")
    print(f"Safety tests: {system_result['safety_tests_passed']}/{system_result['safety_tests_total']}")

    # Generate and print full report
    full_report = system_tester.generate_test_report()
    print("\nDetailed Report:")
    print(full_report)

if __name__ == '__main__':
    main()
```

## Formal Verification for Safety-Critical Systems

For safety-critical applications, formal verification methods can be used:

```python
# Example: Formal verification for safety properties
class FormalVerificationSystem:
    def __init__(self):
        self.properties = []
        self.models = {}
        self.verifier = self.initialize_verifier()

    def initialize_verifier(self):
        """Initialize formal verification tools"""
        # This would interface with actual verification tools like NuSMV, SPIN, etc.
        # For this example, we'll create a mock verifier
        return MockVerifier()

    def add_safety_property(self, property_name, formal_specification):
        """Add safety property to verify"""
        property_def = {
            'name': property_name,
            'specification': formal_specification,
            'type': 'safety',
            'verified': False,
            'verification_result': None
        }
        self.properties.append(property_def)

    def add_liveness_property(self, property_name, formal_specification):
        """Add liveness property to verify"""
        property_def = {
            'name': property_name,
            'specification': formal_specification,
            'type': 'liveness',
            'verified': False,
            'verification_result': None
        }
        self.properties.append(property_def)

    def verify_system(self, system_model):
        """Verify system against all properties"""
        verification_results = []

        for prop in self.properties:
            result = self.verifier.verify_property(prop, system_model)
            prop['verified'] = result['verified']
            prop['verification_result'] = result
            verification_results.append(result)

        return verification_results

    def check_collision_avoidance_property(self):
        """Check that robot never collides with obstacles"""
        # Property: G(!collision_occurs)
        # (Always, it's not the case that collision occurs)
        return {
            'property': 'always_no_collision',
            'logic': 'G(!collision)',
            'description': 'Robot should never collide with obstacles',
            'verification_method': 'model_checking'
        }

    def check_human_safety_property(self):
        """Check that robot maintains safe distance from humans"""
        # Property: G(human_present -> distance_to_human > safe_distance)
        return {
            'property': 'human_safety_distance',
            'logic': 'G(human_present -> distance > safe_distance)',
            'description': 'Robot maintains safe distance from humans',
            'verification_method': 'model_checking'
        }

    def check_system_stability_property(self):
        """Check that system remains stable"""
        # Property: G(stability_maintained)
        return {
            'property': 'system_stability',
            'logic': 'G(stability)',
            'description': 'System maintains stability at all times',
            'verification_method': 'theorem_proving'
        }

class MockVerifier:
    """Mock verifier for demonstration purposes"""
    def verify_property(self, property_def, system_model):
        """Mock verification of property"""
        # In a real system, this would use formal verification tools
        # For this example, return a mock result

        import random

        # Simulate verification process
        verification_time = random.uniform(0.1, 2.0)  # Simulated verification time

        # 80% success rate for demonstration
        verified = random.random() < 0.8

        return {
            'property_name': property_def['name'],
            'verified': verified,
            'verification_time': verification_time,
            'method_used': 'model_checking',  # or theorem_proving, etc.
            'counterexample': None if verified else self.generate_counterexample(property_def),
            'confidence': 0.9 if verified else 0.1
        }

    def generate_counterexample(self, property_def):
        """Generate counterexample for failed property"""
        return {
            'state_sequence': ['initial_state', 'intermediate_state', 'failure_state'],
            'trace': 'Execution trace leading to property violation',
            'cause': 'Specific condition that caused violation'
        }

class PropertyBasedTester:
    """Property-based testing for Physical AI systems"""
    def __init__(self):
        self.properties = {}
        self.test_generators = {}

    def define_property(self, name, property_function, test_generator=None):
        """Define a property to test"""
        self.properties[name] = property_function
        if test_generator:
            self.test_generators[name] = test_generator

    def test_property(self, property_name, num_tests=100):
        """Test a property with generated inputs"""
        if property_name not in self.properties:
            raise ValueError(f"Property {property_name} not defined")

        property_func = self.properties[property_name]
        test_gen = self.test_generators.get(property_name, self.default_test_generator)

        failures = []
        for i in range(num_tests):
            test_input = test_gen()
            try:
                result = property_func(test_input)
                if not result:
                    failures.append({
                        'test_case': test_input,
                        'failure_index': i
                    })
            except Exception as e:
                failures.append({
                    'test_case': test_input,
                    'exception': str(e),
                    'failure_index': i
                })

        success_rate = (num_tests - len(failures)) / num_tests
        return {
            'property': property_name,
            'success_rate': success_rate,
            'failures': failures,
            'total_tests': num_tests
        }

    def default_test_generator(self):
        """Default test input generator"""
        # Generate random test inputs
        return {
            'robot_state': np.random.uniform(-5, 5, 6),  # position, orientation, velocity
            'environment_state': {
                'obstacles': np.random.uniform(0, 10, (5, 3)),  # 5 obstacles with x,y,radius
                'humans': np.random.uniform(0, 10, (2, 3))     # 2 humans with x,y,radius
            },
            'control_inputs': np.random.uniform(-1, 1, 2)  # linear, angular velocity
        }

    def define_safety_property(self):
        """Define safety property for collision avoidance"""
        def collision_avoidance_property(test_input):
            """Property: robot should not collide with obstacles"""
            robot_pos = test_input['robot_state'][:2]
            obstacles = test_input['environment_state']['obstacles']
            robot_radius = 0.3

            for obs in obstacles:
                obs_pos = obs[:2]
                obs_radius = obs[2]
                distance = np.linalg.norm(robot_pos - obs_pos)
                if distance < (robot_radius + obs_radius):
                    return False  # Collision detected
            return True  # No collision

        self.define_property('collision_avoidance', collision_avoidance_property)

    def define_stability_property(self):
        """Define stability property"""
        def stability_property(test_input):
            """Property: robot should maintain balance"""
            robot_orientation = test_input['robot_state'][2]  # orientation angle
            max_tilt = np.pi / 4  # 45 degrees max tilt
            return abs(robot_orientation) <= max_tilt

        self.define_property('stability', stability_property)

    def run_all_property_tests(self):
        """Run all defined property tests"""
        results = {}
        for prop_name in self.properties:
            results[prop_name] = self.test_property(prop_name)
        return results

def run_comprehensive_validation():
    """Run comprehensive validation of Physical AI system"""
    print("=== COMPREHENSIVE VALIDATION OF PHYSICAL AI SYSTEM ===\n")

    # 1. Unit Testing
    print("1. UNIT TESTING")
    unit_success = run_unit_tests()
    print(f"   Result: {'PASS' if unit_success else 'FAIL'}\n")

    # 2. Integration Testing
    print("2. INTEGRATION TESTING")
    integration_tester = IntegrationTestSuite()
    integration_results = integration_tester.run_all_integration_tests()

    integration_success = all(result['success'] for result in integration_results.values())
    print(f"   Result: {'PASS' if integration_success else 'FAIL'}")
    for test_name, result in integration_results.items():
        status = "PASS" if result['success'] else "FAIL"
        print(f"     {test_name}: {status}")
    print()

    # 3. Simulation Testing
    print("3. SIMULATION TESTING")
    sim_runner = SimulationTestRunner(PhysicalAISimulationEnv, num_episodes=100)

    def test_policy(obs):
        # Simple policy for testing
        robot_pos = obs['robot_state'][:2]
        goal_pos = np.array([5.0, 5.0])

        direction = goal_pos - robot_pos
        distance = np.linalg.norm(direction)

        if distance > 0:
            direction_normalized = direction / distance
        else:
            direction_normalized = np.array([0.0, 1.0])

        linear_vel = min(0.5, distance * 0.2)
        angular_vel = np.arctan2(direction_normalized[1], direction_normalized[0]) * 0.5

        return np.array([linear_vel, angular_vel])

    sim_results, sim_report = sim_runner.run_comprehensive_simulation_test(test_policy)
    print(sim_report)
    print()

    # 4. Property-Based Testing
    print("4. PROPERTY-BASED TESTING")
    prop_tester = PropertyBasedTester()
    prop_tester.define_safety_property()
    prop_tester.define_stability_property()

    property_results = prop_tester.run_all_property_tests()

    all_properties_pass = all(result['success_rate'] >= 0.95 for result in property_results.values())
    print(f"   Result: {'PASS' if all_properties_pass else 'FAIL'}")
    for prop_name, result in property_results.items():
        status = "PASS" if result['success_rate'] >= 0.95 else "FAIL"
        print(f"     {prop_name}: {status} ({result['success_rate']:.2%} success)")
    print()

    # 5. System Integration Testing
    print("5. SYSTEM INTEGRATION TESTING")
    system_tester = SystemIntegrationTester()
    system_result = system_tester.run_comprehensive_test("full_system_integration")

    system_success = system_result['success']
    print(f"   Result: {'PASS' if system_success else 'FAIL'}")
    print(f"   Success Rate: {system_result['overall_success_rate']:.2%}")
    print(f"   Functional: {system_result['functional_tests_passed']}/{system_result['functional_tests_total']}")
    print(f"   Safety: {system_result['safety_tests_passed']}/{system_result['safety_tests_total']}")
    print()

    # Overall assessment
    overall_success = all([
        unit_success,
        integration_success,
        sim_results['success_rate'] >= 0.8,  # 80% success in simulation
        all_properties_pass,
        system_success
    ])

    print("=== OVERALL VALIDATION RESULT ===")
    print(f"System Validation: {'PASS' if overall_success else 'FAIL'}")

    if overall_success:
        print("✓ All validation tests passed!")
        print("✓ System is ready for further development/testing")
    else:
        print("⚠ Some validation tests failed")
        print("⚠ System requires additional work before deployment")

    return overall_success

if __name__ == '__main__':
    run_comprehensive_validation()
```

## Exercise: Design Your Own Validation Framework

Consider the following validation design challenge:

1. What specific validation requirements does your Physical AI system have?
2. What safety-critical properties must be verified?
3. How will you balance thoroughness with practical testing constraints?
4. What simulation environments will best represent real-world conditions?
5. How will you handle edge cases and rare scenarios?
6. What metrics will you use to measure validation success?
7. How will you ensure your validation covers all operational scenarios?

## Summary

Validation and testing of Physical AI systems requires a comprehensive, multi-layered approach that addresses the unique challenges of embodied systems:

- **Unit Testing**: Individual component validation with physical constraints
- **Integration Testing**: System-level validation of component interactions
- **Simulation Testing**: Controlled environment testing of complex scenarios
- **Property-Based Testing**: Systematic verification of safety and performance properties
- **Formal Verification**: Mathematical proof of critical safety properties
- **Hardware-in-Loop Testing**: Validation with actual hardware components
- **Regression Testing**: Ensuring new changes don't break existing functionality

The validation process must be iterative and continuous, evolving as the system develops. Safety-critical properties must be verified through multiple approaches, and the system must be tested under a wide range of conditions to ensure robust operation.

In the next lesson, we'll explore how to deploy these validated systems in real-world environments and handle the challenges of field operation.