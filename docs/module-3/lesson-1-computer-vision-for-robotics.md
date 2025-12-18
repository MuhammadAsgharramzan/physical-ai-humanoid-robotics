---
sidebar_position: 1
---

# Computer Vision for Robotics

## Introduction

Computer vision is a critical component of Physical AI systems, enabling robots to perceive and understand their environment. Unlike traditional computer vision applications, robotic vision must operate in real-time, handle dynamic environments, and integrate tightly with motor control systems. This lesson explores computer vision techniques specifically designed for robotics applications.

## Fundamentals of Robotic Computer Vision

### Key Differences from Traditional Computer Vision

Robotic computer vision differs from traditional applications in several important ways:

1. **Real-time Requirements**: Robots must process visual information continuously and respond in real-time
2. **Embodied Perception**: Vision is integrated with the robot's physical movement and actions
3. **Dynamic Environments**: The robot must handle constantly changing scenes
4. **Action-Oriented**: Vision serves specific robotic tasks rather than just recognition

### Visual Processing Pipeline

```python
# Example: Robotic computer vision pipeline
class RoboticVisionPipeline:
    def __init__(self):
        self.image_preprocessor = ImagePreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.object_detector = ObjectDetector()
        self.scene_analyzer = SceneAnalyzer()
        self.action_generator = ActionGenerator()

    def process_frame(self, image):
        """Complete vision processing pipeline for one frame"""
        # 1. Preprocess image
        preprocessed = self.image_preprocessor.process(image)

        # 2. Extract features
        features = self.feature_extractor.extract(preprocessed)

        # 3. Detect objects
        objects = self.object_detector.detect(features)

        # 4. Analyze scene
        scene_info = self.scene_analyzer.analyze(objects, preprocessed)

        # 5. Generate robot actions
        actions = self.action_generator.generate(scene_info)

        return {
            'objects': objects,
            'scene_info': scene_info,
            'actions': actions,
            'features': features
        }
```

## Image Preprocessing for Robotics

### Camera Calibration and Rectification

Robots need accurate camera calibration to convert pixel coordinates to real-world measurements:

```python
# Example: Camera calibration and rectification
import numpy as np
import cv2

class CameraCalibrator:
    def __init__(self):
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.rotation_vector = None
        self.translation_vector = None

    def calibrate_camera(self, object_points, image_points, image_size):
        """Calibrate camera using chessboard pattern"""
        self.camera_matrix, self.distortion_coeffs, self.rotation_vector, self.translation_vector = \
            cv2.calibrateCamera(
                object_points, image_points, image_size, None, None
            )
        return self.camera_matrix, self.distortion_coeffs

    def undistort_image(self, image):
        """Remove lens distortion from image"""
        if self.camera_matrix is not None and self.distortion_coeffs is not None:
            return cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)
        return image

    def convert_pixel_to_world(self, pixel_coords, depth):
        """Convert pixel coordinates to world coordinates"""
        if self.camera_matrix is not None:
            # Convert to normalized coordinates
            normalized = np.linalg.inv(self.camera_matrix).dot(
                np.array([pixel_coords[0], pixel_coords[1], 1])
            )

            # Scale by depth
            world_coords = normalized * depth
            return world_coords
        return None
```

### Image Enhancement for Robotic Applications

Robots often operate in challenging lighting conditions:

```python
# Example: Image enhancement for robotics
class ImageEnhancer:
    def __init__(self):
        self.gamma = 1.0
        self.contrast = 1.0
        self.brightness = 0

    def enhance_for_robot_vision(self, image):
        """Enhance image for better robotic perception"""
        # Adjust gamma for lighting conditions
        enhanced = self.adjust_gamma(image, self.gamma)

        # Enhance contrast
        enhanced = self.enhance_contrast(enhanced, self.contrast)

        # Handle brightness variations
        enhanced = self.adjust_brightness(enhanced, self.brightness)

        # Noise reduction for cleaner processing
        enhanced = self.reduce_noise(enhanced)

        return enhanced

    def adjust_gamma(self, image, gamma):
        """Adjust image gamma"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def enhance_contrast(self, image, contrast):
        """Enhance image contrast"""
        return cv2.convertScaleAbs(image, alpha=contrast, beta=0)

    def adjust_brightness(self, image, brightness):
        """Adjust image brightness"""
        return cv2.convertScaleAbs(image, alpha=1, beta=brightness)

    def reduce_noise(self, image):
        """Reduce noise while preserving edges"""
        return cv2.bilateralFilter(image, 9, 75, 75)
```

## Feature Detection and Extraction

### Key Feature Detection Algorithms

Robots need robust feature detection to identify landmarks and objects:

```python
# Example: Feature detection for robotic applications
class FeatureDetector:
    def __init__(self):
        self.detector_type = 'orb'  # Can be 'orb', 'sift', 'surf', 'akaze'
        self.detector = self.initialize_detector()

    def initialize_detector(self):
        """Initialize the appropriate feature detector"""
        if self.detector_type == 'orb':
            return cv2.ORB_create(nfeatures=500)
        elif self.detector_type == 'sift':
            return cv2.SIFT_create()
        elif self.detector_type == 'akaze':
            return cv2.AKAZE_create()
        else:
            return cv2.ORB_create()

    def detect_features(self, image):
        """Detect features in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, descriptors1, descriptors2):
        """Match features between two images"""
        # Use FLANN matcher for efficiency
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        return good_matches
```

### Visual Odometry and SLAM Features

For navigation, robots need to track their position using visual features:

```python
# Example: Visual odometry feature tracking
class VisualOdometry:
    def __init__(self):
        self.prev_frame = None
        self.prev_keypoints = None
        self.current_position = np.array([0, 0, 0])  # x, y, theta
        self.feature_detector = FeatureDetector()

    def process_frame(self, current_frame):
        """Process a frame for visual odometry"""
        if self.prev_frame is None:
            # Initialize with first frame
            self.prev_frame = current_frame
            self.prev_keypoints, _ = self.feature_detector.detect_features(current_frame)
            return self.current_position

        # Detect features in current frame
        curr_keypoints, curr_descriptors = self.feature_detector.detect_features(current_frame)

        # Match features between frames
        matches = self.feature_detector.match_features(
            self.get_descriptors_for_keypoints(self.prev_keypoints, self.prev_frame),
            curr_descriptors
        )

        # Estimate motion from feature correspondences
        if len(matches) >= 10:  # Need minimum matches for reliable estimation
            prev_matched = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            curr_matched = np.float32([curr_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate transformation
            transformation, mask = cv2.estimateAffinePartial2D(prev_matched, curr_matched)

            if transformation is not None:
                # Update position based on transformation
                self.update_position(transformation)

        # Update for next iteration
        self.prev_frame = current_frame
        self.prev_keypoints = curr_keypoints

        return self.current_position

    def get_descriptors_for_keypoints(self, keypoints, frame):
        """Get descriptors for specific keypoints"""
        # This is a simplified version - in practice, you'd recompute or store descriptors
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        _, descriptors = self.feature_detector.detector.detectAndCompute(gray, None)
        return descriptors

    def update_position(self, transformation):
        """Update robot position based on visual transformation"""
        # Extract translation and rotation from transformation matrix
        dx = transformation[0, 2]
        dy = transformation[1, 2]
        dtheta = np.arctan2(transformation[1, 0], transformation[0, 0])

        # Update position (simplified odometry)
        self.current_position[0] += dx
        self.current_position[1] += dy
        self.current_position[2] += dtheta
```

## Object Detection for Robotics

### Real-time Object Detection

Robots need to detect objects quickly and accurately:

```python
# Example: Real-time object detection for robotics
import time

class RealTimeObjectDetector:
    def __init__(self):
        self.detection_model = self.load_detection_model()
        self.object_classes = self.get_object_classes()
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4

    def load_detection_model(self):
        """Load a pre-trained object detection model"""
        # For this example, we'll use OpenCV's DNN module with a pre-trained model
        # In practice, you might use YOLO, SSD, or other models
        try:
            # Try to load a pre-trained model (simplified)
            return cv2.dnn.readNetFromDarknet("yolo_config.cfg", "yolo_weights.weights")
        except:
            # Fallback to a simpler approach
            return None

    def detect_objects(self, image):
        """Detect objects in the image"""
        start_time = time.time()

        height, width = image.shape[:2]

        # Create blob from image
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

        if self.detection_model is not None:
            # Set blob as input to the network
            self.detection_model.setInput(blob)

            # Run forward pass
            layer_names = self.detection_model.getLayerNames()
            output_names = [layer_names[i[0] - 1] for i in self.detection_model.getUnconnectedOutLayers()]
            outputs = self.detection_model.forward(output_names)

            # Process outputs
            boxes, confidences, class_ids = self.process_detections(outputs, width, height)

            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

            # Format results
            detections = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    detections.append({
                        'class': self.object_classes[class_ids[i]],
                        'confidence': confidences[i],
                        'bbox': [x, y, x + w, y + h],
                        'center': [x + w/2, y + h/2]
                    })
        else:
            # Fallback detection using traditional methods
            detections = self.traditional_detection_fallback(image)

        processing_time = time.time() - start_time
        return detections, processing_time

    def process_detections(self, outputs, width, height):
        """Process the outputs from the detection network"""
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence_threshold:
                    # Convert to actual coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        return boxes, confidences, class_ids

    def traditional_detection_fallback(self, image):
        """Fallback detection using traditional computer vision"""
        # This is a simplified example - in practice, you might use Haar cascades,
        # HOG descriptors, or other traditional methods
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect edges
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    'class': 'object',
                    'confidence': 0.6,  # Default confidence for fallback
                    'bbox': [x, y, x + w, y + h],
                    'center': [x + w/2, y + h/2]
                })

        return detections

    def get_object_classes(self):
        """Get the list of object classes the model can detect"""
        # Common COCO dataset classes
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
```

## ROS2 Implementation: Computer Vision for Robotics

Here's a complete ROS2 implementation of computer vision for robotics:

```python
# computer_vision_robotics.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, Pose
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class ComputerVisionRobotics(Node):
    def __init__(self):
        super().__init__('computer_vision_robotics')

        # Publishers
        self.detection_pub = self.create_publisher(String, '/object_detections', 10)
        self.feature_pub = self.create_publisher(String, '/feature_matches', 10)
        self.pose_pub = self.create_publisher(Pose, '/camera_pose', 10)
        self.processing_time_pub = self.create_publisher(Float32, '/vision_processing_time', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_callback, 10
        )

        # Vision components
        self.cv_bridge = CvBridge()
        self.image_enhancer = ImageEnhancer()
        self.feature_detector = FeatureDetector()
        self.object_detector = RealTimeObjectDetector()
        self.visual_odometry = VisualOdometry()

        # Camera calibration
        self.camera_matrix = None
        self.distortion_coeffs = None

        # Processing state
        self.last_image_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

        # Processing parameters
        self.processing_enabled = True
        self.detection_frequency = 0.5  # Process detection every 0.5 seconds

    def camera_info_callback(self, msg):
        """Handle camera info for calibration"""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.distortion_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        """Handle incoming camera images"""
        current_time = time.time()
        self.frame_count += 1

        # Calculate FPS
        if current_time - self.last_image_time > 1.0:
            self.fps = self.frame_count / (current_time - self.last_image_time)
            self.frame_count = 0
            self.last_image_time = current_time

        if not self.processing_enabled:
            return

        try:
            # Convert ROS image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')
            return

        # 1. IMAGE ENHANCEMENT
        enhanced_image = self.image_enhancer.enhance_for_robot_vision(cv_image)

        # 2. FEATURE DETECTION AND TRACKING
        if self.frame_count % 5 == 0:  # Process features every 5 frames to save computation
            features_start_time = time.time()
            keypoints, descriptors = self.feature_detector.detect_features(enhanced_image)

            # Publish feature information
            feature_info = f"Features detected: {len(keypoints) if keypoints else 0}"
            self.feature_pub.publish(String(data=feature_info))

            features_time = time.time() - features_start_time

        # 3. OBJECT DETECTION (less frequent to save computation)
        if current_time % self.detection_frequency < 0.05:  # Process every 0.5 seconds
            detection_start_time = time.time()
            detections, detection_time = self.object_detector.detect_objects(enhanced_image)

            # Publish detection results
            detection_results = []
            for detection in detections:
                detection_results.append(
                    f"{detection['class']}: {detection['confidence']:.2f} at {detection['center']}"
                )

            if detection_results:
                self.detection_pub.publish(String(data=" | ".join(detection_results)))
            else:
                self.detection_pub.publish(String(data="No objects detected"))

            total_detection_time = time.time() - detection_start_time
        else:
            total_detection_time = 0

        # 4. VISUAL ODOMETRY
        position = self.visual_odometry.process_frame(enhanced_image)

        # Publish camera pose
        pose_msg = Pose()
        pose_msg.position.x = float(position[0])
        pose_msg.position.y = float(position[1])
        pose_msg.position.z = 0.0  # Assume ground level
        # Simple orientation representation
        pose_msg.orientation.z = float(position[2])
        pose_msg.orientation.w = 1.0
        self.pose_pub.publish(pose_msg)

        # 5. PERFORMANCE REPORTING
        total_processing_time = features_time + total_detection_time
        self.processing_time_pub.publish(Float32(data=total_processing_time))

        # Log performance
        self.get_logger().debug(
            f'Vision processing - Features: {features_time:.3f}s, '
            f'Detection: {total_detection_time:.3f}s, '
            f'Total: {total_processing_time:.3f}s, '
            f'FPS: {self.fps:.1f}'
        )

    def undistort_image(self, image):
        """Undistort image using camera calibration"""
        if self.camera_matrix is not None and self.distortion_coeffs is not None:
            return cv2.undistort(image, self.camera_matrix, self.distortion_coeffs.reshape(-1))
        return image

class DepthEstimator:
    """Estimate depth from stereo vision or monocular cues"""
    def __init__(self):
        self.stereo_bm = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        self.depth_map = None

    def estimate_depth_stereo(self, left_image, right_image):
        """Estimate depth using stereo vision"""
        gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY) if len(left_image.shape) == 3 else left_image
        gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY) if len(right_image.shape) == 3 else right_image

        # Compute disparity map
        disparity = self.stereo_bm.compute(gray_left, gray_right).astype(np.float32) / 16.0

        # Convert disparity to depth (simplified)
        baseline = 0.1  # Camera baseline in meters
        focal_length = 500  # Focal length in pixels (example)
        depth_map = (baseline * focal_length) / (disparity + 1e-6)  # Add small value to avoid division by zero

        return depth_map

    def estimate_depth_monocular(self, image, known_object_size):
        """Estimate depth using monocular cues and known object size"""
        # Detect objects and use size constancy principle
        detector = RealTimeObjectDetector()
        detections, _ = detector.detect_objects(image)

        depth_estimates = {}
        for detection in detections:
            bbox = detection['bbox']
            object_size_pixels = max(bbox[2] - bbox[0], bbox[3] - bbox[1])  # max of width/height

            # If we know the real-world size of this object class
            if detection['class'] in known_object_size:
                real_size = known_object_size[detection['class']]
                # Simple inverse relationship: larger objects appear closer
                distance = (real_size * focal_length) / object_size_pixels
                depth_estimates[detection['center']] = distance

        return depth_estimates

class VisualServoingController:
    """Control robot motion based on visual feedback"""
    def __init__(self):
        self.target_position = None
        self.current_position = np.array([0, 0])
        self.gain = 0.1

    def set_target(self, target_pixel):
        """Set the visual target in pixel coordinates"""
        self.target_position = target_pixel

    def compute_control(self, current_feature_position):
        """Compute control commands based on visual error"""
        if self.target_position is None:
            return np.array([0, 0])

        # Calculate error in pixel space
        pixel_error = np.array(self.target_position) - np.array(current_feature_position)

        # Convert pixel error to world coordinates using camera calibration
        # For simplicity, we'll use a linear approximation
        world_error = pixel_error * self.gain

        # Generate velocity commands
        linear_vel = min(0.3, np.linalg.norm(world_error))  # Limit linear velocity
        angular_vel = np.arctan2(world_error[1], world_error[0]) * 0.5  # Proportional control

        return np.array([linear_vel, angular_vel])

def main(args=None):
    rclpy.init(args=args)
    vision_node = ComputerVisionRobotics()

    try:
        rclpy.spin(vision_node)
    except KeyboardInterrupt:
        pass
    finally:
        vision_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Computer Vision Techniques

### Simultaneous Localization and Mapping (SLAM)

SLAM is crucial for autonomous robot navigation:

```python
# Example: Simple visual SLAM implementation
class VisualSLAM:
    def __init__(self):
        self.map_points = []
        self.camera_poses = []
        self.feature_trackers = {}
        self.current_pose = np.eye(4)  # 4x4 identity matrix
        self.keyframe_threshold = 0.1  # Movement threshold for keyframes

    def process_frame(self, image, timestamp):
        """Process a frame for SLAM"""
        # Detect features
        keypoints, descriptors = self.detect_features(image)

        # Track features from previous frames
        tracked_features = self.track_features(keypoints, descriptors)

        # Estimate camera motion
        motion_estimate = self.estimate_motion(tracked_features)

        # Update current pose
        self.current_pose = self.update_pose(self.current_pose, motion_estimate)

        # Add keyframe if moved enough
        if self.should_add_keyframe():
            self.add_keyframe(image, self.current_pose, tracked_features)

        # Optimize map (bundle adjustment would happen here)
        self.optimize_map()

        return {
            'current_pose': self.current_pose,
            'map_points': len(self.map_points),
            'tracked_features': len(tracked_features)
        }

    def detect_features(self, image):
        """Detect features in the image"""
        detector = FeatureDetector()
        return detector.detect_features(image)

    def track_features(self, current_keypoints, current_descriptors):
        """Track features across frames"""
        # This would implement feature tracking algorithms
        # like KLT tracker or descriptor matching
        return []

    def estimate_motion(self, tracked_features):
        """Estimate camera motion from tracked features"""
        # Use RANSAC to estimate essential matrix
        # and decompose to rotation and translation
        return np.eye(4)

    def update_pose(self, current_pose, motion):
        """Update camera pose with new motion"""
        return current_pose @ motion

    def should_add_keyframe(self):
        """Determine if a new keyframe should be added"""
        # Check if enough movement has occurred
        return True  # Simplified

    def add_keyframe(self, image, pose, features):
        """Add a new keyframe to the map"""
        self.camera_poses.append(pose)
        # Add features to map if they are stable
        pass

    def optimize_map(self):
        """Optimize the map and poses"""
        # This would implement bundle adjustment
        # or graph optimization
        pass
```

### 3D Reconstruction from Multiple Views

Robots can build 3D models of their environment:

```python
# Example: 3D reconstruction from multiple views
class MultiViewReconstructor:
    def __init__(self):
        self.views = []
        self.point_cloud = []
        self.camera_poses = []

    def add_view(self, image, camera_pose):
        """Add a new view to the reconstruction"""
        features = self.extract_features(image)

        self.views.append({
            'image': image,
            'pose': camera_pose,
            'features': features
        })

    def extract_features(self, image):
        """Extract features from image"""
        detector = FeatureDetector()
        keypoints, descriptors = detector.detect_features(image)

        return {
            'keypoints': keypoints,
            'descriptors': descriptors
        }

    def triangulate_points(self):
        """Triangulate 3D points from multiple views"""
        # Match features across views
        matches = self.match_features_across_views()

        # Triangulate matched points
        for match in matches:
            point_3d = self.triangulate_point(match)
            if point_3d is not None:
                self.point_cloud.append(point_3d)

    def triangulate_point(self, feature_match):
        """Triangulate a 3D point from feature matches"""
        # Get camera matrices for the views
        P1 = self.get_projection_matrix(self.camera_poses[feature_match['view1']])
        P2 = self.get_projection_matrix(self.camera_poses[feature_match['view2']])

        # Triangulate using DLT algorithm
        point_3d = cv2.triangulatePoints(
            P1, P2,
            feature_match['point1'],
            feature_match['point2']
        )

        # Convert from homogeneous to cartesian coordinates
        point_3d = point_3d[:3] / point_3d[3]
        return point_3d

    def get_projection_matrix(self, pose):
        """Get projection matrix from camera pose"""
        # Combine rotation, translation, and intrinsic matrix
        R = pose[:3, :3]
        t = pose[:3, 3]

        # Assuming known intrinsic matrix K
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])  # Example intrinsic matrix

        # Projection matrix P = K[R|t]
        P = K @ np.hstack((R, t.reshape(3, 1)))
        return P
```

## Lab: Implementing Robotic Computer Vision

In this lab, you'll implement a computer vision system for robotics:

```python
# lab_robotic_vision.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import cv2
import numpy as np

class RoboticVisionLab(Node):
    def __init__(self):
        super().__init__('robotic_vision_lab')

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.target_pub = self.create_publisher(Point, '/target_location', 10)
        self.status_pub = self.create_publisher(String, '/vision_status', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        # Vision components
        self.cv_bridge = CvBridge()
        self.object_detector = RealTimeObjectDetector()
        self.visual_servoing = VisualServoingController()

        # Lab parameters
        self.vision_mode = 'object_tracking'  # object_tracking, color_following, face_following
        self.target_object = 'person'  # Object class to track
        self.color_lower = np.array([0, 50, 50])  # HSV lower bound for color tracking
        self.color_upper = np.array([10, 255, 255])  # HSV upper bound for color tracking

        # Control loop
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

        # State variables
        self.latest_image = None
        self.tracked_object = None
        self.object_position = None

    def image_callback(self, msg):
        """Handle incoming camera images"""
        try:
            self.latest_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Image callback error: {e}')

    def control_loop(self):
        """Main control loop for vision-based navigation"""
        if self.latest_image is None:
            return

        # Process image based on current mode
        if self.vision_mode == 'object_tracking':
            self.process_object_tracking()
        elif self.vision_mode == 'color_following':
            self.process_color_following()
        elif self.vision_mode == 'face_following':
            self.process_face_following()

        # Generate control commands based on tracking results
        if self.object_position is not None:
            cmd = self.generate_navigation_command()
            self.cmd_pub.publish(cmd)

        # Publish status
        status_msg = String()
        status_msg.data = f"Mode: {self.vision_mode}, Target: {self.tracked_object}, Position: {self.object_position}"
        self.status_pub.publish(status_msg)

    def process_object_tracking(self):
        """Process object tracking"""
        detections, _ = self.object_detector.detect_objects(self.latest_image)

        # Find the target object
        for detection in detections:
            if detection['class'] == self.target_object and detection['confidence'] > 0.5:
                self.tracked_object = detection['class']
                self.object_position = detection['center']

                # Publish target location
                target_point = Point()
                target_point.x = float(self.object_position[0])
                target_point.y = float(self.object_position[1])
                target_point.z = detection['confidence']
                self.target_pub.publish(target_point)

                return

        # If no target found, clear tracking
        self.tracked_object = None
        self.object_position = None

    def process_color_following(self):
        """Process color-based object following"""
        # Convert BGR to HSV
        hsv = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2HSV)

        # Create mask for target color
        mask = cv2.inRange(hsv, self.color_lower, self.color_upper)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            if cv2.contourArea(largest_contour) > 500:  # Minimum area threshold
                # Calculate centroid
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    self.tracked_object = 'color_object'
                    self.object_position = [cx, cy]

                    # Publish target location
                    target_point = Point()
                    target_point.x = float(cx)
                    target_point.y = float(cy)
                    target_point.z = 1.0  # High confidence for color detection
                    self.target_pub.publish(target_point)
                    return

        # If no color object found
        self.tracked_object = None
        self.object_position = None

    def process_face_following(self):
        """Process face following (simplified implementation)"""
        # Use Haar cascade for face detection
        gray = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2GRAY)

        # Create a simple face detector (in practice, use cv2.CascadeClassifier)
        # For this example, we'll use a simple skin color detection
        hsv = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2HSV)

        # Skin color range in HSV
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])

        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find largest contour (likely the face)
            largest_contour = max(contours, key=cv2.contourArea)

            if cv2.contourArea(largest_contour) > 1000:  # Minimum area for face
                # Calculate centroid
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    self.tracked_object = 'face'
                    self.object_position = [cx, cy]

                    # Publish target location
                    target_point = Point()
                    target_point.x = float(cx)
                    target_point.y = float(cy)
                    target_point.z = 0.8  # Medium confidence
                    self.target_pub.publish(target_point)
                    return

        # If no face found
        self.tracked_object = None
        self.object_position = None

    def generate_navigation_command(self):
        """Generate navigation command based on object position"""
        cmd = Twist()

        if self.object_position is None:
            # No object detected, stop
            return cmd

        # Get image center
        image_center_x = self.latest_image.shape[1] / 2
        object_x = self.object_position[0]

        # Calculate error
        x_error = object_x - image_center_x

        # Proportional control for angular velocity
        cmd.angular.z = -0.002 * x_error  # Negative for correct direction

        # Move forward if object is in center (within threshold)
        center_threshold = 50  # pixels
        if abs(x_error) < center_threshold:
            cmd.linear.x = 0.2  # Move forward
        else:
            cmd.linear.x = 0.05  # Slow forward to approach

        # Limit angular velocity
        cmd.angular.z = max(-0.5, min(0.5, cmd.angular.z))

        return cmd

def main(args=None):
    rclpy.init(args=args)
    lab = RoboticVisionLab()

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

## Exercise: Design Your Own Computer Vision Application

Consider the following design challenge:

1. What specific robotic task would benefit from computer vision?
2. What objects or features need to be detected?
3. What are the real-time constraints for this application?
4. How would you handle different lighting conditions?
5. What control strategy would you use based on visual feedback?
6. How would you ensure robustness in dynamic environments?

## Summary

Computer vision is fundamental to robotic perception, enabling robots to understand and interact with their environment. Key concepts include:

- **Real-time Processing**: Vision systems must operate continuously and respond quickly
- **Embodied Perception**: Vision is integrated with the robot's physical actions
- **Feature Detection**: Identifying reliable features for tracking and mapping
- **Object Detection**: Recognizing and locating objects of interest
- **Visual Odometry**: Estimating motion from visual information
- **SLAM**: Building maps and localizing simultaneously

The integration of computer vision with ROS2 enables the development of sophisticated perception systems for robotics. Understanding these concepts is essential for developing robots that can perceive and navigate in complex environments.

In the next lesson, we'll explore machine learning techniques specifically applied to robot control and decision-making.