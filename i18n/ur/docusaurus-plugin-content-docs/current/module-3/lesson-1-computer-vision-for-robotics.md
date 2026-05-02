---
sidebar_position: 1
---

# روبوٹکس کے لیے کمپیوٹر وژن (Computer Vision for Robotics)

## تعارف (Introduction)

کمپیوٹر وژن فزیکل اے آئی (Physical AI) سسٹمز کا ایک اہم حصہ ہے، جو روبوٹس کو اپنے ماحول کو سمجھنے اور محسوس کرنے کے قابل بناتا ہے۔ روایتی کمپیوٹر وژن ایپلی کیشنز کے برعکس، روبوٹک وژن کو ریئل ٹائم (real-time) میں کام کرنا پڑتا ہے، متحرک ماحول کو سنبھالنا ہوتا ہے، اور موٹر کنٹرول سسٹمز کے ساتھ گہرا تعلق رکھنا ہوتا ہے۔ یہ سبق کمپیوٹر وژن کی ان تکنیکوں کا احاطہ کرتا ہے جو خاص طور پر روبوٹکس ایپلی کیشنز کے لیے ڈیزائن کی گئی ہیں۔

## روبوٹک کمپیوٹر وژن کے بنیادی اصول (Fundamentals of Robotic Computer Vision)

### روایتی کمپیوٹر وژن سے فرق (Key Differences from Traditional Computer Vision)

روبوٹک کمپیوٹر وژن روایتی ایپلی کیشنز سے کئی اہم طریقوں سے مختلف ہے:

1. **ریئل ٹائم ضروریات (Real-time Requirements)**: روبوٹس کو بصری معلومات کو مسلسل پروسیس کرنا پڑتا ہے اور فوری ردعمل دینا ہوتا ہے۔
2. **مجسم ادراک (Embodied Perception)**: وژن روبوٹ کی جسمانی حرکت اور اعمال کے ساتھ جڑا ہوتا ہے۔
3. **متحرک ماحول (Dynamic Environments)**: روبوٹ کو مسلسل بدلتے ہوئے مناظر کو سنبھالنا ہوتا ہے۔
4. **عمل پر مبنی (Action-Oriented)**: وژن کا مقصد صرف پہچاننا نہیں بلکہ مخصوص روبوٹک کاموں کو سرانجام دینا ہوتا ہے۔

### بصری پروسیسنگ پائپ لائن (Visual Processing Pipeline)

```python
# مثال: روبوٹک کمپیوٹر وژن پائپ لائن
class RoboticVisionPipeline:
    def __init__(self):
        self.image_preprocessor = ImagePreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.object_detector = ObjectDetector()
        self.scene_analyzer = SceneAnalyzer()
        self.action_generator = ActionGenerator()

    def process_frame(self, image):
        """ایک فریم کے لیے مکمل وژن پروسیسنگ پائپ لائن"""
        # 1. تصویر کی پری پروسیسنگ (Preprocess image)
        preprocessed = self.image_preprocessor.process(image)

        # 2. خصوصیات نکالنا (Extract features)
        features = self.feature_extractor.extract(preprocessed)

        # 3. اشیاء کی شناخت (Detect objects)
        objects = self.object_detector.detect(features)

        # 4. منظر کا تجزیہ (Analyze scene)
        scene_info = self.scene_analyzer.analyze(objects, preprocessed)

        # 5. روبوٹ کے اعمال تیار کرنا (Generate robot actions)
        actions = self.action_generator.generate(scene_info)

        return {
            'objects': objects,
            'scene_info': scene_info,
            'actions': actions,
            'features': features
        }
```

## روبوٹکس کے لیے امیج پری پروسیسنگ (Image Preprocessing for Robotics)

### کیمرہ کیلیبریشن اور ریکٹی فیکیشن (Camera Calibration and Rectification)

روبوٹس کو درست کیمرہ کیلیبریشن کی ضرورت ہوتی ہے تاکہ وہ پکسل کوآرڈینیٹس کو حقیقی دنیا کی پیمائش میں تبدیل کر سکیں:

```python
# مثال: کیمرہ کیلیبریشن اور ریکٹی فیکیشن
import numpy as np
import cv2

class CameraCalibrator:
    def __init__(self):
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.rotation_vector = None
        self.translation_vector = None

    def calibrate_camera(self, object_points, image_points, image_size):
        """شطرنج کے پیٹرن کا استعمال کرتے ہوئے کیمرہ کیلیبریٹ کریں"""
        self.camera_matrix, self.distortion_coeffs, self.rotation_vector, self.translation_vector = \
            cv2.calibrateCamera(
                object_points, image_points, image_size, None, None
            )
        return self.camera_matrix, self.distortion_coeffs

    def undistort_image(self, image):
        """تصویر سے لینس کی خرابی (distortion) ختم کریں"""
        if self.camera_matrix is not None and self.distortion_coeffs is not None:
            return cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)
        return image

    def convert_pixel_to_world(self, pixel_coords, depth):
        """پکسل کوآرڈینیٹس کو حقیقی دنیا کے کوآرڈینیٹس میں تبدیل کریں"""
        if self.camera_matrix is not None:
            # نارملائزڈ کوآرڈینیٹس میں تبدیل کریں
            normalized = np.linalg.inv(self.camera_matrix).dot(
                np.array([pixel_coords[0], pixel_coords[1], 1])
            )

            # ڈیپتھ (depth) کے لحاظ سے اسکیل کریں
            world_coords = normalized * depth
            return world_coords
        return None
```

### روبوٹک ایپلی کیشنز کے لیے امیج انہانسمنٹ (Image Enhancement for Robotic Applications)

روبوٹس کو اکثر مشکل روشنی والے حالات میں کام کرنا پڑتا ہے:

```python
# مثال: روبوٹکس کے لیے امیج انہانسمنٹ
class ImageEnhancer:
    def __init__(self):
        self.gamma = 1.0
        self.contrast = 1.0
        self.brightness = 0

    def enhance_for_robot_vision(self, image):
        """روبوٹک ادراک کو بہتر بنانے کے لیے تصویر کو بہتر بنائیں"""
        # روشنی کے حالات کے لیے گاما (gamma) ایڈجسٹ کریں
        enhanced = self.adjust_gamma(image, self.gamma)

        # کنٹراسٹ (contrast) بہتر بنائیں
        enhanced = self.enhance_contrast(enhanced, self.contrast)

        # چمک (brightness) کو سنبھالیں
        enhanced = self.adjust_brightness(enhanced, self.brightness)

        # شور (noise) کو کم کریں تاکہ پروسیسنگ صاف ہو
        enhanced = self.reduce_noise(enhanced)

        return enhanced

    def adjust_gamma(self, image, gamma):
        """تصویر کا گاما ایڈجسٹ کریں"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

## فیچر ڈیٹیکشن اور ایکسٹریکشن (Feature Detection and Extraction)

### اہم فیچر ڈیٹیکشن الگورتھم (Key Feature Detection Algorithms)

روبوٹس کو نشانات (landmarks) اور اشیاء کی شناخت کے لیے مضبوط فیچر ڈیٹیکشن کی ضرورت ہوتی ہے:

```python
# مثال: روبوٹک ایپلی کیشنز کے لیے فیچر ڈیٹیکشن
class FeatureDetector:
    def __init__(self):
        self.detector_type = 'orb'  # 'orb', 'sift', 'akaze' ہو سکتا ہے
        self.detector = self.initialize_detector()

    def initialize_detector(self):
        """مناسب فیچر ڈیٹیکٹر کو شروع کریں"""
        if self.detector_type == 'orb':
            return cv2.ORB_create(nfeatures=500)
        elif self.detector_type == 'sift':
            return cv2.SIFT_create()
        else:
            return cv2.ORB_create()

    def detect_features(self, image):
        """تصویر میں فیچرز تلاش کریں"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors
```

## روبوٹکس کے لیے آبجیکٹ ڈیٹیکشن (Object Detection for Robotics)

### ریئل ٹائم آبجیکٹ ڈیٹیکشن (Real-time Object Detection)

روبوٹس کو اشیاء کو تیزی سے اور درست طریقے سے پہچاننے کی ضرورت ہوتی ہے:

```python
# مثال: روبوٹکس کے لیے ریئل ٹائم آبجیکٹ ڈیٹیکشن
class RealTimeObjectDetector:
    def __init__(self):
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4

    def detect_objects(self, image):
        """تصویر میں اشیاء کی شناخت کریں"""
        # یہاں ڈیپ لرننگ ماڈل (YOLO یا SSD) کا استعمال کیا جاتا ہے
        # فی الحال روایتی طریقہ دکھایا گیا ہے
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    'class': 'object',
                    'bbox': [x, y, x + w, y + h],
                    'center': [x + w/2, y + h/2]
                })
        return detections

## روبوٹکس کے لیے ROS2 امپلیمینٹیشن (ROS2 Implementation)

یہاں روبوٹکس کے لیے کمپیوٹر وژن کی ایک مکمل ROS2 امپلیمینٹیشن دی گئی ہے:

```python
# computer_vision_robotics.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class ComputerVisionRobotics(Node):
    def __init__(self):
        super().__init__('computer_vision_robotics')

        # پبلشرز (Publishers)
        self.detection_pub = self.create_publisher(String, '/object_detections', 10)

        # سبسکرائبرز (Subscribers)
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        # وژن اجزاء (Vision components)
        self.cv_bridge = CvBridge()
        self.image_enhancer = ImageEnhancer()
        self.object_detector = RealTimeObjectDetector()

    def image_callback(self, msg):
        """آنے والی کیمرہ تصاویر کو ہینڈل کریں"""
        try:
            # ROS امیج کو OpenCV فارمیٹ میں تبدیل کریں
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'تصویر تبدیل کرنے میں غلطی: {e}')
            return

        # 1. امیج انہانسمنٹ (IMAGE ENHANCEMENT)
        enhanced_image = self.image_enhancer.enhance_for_robot_vision(cv_image)

        # 2. آبجیکٹ ڈیٹیکشن (OBJECT DETECTION)
        detections = self.object_detector.detect_objects(enhanced_image)

        # نتائج پبلش کریں
        self.get_logger().info(f'شناخت شدہ اشیاء: {len(detections)}')

## بصری اوڈومیٹری اور سلیم (Visual Odometry and SLAM)

نیویگیشن کے لیے، روبوٹس کو بصری فیچرز کا استعمال کرتے ہوئے اپنی پوزیشن کو ٹریک کرنے کی ضرورت ہوتی ہے:

```python
# مثال: بصری اوڈومیٹری فیچر ٹریکنگ
class VisualOdometry:
    def __init__(self):
        self.current_position = np.array([0, 0, 0])  # x, y, theta

    def process_frame(self, current_frame):
        """بصری اوڈومیٹری کے لیے فریم پروسیس کریں"""
        # فیچر میچنگ اور پوزیشن اپ ڈیٹ یہاں کی جاتی ہے
        return self.current_position
```

### سلیمانٹینیئس لوکلائزیشن اینڈ میپنگ (SLAM)

سلیم (SLAM) خود مختار روبوٹ نیویگیشن کے لیے انتہائی اہم ہے:

```python
# مثال: سادہ بصری سلیم (Visual SLAM) امپلیمینٹیشن
class VisualSLAM:
    def __init__(self):
        self.map_points = []
        self.current_pose = np.eye(4)

    def process_frame(self, image):
        """سلیم کے لیے فریم پروسیس کریں"""
        # نقشہ بنانا اور لوکلائزیشن ساتھ ساتھ کرنا
        return self.current_pose

## تھری ڈی ری کنسٹرکشن (3D Reconstruction)

روبوٹس اپنے ماحول کا تھری ڈی ماڈل تیار کر سکتے ہیں:

```python
# مثال: ملٹی ویو سے تھری ڈی ری کنسٹرکشن
class MultiViewReconstructor:
    def __init__(self):
        self.point_cloud = []

    def triangulate_points(self, image1, image2):
        """دو تصاویر سے تھری ڈی پوائنٹس تیار کریں"""
        # ٹرائی اینگولیشن (triangulation) کا عمل یہاں ہوتا ہے
        return self.point_cloud
```

## جدید کمپیوٹر وژن تکنیکیں (Advanced Computer Vision Techniques)

جدید روبوٹکس میں ڈیپ لرننگ (Deep Learning) اور ٹرانسفارمرز (Transformers) کا استعمال تیزی سے بڑھ رہا ہے:

*   **وژن ٹرانسفارمرز (ViT)**: تصویر کے مختلف حصوں کے درمیان تعلق کو سمجھنے کے لیے۔
*   **سیمنٹک سیگمنٹیشن**: ہر پکسل کو ایک مخصوص کلاس (جیسے 'فرش'، 'رکاوٹ') دینے کے لیے۔
*   **ڈیپتھ ایسٹیمیشن**: ایک ہی کیمرے سے فاصلے کا اندازہ لگانے کے لیے۔
```

## لیب: روبوٹک کمپیوٹر وژن کی امپلیمینٹیشن (Lab: Implementing Robotic Computer Vision)

اس لیب میں، آپ روبوٹکس کے لیے ایک کمپیوٹر وژن سسٹم امپلیمینٹ کریں گے:

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

        # پبلشرز (Publishers)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.target_pub = self.create_publisher(Point, '/target_location', 10)
        self.status_pub = self.create_publisher(String, '/vision_status', 10)

        # سبسکرائبرز (Subscribers)
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        # وژن اجزاء (Vision components)
        self.cv_bridge = CvBridge()
        self.object_detector = RealTimeObjectDetector()
        self.visual_servoing = VisualServoingController()

        # لیب پیرامیٹرز (Lab parameters)
        self.vision_mode = 'object_tracking'  # آبجیکٹ ٹریکنگ، کلر فالوونگ، فیس فالوونگ
        self.target_object = 'person'  # ٹریک کرنے کے لیے آبجیکٹ کلاس
        self.color_lower = np.array([0, 50, 50])  # کلر ٹریکنگ کے لیے HSV لوئر باؤنڈ
        self.color_upper = np.array([10, 255, 255])  # کلر ٹریکنگ کے لیے HSV اپر باؤنڈ

        # کنٹرول لوپ (Control loop)
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

        # اسٹیٹ ویری ایبلز (State variables)
        self.latest_image = None
        self.tracked_object = None
        self.object_position = None

    def image_callback(self, msg):
        """آنے والی کیمرہ تصاویر کو ہینڈل کریں"""
        try:
            self.latest_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'تصویر کی کال بیک میں غلطی: {e}')

    def control_loop(self):
        """وژن پر مبنی نیویگیشن کے لیے مین کنٹرول لوپ"""
        if self.latest_image is None:
            return

        # موجودہ موڈ کی بنیاد پر تصویر پروسیس کریں
        if self.vision_mode == 'object_tracking':
            self.process_object_tracking()
        elif self.vision_mode == 'color_following':
            self.process_color_following()
        elif self.vision_mode == 'face_following':
            self.process_face_following()

        # ٹریکنگ کے نتائج کی بنیاد پر کنٹرول کمانڈز تیار کریں
        if self.object_position is not None:
            cmd = self.generate_navigation_command()
            self.cmd_pub.publish(cmd)

        # اسٹیٹس پبلش کریں
        status_msg = String()
        status_msg.data = f"Mode: {self.vision_mode}, Position: {self.object_position}"
        self.status_pub.publish(status_msg)

    def process_object_tracking(self):
        """آبجیکٹ ٹریکنگ پروسیس کریں"""
        detections = self.object_detector.detect_objects(self.latest_image)

        for detection in detections:
            if detection['class'] == self.target_object:
                self.tracked_object = detection['class']
                self.object_position = detection['center']
                return

        self.tracked_object = None
        self.object_position = None

    def generate_navigation_command(self):
        """آبجیکٹ کی پوزیشن کی بنیاد پر نیویگیشن کمانڈ تیار کریں"""
        cmd = Twist()
        if self.object_position is None:
            return cmd

        image_center_x = self.latest_image.shape[1] / 2
        object_x = self.object_position[0]
        error = object_x - image_center_x

        cmd.angular.z = -0.002 * error
        if abs(error) < 50:
            cmd.linear.x = 0.2
        
        return cmd
```

## خلاصہ (Summary)

کمپیوٹر وژن روبوٹک ادراک (perception) کے لیے بنیادی اہمیت رکھتا ہے، جو روبوٹس کو اپنے ماحول کو سمجھنے اور اس کے ساتھ تعامل کرنے کے قابل بناتا ہے۔ اہم تصورات میں شامل ہیں:

- **ریئل ٹائم پروسیسنگ**: وژن سسٹمز کو مسلسل کام کرنا اور تیزی سے جواب دینا چاہیے۔
- **مجسم ادراک (Embodied Perception)**: وژن روبوٹ کے جسمانی اعمال کے ساتھ مربوط ہوتا ہے۔
- **فیچر ڈیٹیکشن**: ٹریکنگ اور میپنگ کے لیے قابل بھروسہ فیچرز کی شناخت۔
- **آبجیکٹ ڈیٹیکشن**: دلچسپی کی اشیاء کو پہچاننا اور ان کا مقام معلوم کرنا۔
- **بصری اوڈومیٹری**: بصری معلومات سے حرکت کا اندازہ لگانا۔
- **سلیم (SLAM)**: بیک وقت نقشے بنانا اور لوکلائزیشن کرنا۔

کمپیوٹر وژن کا ROS2 کے ساتھ انضمام روبوٹکس کے لیے جدید ادراک کے نظام کی ترقی کو ممکن بناتا ہے۔





