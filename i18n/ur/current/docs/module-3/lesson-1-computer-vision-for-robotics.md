---
sidebar_position: 1
---

# روبوٹکس کے لیے کمپیوٹر وژن

## تعارف

کمپیوٹر وژن جسمانی مصنوعی ذہانت کے نظاموں کا ایک اہم جزو ہے، جو روبوٹس کو اپنے ماحول کا ادراک کرنے اور سمجھنے کے قابل بناتا ہے۔ روایتی کمپیوٹر وژن کے اطلاقات کے برعکس، روبوٹک وژن کو حقیقی وقت میں کام کرنا ہوتا ہے، متحرک ماحولوں کو سنبھالنا ہوتا ہے، اور موٹر کنٹرول سسٹم کے ساتھ تنگی سے ضم ہونا ہوتا ہے۔ یہ سبق روبوٹکس اطلاقات کے لیے خاص طور پر ڈیزائن کردہ کمپیوٹر وژن کی تکنیکوں کا جائزہ لیتا ہے۔

## روبوٹک کمپیوٹر وژن کے بنیادیات

### روایتی کمپیوٹر وژن سے کلیدی فروق

روبوٹک کمپیوٹر وژن مختلف اہم طریقوں میں روایتی اطلاقات سے مختلف ہے:

1. **حقیقی وقت کی ضروریات**: روبوٹس کو بصری معلومات کو مسلسل عمل کرنا ہوتا ہے اور حقیقی وقت میں جواب دینا ہوتا ہے
2. **جسمانی ادراک**: وژن روبوٹ کی جسمانی حرکت اور اعمال کے ساتھ ضم ہے
3. **متحرک ماحول**: روبوٹ کو ہمیشہ تبدیل ہوتے مناظر کو سنبھالنا ہوتا ہے
4. **عمل کے لیے موزوں**: وژن صرف پہچان سے زیادہ مخصوص روبوٹک کاموں کے لیے کام کرتا ہے

### بصری پروسیسنگ پائپ لائن

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
        # 1. تصویر کی پیش پروسیسنگ
        preprocessed = self.image_preprocessor.process(image)

        # 2. خصوصیات نکالیں
        features = self.feature_extractor.extract(preprocessed)

        # 3. اشیاء کا پتہ لگائیں
        objects = self.object_detector.detect(features)

        # 4. منظر کا تجزیہ کریں
        scene_info = self.scene_analyzer.analyze(objects, preprocessed)

        # 5. روبوٹ کے اعمال تیار کریں
        actions = self.action_generator.generate(scene_info)

        return {
            'objects': objects,
            'scene_info': scene_info,
            'actions': actions,
            'features': features
        }
```

## روبوٹکس کے لیے تصویر کی پیش پروسیسنگ

### کیمرہ کیلیبریشن اور ریکٹیفکیشن

روبوٹس کو پکسل کوآرڈینیٹس کو حقیقی دنیا کے پیمائش میں تبدیل کرنے کے لیے درست کیمرہ کیلیبریشن کی ضرورت ہوتی ہے:

```python
# مثال: کیمرہ کیلیبریشن اور ریکٹیفکیشن
import numpy as np
import cv2

class CameraCalibrator:
    def __init__(self):
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.rotation_vector = None
        self.translation_vector = None

    def calibrate_camera(self, object_points, image_points, image_size):
        """چیس بورڈ پیٹرن کا استعمال کرتے ہوئے کیمرہ کیلیبریٹ کریں"""
        self.camera_matrix, self.distortion_coeffs, self.rotation_vector, self.translation_vector = \
            cv2.calibrateCamera(
                object_points, image_points, image_size, None, None
            )
        return self.camera_matrix, self.distortion_coeffs

    def undistort_image(self, image):
        """تصویر سے لینس کی بگاڑ کو ہٹائیں"""
        if self.camera_matrix is not None and self.distortion_coeffs is not None:
            return cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)
        return image

    def convert_pixel_to_world(self, pixel_coords, depth):
        """پکسل کوآرڈینیٹس کو ورلڈ کوآرڈینیٹس میں تبدیل کریں"""
        if self.camera_matrix is not None:
            # نارملائزڈ کوآرڈینیٹس میں تبدیل کریں
            normalized = np.linalg.inv(self.camera_matrix).dot(
                np.array([pixel_coords[0], pixel_coords[1], 1])
            )

            # گہرائی کے ذریعے اسکیل کریں
            world_coords = normalized * depth
            return world_coords
        return None
```

### روبوٹک اطلاقات کے لیے تصویر کو بہتر بنانا

روبوٹس اکثر چیلنجنگ روشنی کی حالت میں کام کرتے ہیں:

```python
# مثال: روبوٹکس کے لیے تصویر کو بہتر بنانا
class ImageEnhancer:
    def __init__(self):
        self.gamma = 1.0
        self.contrast = 1.0
        self.brightness = 0

    def enhance_for_robot_vision(self, image):
        """بہتر روبوٹک ادراک کے لیے تصویر کو بہتر بنائیں"""
        # روشنی کی حالت کے لیے گاما ایڈجسٹ کریں
        enhanced = self.adjust_gamma(image, self.gamma)

        # کنٹراسٹ بہتر بنائیں
        enhanced = self.enhance_contrast(enhanced, self.contrast)

        # چمک کی تبدیلیوں کو سنبھالیں
        enhanced = self.adjust_brightness(enhanced, self.brightness)

        # صاف پروسیسنگ کے لیے نوائز کم کریں
        enhanced = self.reduce_noise(enhanced)

        return enhanced

    def adjust_gamma(self, image, gamma):
        """تصویر کا گاما ایڈجسٹ کریں"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def enhance_contrast(self, image, contrast):
        """تصویر کا کنٹراسٹ بہتر بنائیں"""
        return cv2.convertScaleAbs(image, alpha=contrast, beta=0)

    def adjust_brightness(self, image, brightness):
        """تصویر کی چمک ایڈجسٹ کریں"""
        return cv2.convertScaleAbs(image, alpha=1, beta=brightness)

    def reduce_noise(self, image):
        """کناروں کو محفوظ رکھتے ہوئے نوائز کم کریں"""
        return cv2.bilateralFilter(image, 9, 75, 75)
```

## خصوصیات کا پتہ لگانا اور نکالنا

### کلیدی خصوصیات کا پتہ لگانے کے الگورتھم

روبوٹس کو نمایاں مقامات اور اشیاء کی پہچان کے لیے مضبوط خصوصیات کا پتہ لگانے کی ضرورت ہوتی ہے:

```python
# مثال: روبوٹک اطلاقات کے لیے خصوصیات کا پتہ لگانا
class FeatureDetector:
    def __init__(self):
        self.detector_type = 'orb'  # 'orb', 'sift', 'surf', 'akaze' ہو سکتا ہے
        self.detector = self.initialize_detector()

    def initialize_detector(self):
        """مناسب خصوصیات کا پتہ لگانے والا ایجنٹ شروع کریں"""
        if self.detector_type == 'orb':
            return cv2.ORB_create(nfeatures=500)
        elif self.detector_type == 'sift':
            return cv2.SIFT_create()
        elif self.detector_type == 'akaze':
            return cv2.AKAZE_create()
        else:
            return cv2.ORB_create()

    def detect_features(self, image):
        """تصویر میں خصوصیات کا پتہ لگائیں"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, descriptors1, descriptors2):
        """دو تصویروں کے درمیان خصوصیات کو میچ کریں"""
        # کارکردگی کے لیے FLANN میچر کا استعمال کریں
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        # لو کے تناسب ٹیسٹ لاگو کریں
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        return good_matches
```

### وژوئل اوڈومیٹری اور SLAM خصوصیات

نیویگیشن کے لیے، روبوٹس کو بصری خصوصیات کا استعمال کرتے ہوئے اپنی پوزیشن ٹریک کرنے کی ضرورت ہوتی ہے:

```python
# مثال: وژوئل اوڈومیٹری خصوصیات ٹریکنگ
class VisualOdometry:
    def __init__(self):
        self.prev_frame = None
        self.prev_keypoints = None
        self.current_position = np.array([0, 0, 0])  # x, y, theta
        self.feature_detector = FeatureDetector()

    def process_frame(self, current_frame):
        """وژوئل اوڈومیٹری کے لیے ایک فریم کو عمل کریں"""
        if self.prev_frame is None:
            # پہلے فریم کے ساتھ شروع کریں
            self.prev_frame = current_frame
            self.prev_keypoints, _ = self.feature_detector.detect_features(current_frame)
            return self.current_position

        # موجودہ فریم میں خصوصیات کا پتہ لگائیں
        curr_keypoints, curr_descriptors = self.feature_detector.detect_features(current_frame)

        # فریموں کے درمیان خصوصیات کو میچ کریں
        matches = self.feature_detector.match_features(
            self.get_descriptors_for_keypoints(self.prev_keypoints, self.prev_frame),
            curr_descriptors
        )

        # خصوصیات کے مطابقت کی بنیاد پر حرکت کا تخمینہ لگائیں
        if len(matches) >= 10:  # قابل اعتماد تخمینے کے لیے کم از کم میچز کی ضرورت ہے
            prev_matched = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            curr_matched = np.float32([curr_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # ٹرانسفارمیشن کا تخمینہ لگائیں
            transformation, mask = cv2.estimateAffinePartial2D(prev_matched, curr_matched)

            if transformation is not None:
                # ٹرانسفارمیشن کی بنیاد پر پوزیشن اپ ڈیٹ کریں
                self.update_position(transformation)

        # اگلی تکرار کے لیے اپ ڈیٹ کریں
        self.prev_frame = current_frame
        self.prev_keypoints = curr_keypoints

        return self.current_position

    def get_descriptors_for_keypoints(self, keypoints, frame):
        """مخصوص کی پوائنٹس کے لیے ڈسکرپٹرز حاصل کریں"""
        # یہ ایک سادہ ورژن ہے - عمل میں، آپ ڈسکرپٹرز کو دوبارہ کمپیوٹ یا اسٹور کریں گے
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        _, descriptors = self.feature_detector.detector.detectAndCompute(gray, None)
        return descriptors

    def update_position(self, transformation):
        """بصری ٹرانسفارمیشن کی بنیاد پر روبوٹ کی پوزیشن اپ ڈیٹ کریں"""
        # ٹرانسفارمیشن میٹرکس سے ترجمہ اور گھمائش نکالیں
        dx = transformation[0, 2]
        dy = transformation[1, 2]
        dtheta = np.arctan2(transformation[1, 0], transformation[0, 0])

        # پوزیشن اپ ڈیٹ کریں (سادہ اوڈومیٹری)
        self.current_position[0] += dx
        self.current_position[1] += dy
        self.current_position[2] += dtheta
```

## روبوٹکس کے لیے اشیاء کا پتہ لگانا

### حقیقی وقت میں اشیاء کا پتہ لگانا

روبوٹس کو اشیاء کو جلدی اور درست طریقے سے ڈھونڈنے کی ضرورت ہوتی ہے:

```python
# مثال: روبوٹکس کے لیے حقیقی وقت میں اشیاء کا پتہ لگانا
import time

class RealTimeObjectDetector:
    def __init__(self):
        self.detection_model = self.load_detection_model()
        self.object_classes = self.get_object_classes()
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4

    def load_detection_model(self):
        """ایک پیش تربیت یافتہ اشیاء کا پتہ لگانے والا ماڈل لوڈ کریں"""
        # اس مثال کے لیے، ہم ایک پیش تربیت یافتہ ماڈل کے ساتھ اوپن سی وی کا DNN ماڈیول استعمال کریں گے
        # عمل میں، آپ YOLO، SSD، یا دیگر ماڈلز استعمال کر سکتے ہیں
        try:
            # ایک پیش تربیت یافتہ ماڈل لوڈ کرنے کی کوشش کریں (سادہ)
            return cv2.dnn.readNetFromDarknet("yolo_config.cfg", "yolo_weights.weights")
        except:
            # دیگر طریقے کے لیے فیل بیک
            return None

    def detect_objects(self, image):
        """تصویر میں اشیاء کا پتہ لگائیں"""
        start_time = time.time()

        height, width = image.shape[:2]

        # تصویر سے بLOB بنائیں
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

        if self.detection_model is not None:
            # نیٹ ورک کے ان پٹ کے طور پر بLOB سیٹ کریں
            self.detection_model.setInput(blob)

            # فارورڈ پاس چلائیں
            layer_names = self.detection_model.getLayerNames()
            output_names = [layer_names[i[0] - 1] for i in self.detection_model.getUnconnectedOutLayers()]
            outputs = self.detection_model.forward(output_names)

            # آؤٹ پٹس کو عمل کریں
            boxes, confidences, class_ids = self.process_detections(outputs, width, height)

            # غیر زیادہ اہم سپریشن لاگو کریں
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

            # نتائج کی شکل دیں
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
            # روایتی طریقوں کا استعمال کرتے ہوئے فیل بیک کا پتہ لگانا
            detections = self.traditional_detection_fallback(image)

        processing_time = time.time() - start_time
        return detections, processing_time

    def process_detections(self, outputs, width, height):
        """تشخیص نیٹ ورک سے آؤٹ پٹس کو عمل کریں"""
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence_threshold:
                    # اصل کوآرڈینیٹس میں تبدیل کریں
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
        """روایتی کمپیوٹر وژن کا استعمال کرتے ہوئے فیل بیک کا پتہ لگانا"""
        # یہ ایک سادہ مثال ہے - عمل میں، آپ Haar cascades،
        # HOG ڈسکرپٹرز، یا دیگر روایتی طریقے استعمال کر سکتے ہیں
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # کنارے ڈھونڈیں
        edges = cv2.Canny(gray, 50, 150)

        # کنٹور تلاش کریں
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # کم از کم رقبہ کی حد
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    'class': 'object',
                    'confidence': 0.6,  # فیل بیک کے لیے ڈیفالٹ یقین
                    'bbox': [x, y, x + w, y + h],
                    'center': [x + w/2, y + h/2]
                })

        return detections

    def get_object_classes(self):
        """وہ اشیاء کی کلاسز حاصل کریں جن کا پتہ لگانے کا ماڈل کر سکتا ہے"""
        # عام COCO ڈیٹا سیٹ کلاسز
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

## ROS2 نفاذ: روبوٹکس کے لیے کمپیوٹر وژن

یہاں روبوٹکس کے لیے کمپیوٹر وژن کا ایک مکمل ROS2 نفاذ ہے:

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

        # پبلشرز
        self.detection_pub = self.create_publisher(String, '/object_detections', 10)
        self.feature_pub = self.create_publisher(String, '/feature_matches', 10)
        self.pose_pub = self.create_publisher(Pose, '/camera_pose', 10)
        self.processing_time_pub = self.create_publisher(Float32, '/vision_processing_time', 10)

        # سبسکرائبرز
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_callback, 10
        )

        # وژن کمپوننٹس
        self.cv_bridge = CvBridge()
        self.image_enhancer = ImageEnhancer()
        self.feature_detector = FeatureDetector()
        self.object_detector = RealTimeObjectDetector()
        self.visual_odometry = VisualOdometry()

        # کیمرہ کیلیبریشن
        self.camera_matrix = None
        self.distortion_coeffs = None

        # پروسیسنگ کی حالت
        self.last_image_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

        # پروسیسنگ پیرامیٹر
        self.processing_enabled = True
        self.detection_frequency = 0.5  # ہر 0.5 سیکنڈ میں پروسیسنگ کریں

    def camera_info_callback(self, msg):
        """کیلیبریشن کے لیے کیمرہ انفارمیشن کو سنبھالیں"""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.distortion_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        """آنے والی کیمرہ تصویروں کو سنبھالیں"""
        current_time = time.time()
        self.frame_count += 1

        # FPS کا حساب لگائیں
        if current_time - self.last_image_time > 1.0:
            self.fps = self.frame_count / (current_time - self.last_image_time)
            self.frame_count = 0
            self.last_image_time = current_time

        if not self.processing_enabled:
            return

        try:
            # ROS تصویر کو اوپن سی وی فارمیٹ میں تبدیل کریں
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')
            return

        # 1. تصویر کو بہتر بنانا
        enhanced_image = self.image_enhancer.enhance_for_robot_vision(cv_image)

        # 2. خصوصیات کا پتہ لگانا اور ٹریکنگ
        if self.frame_count % 5 == 0:  # ہر 5 فریمز میں خصوصیات کو عمل کریں تاکہ کمپیوٹیشن بچائی جا سکے
            features_start_time = time.time()
            keypoints, descriptors = self.feature_detector.detect_features(enhanced_image)

            # خصوصیات کی معلومات شائع کریں
            feature_info = f"Features detected: {len(keypoints) if keypoints else 0}"
            self.feature_pub.publish(String(data=feature_info))

            features_time = time.time() - features_start_time

        # 3. اشیاء کا پتہ لگانا (کم فریکوینسی سے کمپیوٹیشن بچانے کے لیے)
        if current_time % self.detection_frequency < 0.05:  # ہر 0.5 سیکنڈ میں پروسیسنگ کریں
            detection_start_time = time.time()
            detections, detection_time = self.object_detector.detect_objects(enhanced_image)

            # ڈیٹکشن کے نتائج شائع کریں
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

        # 4. وژوئل اوڈومیٹری
        position = self.visual_odometry.process_frame(enhanced_image)

        # کیمرہ پوزیشن شائع کریں
        pose_msg = Pose()
        pose_msg.position.x = float(position[0])
        pose_msg.position.y = float(position[1])
        pose_msg.position.z = 0.0  # زمینی سطح کا فرض کریں
        # سادہ جہت کی نمائندگی
        pose_msg.orientation.z = float(position[2])
        pose_msg.orientation.w = 1.0
        self.pose_pub.publish(pose_msg)

        # 5. کارکردگی کی رپورٹنگ
        total_processing_time = features_time + total_detection_time
        self.processing_time_pub.publish(Float32(data=total_processing_time))

        # کارکردگی لاگ کریں
        self.get_logger().debug(
            f'Vision processing - Features: {features_time:.3f}s, '
            f'Detection: {total_detection_time:.3f}s, '
            f'Total: {total_processing_time:.3f}s, '
            f'FPS: {self.fps:.1f}'
        )

    def undistort_image(self, image):
        """کیمرہ کیلیبریشن کا استعمال کرتے ہوئے تصویر کو بے نقاب کریں"""
        if self.camera_matrix is not None and self.distortion_coeffs is not None:
            return cv2.undistort(image, self.camera_matrix, self.distortion_coeffs.reshape(-1))
        return image

class DepthEstimator:
    """سٹیریو وژن یا مونوکولر اشاروں سے گہرائی کا تخمینہ لگائیں"""
    def __init__(self):
        self.stereo_bm = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        self.depth_map = None

    def estimate_depth_stereo(self, left_image, right_image):
        """سٹیریو وژن کا استعمال کرتے ہوئے گہرائی کا تخمینہ لگائیں"""
        gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY) if len(left_image.shape) == 3 else left_image
        gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY) if len(right_image.shape) == 3 else right_image

        # ڈسپیرٹی میپ کمپیوٹ کریں
        disparity = self.stereo_bm.compute(gray_left, gray_right).astype(np.float32) / 16.0

        # ڈسپیرٹی کو گہرائی میں تبدیل کریں (سادہ)
        baseline = 0.1  # میٹر میں کیمرہ بیس لائن
        focal_length = 500  # پکسل میں فوکل لمبائی (مثال)
        depth_map = (baseline * focal_length) / (disparity + 1e-6)  # تقسیم صفر سے بچنے کے لیے چھوٹی قیمت شامل کریں

        return depth_map

    def estimate_depth_monocular(self, image, known_object_size):
        """مونوکولر اشاروں اور جانی چیز کے سائز کا استعمال کرتے ہوئے گہرائی کا تخمینہ لگائیں"""
        # اشیاء کا پتہ لگائیں اور سائز کی مستقل اصول کا استعمال کریں
        detector = RealTimeObjectDetector()
        detections, _ = detector.detect_objects(image)

        depth_estimates = {}
        for detection in detections:
            bbox = detection['bbox']
            object_size_pixels = max(bbox[2] - bbox[0], bbox[3] - bbox[1])  # چوڑائی/اونچائی کا زیادہ سے زیادہ

            # اگر ہمیں اس چیز کی کلاس کا حقیقی دنیا کا سائز معلوم ہے
            if detection['class'] in known_object_size:
                real_size = known_object_size[detection['class']]
                # سادہ معکوس رشتہ: بڑی چیزیں قریب نظر آتی ہیں
                distance = (real_size * focal_length) / object_size_pixels
                depth_estimates[detection['center']] = distance

        return depth_estimates

class VisualServoingController:
    """بصری فیڈ بیک کی بنیاد پر روبوٹ حرکت کنٹرول کریں"""
    def __init__(self):
        self.target_position = None
        self.current_position = np.array([0, 0])
        self.gain = 0.1

    def set_target(self, target_pixel):
        """پکسل کوآرڈینیٹس میں بصری ہدف سیٹ کریں"""
        self.target_position = target_pixel

    def compute_control(self, current_feature_position):
        """بصری خامی کی بنیاد پر کنٹرول کمانڈز کمپیوٹ کریں"""
        if self.target_position is None:
            return np.array([0, 0])

        # پکسل سپیس میں خامی کا حساب لگائیں
        pixel_error = np.array(self.target_position) - np.array(current_feature_position)

        # کیمرہ کیلیبریشن کا استعمال کرتے ہوئے پکسل خامی کو ورلڈ کوآرڈینیٹس میں تبدیل کریں
        # سادگی کے لیے، ہم لکیری تقریب استعمال کریں گے
        world_error = pixel_error * self.gain

        # رفتار کمانڈز تیار کریں
        linear_vel = min(0.3, np.linalg.norm(world_error))  # لکیری رفتار کو محدود کریں
        angular_vel = np.arctan2(world_error[1], world_error[0]) * 0.5  # تناسب کنٹرول

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

## اعلی درجے کی کمپیوٹر وژن تکنیکیں

### ہم وقتی مقامیت اور نقشہ سازی (SLAM)

SLAM خودمختار روبوٹ نیویگیشن کے لیے اہم ہے:

```python
# مثال: سادہ بصری SLAM نفاذ
class VisualSLAM:
    def __init__(self):
        self.map_points = []
        self.camera_poses = []
        self.feature_trackers = {}
        self.current_pose = np.eye(4)  # 4x4 شناخت میٹرکس
        self.keyframe_threshold = 0.1

    def process_frame(self, image, timestamp):
        """SLAM کے لیے ایک فریم کو عمل کریں"""
        # خصوصیات کا پتہ لگائیں
        keypoints, descriptors = self.detect_features(image)

        # پچھلے فریمز سے خصوصیات ٹریک کریں
        tracked_features = self.track_features(keypoints, descriptors)

        # کیمرہ حرکت کا تخمینہ لگائیں
        motion_estimate = self.estimate_motion(tracked_features)

        # موجودہ پوزیشن اپ ڈیٹ کریں
        self.current_pose = self.update_pose(self.current_pose, motion_estimate)

        # اگر کافی حرکت ہوئی تو کی فریم شامل کریں
        if self.should_add_keyframe():
            self.add_keyframe(image, self.current_pose, tracked_features)

        # نقشہ کو بہتر بنائیں (بندل ایڈجسٹمنٹ یہاں ہوگا)
        self.optimize_map()

        return {
            'current_pose': self.current_pose,
            'map_points': len(self.map_points),
            'tracked_features': len(tracked_features)
        }

    def detect_features(self, image):
        """تصویر میں خصوصیات کا پتہ لگائیں"""
        detector = FeatureDetector()
        return detector.detect_features(image)

    def track_features(self, current_keypoints, current_descriptors):
        """فریموں کے درمیان خصوصیات ٹریک کریں"""
        # یہ خصوصیات ٹریکنگ الگورتھم نافذ کرے گا
        # جیسے KLT ٹریکر یا ڈسکرپٹر میچنگ
        return []

    def estimate_motion(self, tracked_features):
        """ٹریک کی گئی خصوصیات سے کیمرہ حرکت کا تخمینہ لگائیں"""
        # RANSAC کا استعمال کرتے ہوئے اساسی میٹرکس کا تخمینہ لگائیں
        # اور گھمائش اور ترجمہ میں ڈیکومپوز کریں
        return np.eye(4)

    def update_pose(self, current_pose, motion):
        """نئی حرکت کے ساتھ کیمرہ پوزیشن اپ ڈیٹ کریں"""
        return current_pose @ motion

    def should_add_keyframe(self):
        """یہ تعین کریں کہ کیا ایک نیا کی فریم شامل کیا جانا چاہیے"""
        # یہ چیک کریں کہ کافی حرکت ہوئی ہے
        return True  # سادہ

    def add_keyframe(self, image, pose, features):
        """نقشے میں ایک نیا کی فریم شامل کریں"""
        self.camera_poses.append(pose)
        # اگر خصوصیات مستحکم ہیں تو نقشے میں خصوصیات شامل کریں
        pass

    def optimize_map(self):
        """نقشہ اور پوزیشن کو بہتر بنائیں"""
        # یہ بندل ایڈجسٹمنٹ نافذ کرے گا
        # یا گراف کی اصلاح
        pass
```

### متعدد نظروں سے 3D تعمیر

روبوٹس اپنے ماحول کے 3D ماڈل تیار کر سکتے ہیں:

```python
# مثال: متعدد نظروں سے 3D تعمیر
class MultiViewReconstructor:
    def __init__(self):
        self.views = []
        self.point_cloud = []
        self.camera_poses = []

    def add_view(self, image, camera_pose):
        """تعمیر میں ایک نئی نظر شامل کریں"""
        features = self.extract_features(image)

        self.views.append({
            'image': image,
            'pose': camera_pose,
            'features': features
        })

    def extract_features(self, image):
        """تصویر سے خصوصیات نکالیں"""
        detector = FeatureDetector()
        keypoints, descriptors = detector.detect_features(image)

        return {
            'keypoints': keypoints,
            'descriptors': descriptors
        }

    def triangulate_points(self):
        """متعدد نظروں سے 3D پوائنٹس ٹرائی اینگولیٹ کریں"""
        # نظروں کے درمیان خصوصیات میچ کریں
        matches = self.match_features_across_views()

        # میچ کی گئی خصوصیات ٹرائی اینگولیٹ کریں
        for match in matches:
            point_3d = self.triangulate_point(match)
            if point_3d is not None:
                self.point_cloud.append(point_3d)

    def triangulate_point(self, feature_match):
        """خصوصیات کے میچ سے 3D پوائنٹ ٹرائی اینگولیٹ کریں"""
        # نظروں کے لیے کیمرہ میٹرکس حاصل کریں
        P1 = self.get_projection_matrix(self.camera_poses[feature_match['view1']])
        P2 = self.get_projection_matrix(self.camera_poses[feature_match['view2']])

        # DLT الگورتھم کا استعمال کرتے ہوئے ٹرائی اینگولیٹ کریں
        point_3d = cv2.triangulatePoints(
            P1, P2,
            feature_match['point1'],
            feature_match['point2']
        )

        # ہوموجینس سے کارٹیزین کوآرڈینیٹس میں تبدیل کریں
        point_3d = point_3d[:3] / point_3d[3]
        return point_3d

    def get_projection_matrix(self, pose):
        """کیمرہ پوزیشن سے پروجیکشن میٹرکس حاصل کریں"""
        # گھمائش، ترجمہ، اور اندرونی میٹرکس کو جوڑیں
        R = pose[:3, :3]
        t = pose[:3, 3]

        # معلوم اندرونی میٹرکس K کا فرض
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])  # مثال کا اندرونی میٹرکس

        # پروجیکشن میٹرکس P = K[R|t]
        P = K @ np.hstack((R, t.reshape(3, 1)))
        return P
```

## لیب: روبوٹک کمپیوٹر وژن نافذ کرنا

اس لیب میں، آپ روبوٹکس کے لیے ایک کمپیوٹر وژن سسٹم نافذ کریں گے:

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

        # پبلشرز
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.target_pub = self.create_publisher(Point, '/target_location', 10)
        self.status_pub = self.create_publisher(String, '/vision_status', 10)

        # سبسکرائبرز
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        # وژن کمپوننٹس
        self.cv_bridge = CvBridge()
        self.object_detector = RealTimeObjectDetector()
        self.visual_servoing = VisualServoingController()

        # لیب پیرامیٹر
        self.vision_mode = 'object_tracking'  # object_tracking, color_following, face_following
        self.target_object = 'person'  # ٹریک کرنے کے لیے چیز کی کلاس
        self.color_lower = np.array([0, 50, 50])  # رنگ ٹریکنگ کے لیے HSV کم از کم حد
        self.color_upper = np.array([10, 255, 255])  # رنگ ٹریکنگ کے لیے HSV زیادہ سے زیادہ حد

        # کنٹرول لوپ
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

        # حالت کے متغیرات
        self.latest_image = None
        self.tracked_object = None
        self.object_position = None

    def image_callback(self, msg):
        """آنے والی کیمرہ تصویروں کو سنبھالیں"""
        try:
            self.latest_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Image callback error: {e}')

    def control_loop(self):
        """بصری بیسڈ نیویگیشن کے لیے اصل کنٹرول لوپ"""
        if self.latest_image is None:
            return

        # موجودہ موڈ کی بنیاد پر تصویر کو عمل کریں
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

        # حالت شائع کریں
        status_msg = String()
        status_msg.data = f"Mode: {self.vision_mode}, Target: {self.tracked_object}, Position: {self.object_position}"
        self.status_pub.publish(status_msg)

    def process_object_tracking(self):
        """آبجیکٹ ٹریکنگ عمل کریں"""
        detections, _ = self.object_detector.detect_objects(self.latest_image)

        # ہدف کی چیز تلاش کریں
        for detection in detections:
            if detection['class'] == self.target_object and detection['confidence'] > 0:
                self.tr:
                self.tracked_object = detection['class']
                self.object_position = detection['center']

                # ہدف کی جگہ شائع کریں
                target_point = Point()
                target_point.x = float(self.object_position[0])
                target_point.y = float(self.object_position[1])
                target_point.z = detection['confidence']
                self.target_pub.publish(target_point)

                return

        # اگر کوئی ہدف نہیں ملا، ٹریکنگ صاف کریں
        self.tracked_object = None
        self.object_position = None

    def process_color_following(self):
        """رنگ بیسڈ چیز کے پیچھے چلنے کو عمل کریں"""
        # BGR سے HSV میں تبدیل کریں
        hsv = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2HSV)

        # ہدف رنگ کے لیے ماسک بنائیں
        mask = cv2.inRange(hsv, self.color_lower, self.color_upper)

        # کنٹور تلاش کریں
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # سب سے بڑا کنٹور تلاش کریں
            largest_contour = max(contours, key=cv2.contourArea)

            if cv2.contourArea(largest_contour) > 500:  # کم از کم رقبہ کی حد
                # مرکز کا حساب لگائیں
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    self.tracked_object = 'color_object'
                    self.object_position = [cx, cy]

                    # ہدف کی جگہ شائع کریں
                    target_point = Point()
                    target_point.x = float(cx)
                    target_point.y = float(cy)
                    target_point.z = 1.0  # رنگ کی شناخت کے لیے زیادہ یقین
                    self.target_pub.publish(target_point)
                    return

        # اگر کوئی رنگ کی چیز نہیں ملی
        self.tracked_object = None
        self.object_position = None

    def process_face_following(self):
        """چہرہ فالو کو عمل کریں (سادہ نفاذ)"""
        # چہرہ ڈیٹکشن کے لیے Haar کیسکیڈ کا استعمال کریں
        gray = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2GRAY)

        # ایک سادہ چہرہ ڈیٹیکٹر بنائیں (عمل میں، cv2.CascadeClassifier استعمال کریں)
        # اس مثال کے لیے، ہم ایک سادہ جلد کے رنگ کی شناخت استعمال کریں گے
        hsv = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2HSV)

        # HSV میں جلد کے رنگ کی حد
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])

        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # شور کو کم کرنے کے لیے مورفولوجی آپریشنز لاگو کریں
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # کنٹور تلاش کریں
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # سب سے بڑا کنٹور تلاش کریں (ممکنہ طور پر چہرہ)
            largest_contour = max(contours, key=cv2.contourArea)

            if cv2.contourArea(largest_contour) > 1000:  # چہرے کے لیے کم از کم رقبہ
                # مرکز کا حساب لگائیں
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    self.tracked_object = 'face'
                    self.object_position = [cx, cy]

                    # ہدف کی جگہ شائع کریں
                    target_point = Point()
                    target_point.x = float(cx)
                    target_point.y = float(cy)
                    target_point.z = 0.8  # درمیانی یقین
                    self.target_pub.publish(target_point)
                    return

        # اگر کوئی چہرہ نہیں ملا
        self.tracked_object = None
        self.object_position = None

    def generate_navigation_command(self):
        """چیز کی پوزیشن کی بنیاد پر نیویگیشن کمانڈ تیار کریں"""
        cmd = Twist()

        if self.object_position is None:
            # کوئی چیز نہیں ملی، رکیں
            return cmd

        # تصویر کا مرکز حاصل کریں
        image_center_x = self.latest_image.shape[1] / 2
        object_x = self.object_position[0]

        # خامی کا حساب لگائیں
        x_error = object_x - image_center_x

        # زاویہ کی رفتار کے لیے تناسب کنٹرول
        cmd.angular.z = -0.002 * x_error  # درست سمت کے لیے منفی

        # چیز مرکز میں ہونے پر (حد کے اندر) آگے بڑھیں
        center_threshold = 50  # پکسلز
        if abs(x_error) < center_threshold:
            cmd.linear.x = 0.2  # آگے بڑھیں
        else:
            cmd.linear.x = 0.05  # قریب آنے کے لیے آہستہ آگے

        # زاویہ کی رفتار کو محدود کریں
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

## مشق: اپنا کمپیوٹر وژن اطلاقہ ڈیزائن کریں

مندرجہ ذیل ڈیزائن چیلنج پر غور کریں:

1. کون سا مخصوص روبوٹک کام کمپیوٹر وژن سے فائدہ اٹھائے گا؟
2. کون سی اشیاء یا خصوصیات کا پتہ لگانا ضروری ہے؟
3. اس اطلاقے کے لیے حقیقی وقت کی پابندیاں کیا ہیں؟
4. آپ مختلف روشنی کی حالت کو کیسے سنبھالیں گے؟
5. آپ بصری فیڈ بیک کی بنیاد پر کون سی کنٹرول حکمت عمل استعمال کریں گے؟
6. آپ متحرک ماحول میں مضبوطی کیسے یقینی بنائیں گے؟

## خلاصہ

کمپیوٹر وژن روبوٹک ادراک کے لیے بنیادی ہے، جو روبوٹس کو اپنے ماحول کو سمجھنے اور اس کے ساتھ بات چیت کرنے کے قابل بناتا ہے۔ کلیدی تصورات شامل ہیں:

- **حقیقی وقت پروسیسنگ**: وژن سسٹم کو مسلسل کام کرنا چاہیے اور جلدی جواب دینا چاہیے
- **جسمانی ادراک**: وژن روبوٹ کی جسمانی اعمال کے ساتھ ضم ہے
- **خصوصیات کا پتہ لگانا**: ٹریکنگ اور نقشہ سازی کے لیے قابل اعتماد خصوصیات کی شناخت
- **اشیاء کا پتہ لگانا**: دلچسپ اشیاء کو پہچاننا اور مقام متعین کرنا
- **بصری اوڈومیٹری**: بصری معلومات سے حرکت کا تخمینہ لگانا
- **SLAM**: ایک ہی وقت میں نقشے بنانا اور مقامیت کرنا

کمپیوٹر وژن کا ROS2 کے ساتھ انضمام روبوٹکس کے لیے جامع ادراک کے سسٹم ترقی دینے کو فعال کرتا ہے۔ ان تصورات کو سمجھنا پیچیدہ ماحول میں ادراک اور نیویگیٹ کرنے والے روبوٹس ترقی دینے کے لیے ضروری ہے۔

اگلے سبق میں، ہم روبوٹ کنٹرول اور فیصلہ سازی پر خاص طور پر لاگو کردہ مشین لرننگ کی تکنیکوں کا جائزہ لیں گے۔