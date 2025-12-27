---
sidebar_position: 3
---

# کامیاب انسان نما روبوٹس کی مثالیں

## تعارف

یہ سبق جسمانی مصنوعی ذہانت کے اصولوں کے موثر نفاذ کا مظاہرہ کرنے والے حقیقی دنیا کے انسان نما روبوٹس کا جائزہ لیتا ہے۔ ہم ان کے ڈیزائن، کارکردگی، اور ادراک-عمل لوپ کا تجزیہ کریں گے جو ان کی نمایاں صلاحیات کو فعال کرتے ہیں۔

## ہونڈا ASIMO: انسان نما روبوٹکس کا آغاز

### جائزہ
ASIMO (اعلی درجے کا اقدام میں نوآورانہ موبائلیٹی) ہونڈا کا نمایاں انسان نما روبوٹ تھا، جو دوہری ٹانگوں والی چلنے اور انسان-روبوٹ بات چیت میں دو دہائیوں سے زائد کی ترقی کی نمائندگی کرتا ہے۔

### کلیدی ٹیکنالوجیز

#### موافق چلنے کا سسٹم
```python
# مثال: ASIMO جیسا موافق چلنے والا کنٹرولر
class AdaptiveWalkingController:
    def __init__(self):
        self.step_length = 0.3  # میٹر
        self.step_height = 0.05  # میٹر
        self.walking_speed = 0.5  # میٹر/سیکنڈ
        self.balance_threshold = 0.1  # قابل قبول جھکاؤ

    def calculate_step_pattern(self, terrain_data):
        """زمین کی بنیاد پر موافق قدم کا نمونہ کا حساب لگائیں"""
        step_pattern = []

        for i in range(10):  # 10 قدم آگے منصوبہ بند کریں
            step = {
                'position': self.calculate_next_step_position(i),
                'height': self.calculate_step_height(terrain_data, i),
                'timing': self.calculate_step_timing(i)
            }
            step_pattern.append(step)

        return step_pattern

    def calculate_next_step_position(self, step_index):
        """اگلے قدم کو رکھنے کا مقام کا حساب لگائیں"""
        # سادہ حساب
        base_position = self.current_position
        step_offset = self.step_length * step_index
        return base_position + step_offset

    def calculate_step_height(self, terrain_data, step_index):
        """زمین کی بنیاد پر قدم کی اونچائی کو ایڈجسٹ کریں"""
        # زمین کے ڈیٹا کو آگے دیکھیں
        terrain_height = terrain_data.get_height_at_position(
            self.calculate_next_step_position(step_index)
        )
        return self.step_height + terrain_height

    def calculate_step_timing(self, step_index):
        """ہر قدم کے لیے ٹائمنگ کا حساب لگائیں"""
        # استحکام کی ضروریات کے مطابق ٹائمنگ کو ایڈجسٹ کریں
        return 0.8 + (step_index % 2) * 0.1  # استحکام کے لیے متبادل ٹائمنگ
```

#### ذہین رویہ سسٹم
```python
# مثال: ASIMO کا ذہین رویہ سسٹم
class IntelligentBehaviorSystem:
    def __init__(self):
        self.behaviors = {
            'greeting': self.greeting_behavior,
            'walking': self.walking_behavior,
            'object_interaction': self.object_interaction_behavior,
            'human_following': self.human_following_behavior
        }

    def greeting_behavior(self, detected_person):
        """سلام کی ترتیب انجام دیں"""
        actions = [
            {'type': 'head_turn', 'target': detected_person['position']},
            {'type': 'wave', 'arm': 'right'},
            {'type': 'speak', 'message': 'Hello, nice to meet you!'}
        ]
        return actions

    def walking_behavior(self, destination):
        """رکاوٹ سے بچاؤ کے ساتھ منزل کی طرف نیویگیٹ کریں"""
        path = self.plan_path(destination)
        walking_pattern = self.generate_walking_pattern(path)
        return walking_pattern

    def object_interaction_behavior(self, object_info):
        """دریافت شدہ اشیاء کے ساتھ بات چیت کریں"""
        if object_info['type'] == 'ball':
            return [{'type': 'kick', 'direction': object_info['direction']}]
        elif object_info['type'] == 'cup':
            return [{'type': 'grasp', 'object': object_info}]
        else:
            return []

    def human_following_behavior(self, person_position):
        """محفوظ فاصلے پر انسان کا پیچھا کریں"""
        follow_distance = 1.0  # میٹر
        target_position = self.calculate_follow_position(person_position, follow_distance)
        return self.walking_behavior(target_position)
```

## بوسٹن ڈائینامکس Atlas: متحرک انسان نما صلاحیات

### جائزہ
Atlas متحرک انسان نما روبوٹکس کی انتہا کی نمائندگی کرتا ہے، جو دوڑنے، کودنے، اور پیچیدہ کرتب بازی کے کام انجام دینے کا قابل ہے۔

### متحرک کنٹرول سسٹم
```python
# مثال: Atlas جیسا متحرک کنٹرول سسٹم
import numpy as np
from scipy import signal

class DynamicControlSystem:
    def __init__(self):
        self.mass = 80  # kg
        self.com_height = 0.8  # میٹر
        self.gravity = 9.81
        self.control_frequency = 1000  # Hz

        # حالت کی تخمینہ کاری
        self.com_position = np.zeros(3)
        self.com_velocity = np.zeros(3)
        self.com_acceleration = np.zeros(3)

        # مطلوبہ ٹریجکٹریز
        self.desired_com_trajectory = []
        self.desired_joint_trajectory = []

    def compute_force_control(self, desired_com_state, current_com_state, dt):
        """متحرک حرکت کے لیے ضروری قوتوں کا حساب لگائیں"""
        # توازن کے لیے لکیری الٹا پینڈولم ماڈل
        com_error = desired_com_state['position'] - current_com_state['position']
        vel_error = desired_com_state['velocity'] - current_com_state['velocity']

        # فیڈ بیک کنٹرول گینز
        Kp = 100.0  # تناسب کا گیان
        Kd = 20.0   # ڈیریویٹیو گیان

        # اصلاحی قوت کا حساب لگائیں
        corrective_force = Kp * com_error + Kd * vel_error

        # گریویٹی کی تلافی شامل کریں
        gravity_force = np.array([0, 0, self.mass * self.gravity])

        total_force = corrective_force + gravity_force
        return total_force

    def compute_balance_control(self, support_polygon, com_position):
        """سپورٹ پولی گون کی بنیاد پر توازن کنٹرول کا حساب لگائیں"""
        # مطلوبہ CoM پروجیکشن کے قریب ترین نکتہ تلاش کریں
        com_projection = com_position[:2]  # X، Y کوآرڈینیٹس
        nearest_point = self.find_nearest_point_in_polygon(
            support_polygon, com_projection
        )

        # توازن کی اصلاح کا حساب لگائیں
        balance_correction = nearest_point - com_projection
        return balance_correction

    def find_nearest_point_in_polygon(self, polygon, point):
        """پولی گون میں دیے گئے نکتے کے قریب ترین نکتہ تلاش کریں"""
        # سادہ نفاذ
        # عمل میں، یہ تعداداتی جیومیٹری الگورتھم استعمال کرے گا
        return polygon[0]  # جگہ کا نمونہ
```

## سافٹ بینک روبوٹکس Pepper: سماجی انسان نما روبوٹ

### جائزہ
Pepper سماجی بات چیت اور جذباتی ذہانت پر مرکوز ہے، جو اسے سروس اطلاقات کے لیے مناسب بناتا ہے۔

### سماجی بات چیت سسٹم
```python
# مثال: Pepper جیسا سماجی بات چیت سسٹم
class SocialInteractionSystem:
    def __init__(self):
        self.emotion_detector = EmotionDetector()
        self.speech_recognizer = SpeechRecognizer()
        self.natural_language_processor = NaturalLanguageProcessor()
        self.behavior_selector = BehaviorSelector()

    def process_human_interaction(self, sensor_data):
        """انسانی بات چیت کو عمل کریں اور مناسب جواب تیار کریں"""
        # چہرے کے اظہار سے جذبات کا پتہ لگائیں
        emotions = self.emotion_detector.analyze_facial_expressions(
            sensor_data['face_image']
        )

        # تقریر کو پہچانیں
        speech_text = self.speech_recognizer.recognize_speech(
            sensor_data['audio']
        )

        # قدرتی زبان کو عمل کریں
        intent = self.natural_language_processor.extract_intent(speech_text)

        # مناسب رویہ منتخب کریں
        behavior = self.behavior_selector.select_behavior(
            emotions, intent, sensor_data['context']
        )

        return behavior

class EmotionDetector:
    def analyze_facial_expressions(self, face_image):
        """چہرے کے اظہار سے جذبات کا پتہ لگائیں"""
        # جذبات کو طبقہ بند کرنے کے لیے گہرائی سیکھنے کا ماڈل استعمال کریں
        emotions = {
            'happy': 0.7,
            'sad': 0.1,
            'angry': 0.05,
            'surprised': 0.15
        }
        return emotions

class SpeechRecognizer:
    def recognize_speech(self, audio_data):
        """تقریر کو متن میں تبدیل کریں"""
        # تقریر-سے-متن API استعمال کریں
        return "Hello, how can I help you?"

class NaturalLanguageProcessor:
    def extract_intent(self, text):
        """قدرتی زبان سے ارادہ نکالیں"""
        # صارف کا ارادہ سمجھنے کے لیے NLP تکنیک استعمال کریں
        if "help" in text.lower():
            return "request_assistance"
        elif "weather" in text.lower():
            return "request_weather"
        else:
            return "general_conversation"

class BehaviorSelector:
    def select_behavior(self, emotions, intent, context):
        """مناسب سماجی رویہ منتخب کریں"""
        if intent == "request_assistance":
            return self.assist_behavior()
        elif emotions['happy'] > 0.5:
            return self.celebrate_behavior()
        else:
            return self.neutral_behavior()

    def assist_behavior(self):
        return {'action': 'lean_forward', 'gesture': 'open_hands', 'speech': 'How can I assist you?'}

    def celebrate_behavior(self):
        return {'action': 'raise_arms', 'gesture': 'thumbs_up', 'speech': 'Great!'}

    def neutral_behavior(self):
        return {'action': 'maintain_posture', 'gesture': 'nod', 'speech': 'I understand.'}
```

## ROS2 نفاذ: انسان نما روبوٹ کنٹرول آرکیٹیکچر

یہاں ایک جامع ROS2 نفاذ ہے جو دکھاتا ہے کہ یہ تصورات کیسے ضم ہوتے ہیں:

```python
# humanoid_robot_control.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, LaserScan, Image
from geometry_msgs.msg import Twist, Pose, Point
from std_msgs.msg import String, Float32
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
import numpy as np
import cv2
import math
from collections import deque

class HumanoidRobotControl(Node):
    def __init__(self):
        super().__init__('humanoid_robot_control')

        # پبلشرز
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.base_cmd_pub = self.create_publisher(Twist, '/base_velocity_commands', 10)
        self.head_cmd_pub = self.create_publisher(Point, '/head_look_at', 10)
        self.speech_pub = self.create_publisher(String, '/tts_input', 10)
        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        # سبسکرائبرز
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10
        )
        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10
        )
        self.audio_sub = self.create_subscription(
            String, '/speech_to_text', self.audio_callback, 10
        )

        # سسٹم کمپوننٹس
        self.cv_bridge = CvBridge()
        self.social_system = SocialInteractionSystem()
        self.walking_controller = AdaptiveWalkingController()
        self.dynamic_controller = DynamicControlSystem()
        self.behavior_selector = BehaviorSelector()

        # ڈیٹا اسٹوریج
        self.joint_states = None
        self.imu_data = None
        self.laser_data = None
        self.camera_data = None
        self.audio_data = None

        # روبوٹ کی حالت
        self.robot_state = {
            'position': np.array([0.0, 0.0, 0.0]),
            'orientation': 0.0,
            'balance': 1.0,  # 1.0 = مکمل طور پر متوازن، 0.0 = گر گیا
            'battery_level': 1.0,
            'interaction_mode': 'idle'
        }

        # کنٹرول لوپ
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100 Hz

        # سیکھنے کے لیے رویہ کی تاریخ
        self.behavior_history = deque(maxlen=100)

    def joint_state_callback(self, msg):
        """جوائنٹ حالت کی تازہ کاریوں کو سنبھالیں"""
        self.joint_states = msg
        self.update_robot_position_from_joints(msg)

    def imu_callback(self, msg):
        """توازن کنٹرول کے لیے IMU ڈیٹا کو سنبھالیں"""
        self.imu_data = msg
        self.update_balance_from_imu(msg)

    def laser_callback(self, msg):
        """نیویگیشن کے لیے لیزر اسکین کو سنبھالیں"""
        self.laser_data = msg

    def camera_callback(self, msg):
        """ادراک کے لیے کیمرہ ڈیٹا کو سنبھالیں"""
        try:
            self.camera_data = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Camera callback error: {e}')

    def audio_callback(self, msg):
        """بات چیت کے لیے آڈیو ان پٹ کو سنبھالیں"""
        self.audio_data = msg.data

    def control_loop(self):
        """ادراک-عمل سائیکل کو نافذ کرنے والا اصل کنٹرول لوپ"""
        # یقینی بنائیں کہ ہمارے پاس ضروری سینسر ڈیٹا ہے
        if not all([self.joint_states, self.imu_data]):
            return

        # 1. ادراک کا مرحلہ
        perceptual_state = self.process_perception()

        # 2. سوچ بچار کا مرحلہ
        cognitive_state = self.process_cognition(perceptual_state)

        # 3. رویہ منتخب کرنے کا مرحلہ
        behavior = self.select_behavior(cognitive_state)

        # 4. عمل پیدا کرنے کا مرحلہ
        commands = self.generate_commands(behavior)

        # 5. انجام دینے کا مرحلہ
        self.execute_commands(commands)

        # 6. حالت کو اپ ڈیٹ کرنے کا مرحلہ
        self.update_robot_state(commands)

        # 7. حالت کی رپورٹنگ
        self.publish_status()

    def process_perception(self):
        """تمام سینسر ڈیٹا کو ادراک کی حالت میں عمل کریں"""
        perceptual_state = {
            'environment_map': self.create_environment_map(),
            'human_detection': self.detect_humans(),
            'obstacle_distances': self.analyze_obstacles(),
            'balance_state': self.get_balance_state(),
            'battery_status': self.robot_state['battery_level']
        }

        return perceptual_state

    def create_environment_map(self):
        """سینسرز سے ماحولیاتی نمائندگی بنائیں"""
        if self.laser_data:
            # لیزر ڈیٹا سے سادہ آکوپنسی گرڈ بنائیں
            angles = np.linspace(
                self.laser_data.angle_min,
                self.laser_data.angle_max,
                len(self.laser_data.ranges)
            )
            ranges = np.array(self.laser_data.ranges)

            # جائز رینج فلٹر کریں
            valid_mask = (ranges > 0) & (ranges < self.laser_data.range_max)
            valid_angles = angles[valid_mask]
            valid_ranges = ranges[valid_mask]

            # کارٹیزین کوآرڈینیٹس میں تبدیل کریں
            x_coords = valid_ranges * np.cos(valid_angles)
            y_coords = valid_ranges * np.sin(valid_angles)

            return np.column_stack([x_coords, y_coords])

        return np.array([])

    def detect_humans(self):
        """کیمرہ امیج میں انسانوں کا پتہ لگائیں"""
        if self.camera_data is not None:
            # سادہ HOG-بیسڈ انسان کا پتہ لگانا (عمل میں، گہرائی سیکھنے استعمال کریں)
            gray = cv2.cvtColor(self.camera_data, cv2.COLOR_BGR2GRAY)

            # انسان کا پتہ لگانے کے لیے HOG ڈیسکرپٹر استعمال کریں
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

            boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8))

            humans = []
            for (x, y, w, h) in boxes:
                humans.append({
                    'bbox': (x, y, w, h),
                    'center': (x + w//2, y + h//2),
                    'confidence': float(weights[len(humans)])
                })

            return humans

        return []

    def analyze_obstacles(self):
        """لیزر اسکینر سے رکاوٹ ڈیٹا کا تجزیہ کریں"""
        if self.laser_data:
            ranges = np.array(self.laser_data.ranges)
            valid_ranges = ranges[(ranges > 0) & (ranges < self.laser_data.range_max)]

            if len(valid_ranges) > 0:
                return {
                    'closest': min(valid_ranges),
                    'front_clear': all(r > 1.0 for r in self.laser_data.ranges[300:600] if r > 0),
                    'left_clear': all(r > 0.8 for r in self.laser_data.ranges[0:180] if r > 0),
                    'right_clear': all(r > 0.8 for r in self.laser_data.ranges[540:720] if r > 0),
                    'density': len(valid_ranges) / len(self.laser_data.ranges)  # رکاوٹ کی کثافت
                }

        return {
            'closest': float('inf'),
            'front_clear': True,
            'left_clear': True,
            'right_clear': True,
            'density': 0.0
        }

    def get_balance_state(self):
        """IMU سے موجودہ توازن کی حالت حاصل کریں"""
        if self.imu_data:
            # کووٹرین سے جہت نکالیں
            orientation = self.imu_data.orientation
            # سادہ توازن کا حساب (عمل میں، مناسب کووٹرین ریاضی استعمال کریں گے)
            tilt_magnitude = abs(orientation.x) + abs(orientation.y)
            balance_score = max(0.0, 1.0 - tilt_magnitude * 5)  # [0,1] تک معمول کریں
            return balance_score

        return 1.0  # ڈیفالٹ طور پر متوازن

    def process_cognition(self, perceptual_state):
        """فیصلہ کرنے کے لیے ادراک کی حالت کو عمل کریں"""
        cognitive_state = {
            'threat_level': self.assess_threats(perceptual_state),
            'social_opportunities': self.assess_social_opportunities(perceptual_state),
            'navigation_state': self.assess_navigation_state(perceptual_state),
            'battery_considerations': perceptual_state['battery_status'] < 0.2
        }

        return cognitive_state

    def assess_threats(self, perceptual_state):
        """روبوٹ کی سلامتی کے لیے ممکنہ خطرات کا جائزہ لیں"""
        threat_level = 0.0

        # فوری ٹکر کی جانچ کریں
        if perceptual_state['obstacle_distances']['closest'] < 0.3:
            threat_level += 0.8

        # توازن کے مسائل کی جانچ کریں
        if perceptual_state['balance_state'] < 0.3:
            threat_level += 0.9

        # کم بیٹری کی جانچ کریں
        if perceptual_state['battery_status'] < 0.1:
            threat_level += 0.2

        return min(threat_level, 1.0)

    def assess_social_opportunities(self, perceptual_state):
        """سماجی بات چیت کے مواقعوں کا جائزہ لیں"""
        social_score = 0.0

        # دریافت شدہ انسانوں کی گنتی کریں
        human_count = len(perceptual_state['human_detection'])
        social_score += min(human_count * 0.3, 0.5)  # انسانوں کے لیے زیادہ سے زیادہ 0.5

        # جانچ کریں کہ کیا انسان بات چیت کے فاصلے میں ہیں
        if human_count > 0 and perceptual_state['obstacle_distances']['closest'] > 1.5:
            social_score += 0.3

        return min(social_score, 1.0)

    def assess_navigation_state(self, perceptual_state):
        """موجودہ نیویگیشن کی صورتحال کا جائزہ لیں"""
        return {
            'path_clear': perceptual_state['obstacle_distances']['front_clear'],
            'obstacle_density': perceptual_state['obstacle_distances']['density'],
            'safe_to_move': (perceptual_state['balance_state'] > 0.7 and
                           perceptual_state['obstacle_distances']['closest'] > 0.5)
        }

    def select_behavior(self, cognitive_state):
        """سوچ بچار کی حالت کی بنیاد پر مناسب رویہ منتخب کریں"""
        # رویہ ترجیح سلسلہ
        if cognitive_state['threat_level'] > 0.7:
            return self.emergency_behavior(cognitive_state)
        elif cognitive_state['social_opportunities'] > 0.5 and self.robot_state['interaction_mode'] != 'avoiding_interaction':
            return self.social_behavior(cognitive_state)
        elif cognitive_state['navigation_state']['safe_to_move']:
            return self.navigation_behavior(cognitive_state)
        else:
            return self.waiting_behavior(cognitive_state)

    def emergency_behavior(self, cognitive_state):
        """زیادہ ترجیح والے ہنگامی رویہ"""
        self.get_logger().warn('EMERGENCY: Activating safety protocol')
        self.robot_state['interaction_mode'] = 'emergency'

        return {
            'type': 'emergency_stop',
            'action': 'stop_all_motors',
            'priority': 'critical',
            'recovery_plan': 'assess_damage_and_recover_balance'
        }

    def social_behavior(self, cognitive_state):
        """سماجی بات چیت کا رویہ"""
        self.robot_state['interaction_mode'] = 'social'

        return {
            'type': 'social_interaction',
            'action': 'approach_human_and_greet',
            'priority': 'high',
            'social_elements': ['head_turn', 'gesture', 'speech']
        }

    def navigation_behavior(self, cognitive_state):
        """نیویگیشن رویہ"""
        self.robot_state['interaction_mode'] = 'navigating'

        nav_state = cognitive_state['navigation_state']
        if nav_state['path_clear']:
            return {
                'type': 'navigation',
                'action': 'move_forward',
                'priority': 'medium',
                'speed': 'normal'
            }
        else:
            return {
                'type': 'navigation',
                'action': 'obstacle_avoidance',
                'priority': 'medium',
                'speed': 'careful'
            }

    def waiting_behavior(self, cognitive_state):
        """انتظار/بے کاری رویہ"""
        self.robot_state['interaction_mode'] = 'idle'

        return {
            'type': 'idle',
            'action': 'maintain_balance_and_scan',
            'priority': 'low',
            'activity': 'passive_monitoring'
        }

    def generate_commands(self, behavior):
        """اعلی درجے کے رویہ سے کم درجے کے کمانڈز پیدا کریں"""
        commands = {
            'joint_commands': JointState(),
            'base_velocity': Twist(),
            'head_commands': Point(),
            'speech_commands': String()
        }

        if behavior['type'] == 'emergency_stop':
            # تمام حرکت کو روکیں
            commands['base_velocity'] = Twist()
            commands['joint_commands'].position = list(self.joint_states.position)  # پوزیشن پکڑیں

        elif behavior['type'] == 'social_interaction':
            # انسان کی طرف بڑھیں اور بات چیت کریں
            commands['base_velocity'].linear.x = 0.2  # آہستہ آگے بڑھیں
            commands['head_commands'].x = 0.0  # آگے دیکھیں
            commands['head_commands'].y = 0.0
            commands['speech_commands'].data = "Hello! How can I help you today?"

        elif behavior['type'] == 'navigation':
            if behavior['action'] == 'move_forward':
                commands['base_velocity'].linear.x = 0.3 if behavior['speed'] == 'normal' else 0.1
            elif behavior['action'] == 'obstacle_avoidance':
                # سادہ رکاوٹ سے بچنا
                if self.laser_data:
                    left_clear = all(r > 0.8 for r in self.laser_data.ranges[0:180] if r > 0)
                    right_clear = all(r > 0.8 for r in self.laser_data.ranges[540:720] if r > 0)

                    if left_clear:
                        commands['base_velocity'].angular.z = 0.3  # بائیں مڑیں
                    elif right_clear:
                        commands['base_velocity'].angular.z = -0.3  # دائیں مڑیں
                    else:
                        commands['base_velocity'].linear.x = 0.0  # رکیں

        elif behavior['type'] == 'idle':
            # پوزیشن برقرار رکھیں اور ماحول اسکین کریں
            commands['base_velocity'] = Twist()  # کوئی حرکت نہیں
            commands['head_commands'].x = 0.1  # معمولی اسکیننگ حرکت
            commands['head_commands'].y = 0.0

        return commands

    def execute_commands(self, commands):
        """پیدا کردہ کمانڈز انجام دیں"""
        # جوائنٹ کمانڈز شائع کریں
        if commands['joint_commands'].position:
            self.joint_cmd_pub.publish(commands['joint_commands'])

        # بیس رفتار کے کمانڈز شائع کریں
        self.base_cmd_pub.publish(commands['base_velocity'])

        # سر کے کمانڈز شائع کریں
        self.head_cmd_pub.publish(commands['head_commands'])

        # تقریر کے کمانڈز شائع کریں
        if commands['speech_commands'].data:
            self.speech_pub.publish(commands['speech_commands'])

    def update_robot_state(self, commands):
        """کمانڈز اور سینسرز کی بنیاد پر اندرونی روبوٹ کی حالت کو اپ ڈیٹ کریں"""
        # بیس رفتار کی بنیاد پر پوزیشن کو اپ ڈیٹ کریں
        dt = 0.01  # کنٹرول لوپ ٹائم سٹیپ
        vel = commands['base_velocity']

        # پوزیشن کو اپ ڈیٹ کریں (سادہ انٹیگریشن)
        self.robot_state['position'][0] += vel.linear.x * dt
        self.robot_state['position'][1] += vel.linear.y * dt
        self.robot_state['orientation'] += vel.angular.z * dt

        # IMU سے توازن کو اپ ڈیٹ کریں
        self.robot_state['balance'] = self.get_balance_state()

        # بیٹری کو اپ ڈیٹ کریں (سیمیولیٹڈ ڈسچارج)
        self.robot_state['battery_level'] = max(0.0, self.robot_state['battery_level'] - 0.0001)

    def update_robot_position_from_joints(self, joint_state):
        """جوائنٹس سے روبوٹ کی پوزیشن کو اپ ڈیٹ کریں"""
        # عمل میں، یہ فارورڈ کنیمیٹکس استعمال کرے گا
        # اس مثال کے لیے، ہم صرف جوائنٹ پوزیشنز کو محفوظ کریں گے
        pass

    def update_balance_from_imu(self, imu_msg):
        """IMU ڈیٹا سے توازن کی حالت کو اپ ڈیٹ کریں"""
        # توازن کا تعین کرنے کے لیے IMU ڈیٹا کو عمل کریں
        # عمل میں، یہ مناسب کووٹرین ریاضی شامل کرے گا
        pass

    def publish_status(self):
        """روبوٹ کی حالت شائع کریں"""
        status_msg = String()
        status_msg.data = f"Mode: {self.robot_state['interaction_mode']}, " \
                         f"Balance: {self.robot_state['balance']:.2f}, " \
                         f"Battery: {self.robot_state['battery_level']:.2f}"
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    robot_control = HumanoidRobotControl()

    try:
        rclpy.spin(robot_control)
    except KeyboardInterrupt:
        pass
    finally:
        robot_control.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## انسان نما روبوٹ کے نقطہ نظر کا موازنہ

### ہونڈا ASIMO بمقابلہ بوسٹن ڈائینامکس Atlas بمقابلہ سافٹ بینک Pepper

| خصوصیت | ASIMO | Atlas | Pepper |
|---------|-------|-------|--------|
| **اہم توجہ** | انسانی بات چیت، چلنا | متحرک حرکت | سماجی بات چیت |
| **چلنے کی صلاحیت** | مستحکم دو پائوں والی چلنا | متحرک دوڑنا/کودنا | پہیوں والی بنیاد |
| **کنٹرول سسٹم** | پیش گوئی والے کنٹرول | حقیقی وقت میں متحرک کنٹرول | رویہ-بیسڈ |
| **سینسرز** | کیمرے، پوزیشن سینسرز | کیمرے، IMU، لیڈار | کیمرے، مائیکروفون، چھونے کے سینسرز |
| **ایپلی کیشنز** | گائیڈ، اسسٹنٹ | تحقیق، مخصوص کام | سروس، ہم منصب |
| **اہم نوآوری** | موافق چلنا | متحرک توازن | سماجی مصنوعی ذہانت |

## حقیقی دنیا کے انسان نما روبوٹس سے سبق

### 1. تخصص کی اہمیت
ہر کامیاب انسان نما روبوٹ نے ہر چیز میں بہترین ہونے کی کوشش کرنے کے بجائے مخصوص صلاحیات پر توجہ مرکوز کی:

```python
# مثال: مخصوص کنٹرول موڈز
class SpecializedControlModes:
    def __init__(self):
        self.modes = {
            'stable_locomotion': StableLocomotionMode(),
            'dynamic_movement': DynamicMovementMode(),
            'social_interaction': SocialInteractionMode(),
            'manipulation': ManipulationMode()
        }
        self.current_mode = 'stable_locomotion'

    def switch_mode(self, new_mode):
        """مخصوص کنٹرول موڈز کے درمیان سوئچ کریں"""
        if new_mode in self.modes:
            self.modes[self.current_mode].deactivate()
            self.current_mode = new_mode
            self.modes[self.current_mode].activate()

    def execute_current_mode(self, sensor_data):
        """موجودہ مخصوص موڈ انجام دیں"""
        return self.modes[self.current_mode].execute(sensor_data)
```

### 2. متعدد سسٹم کا انضمام
کامیاب انسان نما روبوٹس متعدد پیچیدہ سسٹم کو ضم کرتے ہیں:

```python
# مثال: سسٹم انضمام فریم ورک
class SystemIntegrationFramework:
    def __init__(self):
        self.perception_system = PerceptionSystem()
        self.cognition_system = CognitionSystem()
        self.action_system = ActionSystem()
        self.learning_system = LearningSystem()
        self.safety_system = SafetySystem()

    def integrated_cycle(self, sensor_data):
        """انضمام شدہ ادراک-سوچ بچار-عمل سائیکل انجام دیں"""
        # پہلے سلامتی کی جانچ
        if not self.safety_system.is_safe_to_proceed(sensor_data):
            return self.safety_system.emergency_protocol()

        # ادراک
        perceptual_data = self.perception_system.process(sensor_data)

        # سوچ بچار
        cognitive_output = self.cognition_system.process(perceptual_data)

        # سیکھنے کی انضمام کے ساتھ عمل منتخب کریں
        action = self.action_system.select_action(cognitive_output)
        action = self.learning_system.adapt_action(action, perceptual_data)

        # عمل انجام دیں
        result = self.action_system.execute(action)

        # سیکھنے کا سسٹم اپ ڈیٹ کریں
        self.learning_system.update(action, result)

        return result
```

### 3. تدریجی صلاحیت کی ترقی
انسان نما روبوٹس عام طور پر صلاحیات کو مرحلہ وار ترقی دیتے ہیں:

```python
# مثال: صلاحیت کی ترقی فریم ورک
class CapabilityDevelopmentFramework:
    def __init__(self):
        self.capabilities = {
            'basic_balance': BasicBalanceCapability(),
            'simple_locomotion': SimpleLocomotionCapability(),
            'object_interaction': ObjectInteractionCapability(),
            'complex_navigation': ComplexNavigationCapability(),
            'social_interaction': SocialInteractionCapability()
        }

        # انحصار گراف کی وضاحت کریں
        self.dependencies = {
            'simple_locomotion': ['basic_balance'],
            'object_interaction': ['basic_balance'],
            'complex_navigation': ['simple_locomotion'],
            'social_interaction': ['basic_balance', 'simple_locomotion']
        }

    def develop_capability(self, capability_name):
        """انحصاروں کے مطابق صلاحیت کو ترقی دیں"""
        prerequisites = self.dependencies.get(capability_name, [])

        for prereq in prerequisites:
            if not self.capabilities[prereq].is_developed():
                self.develop_capability(prereq)

        # اب درخواست کردہ صلاحیت کو ترقی دیں
        self.capabilities[capability_name].develop()
```

## لیب: انسان نما روبوٹ کے رویوں کا تجزیہ

اس لیب میں، آپ حقیقی انسان نما روبوٹس سے متاثر کردہ رویوں کا تجزیہ اور نفاذ کریں گے:

```python
# lab_humanoid_analysis.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import String, Bool
import numpy as np

class HumanoidAnalysisLab(Node):
    def __init__(self):
        super().__init__('humanoid_analysis_lab')

        # پبلشرز
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/lab_status', 10)
        self.behavior_pub = self.create_publisher(String, '/selected_behavior', 10)

        # سبسکرائبرز
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        # ڈیٹا اسٹوریج
        self.joint_data = None
        self.imu_data = None
        self.scan_data = None

        # لیب پیرامیٹر
        self.analysis_mode = 'asimo'  # asimo, atlas, pepper
        self.behavior_state = 'idle'
        self.balance_threshold = 0.2

        # کنٹرول لوپ
        self.control_timer = self.create_timer(0.05, self.control_loop)

    def joint_callback(self, msg):
        """جوائنٹ حالت ڈیٹا کو سنبھالیں"""
        self.joint_data = msg

    def imu_callback(self, msg):
        """IMU ڈیٹا کو سنبھالیں"""
        self.imu_data = msg

    def scan_callback(self, msg):
        """لیزر اسکین ڈیٹا کو سنبھالیں"""
        self.scan_data = msg

    def control_loop(self):
        """مختلف انسان نما نقطہ نظر کا تجزیہ کرنے والا اصل کنٹرول لوپ"""
        if not all([self.joint_data, self.imu_data, self.scan_data]):
            return

        # منتخب نقطہ نظر کے مطابق تجزیہ کریں
        if self.analysis_mode == 'asimo':
            behavior = self.asimo_approach()
        elif self.analysis_mode == 'atlas':
            behavior = self.atlas_approach()
        elif self.analysis_mode == 'pepper':
            behavior = self.pepper_approach()
        else:
            behavior = self.default_approach()

        # رویہ انجام دیں
        command = self.behavior_to_command(behavior)
        self.cmd_pub.publish(command)

        # اپ ڈیٹ اور حالت شائع کریں
        self.update_behavior_state(behavior)
        self.behavior_pub.publish(String(data=behavior['type']))
        self.status_pub.publish(
            String(data=f"Mode: {self.analysis_mode}, Behavior: {behavior['type']}")
        )

    def asimo_approach(self):
        """ASIMO متاثر نقطہ نظر: مستحکم، قابل پیش گوئی رویہ"""
        # رکاوٹ سے بچاؤ کے ساتھ مستحکم چلنے پر توجہ دیں
        if self.is_unbalanced():
            return {'type': 'balance_correction', 'priority': 'critical'}
        elif self.obstacle_ahead():
            return {'type': 'obstacle_avoidance', 'priority': 'high'}
        else:
            return {'type': 'steady_locomotion', 'priority': 'normal'}

    def atlas_approach(self):
        """Atlas متاثر نقطہ نظر: متحرک، زیادہ کارکردگی"""
        # متحرک حرکت کی صلاحیات پر توجہ دیں
        if self.is_unbalanced():
            return {'type': 'dynamic_recovery', 'priority': 'critical'}
        elif self.path_is_clear():
            return {'type': 'dynamic_locomotion', 'priority': 'high'}
        else:
            return {'type': 'careful_navigation', 'priority': 'normal'}

    def pepper_approach(self):
        """Pepper متاثر نقطہ نظر: سماجی، تعاملی"""
        # انسانی بات چیت اور مشغولیت پر توجہ دیں
        if self.human_detected():
            return {'type': 'social_interaction', 'priority': 'high'}
        elif self.is_safe():
            return {'type': 'approachable_posture', 'priority': 'normal'}
        else:
            return {'type': 'cautious_behavior', 'priority': 'high'}

    def default_approach(self):
        """ڈیفالٹ نقطہ نظر: بنیادی کارکردگی"""
        return {'type': 'basic_operation', 'priority': 'normal'}

    def is_unbalanced(self):
        """IMU ڈیٹا کا استعمال کرتے ہوئے چیک کریں کہ کیا روبوٹ غیر متوازن ہے"""
        if self.imu_data:
            # سادہ توازن چیک
            orientation = self.imu_data.orientation
            tilt = abs(orientation.x) + abs(orientation.y)
            return tilt > self.balance_threshold
        return False

    def obstacle_ahead(self):
        """لیزر ڈیٹا کا استعمال کرتے ہوئے سامنے کی رکاوٹوں کی جانچ کریں"""
        if self.scan_data:
            front_ranges = self.scan_data.ranges[300:600]
            return any(r < 0.8 for r in front_ranges if r > 0)
        return False

    def path_is_clear(self):
        """متحرک حرکت کے لیے راستہ صاف ہے یا نہیں چیک کریں"""
        if self.scan_data:
            front_ranges = self.scan_data.ranges[300:600]
            return all(r > 1.5 for r in front_ranges if r > 0)
        return False

    def human_detected(self):
        """انسان کا پتہ لگانا (حقیقی سسٹم میں، کیمرہ استعمال کرے گا)"""
        # اس لیب کے لیے، قرب کی بنیاد پر سیمیولیٹ کریں
        if self.scan_data:
            close_ranges = [r for r in self.scan_data.ranges if 0 < r < 2.0]
            return len(close_ranges) > 5  # اگر متعدد قریب کی ریڈنگز ہیں، تو انسان تصور کریں
        return False

    def is_safe(self):
        """چیک کریں کہ کیا موجودہ صورتحال محفوظ ہے"""
        return not self.is_unbalanced() and not self.obstacle_ahead()

    def behavior_to_command(self, behavior):
        """رویہ کو روبوٹ کمانڈ میں تبدیل کریں"""
        cmd = Twist()

        if behavior['type'] == 'balance_correction':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.3  # سمت کو درست کریں
        elif behavior['type'] == 'obstacle_avoidance':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5  # بچنے کے لیے مڑیں
        elif behavior['type'] == 'steady_locomotion':
            cmd.linear.x = 0.2  # مستحکم فارورڈ موشن
            cmd.angular.z = 0.0
        elif behavior['type'] == 'dynamic_locomotion':
            cmd.linear.x = 0.5  # تیز موشن
            cmd.angular.z = 0.0
        elif behavior['type'] == 'social_interaction':
            cmd.linear.x = 0.1  # آہستہ قریب آئیں
            cmd.angular.z = 0.0
        elif behavior['type'] == 'approachable_posture':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.1  # قریب آنے کے قابل نظر آنے کے لیے معمولی مڑنا

        return cmd

    def update_behavior_state(self, behavior):
        """اندرونی رویہ کی حالت کو اپ ڈیٹ کریں"""
        self.behavior_state = behavior['type']

def main(args=None):
    rclpy.init(args=args)
    lab = HumanoidAnalysisLab()

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

## مشق: اپنا انسان نما روبوٹ ڈیزائن کریں

مندرجہ ذیل ڈیزائن چیلنج پر غور کریں:

1. آپ کے روبوٹ کا بنیادی فنکشن کیا ہوگا؟
2. تین نقطہ نظر میں سے کون سا (ASIMO کی مستحکم، Atlas کی متحرک، یا Pepper کی سماجی بات چیت) سب سے متعلق ہوگا؟
3. آپ کے روبوٹ کے فنکشن کے لیے کون سے سینسرز ضروری ہوں گے؟
4. آپ اپنی مخصوص اطلاق کے لیے ادراک، سوچ بچار، اور عمل کو کیسے ضم کریں گے؟
5. کون سی منفرد صلاحیات آپ کے روبوٹ کو موجودہ ڈیزائن سے ممتاز کریں گی؟

## خلاصہ

حقیقی دنیا کے انسان نما روبوٹس جسمانی مصنوعی ذہانت کے اصولوں کے نفاذ کے مختلف نقطہ نظر کا مظاہرہ کرتے ہیں:

- **ہونڈا ASIMO**: مستحکم، قابل پیش گوئی دو پائوں والی چلنے اور انسانی بات چیت پر زور دیا
- **بوسٹن ڈائینامکس Atlas**: متحرک حرکت کی صلاحیات اور زیادہ کارکردگی والے کنٹرول پر توجہ مرکوز کی
- **سافٹ بینک Pepper**: سماجی بات چیت اور جذباتی ذہانت میں تخصص کیا

ان روبوٹس سے کلیدی سبق یہ ہیں:
- تخصص اور مرکوز صلاحیات کی اہمیت
- ادراک، سوچ بچار، اور عمل کے درمیان تنگ انضمام کی ضرورت
- صلاحیات کی تدریجی ترقی کی قدر
- انسان نما سسٹم میں سلامتی اور توازن کا اہم کردار

یہ مثالیں آپ کے اپنے انسان نما روبوٹ سسٹم ڈیزائن اور نافذ کرنے کے لیے قیمتی بصیرت فراہم کرتی ہیں۔ ان کامیاب روبوٹس کے ذریعہ کیے گئے تنازعات اور ڈیزائن کے فیصلوں کو سمجھنا آپ کی اپنی ترقی کی کوششوں کی رہنمائی کر سکتا ہے۔

اگلے سبق میں، ہم اس کا جائزہ لیں گے کہ جسمانیت انسان نما روبوٹس میں سیکھنے اور ذہانت کو کیسے متاثر کرتی ہے۔