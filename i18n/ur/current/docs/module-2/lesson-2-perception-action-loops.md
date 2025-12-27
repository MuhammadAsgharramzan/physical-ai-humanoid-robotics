---
sidebar_position: 2
---

# جامع ذہانت میں ادراک-عمل لوپ

## تعارف

ادراک-عمل لوپ وہ اصل میکنزم ہے جس کے ذریعے جامع ایجنٹس اپنے ماحول کے ساتھ بات چیت کرتے ہیں۔ روایتی مصنوعی ذہانت کے نظاموں کے برعکس جو معلومات کو الگ الگ اقدامات میں عمل کرتے ہیں، جامع ایجنٹس مسلسل اپنے ماحول کا ادراک کرتے ہیں اور اس پر عمل کرتے ہیں، جس سے متحرک فیڈ بیک لوپ تشکیل پاتے ہیں جو سادہ قواعد سے پیچیدہ رویے کو نمودار کرنے کے قابل بناتے ہیں۔

## ادراک-عمل فریم ورک

ادراک-عمل لوپ کو اس طرح بیان کیا جا سکتا ہے:

```
ماحول → ادراک → سوچ بچار → عمل → ماحول
     ↑__________________________________________|
```

یہ فریم ورک کلاسیکل مصنوعی ذہانت کے نقطہ نظر سے کئی اہم طریقوں میں مختلف ہے:

- **مسلسل پروسیسنگ**: بیچ پروسیسنگ کے بجائے، سسٹم حقیقی وقت میں کام کرتا ہے
- ** jamsی سوچ بچار**: سوچ بچار کے عمل جسمانی بات چیت میں جڑے ہوتے ہیں
- **متحرک مطابقت**: رویہ ماحولیاتی فیڈ بیک کی بنیاد پر مسلسل موافق ہوتا ہے
- **نمودار خصوصیات**: سادہ ادراک-عمل کے قواعد سے پیچیدہ رویے نمودار ہوتے ہیں

## ادراک-عمل لوپ کی اقسام

### 1. ری ایکٹو لوپ

ادراک-عمل لوپ کی سب سے سادہ شکل حسی ان پٹ کا براہ راست جواب دیتی ہے:

```python
# مثال: ری ایکٹو ادراک-عمل لوپ
class ReactiveAgent:
    def __init__(self):
        self.sensors = ['proximity', 'light', 'sound']
        self.actuators = ['wheels', 'gripper', 'speaker']

    def perceive(self, sensor_data):
        """فوری حسی ان پٹ کو عمل کریں"""
        return sensor_data

    def act(self, perception):
        """فوری جواب تیار کریں"""
        action = {}

        # سادہ ری ایکٹو قواعد
        if perception.get('proximity', float('inf')) < 0.3:
            action['wheels'] = {'linear': 0.0, 'angular': 0.5}  # دور مڑیں
        elif perception.get('light', 0) > 0.8:
            action['wheels'] = {'linear': 0.3, 'angular': 0.0}  # روشنی کی طرف بڑھیں
        else:
            action['wheels'] = {'linear': 0.2, 'angular': 0.0}  # تلاش کریں

        return action

    def run_loop(self, sensor_data):
        """ایک ادراک-عمل سائیکل انجام دیں"""
        perception = self.perceive(sensor_data)
        action = self.act(perception)
        return action
```

### 2. پیش گوئی والے لوپ

زیادہ ترقی یافتہ ایجنٹس نتائج کی پیش گوئی کے لیے پیش گوئی کے ماڈل استعمال کرتے ہیں:

```python
# مثال: پیش گوئی والے ادراک-عمل لوپ
import numpy as np

class PredictiveAgent:
    def __init__(self):
        self.internal_model = self.initialize_model()
        self.belief_state = np.zeros(10)  # اندرونی نمائندگی

    def initialize_model(self):
        """اندرونی دنیا کا ماڈل شروع کریں"""
        return {
            'transition_model': {},  # دنیا کیسے تبدیل ہوتی ہے اعمال کے ساتھ
            'observation_model': {}, # مشاہدات حالت سے کیسے متعلق ہیں
            'reward_model': {}       # متوقع نتائج
        }

    def perceive(self, observation, action_taken):
        """مشاہدہ کی بنیاد پر اندرونی ماڈل کو اپ ڈیٹ کریں"""
        # بےز رول یا اسی طرح کے ذریعہ ایمان کی حالت کو اپ ڈیٹ کریں
        self.belief_state = self.update_belief(
            self.belief_state, observation, action_taken
        )
        return self.belief_state

    def predict_outcome(self, candidate_action):
        """ایک ممکنہ عمل کا نتیجہ تلاش کریں"""
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
        """اندرونی ماڈل کی بنیاد پر عمل منتخب کریں"""
        best_action = None
        best_value = float('-inf')

        # ممکنہ اعمال کا جائزہ لیں
        for action in self.get_possible_actions():
            predicted_state, pred_obs, pred_reward = self.predict_outcome(action)
            value = self.calculate_expected_value(pred_reward, predicted_state)

            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def get_possible_actions(self):
        """ممکنہ اعمال واپس کریں"""
        return ['move_forward', 'turn_left', 'turn_right', 'stop']

    def calculate_expected_value(self, reward, state):
        """عمل کی متوقع قیمت کا حساب لگائیں"""
        return reward  # مثال کے لیے سادہ بنایا گیا

    def update_belief(self, current_belief, observation, action):
        """نئی معلومات کی بنیاد پر ایمان کی حالت کو اپ ڈیٹ کریں"""
        # عمل میں، یہ زیادہ ترقی یافتہ طریقے استعمال کرے گا
        return current_belief
```

### 3. سلسلہ دار لوپ

پیچیدہ رویے مختلف وقتی اور مقامی پیمانوں پر ایک سے زیادہ لوپ کے تعامل سے نمودار ہوتے ہیں:

```python
# مثال: سلسلہ دار ادراک-عمل سسٹم
class HierarchicalAgent:
    def __init__(self):
        # مختلف وقتی پیمانے
        self.reflexive_layer = ReflexiveLayer()      # تیز (100Hz)
        self.reactive_layer = ReactiveLayer()        # درمیانی (10Hz)
        self.planning_layer = PlanningLayer()        # سست (1Hz)

    def perceive(self, sensor_data, dt):
        """کئی سطحوں پر ادراک کو عمل کریں"""
        # مختلف وقتی پیمانوں پر عمل کریں
        reflex_action = self.reflexive_layer.process(sensor_data, dt)
        reactive_action = self.reactive_layer.process(sensor_data, dt)
        planned_action = self.planning_layer.process(sensor_data, dt)

        # اعمال کو سلسلہ دار طور پر ضم کریں
        integrated_action = self.integrate_actions(
            reflex_action, reactive_action, planned_action
        )

        return integrated_action

    def integrate_actions(self, reflex, reactive, planned):
        """مختلف سطحوں سے اعمال کو ضم کریں"""
        # تحفظ کے ریفلیکس دیگر اعمال کو نظرانداز کر دیتے ہیں
        if reflex['priority'] == 'high':
            return reflex

        # بصورت دیگر، ری ایکٹو اور منصوبہ بند اعمال کو ملا دیں
        final_action = {}
        for key in set(reflex.keys()) | set(reactive.keys()) | set(planned.keys()):
            if key == 'priority':
                final_action[key] = 'medium'  # ڈیفالٹ ترجیح
            else:
                # وقتی پیمانے کی بنیاد پر وزن دیا گیا امتزاج
                final_action[key] = (
                    0.1 * reflex.get(key, 0) +
                    0.3 * reactive.get(key, 0) +
                    0.6 * planned.get(key, 0)
                )

        return final_action

class ReflexiveLayer:
    def process(self, sensor_data, dt):
        """تیز ریفلیکس جوابات (رکاوٹ سے بچنا، توازن)"""
        action = {'priority': 'high'}

        # فوری تحفظ کے جوابات
        if sensor_data.get('collision_imminent', False):
            action['linear_vel'] = 0.0
            action['angular_vel'] = 0.0

        return action

class ReactiveLayer:
    def process(self, sensor_data, dt):
        """درمیانی مدت کے ری ایکٹو رویے"""
        action = {'priority': 'medium'}

        # رکاوٹ سے بچنا
        if sensor_data.get('obstacle_distance', float('inf')) < 0.5:
            action['angular_vel'] = 0.3  # دور مڑیں

        return action

class PlanningLayer:
    def process(self, sensor_data, dt):
        """طویل مدت کا مقصد کے مطابق رویہ"""
        action = {'priority': 'low'}

        # مقصد کی طرف نیویگیٹ کرنا
        action['linear_vel'] = 0.2  # مقصد کی طرف جاری رکھیں

        return action
```

## ROS2 کے ساتھ حقیقی دنیا کا نفاذ

یہاں دیکھیں کہ ROS2 کا استعمال کرتے ہوئے ادراک-عمل لوپ کیسے نافذ کیے جاتے ہیں:

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

        # پبلشرز
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/system_status', 10)

        # سبسکرائبرز
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )

        # سینسر ڈیٹا اسٹوریج
        self.scan_data = None
        self.image_data = None
        self.imu_data = None

        # پروسیسنگ کمپوننٹس
        self.cv_bridge = CvBridge()
        self.perception_processor = PerceptionProcessor()
        self.action_selector = ActionSelector()
        self.predictive_model = PredictiveModel()

        # لوپ ٹائمنگ
        self.loop_timer = self.create_timer(0.05, self.perception_action_loop)  # 20 Hz
        self.last_loop_time = self.get_clock().now()

        # سسٹم کی حالت
        self.system_state = {
            'position': np.array([0.0, 0.0, 0.0]),
            'orientation': 0.0,
            'velocity': np.array([0.0, 0.0, 0.0]),
            'goal': np.array([5.0, 5.0, 0.0])
        }

    def scan_callback(self, msg):
        """لیزر اسکین ڈیٹا کو سنبھالیں"""
        self.scan_data = msg

    def image_callback(self, msg):
        """کیمرہ امیج ڈیٹا کو سنبھالیں"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_data = cv_image
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def imu_callback(self, msg):
        """IMU ڈیٹا کو سنبھالیں"""
        self.imu_data = msg
        # IMU ڈیٹا کے ساتھ سسٹم کی حالت کو اپ ڈیٹ کریں
        self.update_orientation_from_imu(msg)

    def perception_action_loop(self):
        """اصل ادراک-عمل لوپ"""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_loop_time).nanoseconds / 1e9
        self.last_loop_time = current_time

        # یقینی بنائیں کہ ہمارے پاس سینسر ڈیٹا ہے
        if not all([self.scan_data, self.image_data, self.imu_data]):
            return

        # 1. ادراک کا مرحلہ
        perceptual_state = self.process_perception()

        # 2. سوچ بچار/پیش گوئی کا مرحلہ
        cognitive_state = self.process_cognition(perceptual_state, dt)

        # 3. عمل منتخب کرنے کا مرحلہ
        action = self.select_action(cognitive_state)

        # 4. عمل انجام دینے کا مرحلہ
        self.execute_action(action)

        # 5. سسٹم کی حالت کو اپ ڈیٹ کریں
        self.update_system_state(action, dt)

        # 6. حالت شائع کریں
        self.status_pub.publish(
            String(data=f"Loop executing - Position: {self.system_state['position']}")
        )

    def process_perception(self):
        """تمام سینسر ڈیٹا کو مربوط ادراک کی حالت میں عمل کریں"""
        perceptual_state = {
            'environment_map': self.create_environment_map(),
            'object_detections': self.detect_objects_in_image(),
            'obstacle_distances': self.extract_obstacle_distances(),
            'current_pose': self.system_state['position'],
            'current_orientation': self.system_state['orientation']
        }

        return perceptual_state

    def create_environment_map(self):
        """لیزر ڈیٹا سے آکوپنسی گرڈ بنائیں"""
        if self.scan_data:
            # لیزر اسکین کو سادہ آکوپنسی نمائندگی میں تبدیل کریں
            angles = np.linspace(
                self.scan_data.angle_min,
                self.scan_data.angle_max,
                len(self.scan_data.ranges)
            )
            ranges = np.array(self.scan_data.ranges)

            # غلط رینج کو فلٹر کریں
            valid_mask = (ranges > 0) & (ranges < self.scan_data.range_max)
            valid_angles = angles[valid_mask]
            valid_ranges = ranges[valid_mask]

            # روبوٹ کے مطابق کارٹیزین کوآرڈینیٹس میں تبدیل کریں
            x_points = valid_ranges * np.cos(valid_angles)
            y_points = valid_ranges * np.sin(valid_angles)

            return np.column_stack([x_points, y_points])

        return np.array([])

    def detect_objects_in_image(self):
        """کیمرہ امیج میں اشیاء کا پتہ لگائیں"""
        if self.image_data is not None:
            # مثال کے لیے سادہ رنگ کی بنیاد پر اشیاء کا پتہ لگانا
            hsv = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2HSV)

            # لال اشیاء کا پتہ لگائیں (مثال)
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)

            # بلیک کے لیے اعلی ریڈ ویلیوز چیک کریں (hue > 170)
            lower_red2 = np.array([170, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

            mask = mask1 + mask2
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            objects = []
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # کم از کم سائز کی حد
                    # مرکز کا حساب لگائیں
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        objects.append({'x': cx, 'y': cy, 'type': 'red_object'})

            return objects

        return []

    def extract_obstacle_distances(self):
        """مختلف سمت میں کم از کم فاصلے نکالیں"""
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
        """فیصلہ کرنے کے لیے ادراک کی حالت کو عمل کریں"""
        cognitive_state = {
            'threat_level': self.assess_threats(perceptual_state),
            'navigation_state': self.assess_navigation(perceptual_state),
            'object_interest': self.assess_object_interest(perceptual_state),
            'predicted_environment': self.predict_environment(perceptual_state, dt)
        }

        return cognitive_state

    def assess_threats(self, perceptual_state):
        """ماحول میں ممکنہ خطرات کا جائزہ لیں"""
        threat_level = 0.0

        # قریبی رکاوٹوں کی جانچ کریں
        if perceptual_state['obstacle_distances']['min'] < 0.5:
            threat_level += 0.8
        elif perceptual_state['obstacle_distances']['min'] < 1.0:
            threat_level += 0.3

        # غیر مستحکم جہت کی جانچ کریں (اگر دستیاب ہو)
        if hasattr(self, 'tilt_angle') and abs(self.tilt_angle) > 0.5:
            threat_level += 0.9

        return min(threat_level, 1.0)  # 0 اور 1 کے درمیان محدود کریں

    def assess_navigation(self, perceptual_state):
        """نیویگیشن کی حالت اور مقصد کی پیشرفت کا جائزہ لیں"""
        # مقصد کی طرف سمت کا حساب لگائیں
        current_pos = self.system_state['position']
        goal_pos = self.system_state['goal']

        direction_to_goal = goal_pos - current_pos
        distance_to_goal = np.linalg.norm(direction_to_goal)

        # چیک کریں کہ مقصد کی طرف راستہ صاف ہے یا نہیں
        path_clear = self.is_path_clear(perceptual_state, direction_to_goal)

        return {
            'distance_to_goal': distance_to_goal,
            'direction_to_goal': direction_to_goal,
            'path_clear': path_clear,
            'progress': 1.0 / (1.0 + distance_to_goal)  # زیادہ ویلیو = مقصد کے قریب
        }

    def is_path_clear(self, perceptual_state, direction):
        """چیک کریں کہ دی گئی سمت میں راستہ رکاوٹوں سے صاف ہے یا نہیں"""
        # سادہ چیک - حقیقت میں زیادہ ترقی یافتہ راستہ منصوبہ بندی استعمال کرے گا
        front_distance = perceptual_state['obstacle_distances']['front']
        return front_distance > 1.0

    def assess_object_interest(self, perceptual_state):
        """دریافت شدہ اشیاء میں دلچسپی کا جائزہ لیں"""
        if perceptual_state['object_detections']:
            # اس مثال کے لیے، کوئی بھی دریافت شدہ چیز دلچسپ ہے
            return {
                'object_count': len(perceptual_state['object_detections']),
                'nearest_object': self.find_nearest_object(perceptual_state['object_detections'])
            }
        return {'object_count': 0, 'nearest_object': None}

    def find_nearest_object(self, objects):
        """روبوٹ کے قریب ترین چیز تلاش کریں"""
        # سادہ - تصور کریں کہ اشیاء کے پاس پوزیشن کی معلومات ہے
        if objects:
            # ایک حقیقی سسٹم میں، یہ امیج کوآرڈینیٹس کو ورلڈ کوآرڈینیٹس میں تبدیل کرے گا
            return objects[0]  # پہلی چیز کو مثال کے طور پر واپس کریں
        return None

    def predict_environment(self, perceptual_state, dt):
        """پیش گوئی کریں کہ ماحول کیسے تبدیل ہوگا"""
        # موجودہ حالت کی بنیاد پر سادہ پیش گوئی
        predicted_state = perceptual_state.copy()

        # پیش گوئی کریں کہ رکاوٹیں قریب آ سکتی ہیں اگر روبوٹ ان کی طرف بڑھ رہا ہے
        if self.system_state['velocity'][0] > 0:  # آگے بڑھ رہا ہے
            for key in ['front', 'left', 'right', 'min']:
                if key in predicted_state['obstacle_distances']:
                    predicted_state['obstacle_distances'][key] = max(
                        0.1, predicted_state['obstacle_distances'][key] -
                        self.system_state['velocity'][0] * dt
                    )

        return predicted_state

    def select_action(self, cognitive_state):
        """سوچ بچار کی حالت کی بنیاد پر عمل منتخب کریں"""
        # عمل منتخب کرنے کا سلسلہ
        # 1. تحفظ (سب سے زیادہ ترجیح)
        if cognitive_state['threat_level'] > 0.7:
            return self.emergency_action(cognitive_state)

        # 2. مقصد کی طرف نیویگیٹ کرنا
        elif cognitive_state['navigation_state']['path_clear']:
            return self.navigate_toward_goal(cognitive_state)

        # 3. رکاوٹ سے بچنا
        elif not cognitive_state['navigation_state']['path_clear']:
            return self.avoid_obstacles(cognitive_state)

        # 4. تلاش
        else:
            return self.explore_environment(cognitive_state)

    def emergency_action(self, cognitive_state):
        """زیادہ ترجیح والی ایمرجنسی کا عمل"""
        cmd = Twist()
        cmd.linear.x = 0.0  # فوری طور پر رکیں
        cmd.angular.z = 0.5  # خطرے سے بچنے کے لیے مڑیں

        self.get_logger().warn('EMERGENCY ACTION: High threat detected')
        return cmd

    def navigate_toward_goal(self, cognitive_state):
        """مقصد کی طرف نیویگیٹ کریں"""
        cmd = Twist()

        # مقصد کی طرف بڑھیں
        direction = cognitive_state['navigation_state']['direction_to_goal']
        cmd.linear.x = 0.3  # فارورڈ سپیڈ

        # سمت کے لیے سادہ پروپورشل کنٹرولر
        desired_angle = math.atan2(direction[1], direction[0])
        current_angle = self.system_state['orientation']
        angle_error = desired_angle - current_angle

        # زاویہ کی خرابی کو [-π, π] تک محدود کریں
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi

        cmd.angular.z = 1.0 * angle_error  # پروپورشل کنٹرول

        return cmd

    def avoid_obstacles(self, cognitive_state):
        """رکاوٹوں سے بچیں"""
        cmd = Twist()

        obstacle_distances = cognitive_state['perceptual_state']['obstacle_distances']

        # قریب ترین رکاوٹ سے دور مڑیں
        if obstacle_distances['front'] < 0.8:
            # سامنے رکاوٹ ہے - مڑیں
            if obstacle_distances['left'] > obstacle_distances['right']:
                cmd.angular.z = 0.5  # بائیں مڑیں
            else:
                cmd.angular.z = -0.5  # دائیں مڑیں
        elif obstacle_distances['front'] < 1.5:
            # رکاوٹوں کی طرف بڑھتے وقت سست کریں
            cmd.linear.x = 0.1
        else:
            # کوئی فوری رکاوٹ نہیں
            cmd.linear.x = 0.3

        return cmd

    def explore_environment(self, cognitive_state):
        """ماحول کو تلاش کریں"""
        cmd = Twist()
        cmd.linear.x = 0.2  # سست تلاش
        cmd.angular.z = 0.1  # معمولی مڑنا

        return cmd

    def execute_action(self, action):
        """منتخب کردہ عمل انجام دیں"""
        if isinstance(action, Twist):
            self.cmd_vel_pub.publish(action)
        else:
            # اگر عمل پہلے سے Twist پیغام نہیں ہے، تبدیل کریں
            cmd = Twist()
            if 'linear' in action:
                cmd.linear.x = action['linear']
            if 'angular' in action:
                cmd.angular.z = action['angular']
            self.cmd_vel_pub.publish(cmd)

    def update_system_state(self, action, dt):
        """عمل اور وقت کی بنیاد پر اندرونی سسٹم کی حالت کو اپ ڈیٹ کریں"""
        # رفتار کی بنیاد پر پوزیشن کو اپ ڈیٹ کریں
        if isinstance(action, Twist):
            linear_vel = action.linear.x
            angular_vel = action.angular.z
        else:
            linear_vel = action.get('linear', 0.0)
            angular_vel = action.get('angular', 0.0)

        # جہت کو اپ ڈیٹ کریں
        self.system_state['orientation'] += angular_vel * dt

        # پوزیشن کو اپ ڈیٹ کریں (سادہ ڈیڈ ریکننگ)
        self.system_state['position'][0] += linear_vel * math.cos(self.system_state['orientation']) * dt
        self.system_state['position'][1] += linear_vel * math.sin(self.system_state['orientation']) * dt

        # رفتار کو اپ ڈیٹ کریں
        self.system_state['velocity'][0] = linear_vel * math.cos(self.system_state['orientation'])
        self.system_state['velocity'][1] = linear_vel * math.sin(self.system_state['orientation'])

    def update_orientation_from_imu(self, imu_msg):
        """IMU ڈیٹا سے جہت کو اپ ڈیٹ کریں"""
        # کووٹرین کو ایولر اینگل میں تبدیل کریں (سادہ)
        # عمل میں، مناسب کووٹرین ریاضی استعمال کریں گے
        self.system_state['orientation'] = math.atan2(
            2 * (imu_msg.orientation.w * imu_msg.orientation.z +
                 imu_msg.orientation.x * imu_msg.orientation.y),
            1 - 2 * (imu_msg.orientation.y**2 + imu_msg.orientation.z**2)
        )

class PerceptionProcessor:
    """ادراک کے ڈیٹا کو عمل کرنے کا جزو"""
    def __init__(self):
        self.feature_extractors = {}
        self.scene_understanding = None

    def process_sensor_data(self, sensor_data):
        """خام سینسر ڈیٹا کو معنی خیز ادراک میں عمل کریں"""
        processed_data = {}

        # مختلف سینسرز سے خصوصیات نکالیں
        if 'image' in sensor_data:
            processed_data['visual_features'] = self.extract_visual_features(sensor_data['image'])

        if 'laser' in sensor_data:
            processed_data['spatial_features'] = self.extract_spatial_features(sensor_data['laser'])

        if 'imu' in sensor_data:
            processed_data['inertial_features'] = self.extract_inertial_features(sensor_data['imu'])

        return processed_data

    def extract_visual_features(self, image):
        """امیج سے بصری خصوصیات نکالیں"""
        # اصل خصوصیات نکالنے کے لیے جگہ کا نمونہ
        return {'features': [], 'objects': []}

    def extract_spatial_features(self, laser_data):
        """لیزر ڈیٹا سے جگہ کی خصوصیات نکالیں"""
        # اصل خصوصیات نکالنے کے لیے جگہ کا نمونہ
        return {'obstacles': [], 'free_space': []}

    def extract_inertial_features(self, imu_data):
        """IMU ڈیٹا سے انرٹیل خصوصیات نکالیں"""
        # اصل خصوصیات نکالنے کے لیے جگہ کا نمونہ
        return {'orientation': 0.0, 'acceleration': [0, 0, 0]}

class ActionSelector:
    """حالت کی بنیاد پر اعمال منتخب کرنے کا جزو"""
    def __init__(self):
        self.policy_network = None
        self.value_function = None

    def select_action(self, state, policy_type='reactive'):
        """موجودہ حالت کی بنیاد پر عمل منتخب کریں"""
        if policy_type == 'reactive':
            return self.reactive_policy(state)
        elif policy_type == 'planning':
            return self.planning_policy(state)
        else:
            return self.default_policy(state)

    def reactive_policy(self, state):
        """سادہ ری ایکٹو پالیسی"""
        # ری ایکٹو عمل منتخب کرنے کے لیے جگہ کا نمونہ
        return {'linear': 0.0, 'angular': 0.0}

    def planning_policy(self, state):
        """منصوبہ بندی پر مبنی پالیسی"""
        # منصوبہ بندی پر مبنی عمل منتخب کرنے کے لیے جگہ کا نمونہ
        return {'linear': 0.0, 'angular': 0.0}

    def default_policy(self, state):
        """ڈیفالٹ فال بیک پالیسی"""
        return {'linear': 0.0, 'angular': 0.0}

class PredictiveModel:
    """ماحولیاتی تبدیلیوں کی پیش گوئی کا جزو"""
    def __init__(self):
        self.dynamics_model = None
        self.uncertainty_model = None

    def predict_next_state(self, current_state, action):
        """عمل کی بنیاد پر اگلی ماحولیاتی حالت کی پیش گوئی کریں"""
        # پیش گوئی کے لیے جگہ کا نمونہ
        return current_state

    def predict_sensor_observations(self, predicted_state):
        """پیش گوئی کریں کہ سینسرز متوقع حالت میں کیا مشاہدہ کریں گے"""
        # سینسر پیش گوئی کے لیے جگہ کا نمونہ
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

## ادراک-عمل لوپ سے نمودار ہونے والے رویے

سادہ ادراک-عمل کے قواعد سے پیچیدہ رویے نمودار ہو سکتے ہیں:

```python
# مثال: سادہ قواعد سے نمودار ہونے والے فلوکنگ رویے
class FlockingAgent:
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.max_speed = 2.0
        self.perception_radius = 5.0

    def update(self, neighbors, dt):
        """ہمسائیوں اور ماحول کی بنیاد پر ایجنٹ کو اپ ڈیٹ کریں"""
        # فلوکنگ کے قواعد لاگو کریں
        alignment = self.align_with_neighbors(neighbors)
        cohesion = self.move_toward_center(neighbors)
        separation = self.avoid_crowding(neighbors)

        # رویے کو ضم کریں
        acceleration = 0.5 * alignment + 0.3 * cohesion + 0.7 * separation

        # رفتار اور پوزیشن کو اپ ڈیٹ کریں
        self.velocity += acceleration * dt
        if np.linalg.norm(self.velocity) > self.max_speed:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.max_speed

        self.position += self.velocity * dt

    def align_with_neighbors(self, neighbors):
        """ہمسائیوں کے اوسط ہیڈنگ کے ساتھ ہم آہنگ کریں"""
        if not neighbors:
            return np.zeros(2)

        avg_heading = np.mean([n.velocity for n in neighbors], axis=0)
        return (avg_heading - self.velocity) * 0.05

    def move_toward_center(self, neighbors):
        """ہمسائیوں کے اوسط پوزیشن کی طرف بڑھیں"""
        if not neighbors:
            return np.zeros(2)

        center = np.mean([n.position for n in neighbors], axis=0)
        direction = center - self.position
        return (direction - self.velocity) * 0.02

    def avoid_crowding(self, neighbors):
        """مقامی فلوک میٹس کے ہجوم سے بچیں"""
        if not neighbors:
            return np.zeros(2)

        repulsion = np.zeros(2)
        for neighbor in neighbors:
            distance = np.linalg.norm(neighbor.position - self.position)
            if distance < 2.0:  # بہت قریب
                repulsion += (self.position - neighbor.position) / (distance + 0.01)

        return repulsion * 0.1

# مثال: فوریج رویہ
class ForagingAgent:
    def __init__(self):
        self.state = 'searching'  # searching, approaching, collecting, returning
        self.cargo = 0
        self.cargo_capacity = 10
        self.goal_location = None

    def perception_action_cycle(self, sensor_data):
        """فوریج کے لیے ادراک-عمل سائیکل"""
        # ادراک
        food_detected = sensor_data.get('food_nearby', False)
        nest_detected = sensor_data.get('nest_nearby', False)
        obstacle_ahead = sensor_data.get('obstacle_ahead', False)

        # ادراک کی بنیاد پر حالت میں تبدیلی
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
                self.cargo = 0  # کارگو چھوڑ دیں

        # حالت کی بنیاد پر عمل منتخب کریں
        return self.select_action(obstacle_ahead)

    def select_action(self, obstacle_ahead):
        """موجودہ حالت کی بنیاد پر عمل منتخب کریں"""
        action = {'linear': 0.0, 'angular': 0.0}

        if obstacle_ahead:
            action['angular'] = 0.5  # بچنے کے لیے مڑیں
        elif self.state == 'searching':
            action['linear'] = 0.3
            action['angular'] = 0.1  # سرپل تلاش کا نمونہ
        elif self.state == 'approaching':
            action['linear'] = 0.2  # احتیاط سے قریب آنا
        elif self.state == 'collecting':
            action['linear'] = 0.0  # جمع کرنے کے لیے سٹیل رہیں
        elif self.state == 'returning':
            action['linear'] = 0.25  # مستحکم واپسی

        return action
```

## لیب: ادراک-عمل لوپ کا نفاذ

اس لیب میں، آپ ایک سادہ ادراک-عمل لوپ نافذ کریں گے:

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

        # پبلشرز
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/lab_status', 10)
        self.emergency_pub = self.create_publisher(Bool, '/emergency_stop', 10)

        # سبسکرائبرز
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        # ڈیٹا اسٹوریج
        self.scan_data = None
        self.image_data = None
        self.cv_bridge = CvBridge()

        # لیب پیرامیٹر
        self.loop_state = 'exploration'  # exploration, obstacle_avoidance, object_interaction
        self.object_detected = False
        self.target_color = [0, 255, 0]  # سبز چیز کو فالو کریں

        # کنٹرول لوپ
        self.control_timer = self.create_timer(0.1, self.control_loop)

    def scan_callback(self, msg):
        """لیزر اسکین ڈیٹا کو سنبھالیں"""
        self.scan_data = msg

    def image_callback(self, msg):
        """کیمرہ امیج ڈیٹا کو سنبھالیں"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_data = cv_image
        except Exception as e:
            self.get_logger().error(f'Image callback error: {e}')

    def control_loop(self):
        """اصل ادراک-عمل کنٹرول لوپ"""
        if not self.scan_data and not self.image_data:
            return

        # 1. ادراک کا مرحلہ
        perceptions = self.process_perceptions()

        # 2. حالت کا جائزہ مرحلہ
        self.evaluate_state(perceptions)

        # 3. عمل منتخب کرنے کا مرحلہ
        action = self.select_action(perceptions)

        # 4. عمل انجام دینے کا مرحلہ
        self.execute_action(action)

        # 5. حالت کی تازہ کاری
        self.status_pub.publish(
            String(data=f"State: {self.loop_state}, Objects: {int(self.object_detected)}")
        )

    def process_perceptions(self):
        """معنی خیز ادراک نکالنے کے لیے سینسر ڈیٹا کو عمل کریں"""
        perceptions = {}

        # لیزر ڈیٹا کو عمل کریں
        if self.scan_data:
            perceptions['obstacles'] = self.analyze_obstacles()
            perceptions['clear_front'] = self.is_front_clear()

        # امیج ڈیٹا کو عمل کریں
        if self.image_data is not None:
            perceptions['objects'] = self.detect_target_objects()
            perceptions['object_direction'] = self.get_object_direction()

        return perceptions

    def analyze_obstacles(self):
        """رکاوٹوں کے لیے لیزر ڈیٹا کا تجزیہ کریں"""
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
        """چیک کریں کہ سامنے کا راستہ صاف ہے یا نہیں"""
        if self.scan_data:
            front_ranges = self.scan_data.ranges[300:600]
            return all(r > 1.0 for r in front_ranges if r > 0)
        return True

    def detect_target_objects(self):
        """امیج میں ہدف کی چیز کا پتہ لگائیں"""
        if self.image_data is not None:
            # بہتر رنگ کی شناخت کے لیے BGR کو HSV میں تبدیل کریں
            hsv = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2HSV)

            # ہدف کے رنگ (سبز) کے لیے رینج کی وضاحت کریں
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])

            mask = cv2.inRange(hsv, lower_green, upper_green)

            # کنٹور تلاش کریں
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            objects = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # کم از کم رقبہ کی حد
                    # مرکز کا حساب لگائیں
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
        """دریافت شدہ چیز کی سمت حاصل کریں"""
        if self.image_data is not None and self.object_detected:
            objects = self.detect_target_objects()
            if objects:
                # سب سے بڑی چیز استعمال کریں
                largest_obj = max(objects, key=lambda x: x['area'])
                image_center_x = self.image_data.shape[1] / 2
                object_x = largest_obj['x']

                # سمت کا حساب لگائیں (-1 بائیں کے لیے، 1 دائیں کے لیے)
                direction = (object_x - image_center_x) / image_center_x
                return max(-1.0, min(1.0, direction))  # [-1, 1] تک محدود کریں

        return 0.0  # کوئی چیز دریافت نہیں ہوئی

    def evaluate_state(self, perceptions):
        """ادراک کی بنیاد پر موجودہ حالت کا جائزہ لیں"""
        # ہم جو ادراک کرتے ہیں اس کی بنیاد پر حالت کو اپ ڈیٹ کریں
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
        """موجودہ حالت اور ادراک کی بنیاد پر عمل منتخب کریں"""
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
        """دریافت شدہ چیز کو فالو کرنے کا عمل"""
        cmd = Twist()

        if self.object_detected:
            object_direction = perceptions['object_direction']

            # چیز کی طرف بڑھیں اگر یہ کافی دور ہے
            if perceptions['obstacles']['closest'] > 0.8:
                cmd.linear.x = 0.2
            else:
                cmd.linear.x = 0.0  # رکاوٹوں سے بہت قریب ہونے پر رکیں

            # چیز کی طرف مڑیں
            cmd.angular.z = -0.8 * object_direction  # درست سمت کے لیے منفی
        else:
            # اگر کوئی چیز دریافت نہیں ہوئی، اسے تلاش کریں
            cmd.linear.x = 0.0
            cmd.angular.z = 0.3  # چیز تلاش کرنے کے لیے مڑیں

        return cmd

    def avoid_obstacle_action(self, perceptions):
        """رکاوٹوں سے بچنے کا عمل"""
        cmd = Twist()

        obstacles = perceptions['obstacles']

        if obstacles['closest'] < 0.5:
            # ایمرجنسی اسٹاپ یا تیزی سے مڑیں
            cmd.linear.x = 0.0
            if obstacles['left_clear'] and not obstacles['right_clear']:
                cmd.angular.z = 0.5  # بائیں مڑیں
            elif obstacles['right_clear'] and not obstacles['left_clear']:
                cmd.angular.z = -0.5  # دائیں مڑیں
            else:
                cmd.angular.z = 0.3  # بے ترتیب مڑیں
        elif obstacles['closest'] < 1.0:
            # سست کریں اور مڑنے کی تیاری کریں
            cmd.linear.x = 0.1
            if not obstacles['front_clear']:
                cmd.angular.z = 0.2  # معمولی مڑنا
        else:
            # راستہ صاف ہے، آگے بڑھیں
            cmd.linear.x = 0.3

        return cmd

    def explore_action(self, perceptions):
        """تلاش کا عمل"""
        cmd = Twist()

        # سادہ تلاش کا نمونہ
        cmd.linear.x = 0.2  # آگے بڑھیں
        cmd.angular.z = 0.1  # تلاش کے لیے معمولی مڑنا

        return cmd

    def execute_action(self, action):
        """منتخب کردہ عمل انجام دیں"""
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

## مشق: اپنا ادراک-عمل لوپ ڈیزائن کریں

ایک مخصوص کام پر غور کریں اور اس کے لیے ادراک-عمل لوپ ڈیزائن کریں:

1. آپ کے کام کے لیے کون سے سینسرز سب سے متعلق ہوں گے؟
2. ادراک کا مرحلہ کیا شامل کرے گا؟
3. آپ ادراک کی معلومات کو کیسے عمل کریں گے؟
4. کون سے اعمال دستیاب ہوں گے؟
5. آپ مختلف اعمال کے درمیان کیسے منتخب کریں گے؟
6. آپ کے ڈیزائن سے کون سے نمودار رویے نمودار ہو سکتے ہیں؟

## خلاصہ

ادراک-عمل لوپ جامع ذہانت کے لیے بنیادی ہیں، جو روبوٹس کو اپنے ماحول کے ساتھ بات چیت کرنے کے قابل بناتے ہیں ادراک، پروسیسنگ، اور عمل کے مسلسل چکروں کے ذریعے۔ کلیدی تصورات میں شامل ہیں:

- **مسلسل پروسیسنگ**: الگ الگ مصنوعی ذہانت کے نظاموں کے برعکس، جامع ایجنٹس حقیقی وقت میں کام کرتے ہیں
- **کئی لوپ کی اقسام**: ری ایکٹو، پیش گوئی والے، اور سلسلہ دار لوپ مختلف مقاصد کے لیے کام کرتے ہیں
- **نمودار رویے**: سادہ ادراک-عمل کے قواعد سے پیچیدہ رویے نمودار ہوتے ہیں
- **حقیقی وقت کی پابندیاں**: سسٹم کو سخت ٹائمنگ کی ضروریات کے اندر جواب دینا چاہیے
- **ماحولیاتی مطابقت**: رویہ ماحولیاتی فیڈ بیک کی بنیاد پر مسلسل موافق ہوتا ہے

ROS2 کا استعمال کرتے ہوئے ادراک-عمل لوپ کا نفاذ متحرک ماحول میں مؤثر طریقے سے کام کرنے والے ترقی یافتہ جامع سسٹم کی ترقی کو فعال کرتا ہے۔ ان لوپ کو سمجھنا روبوٹس کو ترقی دینے کے لیے اہم ہے جو اپنے ماحول کے ساتھ بات چیت کے ذریعہ سیکھ سکیں اور موافق ہو سکیں۔

اگلے سبق میں، ہم کامیاب انسان نما روبوٹس کی مثالیں تلاش کریں گے اور یہ تجزیہ کریں گے کہ وہ عملاً ادراک-عمل لوپ کو کیسے نافذ کرتے ہیں۔