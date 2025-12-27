---
sidebar_position: 1
---

# روبوٹ کنٹرول کے لیے مسلسل انٹرفیس

## تعارف

مسلسل انٹرفیس انسان-روبوٹ بات چیت کے لیے اہم ہیں، صارفین کو روبوٹس کے ساتھ قدرتی اور کارآمد انداز میں بات چیت کرنے کے قابل بناتے ہیں۔ جسمانی مصنوعی ذہانت کے نظاموں میں، یہ انٹرفیس انسانی نیت اور روبوٹک صلاحیات کے درمیان خلاء کو پُر کرنا چاہیے، جو پیچیدہ روبوٹک نظاموں کو بغیر مخصوص تربیت کے صارفین کے لیے قابل رسائی بناتے ہیں۔ یہ سبق روبوٹ کنٹرول کے لیے مسلسل انٹرفیس ڈیزائن کے مختلف طریقے متعارف کراتا ہے۔

## مسلسل انٹرفیس ڈیزائن کے اصول

### 1. قدرتی میپنگ

قدرتی میپنگ انسانی توقعات کو روبوٹک اعمال کے ساتھ مسلسل طریقے سے جوڑتا ہے:

```python
# مثال: قدرتی میپنگ انٹرفیس ڈیزائن
class NaturalMappingInterface:
    def __init__(self):
        self.action_mappings = {
            'point_at_object': 'move_to_location',
            'wave_hand': 'greet_user',
            'nod_head': 'confirm_action',
            'shake_head': 'deny_action',
            'open_hand': 'release_object',
            'close_hand': 'grasp_object'
        }

    def interpret_human_action(self, human_action):
        """انسانی عمل کی تشریح کریں اور روبوٹ عمل میں میپ کریں"""
        if human_action in self.action_mappings:
            return self.action_mappings[human_action]
        else:
            return 'unknown_action'

    def create_mapping(self, human_action, robot_action):
        """انسانی اور روبوٹ اعمال کے درمیان نیا میپنگ بنائیں"""
        self.action_mappings[human_action] = robot_action

# مثال: جسمانی اشاروں کو روبوٹ کمانڈز میں میپ کرنا
class GestureMapping:
    def __init__(self):
        self.gesture_commands = {
            'move_right': {'linear_x': 0, 'angular_z': -0.5},  # دائیں مڑیں
            'move_left': {'linear_x': 0, 'angular_z': 0.5},   # بائیں مڑیں
            'move_forward': {'linear_x': 0.3, 'angular_z': 0}, # آگے بڑھیں
            'stop': {'linear_x': 0, 'angular_z': 0}            # رکیں
        }

    def get_robot_command(self, gesture):
        """انسانی اشارے سے روبوٹ کمانڈ حاصل کریں"""
        return self.gesture_commands.get(gesture, {'linear_x': 0, 'angular_z': 0})
```

### 2. براہ راست ہیرا پھیرا

براہ راست ہیرا پھیرا صارفین کو قدرتی جسمانی بات چیت کے ذریعے روبوٹس کنٹرول کرنے کی اجازت دیتا ہے:

```python
# مثال: براہ راست ہیرا پھیرا انٹرفیس
class DirectManipulationInterface:
    def __init__(self):
        self.manipulation_modes = {
            'position_control': self.position_control,
            'velocity_control': self.velocity_control,
            'impedance_control': self.impedance_control
        }
        self.current_mode = 'position_control'

    def position_control(self, target_position):
        """راست طور پر روبوٹ پوزیشن کنٹرول کریں"""
        # روبوٹ کو ہدف پوزیشن پر لے جائیں
        return {
            'command_type': 'position',
            'target': target_position,
            'stiffness': 1.0  # درست پوزیشننگ کے لیے زیادہ سختی
        }

    def velocity_control(self, velocity_vector):
        """راست طور پر روبوٹ رفتار کنٹرول کریں"""
        # مخصوص رفتار پر روبوٹ کو چلائیں
        return {
            'command_type': 'velocity',
            'velocity': velocity_vector,
            'stiffness': 0.5  # ہموار حرکت کے لیے میڈیم سختی
        }

    def impedance_control(self, desired_impedance):
        """روبوٹ کی مکینیکل امپیڈنس کنٹرول کریں"""
        # بیرونی قوتوں کے جواب میں روبوٹ کا رد عمل ایڈجسٹ کریں
        return {
            'command_type': 'impedance',
            'impedance': desired_impedance,
            'stiffness': desired_impedance.get('stiffness', 0.3)
        }

    def switch_mode(self, new_mode):
        """ہیرا پھیرا موڈز کے درمیان سوئچ کریں"""
        if new_mode in self.manipulation_modes:
            self.current_mode = new_mode
            return f"Switched to {new_mode}"
        else:
            return f"Mode {new_mode} not available"
```

### 3. مسلسل اور قابل پیش گوئی

مسلسل انٹرفیس صارفین کو روبوٹ کے رویے کا ذہنی ماڈل بنانے میں مدد کرتے ہیں:

```python
# مثال: مسلسل انٹرفیس ڈیزائن
class ConsistentRobotInterface:
    def __init__(self):
        self.command_history = []
        self.user_preferences = {}
        self.response_style = 'consistent'

    def send_command(self, command, parameters):
        """مسلسل فارمیٹ کے ساتھ روبوٹ کو کمانڈ بھیجیں"""
        formatted_command = {
            'timestamp': self.get_timestamp(),
            'command': command,
            'parameters': parameters,
            'user_id': self.get_current_user(),
            'context': self.get_current_context()
        }

        self.command_history.append(formatted_command)
        response = self.execute_command(formatted_command)

        return self.format_response(response)

    def get_current_context(self):
        """موجودہ آپریشنل سیاق و سباق حاصل کریں"""
        return {
            'robot_state': self.get_robot_state(),
            'environment': self.get_environment_state(),
            'task': self.get_current_task()
        }

    def format_response(self, response):
        """جواب کو مسلسل طریقے سے فارمیٹ کریں"""
        return {
            'status': response.get('status', 'unknown'),
            'result': response.get('result', None),
            'confidence': response.get('confidence', 0.0),
            'timestamp': self.get_timestamp()
        }

    def get_robot_state(self):
        """موجودہ روبوٹ کی حالت حاصل کریں"""
        # عمل میں، یہ روبوٹ کی حالت کے ساتھ بات چیت کرے گا
        return {'position': [0, 0, 0], 'battery': 0.8, 'status': 'ready'}

    def get_environment_state(self):
        """موجودہ ماحول کی حالت حاصل کریں"""
        # عمل میں، یہ ادراک کے سسٹم کے ساتھ بات چیت کرے گا
        return {'obstacles': 0, 'lighting': 'good', 'temperature': 22}

    def get_current_task(self):
        """موجودہ کام کی معلومات حاصل کریں"""
        return {'name': 'navigation', 'progress': 0.0, 'goal': [5, 5, 0]}
```

## مسلسل انٹرفیس کی اقسام

### 1. اشارے-مبنی انٹرفیس

اشارے-مبنی انٹرفیس صارفین کو قدرتی ہاتھ اور جسم کی حرکات کے ذریعے روبوٹس کنٹرول کرنے کی اجازت دیتے ہیں:

```python
# مثال: روبوٹ کنٹرول کے لیے اشارے کی پہچان
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class GestureRecognitionInterface:
    def __init__(self):
        self.gesture_classifier = RandomForestClassifier(n_estimators=100)
        self.is_trained = False
        self.gesture_commands = {
            'wave': 'approach_user',
            'point': 'move_to_location',
            'stop': 'stop_robot',
            'come_here': 'move_to_user',
            'follow_me': 'follow_user',
            'wait': 'pause_task'
        }
        self.training_data = []

    def train_gesture_classifier(self, gesture_samples, labels):
        """اشارے کے طبقہ بند کو تربیت دیں"""
        X = np.array(gesture_samples)
        y = np.array(labels)
        self.gesture_classifier.fit(X, y)
        self.is_trained = True

    def recognize_gesture(self, gesture_features):
        """خصوصیات سے اشارے کی پہچان کریں"""
        if not self.is_trained:
            return 'unknown_gesture'

        gesture_features = np.array(gesture_features).reshape(1, -1)
        predicted_gesture = self.gesture_classifier.predict(gesture_features)[0]
        confidence = max(self.gesture_classifier.predict_proba(gesture_features)[0])

        return predicted_gesture, confidence

    def process_gesture(self, gesture_data):
        """اشارے کو عمل کریں اور روبوٹ کمانڈ واپس کریں"""
        gesture, confidence = self.recognize_gesture(gesture_data)

        if confidence > 0.7:  # یقین کی حد
            command = self.gesture_commands.get(gesture, 'unknown_command')
            return {'command': command, 'confidence': confidence}
        else:
            return {'command': 'uncertain_gesture', 'confidence': confidence}

    def add_training_sample(self, gesture_features, gesture_label):
        """اشارے کی پہچان کے لیے تربیت کا نمونہ شامل کریں"""
        self.training_data.append((gesture_features, gesture_label))
```

### 2. آواز-مبنی انٹرفیس

آواز انٹرفیس روبوٹس کے ساتھ قدرتی زبان کی بات چیت کو فعال کرتے ہیں:

```python
# مثال: آواز کمانڈ انٹرفیس
class VoiceCommandInterface:
    def __init__(self):
        self.command_keywords = {
            'move forward': 'move_forward',
            'move backward': 'move_backward',
            'turn left': 'turn_left',
            'turn right': 'turn_right',
            'stop': 'stop',
            'go to': 'navigate_to',
            'come to me': 'come_to_user',
            'follow me': 'follow_user',
            'pick up': 'pick_up_object',
            'put down': 'put_down_object',
            'hello': 'greet',
            'goodbye': 'farewell'
        }
        self.location_keywords = ['kitchen', 'living room', 'bedroom', 'office', 'dining room']
        self.object_keywords = ['cup', 'book', 'phone', 'bottle', 'toy']

    def parse_voice_command(self, voice_text):
        """آواز کمانڈ کو پارس کریں اور مقصد نکالیں"""
        voice_text = voice_text.lower().strip()
        words = voice_text.split()

        # کمانڈ کلیدی الفاظ کی جانچ کریں
        for i in range(len(words)):
            for cmd_len in range(3, 0, -1):  # 3 الفاظ تک کے جملوں کی جانچ کریں
                if i + cmd_len <= len(words):
                    phrase = ' '.join(words[i:i + cmd_len])
                    if phrase in self.command_keywords:
                        command = self.command_keywords[phrase]

                        # اضافی پیرامیٹر نکالیں
                        parameters = self.extract_parameters(words, i + cmd_len)

                        return {
                            'command': command,
                            'parameters': parameters,
                            'confidence': 0.9
                        }

        return {'command': 'unknown', 'parameters': {}, 'confidence': 0.0}

    def extract_parameters(self, words, start_index):
        """کمانڈ سے مقام یا چیز کے پیرامیٹر نکالیں"""
        parameters = {}

        for i in range(start_index, len(words)):
            word = words[i]

            # مقام کی جانچ کریں
            for location in self.location_keywords:
                if location in word or word in location:
                    parameters['location'] = location
                    break

            # چیز کی جانچ کریں
            for obj in self.object_keywords:
                if obj in word or word in obj:
                    parameters['object'] = obj
                    break

        return parameters

    def generate_response(self, command_result):
        """قدرتی زبان کا جواب تیار کریں"""
        responses = {
            'move_forward': "ٹھیک ہے، آگے بڑھ رہا ہوں۔",
            'move_backward': "ٹھیک ہے، پیچھے جا رہا ہوں۔",
            'turn_left': "بائیں مڑ رہا ہوں۔",
            'turn_right': "دائیں مڑ رہا ہوں۔",
            'stop': "رک رہا ہوں۔",
            'navigate_to': " {} کی طرف جا رہا ہوں۔".format(command_result.get('location', 'destination')),
            'unknown': "مجھے وہ کمانڈ سمجھ نہیں آئی۔",
            'uncertain': "کیا آپ دہرائیں گے؟"
        }

        cmd = command_result.get('command', 'unknown')
        return responses.get(cmd, "کمانڈ انجام دی گئی۔")

# مثال: سیاق و سباق کے بارے میں آگہی رکھنے والی آواز انٹرفیس
class ContextAwareVoiceInterface:
    def __init__(self):
        self.voice_interface = VoiceCommandInterface()
        self.context = {
            'current_task': None,
            'user_location': None,
            'robot_location': None,
            'available_objects': [],
            'navigation_goals': []
        }

    def parse_contextual_command(self, voice_text):
        """موجودہ سیاق و سباق کو مدنظر رکھتے ہوئے کمانڈ پارس کریں"""
        basic_result = self.voice_interface.parse_voice_command(voice_text)

        # سیاق و سباق کے ساتھ بہتر بنائیں
        if basic_result['command'] == 'navigate_to' and 'location' not in basic_result['parameters']:
            # سیاق و سباق کا استعمال کرتے ہوئے مقام کا اندازہ لگائیں
            inferred_location = self.infer_location_from_context()
            if inferred_location:
                basic_result['parameters']['location'] = inferred_location

        return basic_result

    def infer_location_from_context(self):
        """موجودہ سیاق و سباق سے مقام کا اندازہ لگائیں"""
        # یہ موجودہ کام، صارف کے مقام، وغیرہ کا استعمال کرے گا
        # مثال کے طور پر، اگر صارف کچن میں ہے اور "چلیں" کہتا ہے لیکن مقام کے بغیر،
        # روبوٹ اگلا منطقی مقام پر جا سکتا ہے
        return None
```

### 3. چھوٹے-مبنی انٹرفیس

چھوٹے انٹرفیس براہ راست، چھوٹے سے بات چیت کو فعال کرتے ہیں:

```python
# مثال: چھوٹے-مبنی انٹرفیس
class TouchInterface:
    def __init__(self):
        self.touch_zones = {
            'head': 'head_touch',
            'chest': 'chest_touch',
            'hand': 'hand_touch',
            'arm': 'arm_touch',
            'shoulder': 'shoulder_touch'
        }
        self.touch_patterns = {
            'single_tap': 'acknowledge',
            'double_tap': 'confirm',
            'long_press': 'activate',
            'swipe_up': 'increase',
            'swipe_down': 'decrease',
            'swipe_left': 'previous',
            'swipe_right': 'next'
        }

    def process_touch(self, location, pattern, duration):
        """چھوٹے کی بات چیت کو عمل کریں"""
        if location in self.touch_zones and pattern in self.touch_patterns:
            action = self.touch_patterns[pattern]
            return {
                'action': action,
                'location': location,
                'duration': duration,
                'command': self.map_touch_to_command(location, action)
            }
        return {'action': 'unknown', 'location': location, 'command': 'none'}

    def map_touch_to_command(self, location, action):
        """چھوٹے کے مقام اور عمل کو روبوٹ کمانڈ میں میپ کریں"""
        command_mapping = {
            ('head', 'acknowledge'): 'nod_head',
            ('head', 'confirm'): 'yes_response',
            ('chest', 'acknowledge'): 'heart_symbol',
            ('hand', 'acknowledge'): 'hand_wave',
            ('shoulder', 'activate'): 'wake_up',
            ('chest', 'long_press'): 'shutdown'
        }
        return command_mapping.get((location, action), 'no_command')
```

## ROS2 نفاذ: مسلسل روبوٹ انٹرفیس

یہاں مسلسل انٹرفیس کا ایک جامع ROS2 نفاذ ہے:

```python
# intuitive_robot_interface.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState, Imu
from geometry_msgs.msg import Twist, Pose, Point
from std_msgs.msg import String, Bool, Float32
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
import numpy as np
import cv2
from collections import deque

class IntuitiveRobotInterface(Node):
    def __init__(self):
        super().__init__('intuitive_robot_interface')

        # پبلشرز
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.speech_pub = self.create_publisher(String, '/tts_input', 10)
        self.interface_status_pub = self.create_publisher(String, '/interface_status', 10)
        self.response_pub = self.create_publisher(String, '/interface_response', 10)

        # سبسکرائبرز
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.voice_cmd_sub = self.create_subscription(
            String, '/speech_to_text', self.voice_command_callback, 10
        )
        self.touch_cmd_sub = self.create_subscription(
            String, '/touch_interface', self.touch_command_callback, 10
        )

        # انٹرفیس کمپوننٹس
        self.cv_bridge = CvBridge()
        self.gesture_interface = GestureRecognitionInterface()
        self.voice_interface = VoiceCommandInterface()
        self.touch_interface = TouchInterface()
        self.direct_manipulation = DirectManipulationInterface()
        self.consistent_interface = ConsistentRobotInterface()

        # ڈیٹا اسٹوریج
        self.image_data = None
        self.joint_data = None
        self.imu_data = None
        self.voice_command = None
        self.touch_command = None

        # انٹرفیس کی حالت
        self.interface_mode = 'gesture'  # gesture, voice, touch, direct
        self.active_interfaces = {
            'gesture': True,
            'voice': True,
            'touch': True,
            'direct': True
        }
        self.user_attention = True  # چاہے روبوٹ صارف کو دیکھ رہا ہو

        # انٹرفیس پروسیسنگ
        self.gesture_buffer = deque(maxlen=10)
        self.voice_buffer = deque(maxlen=5)
        self.response_history = deque(maxlen=20)

        # کنٹرول لوپ
        self.interface_timer = self.create_timer(0.05, self.interface_control_loop)

    def image_callback(self, msg):
        """اشارے کی پہچان کے لیے کیمرہ امیج کو سنبھالیں"""
        try:
            self.image_data = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Image callback error: {e}')

    def joint_callback(self, msg):
        """جوائنٹ اسٹیٹ ڈیٹا کو سنبھالیں"""
        self.joint_data = msg

    def imu_callback(self, msg):
        """اشارے کی پہچان کے لیے IMU ڈیٹا کو سنبھالیں"""
        self.imu_data = msg

    def voice_command_callback(self, msg):
        """آواز کمانڈز کو سنبھالیں"""
        self.voice_command = msg.data
        self.voice_buffer.append(msg.data)

    def touch_command_callback(self, msg):
        """چھوٹے کمانڈز کو سنبھالیں"""
        self.touch_command = msg.data

    def interface_control_loop(self):
        """اصل انٹرفیس کنٹرول لوپ"""
        # تمام فعال انٹرفیسز کو عمل کریں
        if self.active_interfaces['gesture'] and self.image_data is not None:
            self.process_gesture_interface()

        if self.active_interfaces['voice'] and self.voice_command is not None:
            self.process_voice_interface()

        if self.active_interfaces['touch'] and self.touch_command is not None:
            self.process_touch_interface()

        # انٹرفیس کی حالت اپ ڈیٹ کریں
        self.publish_interface_status()

    def process_gesture_interface(self):
        """اشارے-مبنی انٹرفیس کو عمل کریں"""
        # امیج اور IMU ڈیٹا سے اشارے کی خصوصیات نکالیں
        gesture_features = self.extract_gesture_features()

        if gesture_features:
            # اشارے کی پہچان کریں
            gesture_result = self.gesture_interface.process_gesture(gesture_features)

            if gesture_result['confidence'] > 0.7:
                # اشارے کی کمانڈ انجام دیں
                command = gesture_result['command']
                self.execute_robot_command(command)

                # جواب تیار کریں
                response = f"اشارے کی پہچان: {command}"
                self.response_pub.publish(String(data=response))

    def extract_gesture_features(self):
        """اشارے کی پہچان کے لیے خصوصیات نکالیں"""
        if self.image_data is not None:
            # خصوصیات نکالنے کا سادہ طریقہ (عمل میں، زیادہ جامع طریقے استعمال کریں)
            gray = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2GRAY)

            # سادہ رنگ-مبنی ڈیٹکشن کا استعمال کرتے ہوئے ہاتھ کا پتہ لگائیں
            hsv = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 20, 70])
            upper_skin = np.array([20, 255, 255])
            mask = cv2.inRange(hsv, lower_skin, upper_skin)

            # کنٹور تلاش کریں
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # سب سے بڑا کنٹور حاصل کریں (فرض کریں ہاتھ ہے)
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 1000:  # کم از کم سائز
                    # خصوصیات کا حساب لگائیں
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        # شکل کی خصوصیات حاصل کرنے کے لیے تقریباً کنٹور
                        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

                        features = [cx, cy, len(approx), cv2.contourArea(largest_contour)]
                        return features

        return None

    def process_voice_interface(self):
        """آواز-مبنی انٹرفیس کو عمل کریں"""
        if self.voice_command:
            # آواز کمانڈ کو پارس کریں
            command_result = self.voice_interface.parse_voice_command(self.voice_command)

            if command_result['confidence'] > 0.5:
                # آواز کمانڈ انجام دیں
                command = command_result['command']
                parameters = command_result['parameters']

                self.execute_robot_command(command, parameters)

                # زبانی جواب تیار کریں
                response = self.voice_interface.generate_response(command_result)
                self.speech_pub.publish(String(data=response))

            # عمل کے بعد کمانڈ صاف کریں
            self.voice_command = None

    def process_touch_interface(self):
        """چھوٹے-مبنی انٹرفیس کو عمل کریں"""
        if self.touch_command:
            # چھوٹے کمانڈ کو پارس کریں (سادہ)
            parts = self.touch_command.split(':')
            if len(parts) >= 2:
                location = parts[0]
                pattern = parts[1]
                duration = float(parts[2]) if len(parts) > 2 else 0.0

                # چھوٹے کو عمل کریں
                touch_result = self.touch_interface.process_touch(location, pattern, duration)

                if touch_result['command'] != 'none':
                    self.execute_robot_command(touch_result['command'])

            # عمل کے بعد کمانڈ صاف کریں
            self.touch_command = None

    def execute_robot_command(self, command, parameters=None):
        """انٹرفیس ان پٹ کی بنیاد پر روبوٹ کمانڈ انجام دیں"""
        if parameters is None:
            parameters = {}

        cmd = Twist()

        # انٹرفیس کمانڈز کو روبوٹ حرکات میں میپ کریں
        if command == 'move_forward':
            cmd.linear.x = 0.3
        elif command == 'move_backward':
            cmd.linear.x = -0.2
        elif command == 'turn_left':
            cmd.angular.z = 0.5
        elif command == 'turn_right':
            cmd.angular.z = -0.5
        elif command == 'stop':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        elif command == 'approach_user':
            # صارف کی طرف جائیں (سادہ - صارف کی پہچان کی ضرورت ہوگی)
            cmd.linear.x = 0.2
        elif command == 'greet':
            # روبوٹ کا سلام کا رویہ
            self.speech_pub.publish(String(data="ہیلو! میں آپ کی کیسے مدد کر سکتا ہوں؟"))
            cmd.angular.z = 0.5  # تصدیق کے لیے تھوڑا مڑیں
        elif command == 'navigate_to':
            # مخصوص مقام پر نیویگیٹ کریں
            location = parameters.get('location', 'unknown')
            self.get_logger().info(f'Navigating to {location}')
            cmd.linear.x = 0.3  # نیویگیشن شروع کرنے کے لیے آگے بڑھیں

        # کمانڈ شائع کریں
        self.cmd_vel_pub.publish(cmd)

    def publish_interface_status(self):
        """موجودہ انٹرفیس کی حالت شائع کریں"""
        status_msg = String()
        status_msg.data = (
            f"Mode: {self.interface_mode}, "
            f"Gesture: {self.active_interfaces['gesture']}, "
            f"Voice: {self.active_interfaces['voice']}, "
            f"Touch: {self.active_interfaces['touch']}, "
            f"Attention: {self.user_attention}"
        )
        self.interface_status_pub.publish(status_msg)

class GestureTracker:
    """اشاروں کو وقت کے ساتھ ٹریک اور تشریح کریں"""
    def __init__(self):
        self.gesture_history = deque(maxlen=20)
        self.current_gesture = None
        self.gesture_threshold = 5  # تصدیق کے لیے کم از کم فریم

    def update_gesture(self, gesture_features):
        """نئی خصوصیات کے ساتھ اشارے ٹریکنگ اپ ڈیٹ کریں"""
        if gesture_features:
            self.gesture_history.append(gesture_features)

            # اشارے کی ترتیب کا تجزیہ کریں
            if len(self.gesture_history) >= self.gesture_threshold:
                self.current_gesture = self.analyze_gesture_sequence()

    def analyze_gesture_sequence(self):
        """اشارے کی خصوصیات کی ترتیب کا تجزیہ کریں"""
        # یہ وقت کے لحاظ سے اشارے کی پہچان کو نافذ کرے گا
        # اشاروں کے عارضی پہلوؤں کو مدنظر رکھتے ہوئے
        return "unknown"

class MultimodalInterfaceFusion:
    """کئی انٹرفیس ماڈلٹیز کو ضم کریں"""
    def __init__(self):
        self.interfaces = {
            'gesture': GestureRecognitionInterface(),
            'voice': VoiceCommandInterface(),
            'touch': TouchInterface()
        }
        self.fusion_weights = {
            'gesture': 0.4,
            'voice': 0.4,
            'touch': 0.2
        }
        self.confidence_threshold = 0.6

    def fuse_inputs(self, gesture_input, voice_input, touch_input):
        """کئی انٹرفیسز سے ان پٹس کو ضم کریں"""
        fused_result = {}

        # ہر ان پٹ ماڈلٹی کو عمل کریں
        gesture_result = self.process_gesture(gesture_input) if gesture_input else None
        voice_result = self.process_voice(voice_input) if voice_input else None
        touch_result = self.process_touch(touch_input) if touch_input else None

        # یقین کی بنیاد پر وزن دی گئی ضم کریں
        results = []
        if gesture_result and gesture_result.get('confidence', 0) > self.confidence_threshold:
            results.append(('gesture', gesture_result))

        if voice_result and voice_result.get('confidence', 0) > self.confidence_threshold:
            results.append(('voice', voice_result))

        if touch_result and touch_result.get('confidence', 0) > self.confidence_threshold:
            results.append(('touch', touch_result))

        # زیادہ یقین والے نتیجے کو منتخب کریں یا اگر مماثل ہوں تو ضم کریں
        if results:
            # یقین کے لحاظ سے ترتیب دیں
            results.sort(key=lambda x: x[1].get('confidence', 0), reverse=True)
            return results[0][1]  # سب سے زیادہ یقین والے نتیجے کو لوٹائیں

        return {'command': 'no_input', 'confidence': 0.0}

    def process_gesture(self, gesture_input):
        """اشارے ان پٹ کو عمل کریں"""
        return self.interfaces['gesture'].process_gesture(gesture_input)

    def process_voice(self, voice_input):
        """آواز ان پٹ کو عمل کریں"""
        return self.interfaces['voice'].parse_voice_command(voice_input)

    def process_touch(self, touch_input):
        """چھوٹے ان پٹ کو عمل کریں"""
        # چھوٹے ان پٹ فارمیٹ کو پارس کریں: "location:pattern:duration"
        parts = touch_input.split(':')
        if len(parts) >= 2:
            location = parts[0]
            pattern = parts[1]
            duration = float(parts[2]) if len(parts) > 2 else 0.0

            result = self.interfaces['touch'].process_touch(location, pattern, duration)
            return {
                'command': result['command'],
                'confidence': 0.8,  # چھوٹے کے لیے زیادہ یقین
                'interface': 'touch'
            }
        return None

def main(args=None):
    rclpy.init(args=args)
    interface_node = IntuitiveRobotInterface()

    try:
        rclpy.spin(interface_node)
    except KeyboardInterrupt:
        pass
    finally:
        interface_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## اعلی درجے کے انٹرفیس کے تصورات

### موافق انٹرفیس

انٹرفیس جو صارف کی ترجیحات اور صلاحیات کے مطابق موافق ہوتے ہیں:

```python
# مثال: موافق انٹرفیس سسٹم
class AdaptiveInterfaceSystem:
    def __init__(self):
        self.user_profiles = {}
        self.interface_preferences = {}
        self.adaptation_engine = self.initialize_adaptation_engine()

    def initialize_adaptation_engine(self):
        """انٹرفیس موافقت کے لیے سسٹم شروع کریں"""
        return {
            'preference_learner': self.learn_user_preferences,
            'ability_assessor': self.assess_user_ability,
            'interface_optimizer': self.optimize_interface
        }

    def learn_user_preferences(self, user_id, interaction_history):
        """تعامل کی تاریخ سے صارف کی ترجیحات سیکھیں"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'preferred_modality': 'voice',  # voice, gesture, touch, etc.
                'interaction_style': 'direct',   # direct, indirect, etc.
                'response_speed': 'normal',      # fast, normal, slow
                'complexity_level': 'simple'     # simple, moderate, complex
            }

        # تاریخ کی بنیاد پر ترجیحات اپ ڈیٹ کریں
        for interaction in interaction_history:
            # تعامل کے نمونوں کا تجزیہ کریں اور ترجیحات اپ ڈیٹ کریں
            pass

    def assess_user_ability(self, user_id):
        """صارف کی مختلف انٹرفیس ماڈلٹیز کا استعمال کرنے کی صلاحیت کا جائزہ لیں"""
        abilities = {
            'motor_skills': 0.8,      # 0-1 سکیل
            'speech_clarity': 0.9,    # 0-1 سکیل
            'visual_acuity': 0.7,     # 0-1 سکیل
            'cognitive_load': 0.3     # 0-1 سکیل (کم بہتر ہے)
        }
        return abilities

    def optimize_interface(self, user_id, context):
        """مخصوص صارف اور سیاق و سباق کے لیے انٹرفیس کو بہتر بنائیں"""
        user_profile = self.user_profiles.get(user_id, {})
        user_abilities = self.assess_user_ability(user_id)

        # بہترین انٹرفیس کنفیگریشن کا تعین کریں
        optimal_config = {
            'primary_modality': self.select_primary_modality(user_abilities),
            'feedback_style': self.select_feedback_style(user_profile),
            'interaction_complexity': self.select_complexity_level(user_abilities)
        }

        return optimal_config

    def select_primary_modality(self, abilities):
        """صارف کی صلاحیات کی بنیاد پر بنیادی انٹرفیس ماڈلٹی منتخب کریں"""
        if abilities['speech_clarity'] > 0.7:
            return 'voice'
        elif abilities['motor_skills'] > 0.7:
            return 'gesture'
        else:
            return 'touch'

    def select_feedback_style(self, profile):
        """صارف کے پروفائل کی بنیاد پر فیڈ بیک انداز منتخب کریں"""
        style_preferences = {
            'direct': {'visual': 0.7, 'auditory': 0.3},
            'cautious': {'visual': 0.5, 'auditory': 0.5},
            'efficient': {'visual': 0.3, 'auditory': 0.7}
        }
        return style_preferences.get(profile.get('interaction_style', 'direct'), {})

    def select_complexity_level(self, abilities):
        """صارف کی صلاحیات کی بنیاد پر انٹرفیس کی پیچیدگی منتخب کریں"""
        avg_ability = sum(abilities.values()) / len(abilities)
        if avg_ability > 0.8:
            return 'complex'
        elif avg_ability > 0.5:
            return 'moderate'
        else:
            return 'simple'
```

### سیاق و سباق کے بارے میں آگاہ انٹرفیس

انٹرفیس جو ماحولیاتی اور صورتحال کے سیاق و سباق کے مطابق موافق ہوتے ہیں:

```python
# مثال: سیاق و سباق کے بارے میں آگاہ انٹرفیس
class ContextAwareInterface:
    def __init__(self):
        self.context_model = self.initialize_context_model()
        self.context_aware_commands = {}

    def initialize_context_model(self):
        """ماحولیاتی سیاق و سباق کا ماڈل شروع کریں"""
        return {
            'location': 'unknown',
            'time_of_day': 'unknown',
            'social_context': 'unknown',  # alone, with family, in public
            'environmental_conditions': {
                'lighting': 'normal',
                'noise_level': 'low',
                'crowd_density': 'low'
            }
        }

    def update_context(self, sensor_data):
        """سینسر ڈیٹا کی بنیاد پر سیاق و سباق اپ ڈیٹ کریں"""
        # مقام اپ ڈیٹ کریں
        if 'location_sensor' in sensor_data:
            self.context_model['location'] = sensor_data['location_sensor']

        # ماحولیاتی حالات اپ ڈیٹ کریں
        if 'light_sensor' in sensor_data:
            light_level = sensor_data['light_sensor']
            self.context_model['environmental_conditions']['lighting'] = (
                'bright' if light_level > 0.8 else 'dim' if light_level < 0.3 else 'normal'
            )

        if 'noise_sensor' in sensor_data:
            noise_level = sensor_data['noise_sensor']
            self.context_model['environmental_conditions']['noise_level'] = (
                'high' if noise_level > 0.7 else 'low'
            )

    def get_adapted_command(self, user_input):
        """موجودہ سیاق و سباق کے مطابق کمانڈ حاصل کریں"""
        base_command = self.parse_user_input(user_input)

        # سیاق و سباق کی بنیاد پر موافق کریں
        if self.context_model['environmental_conditions']['noise_level'] == 'high':
            if base_command['type'] == 'voice':
                # شور میں ماحول، وژوئل تصدیق کو ترجیح دیں
                base_command['require_visual_feedback'] = True

        if self.context_model['environmental_conditions']['lighting'] == 'dim':
            if base_command['type'] == 'gesture':
                # تاریک روشنی میں، آواز یا چھوٹے کو ترجیح دیں
                base_command['suggest_alternative'] = 'voice'

        return base_command

    def parse_user_input(self, user_input):
        """سیاق و سباق کے بارے میں آگاہی کے ساتھ صارف ان پٹ کو پارس کریں"""
        # یہ سیاق و سباق-آگاہی والی پارسنگ نافذ کرے گا
        # موجودہ ماحولیاتی سیاق و سباق کو مدنظر رکھتے ہوئے
        return {'type': 'unknown', 'command': user_input, 'confidence': 0.5}
```

## لیب: مسلسل روبوٹ انٹرفیس نافذ کرنا

اس لیب میں، آپ روبوٹ کنٹرول کے لیے ایک مسلسل انٹرفیس نافذ کریں گے:

```python
# lab_intuitive_interface.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import numpy as np
import cv2

class IntuitiveInterfaceLab(Node):
    def __init__(self):
        super().__init__('intuitive_interface_lab')

        # پبلشرز
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/interface_status', 10)
        self.response_pub = self.create_publisher(String, '/interface_response', 10)

        # سبسکرائبرز
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.voice_sub = self.create_subscription(
            String, '/speech_commands', self.voice_callback, 10
        )

        # انٹرفیس کمپوننٹس
        self.cv_bridge = CvBridge()
        self.image_data = None
        self.scan_data = None
        self.voice_command = None

        # انٹرفیس کی حالت
        self.interface_mode = 'gesture'  # gesture, voice, combined
        self.active_gesture = None
        self.user_attention = False
        self.interface_confidence = 0.0

        # اشارے کی پہچان کے پیرامیٹر
        self.hand_lower = np.array([0, 20, 70])
        self.hand_upper = np.array([20, 255, 255])
        self.min_hand_area = 1000

        # کنٹرول لوپ
        self.control_timer = self.create_timer(0.05, self.interface_control_loop)

    def image_callback(self, msg):
        """اشارے کی پہچان کے لیے کیمرہ امیج کو سنبھالیں"""
        try:
            self.image_data = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Image callback error: {e}')

    def scan_callback(self, msg):
        """قریبی ڈیٹکشن کے لیے لیزر اسکین کو سنبھالیں"""
        self.scan_data = msg

    def voice_callback(self, msg):
        """آواز کمانڈز کو سنبھالیں"""
        self.voice_command = msg.data

    def interface_control_loop(self):
        """اصل انٹرفیس کنٹرول لوپ"""
        if self.interface_mode == 'gesture' and self.image_data is not None:
            self.process_gesture_interface()
        elif self.interface_mode == 'voice' and self.voice_command is not None:
            self.process_voice_interface()
        elif self.interface_mode == 'combined':
            self.process_combined_interface()

        # حالت شائع کریں
        self.publish_interface_status()

        # عمل کردہ کمانڈز صاف کریں
        if self.voice_command:
            self.voice_command = None

    def process_gesture_interface(self):
        """اشارے-مبنی انٹرفیس کو عمل کریں"""
        if self.image_data is None:
            return

        # جلد کی پہچان کے لیے HSV میں تبدیل کریں
        hsv = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hand_lower, self.hand_upper)

        # شور کو کم کرنے کے لیے مورفولوجی آپریشنز لاگو کریں
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # کنٹور تلاش کریں
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # سب سے بڑا کنٹور تلاش کریں (ممکنہ طور پر ہاتھ)
            largest_contour = max(contours, key=cv2.contourArea)

            if cv2.contourArea(largest_contour) > self.min_hand_area:
                # ہاتھ کا مرکز حاصل کریں
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # ہاتھ کی پوزیشن کی بنیاد پر اشارے کی قسم بتائیں
                    gesture = self.classify_gesture(cx, cy, self.image_data.shape)

                    # اشارے کمانڈ انجام دیں
                    cmd = self.gesture_to_command(gesture)
                    self.cmd_pub.publish(cmd)

                    # انٹرفیس کی حالت اپ ڈیٹ کریں
                    self.active_gesture = gesture
                    self.interface_confidence = 0.9
                    self.user_attention = True

                    # جواب شائع کریں
                    response = f"اشارے کی پہچان: {gesture}"
                    self.response_pub.publish(String(data=response))
            else:
                # کوئی اہم ہاتھ نہیں ملا
                self.active_gesture = None
                self.interface_confidence = 0.0
                self.user_attention = False
        else:
            # کوئی کنٹور نہیں ملا
            self.active_gesture = None
            self.interface_confidence = 0.0
            self.user_attention = False

    def classify_gesture(self, x, y, image_shape):
        """ہاتھ کی پوزیشن کی بنیاد پر اشارے کی قسم بتائیں"""
        height, width = image_shape[:2]

        # مختلف اشاروں کے لیے علاقے کی وضاحت کریں
        if x < width * 0.3:  # بائیں تہائی
            if y < height * 0.3:  # بائیں اوپر
                return 'top_left'
            elif y > height * 0.7:  # بائیں نیچے
                return 'bottom_left'
            else:  # بائیں درمیان
                return 'left'
        elif x > width * 0.7:  # دائیں تہائی
            if y < height * 0.3:  # دائیں اوپر
                return 'top_right'
            elif y > height * 0.7:  # دائیں نیچے
                return 'bottom_right'
            else:  # دائیں درمیان
                return 'right'
        else:  # مرکزی علاقہ
            if y < height * 0.3:  # مرکز اوپر
                return 'up'
            elif y > height * 0.7:  # مرکز نیچے
                return 'down'
            else:  # مرکز
                return 'center'

    def gesture_to_command(self, gesture):
        """اشارے کو روبوٹ کمانڈ میں تبدیل کریں"""
        cmd = Twist()

        if gesture == 'left':
            cmd.angular.z = 0.5  # بائیں مڑیں
        elif gesture == 'right':
            cmd.angular.z = -0.5  # دائیں مڑیں
        elif gesture == 'up':
            cmd.linear.x = 0.3  # آگے بڑھیں
        elif gesture == 'down':
            cmd.linear.x = -0.2  # پیچھے جائیں
        elif gesture == 'center':
            cmd.linear.x = 0.1  # آہستہ آگے بڑھیں
        elif gesture in ['top_left', 'top_right', 'bottom_left', 'bottom_right']:
            # ترچھی حرکات
            cmd.linear.x = 0.2
            cmd.angular.z = 0.3 if 'left' in gesture else -0.3

        return cmd

    def process_voice_interface(self):
        """آواز-مبنی انٹرفیس کو عمل کریں"""
        if not self.voice_command:
            return

        # آواز کمانڈز کے لیے سادہ کلیدی الفاظ میچنگ
        voice_command = self.voice_command.lower()

        cmd = Twist()

        if 'forward' in voice_command or 'go' in voice_command:
            cmd.linear.x = 0.3
            response = "آگے بڑھ رہا ہوں"
        elif 'backward' in voice_command or 'back' in voice_command:
            cmd.linear.x = -0.2
            response = "پیچھے جا رہا ہوں"
        elif 'left' in voice_command:
            cmd.angular.z = 0.5
            response = "بائیں مڑ رہا ہوں"
        elif 'right' in voice_command:
            cmd.angular.z = -0.5
            response = "دائیں مڑ رہا ہوں"
        elif 'stop' in voice_command or 'halt' in voice_command:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            response = "رک رہا ہوں"
        else:
            response = f"نامعلوم کمانڈ: {self.voice_command}"

        # کمانڈ اور جواب شائع کریں
        self.cmd_pub.publish(cmd)
        self.response_pub.publish(String(data=response))
        self.interface_confidence = 0.8

    def process_combined_interface(self):
        """مشترکہ اشارے اور آواز انٹرفیس کو عمل کریں"""
        # دستیاب ہونے پر آواز کمانڈز کو ترجیح دیں، بصورت دیگر اشارے استعمال کریں
        if self.voice_command:
            self.process_voice_interface()
        elif self.image_data is not None:
            self.process_gesture_interface()

    def publish_interface_status(self):
        """موجودہ انٹرفیس کی حالت شائع کریں"""
        status = (
            f"Mode: {self.interface_mode}, "
            f"Gesture: {self.active_gesture}, "
            f"Confidence: {self.interface_confidence:.2f}, "
            f"Attention: {self.user_attention}"
        )
        self.status_pub.publish(String(data=status))

def main(args=None):
    rclpy.init(args=args)
    lab = IntuitiveInterfaceLab()

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

## مشق: اپنا مسلسل انٹرفیس ڈیزائن کریں

مندرجہ ذیل ڈیزائن چیلنج پر غور کریں:

1. آپ کا انٹرفیس کون سا روبوٹ کنٹرول کرے گا (موبائل، مینیپولیٹر، انسان نما)?
2. روبوٹ کے اہم کام کون سے ہوں گے؟
3. کون سی انٹرفیس ماڈلٹیز (اشارے، آواز، چھوٹے، وغیرہ) مناسب ہیں؟
4. آپ یہ کیسے یقینی بنائیں گے کہ آپ کا انٹرفیس آپ کے ہدف کے صارفین کے لیے مسلسل ہے؟
5. انٹرفیس مختلف صارف کی صلاحیات یا ترجیحات کے مطابق کیسے موافق ہوگا؟
6. کون سے فیڈ بیک کے طریقے صارفین کو روبوٹ کے جوابات کو سمجھنے میں مدد کریں گے؟
7. آپ مبہم یا متضاد صارف ان پٹس کو کیسے ہینڈل کریں گے؟

## خلاصہ

مسلسل انٹرفیس مؤثر انسان-روبوٹ بات چیت کے لیے اہم ہیں، جو پیچیدہ روبوٹک نظاموں کو قابل رسائی اور آسان استعمال بناتے ہیں۔ کلیدی تصورات شامل ہیں:

- **قدرتی میپنگ**: انسانی اعمال کو روبوٹک جوابات سے مسلسل طریقے سے جوڑنا
- **براہ راست ہیرا پھیرا**: صارفین کو جسمانی بات چیت کے ذریعے روبوٹس کنٹرول کرنے کی اجازت دینا
- **مسلسل ہونا**: بات چیت میں قابل پیش گوئی سلوک برقرار رکھنا
- **کثیر ماڈلٹی انٹرفیس**: مضبوط بات چیت کے لیے متعدد ان پٹ ماڈلٹیز کو ضم کرنا
- **موافق انٹرفیس**: انٹرفیس جو صارف کی ترجیحات اور صلاحیات کے مطابق موافق ہوتے ہیں
- **سیاق و سباق کی آگاہی**: انٹرفیس جو ماحولیاتی حالات کے مطابق موافق ہوتے ہیں

ROS2 میں ان اصولوں کا انضمام جامع، صارف دوست روبوٹ انٹرفیس ترقی دینے کو فعال کرتا ہے جو مختلف صارفین اور صورتحال کے مطابق موافق ہو سکتے ہیں۔ ان تصورات کو سمجھنا روبوٹس ترقی دینے کے لیے ضروری ہے جو انسانوں کے ساتھ قدرتی طریقے سے بات چیت کر سکیں۔

اگلے سبق میں، ہم روبوٹ کی بات چیت کے لیے قدرتی زبان کی پروسیسنگ کا جائزہ لیں گے، جس میں آواز کی پہچان، زبان کی سمجھ، اور گفتگو کے انٹرفیس شامل ہیں۔