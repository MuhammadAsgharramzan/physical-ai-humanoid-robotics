---
sidebar_position: 1
---

# جسمانی مصنوعی ذہانت میں حسی موٹر انضمام

## تعارف

حسی موٹر انضمام وہ بنیادی عمل ہے جس کے ذریعے جسمانی مصنوعی ذہانت کے نظام حسی ان پٹ کو موٹر آؤٹ پٹ کے ساتھ جوڑ کر مربوط، مقصد کے مطابق رویہ پیدا کرتے ہیں۔ روایتی مصنوعی ذہانت کے نظاموں کے برعکس جو معلومات کو علیحدگی میں عمل کرتے ہیں، جسمانی ایجنٹس کو اپنے ماحول کے ساتھ مؤثر بات چیت کے لیے ادراک اور عمل کو مسلسل ضم کرنا چاہیے۔

## حسی موٹر لوپ

حسی موٹر لوپ جامع ذہانت کا اصل میکنزم ہے:

```
ماحول → سینسر → ادراک → عمل کا انتخاب → ایفیکٹرز → ماحول
     ↑______________________________________________________________|
```

یہ مسلسل لوپ اسے قابل بناتا ہے:

- **حقوقی وقت میں مطابقت**: ماحولیاتی تبدیلیوں کا فوری جواب
- **جسمانی سیکھنا**: صرف مشاہدہ کے بجائے بات چیت کے ذریعہ سیکھنا
- **نمودار ہونے والے رویے**: سادہ حسی موٹر قواعد سے پیچیدہ رویے
- ** مضبوطی**: فیڈ بیک کے ذریعہ قدرتی خامی کی اصلاح

## جسمانی مصنوعی ذہانت میں حسی نظام

### سینسر کی اقسام

جسمانی مصنوعی ذہانت کے نظام عام طور پر متعدد حسی ماڈلز کو ضم کرتے ہیں:

1. **پروپریوسیف سینسر**: اندرونی سینسرز جو روبوٹ کی حالت کو پیمائش کرتے ہیں
   - جوائنٹ اینکوڈرز (پوزیشن، رفتار)
   - انرٹیل میزورمنٹ یونٹس (IMU)
   - فورس/ٹورک سینسرز
   - موٹر مانیٹرنگ کے لیے کرنٹ سینسرز

2. **ایکسٹروپریوسیف سینسر**: بیرونی ماحول کے سینسرز
   - کیمرے (وژن)
   - لیڈار (رینج سینسنگ)
   - الٹرا سونک سینسرز (نزدیکی)
   - ٹیکٹائل سینسرز (چھونا)

3. **انٹروپریوسیف سینسرز**: اندرونی حالت کے سینسرز
   - ٹمپریچر سینسرز
   - بیٹری کی سطح کے مانیٹر
   - توانائی کے استعمال کے مانیٹر

### سینسر فیوژن کی مثال
```python
# مثال: حالت کے اندازے کے لیے سینسر فیوژن
import numpy as np
from scipy.spatial.transform import Rotation as R

class SensorFusion:
    def __init__(self):
        self.imu_data = None
        self.camera_data = None
        self.lidar_data = None
        self.joint_encoders = None

        # کیلمن فلٹر پیرامیٹر
        self.state = np.zeros(13)  # [position, orientation, velocity, angular_velocity]
        self.covariance = np.eye(13) * 100  # ابتدائی عدم یقینی

    def update_from_imu(self, linear_accel, angular_vel, dt):
        """IMU ڈیٹا سے حالت کا اندازہ اپ ڈیٹ کریں"""
        # IMU پیمائش کو ضم کریں
        # یہ ایک سادہ مثال ہے - حقیقی نفاذ کیلمن فلٹرنگ کا استعمال کرے گا
        pass

    def update_from_camera(self, visual_features):
        """بصری خصوصیات سے حالت کا اندازہ اپ ڈیٹ کریں"""
        # حالت کی تصحیح کے لیے بصری خصوصیات استعمال کریں
        pass

    def update_from_lidar(self, range_measurements):
        """لیڈار ڈیٹا سے حالت کا اندازہ اپ ڈیٹ کریں"""
        # حالت کی تصدیق کے لیے رینج ڈیٹا استعمال کریں
        pass

    def get_fused_state(self):
        """ضم شدہ حالت کا اندازہ واپس کریں"""
        return self.state
```

## موٹر نظام اور کنٹرول

### ایکچوایٹر کی اقسام

جسمانی مصنوعی ذہانت کے نظام مختلف ایکچوایٹر کی اقسام استعمال کرتے ہیں:

1. **راؤنڈری ایکچوایٹر**: سرو موٹر، سٹیپر موٹر
2. **لینئر ایکچوایٹر**: پنومیٹک، ہائیڈرولک، یا الیکٹرک لینئر ایکچوایٹر
3. **نرم ایکچوایٹر**: پنومیٹک نیٹ ورک، شیپ میموری الائے

### کنٹرول آرکیٹیکچر
```python
# مثال: سلسلہ دار موٹر کنٹرول سسٹم
class MotorController:
    def __init__(self, joint_count):
        self.joint_count = joint_count
        self.joint_positions = np.zeros(joint_count)
        self.joint_velocities = np.zeros(joint_count)
        self.joint_efforts = np.zeros(joint_count)

        # ہر جوائنٹ کے لیے PID کنٹرولر
        self.pid_controllers = [PIDController() for _ in range(joint_count)]

    def compute_joint_commands(self, desired_positions, dt):
        """تمام جوائنٹس کے لیے حکم کا حساب لگائیں"""
        commands = []
        for i in range(self.joint_count):
            command = self.pid_controllers[i].compute(
                desired_positions[i],
                self.joint_positions[i],
                dt
            )
            commands.append(command)
        return commands

    def update_joint_states(self, current_positions, current_velocities):
        """موجودہ جوائنٹ ویلیوز کے ساتھ اندرونی حالت کو اپ ڈیٹ کریں"""
        self.joint_positions = current_positions
        self.joint_velocities = current_velocities

class PIDController:
    def __init__(self, kp=1.0, ki=0.1, kd=0.05):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_error = 0.0
        self.previous_error = 0.0

    def compute(self, desired, current, dt):
        """PID کنٹرول آؤٹ پٹ کا حساب لگائیں"""
        error = desired - current
        self.integral_error += error * dt
        derivative_error = (error - self.previous_error) / dt if dt > 0 else 0

        output = (self.kp * error +
                 self.ki * self.integral_error +
                 self.kd * derivative_error)

        self.previous_error = error
        return output
```

## حسی موٹر کوآرڈی نیشن پیٹرن

### ریفلیکس رویے

سادہ حسی موٹر کوآرڈی نیشن پیٹرن زیادہ پیچیدہ رویوں کی بنیاد بناتے ہیں:

```python
# مثال: ریفلیکس رویے
class ReflexiveBehaviors:
    def __init__(self):
        self.safety_thresholds = {
            'collision_distance': 0.3,  # میٹر
            'temperature_limit': 60,    # سیلسیس
            'current_limit': 10.0       # امپئر
        }

    def collision_avoidance_reflex(self, distance_sensors):
        """رکاوٹ سے بچنے کے لیے سادہ ریفلیکس"""
        if min(distance_sensors) < self.safety_thresholds['collision_distance']:
            # فوری اسٹاپ کمانڈ
            return {'linear_vel': 0.0, 'angular_vel': 0.0}
        else:
            return None  # کوئی ریفلیکس ایکشن کی ضرورت نہیں

    def thermal_protection_reflex(self, temperatures):
        """گرمی سے بچنے کے لیے ریفلیکس"""
        if max(temperatures) > self.safety_thresholds['temperature_limit']:
            # موٹر کی توانائی کم کریں
            return {'power_reduction': 0.5}
        else:
            return None

    def current_limit_reflex(self, currents):
        """موٹر کے نقصان سے بچنے کے لیے ریفلیکس"""
        if max(currents) > self.safety_thresholds['current_limit']:
            # ایمرجنسی اسٹاپ
            return {'emergency_stop': True}
        else:
            return None
```

### ریڈمیک پیٹرن

بہت سے بائیولوجیکل نظام چلنے کی صلاحیت اور دیگر رویوں کے لیے ریڈمیک پیٹرن استعمال کرتے ہیں:

```python
# مثال: ریڈمیک موشن کے لیے سینٹرل پیٹرن جنریٹر
import numpy as np
import math

class CentralPatternGenerator:
    def __init__(self, frequency=1.0, amplitude=1.0):
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = 0.0
        self.time = 0.0

    def update(self, dt):
        """پیٹرن جنریٹر کو اپ ڈیٹ کریں"""
        self.time += dt
        self.phase = self.frequency * self.time * 2 * math.pi
        return self.get_output()

    def get_output(self):
        """پیٹرن جنریٹر کا موجودہ آؤٹ پٹ حاصل کریں"""
        # ریڈمیک آؤٹ پٹ پیدا کریں (مثلاً چلنے کے لیے)
        left_leg = self.amplitude * math.sin(self.phase)
        right_leg = self.amplitude * math.sin(self.phase + math.pi)  # فیز سے باہر
        return {'left_leg': left_leg, 'right_leg': right_leg}
```

## ROS2 نفاذ: حسی موٹر انضمام

یہاں ROS2 کا استعمال کرتے ہوئے حسی موٹر انضمام کی ایک مکمل مثال ہے:

```python
# sensorimotor_integration_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, LaserScan, Imu, Image
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge
import numpy as np
import cv2

class SensorimotorIntegrationNode(Node):
    def __init__(self):
        super().__init__('sensorimotor_integration')

        # پبلشرز
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.safety_pub = self.create_publisher(Bool, '/safety_status', 10)

        # سبسکرائبرز
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        # سینسر ڈیٹا اسٹوریج
        self.joint_states = None
        self.laser_data = None
        self.imu_data = None
        self.image_data = None

        # کنٹرول اجزاء
        self.motor_controller = MotorController(12)  # 12 جوائنٹس کی مثال
        self.reflexive_behaviors = ReflexiveBehaviors()
        self.cpg = CentralPatternGenerator(frequency=0.5)
        self.cv_bridge = CvBridge()

        # کنٹرول ٹائمر
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

        # رویہ کی حالت
        self.current_behavior = 'idle'
        self.behavior_params = {}

    def joint_state_callback(self, msg):
        """جوائنٹ حالت کی تازہ کاریوں کو سنبھالیں"""
        self.joint_states = msg
        self.motor_controller.update_joint_states(
            np.array(msg.position),
            np.array(msg.velocity)
        )

    def laser_callback(self, msg):
        """لیزر اسکین ڈیٹا کو سنبھالیں"""
        self.laser_data = msg

    def imu_callback(self, msg):
        """IMU ڈیٹا کو سنبھالیں"""
        self.imu_data = msg

    def image_callback(self, msg):
        """کیمرہ امیج ڈیٹا کو سنبھالیں"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_data = cv_image
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def control_loop(self):
        """اصل حسی موٹر انضمام لوپ"""
        if not all([self.joint_states, self.laser_data, self.imu_data]):
            return

        # 1. حسی ان پٹ کو عمل کریں
        sensor_data = self.process_sensors()

        # 2. تحفظ کے لیے ریفلیکس رویے لاگو کریں
        reflex_action = self.apply_reflexive_behaviors(sensor_data)
        if reflex_action:
            self.execute_action(reflex_action)
            return  # تحفظ کے ریفلیکس کو ترجیح دیں

        # 3. حالت کے مطابق رویہ منتخب کریں اور انجام دیں
        action = self.select_behavior(sensor_data)
        self.execute_action(action)

        # 4. تحفظ کی حالت کو اپ ڈیٹ کریں
        self.safety_pub.publish(Bool(data=True))

    def process_sensors(self):
        """تمام سینسر ڈیٹا کو ایک مربوط نمائندگی میں عمل کریں"""
        sensor_data = {
            'joint_positions': np.array(self.joint_states.position) if self.joint_states else np.array([]),
            'joint_velocities': np.array(self.joint_states.velocity) if self.joint_states else np.array([]),
            'laser_ranges': np.array(self.laser_data.ranges) if self.laser_data else np.array([]),
            'imu_orientation': self.imu_data.orientation if self.imu_data else None,
            'imu_angular_velocity': self.imu_data.angular_velocity if self.imu_data else None,
            'image_features': self.extract_image_features() if self.image_data is not None else None
        }
        return sensor_data

    def apply_reflexive_behaviors(self, sensor_data):
        """فوری تحفظ کے جوابات کے لیے ریفلیکس رویے لاگو کریں"""
        # رکاوٹ سے بچنے کا ریفلیکس
        if len(sensor_data['laser_ranges']) > 0:
            min_distance = min([r for r in sensor_data['laser_ranges'] if r > 0], default=float('inf'))
            if min_distance < 0.3:  # 30cm تحفظ کا فاصلہ
                self.get_logger().warn(f'Collision imminent: {min_distance:.2f}m')
                return {'linear_vel': 0.0, 'angular_vel': 0.0}

        # توازن کا ریفلیکس
        if sensor_data['imu_orientation']:
            # چیک کریں کہ کیا روبوٹ بہت جھک گیا ہے
            orientation = sensor_data['imu_orientation']
            # سادہ چیک - حقیقت میں مناسب کووٹرین ریاضی استعمال کرے گا
            if abs(orientation.z) > 0.5:  # بہت جھکا ہوا
                self.get_logger().warn('Balance at risk - applying correction')
                return {'linear_vel': 0.0, 'angular_vel': 0.5}  # اصلاح کی کوشش کریں

        return None

    def select_behavior(self, sensor_data):
        """موجودہ حالت اور سینسر ڈیٹا کے مطابق رویہ منتخب کریں"""
        # سینسر ڈیٹا کے مطابق سادہ رویہ کا انتخاب
        if len(sensor_data['laser_ranges']) > 0:
            front_clear = all(r > 1.0 for r in sensor_data['laser_ranges'][300:600] if r > 0)

            if front_clear:
                return self.go_forward_behavior()
            else:
                return self.avoid_obstacle_behavior(sensor_data['laser_ranges'])
        else:
            return {'linear_vel': 0.0, 'angular_vel': 0.0}

    def go_forward_behavior(self):
        """سادہ آگے بڑھنے کا رویہ"""
        return {'linear_vel': 0.3, 'angular_vel': 0.0}

    def avoid_obstacle_behavior(self, laser_ranges):
        """رکاوٹ سے بچنے کا رویہ"""
        # صاف ترین سمت تلاش کریں
        left_clear = sum(r > 1.0 for r in laser_ranges[0:180] if r > 0)
        right_clear = sum(r > 1.0 for r in laser_ranges[540:720] if r > 0)

        if left_clear > right_clear:
            return {'linear_vel': 0.0, 'angular_vel': 0.3}  # بائیں مڑیں
        else:
            return {'linear_vel': 0.0, 'angular_vel': -0.3}  # دائیں مڑیں

    def execute_action(self, action):
        """منتخب کردہ عمل انجام دیں"""
        if 'linear_vel' in action and 'angular_vel' in action:
            cmd = Twist()
            cmd.linear.x = action['linear_vel']
            cmd.angular.z = action['angular_vel']
            self.cmd_vel_pub.publish(cmd)

    def extract_image_features(self):
        """کیمرہ امیج سے متعلقہ خصوصیات نکالیں"""
        # سادہ خصوصیات نکالنے کی مثال
        gray = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # ممکنہ اشیاء کے طور پر کنٹور تلاش کریں
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        features = {
            'contour_count': len(contours),
            'largest_contour_area': max([cv2.contourArea(c) for c in contours]) if contours else 0
        }

        return features

def main(args=None):
    rclpy.init(args=args)
    node = SensorimotorIntegrationNode()

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

## اعلی درجے کا حسی موٹر انضمام

### پیش گوئی کی پروسیسنگ

اعلی درجے کے جسمانی مصنوعی ذہانت کے نظام عمل کے نتائج کی توقع کرنے کے لیے پیش گوئی کے ماڈل استعمال کرتے ہیں:

```python
# مثال: پیش گوئی کا حسی موٹر ماڈل
class PredictiveModel:
    def __init__(self):
        self.sensor_history = []
        self.action_history = []
        self.prediction_model = None

    def update_model(self, current_sensor, action_taken):
        """نئے ڈیٹا کے ساتھ پیش گوئی کا ماڈل اپ ڈیٹ کریں"""
        self.sensor_history.append(current_sensor)
        self.action_history.append(action_taken)

        # اندرونی پیش گوئی ماڈل کو اپ ڈیٹ کریں
        # (عمل میں، یہ مشین لرننگ کی تکنیکوں کا استعمال کرے گا)
        pass

    def predict_sensor_state(self, action_sequence):
        """اعمال کی ترتیب کے دی گئی مستقبل کی حسی حالت کی پیش گوئی کریں"""
        # نتیجہ کی پیش گوئی کے لیے اندرونی ماڈل استعمال کریں
        predicted_state = self.internal_prediction(action_sequence)
        return predicted_state

    def internal_prediction(self, action_sequence):
        """اندرونی پیش گوئی کا میکنزم"""
        # حالیہ تاریخ کی بنیاد پر سادہ پیش گوئی
        return action_sequence[-1] if action_sequence else None
```

### موافق حسی موٹر کوآرڈی نیشن

تجربے کے مطابق اپنی حسی موٹر کوآرڈی نیشن کو موافق کرنے والے نظام:

```python
# مثال: موافق حسی موٹر کوآرڈی نیشن
class AdaptiveSensorimotorSystem:
    def __init__(self):
        self.sensory_weights = np.ones(10)  # ہر حسی ماڈلٹی کے لیے وزن
        self.motor_mapping = np.eye(6)     # حسی سپیس سے موٹر سپیس تک کا نقشہ
        self.performance_history = []

    def adapt_coordination(self, sensory_input, motor_output, performance_feedback):
        """کارکردگی کے مطابق حسی موٹر کوآرڈی نیشن کو موافق کریں"""
        # کسی حسی کو تبدیل کریں جو زیادہ مفید ثابت ہوئی
        self.update_sensory_weights(sensory_input, performance_feedback)

        # موثر کارکردگی کے مطابق موٹر نقشہ کو اپ ڈیٹ کریں
        self.update_motor_mapping(sensory_input, motor_output, performance_feedback)

        # مستقبل کی موافقت کے لیے کارکردگی کو محفوظ کریں
        self.performance_history.append(performance_feedback)

    def update_sensory_weights(self, sensory_input, performance):
        """ مختلف حسی ماڈلٹیز کے لیے وزن اپ ڈیٹ کریں"""
        # اچھی کارکردگی میں شراکت دار حسی کا وزن بڑھائیں
        # کم مفید حسی کا وزن کم کریں
        pass

    def update_motor_mapping(self, sensory_input, motor_output, performance):
        """ حسی ان پٹ سے موٹر آؤٹ پٹ تک کا نقشہ اپ ڈیٹ کریں"""
        # کارکردگی کے مطابق ٹرانسفارمیشن میٹرکس کو ایڈجسٹ کریں
        pass
```

## لیب: حسی موٹر کوآرڈی نیشن کا نفاذ

اس لیب میں، آپ ایک سادہ حسی موٹر کوآرڈی نیشن سسٹم نافذ کریں گے:

```python
# lab_sensorimotor_coordination.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import numpy as np

class SensorimotorLabNode(Node):
    def __init__(self):
        super().__init__('sensorimotor_lab')

        # پبلشرز
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/lab_status', 10)

        # سبسکرائبرز
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )

        # حالت کے متغیرات
        self.scan_data = None
        self.imu_data = None
        self.last_command_time = self.get_clock().now()

        # لیب پیرامیٹر
        self.lab_state = 'exploration'  # exploration, obstacle_avoidance, balance
        self.exploration_pattern = 'random_walk'
        self.balance_threshold = 0.2

        # کنٹرول ٹائمر
        self.control_timer = self.create_timer(0.1, self.control_callback)

    def scan_callback(self, msg):
        """لیزر اسکین ڈیٹا کو سنبھالیں"""
        self.scan_data = msg

    def imu_callback(self, msg):
        """توازن کے لیے IMU ڈیٹا کو سنبھالیں"""
        self.imu_data = msg

    def control_callback(self):
        """اصل کنٹرول کال بیک جو حسی موٹر کوآرڈی نیشن نافذ کرتا ہے"""
        if not self.scan_data or not self.imu_data:
            return

        # پہلے توازن چیک کریں (تحفظ کی ترجیح)
        if self.check_balance():
            # ایمرجنسی توازن کی اصلاح
            cmd = Twist()
            cmd.angular.z = 0.5  # سمت کی اصلاح
            self.cmd_pub.publish(cmd)
            self.status_pub.publish(String(data='BALANCE_CORRECTION'))
            return

        # موجودہ لیب حالت انجام دیں
        if self.lab_state == 'exploration':
            command = self.exploration_behavior()
        elif self.lab_state == 'obstacle_avoidance':
            command = self.obstacle_avoidance_behavior()
        else:
            command = Twist()  # رکیں

        self.cmd_pub.publish(command)

        # حسی ڈیٹا کے مطابق لیب کی حالت کو اپ ڈیٹ کریں
        self.update_lab_state()

    def check_balance(self):
        """چیک کریں کہ کیا روبوٹ محفوظ حد سے زیادہ جھک گیا ہے"""
        if self.imu_data:
            # IMU جہت کا استعمال کرتے ہوئے سادہ توازن چیک
            # عمل میں، مناسب کووٹرین ریاضی استعمال کرے گا
            orientation = self.imu_data.orientation
            tilt_magnitude = abs(orientation.x) + abs(orientation.y)
            return tilt_magnitude > self.balance_threshold
        return False

    def exploration_behavior(self):
        """حسی موٹر کوآرڈی نیشن کا استعمال کرتے ہوئے تلاش کا رویہ نافذ کریں"""
        cmd = Twist()

        # تلاش کو ہدایت کے لیے لیزر ڈیٹا کا استعمال کریں
        if self.scan_data:
            # رکاوٹوں کے لیے سامنے چیک کریں
            front_ranges = self.scan_data.ranges[300:600]  # سامنے 60 ڈگری
            front_clear = all(r > 1.0 for r in front_ranges if r > 0)

            if front_clear:
                cmd.linear.x = 0.3  # آگے بڑھیں
            else:
                # رکاوٹوں سے بچنے کے لیے مڑیں
                left_ranges = self.scan_data.ranges[0:180]
                right_ranges = self.scan_data.ranges[540:720]

                left_clear = sum(r > 1.0 for r in left_ranges if r > 0)
                right_clear = sum(r > 1.0 for r in right_ranges if r > 0)

                if left_clear > right_clear:
                    cmd.angular.z = 0.3  # بائیں مڑیں
                else:
                    cmd.angular.z = -0.3  # دائیں مڑیں

        return cmd

    def obstacle_avoidance_behavior(self):
        """رکاوٹ سے بچنے کا رویہ نافذ کریں"""
        cmd = Twist()

        if self.scan_data:
            # زیادہ پیچیدہ رکاوٹ سے بچنے کا طریقہ
            ranges = self.scan_data.ranges
            min_distance = min([r for r in ranges if r > 0], default=float('inf'))

            if min_distance > 1.0:  # محفوظ فاصلہ
                cmd.linear.x = 0.3
            elif min_distance > 0.5:  # قریب آ رہا ہے
                cmd.linear.x = 0.1
                cmd.angular.z = 0.2  # مڑنا شروع کریں
            else:  # بہت قریب
                cmd.linear.x = 0.0
                cmd.angular.z = 0.5  # تیزی سے مڑیں

        return cmd

    def update_lab_state(self):
        """ حسی ڈیٹا اور کارکردگی کے مطابق لیب کی حالت کو اپ ڈیٹ کریں"""
        if self.scan_data:
            # مختلف ماحولوں کا سامنا کرتے ہوئے حالت تبدیل کریں
            front_ranges = self.scan_data.ranges[300:600]
            obstacles_nearby = any(r < 0.8 for r in front_ranges if r > 0)

            if obstacles_nearby and self.lab_state != 'obstacle_avoidance':
                self.lab_state = 'obstacle_avoidance'
                self.status_pub.publish(String(data='STATE_CHANGED: obstacle_avoidance'))
            elif not obstacles_nearby and self.lab_state != 'exploration':
                self.lab_state = 'exploration'
                self.status_pub.publish(String(data='STATE_CHANGED: exploration'))

def main(args=None):
    rclpy.init(args=args)
    lab_node = SensorimotorLabNode()

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

## مشق: اپنا حسی موٹر پیٹرن ڈیزائن کریں

ایک مخصوص کام (مثلاً کوئی چیز اٹھانا، میز میں نیویگیٹ کرنا، کسی شخص کا پیچھا کرنا) پر غور کریں اور ایک حسی موٹر کوآرڈی نیشن پیٹرن ڈیزائن کریں جو روبوٹ کو اس کام کو مؤثر طریقے سے انجام دینے کے قابل بنائے۔ غور کریں:

1. اس کام کے لیے کون سے سینسرز سب سے اہم ہوں گے؟
2. کون سے موٹر پیٹرن درکار ہوں گے؟
3. آپ حسی ان پٹ کو موٹر آؤٹ پٹ کے ساتھ کیسے منسق کریں گے؟
4. تحفظ کے لیے کون سے ریفلیکس رویے اہم ہوں گے؟
5. نظام تجربے کے مطابق کیسے موافق ہوگا؟

## خلاصہ

حسی موٹر انضمام جامع ذہانت کی بنیاد ہے، جو روبوٹس کو مسلسل ادراک-عمل لوپ کے ذریعے جسمانی دنیا کے ساتھ بات چیت کرنے کے قابل بناتا ہے۔ اہم اجزاء میں شامل ہیں:

- متعدد حسی ماڈلٹیز جو ماحولیاتی معلومات فراہم کرتے ہیں
- موٹر نظام منسق کردہ اعمال انجام دیتے ہیں
- فوری جوابات کے لیے حقیقی وقت کی پروسیسنگ
- تحفظ اور استحکام کے لیے ریفلیکس رویے
- اعمال کے نتائج کی توقع کے لیے پیش گوئی کے ماڈل
- تجربے کے ساتھ بہتر ہونے والی موافق کوآرڈی نیشن

ROS2 کے ذریعہ ان اجزاء کا انضمام حقیقی دنیا کے ماحول میں مؤثر طریقے سے کام کرنے والے جسمانی مصنوعی ذہانت کے نظام کی ترقی کو فعال کرتا ہے۔ حسی موٹر انضمام کو سمجھنا روبوٹس کو ترقی دینے کے لیے ضروری ہے جو اپنے ماحول کے ساتھ بات چیت کے ذریعہ سیکھ سکیں اور موافق ہو سکیں۔

اگلے سبق میں، ہم ادراک-عمل لوپ کو مزید تفصیل سے جانیں گے اور یہ دیکھیں گے کہ وہ سادہ قواعد سے پیچیدہ رویے کیسے نمودار ہوتے ہیں۔