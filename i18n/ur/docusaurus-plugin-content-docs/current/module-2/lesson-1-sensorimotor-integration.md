---
sidebar_position: 1
---

# فزیکل ای آئی میں سینسروموٹر انٹیگریشن

## تعارف

سینسروموٹر انٹیگریشن اس بنیادی عمل کو کہا جاتا ہے جس کے ذریعے فزیکل ای آئی سسٹم حسی ان پٹ کو موٹر آؤٹ پٹ کے ساتھ جوڑ کر ہم آہنگ، مقصد کے مطابق رویہ تخلیق کرتے ہیں۔ روایتی ای آئی سسٹم کے برعکس جو معلومات کو علیحدگی میں پروسیس کرتے ہیں، امبدڈ ایجنٹس کو اپنے ماحول کے ساتھ مؤثر طریقے سے تعامل کرنے کے لیے مستقل طور پر ادراک اور کارروائی کو ضم کرنا ہوتا ہے۔

## سینسروموٹر لوپ

سینسروموٹر لوپ امبدڈ انٹیلی جنس کا اساسی میکانزم ہے:

```
ماحول → سینسرز → ادراک → کارروائی کا انتخاب → ایفیکٹرز → ماحول
     ↑______________________________________________________________|
```

یہ مسلسل لوپ اس کے قابل بناتا ہے:

- **ریل ٹائم مطابقت**: ماحولیاتی تبدیلیوں کا فوری جواب
- **امبدڈ لرننگ**: صرف مشاہدہ کے بجائے تعامل کے ذریعے سیکھنا
- **ایمرجینٹ بیہیویئرز**: سادہ سینسروموٹر کے قواعد سے ابھرنے والے پیچیدہ رویے
- **روبسٹ نیس**: فیڈ بیک کے ذریعے قدرتی خرابی کی اصلاح

## فزیکل ای آئی میں حسی سسٹم

### سینسرز کی اقسام

فزیکل ای آئی سسٹم عام طور پر متعدد حسی ماڈلٹیز کو ضم کرتے ہیں:

1. **پروپریوسیپٹو سینسرز**: داخلی سینسرز جو روبوٹ کی حالت کا پتہ لگاتے ہیں
   - جوئنٹ انکوڈرز (پوزیشن، ویلوسٹی)
   - انرٹیل میزورمینٹ یونٹس (IMU)
   - فورس/ٹورک سینسرز
   - موٹر مانیٹرنگ کے لیے کرنٹ سینسرز

2. **ایکسٹروسیپٹو سینسرز**: بیرونی ماحول کے سینسرز
   - کیمرے (وژن)
   - لیڈار (رینج سینسنگ)
   - الٹرا سونک سینسرز (قربت)
   - ٹیکٹائل سینسرز (چھونا)

3. **انٹرو سیپٹو سینسرز**: داخلی حالت کے سینسرز
   - ٹیمپریچر سینسرز
   - بیٹری لیول مانیٹر
   - توانائی کی کھپت کے مانیٹر

### سینسر فیوژن کی مثال
```python
# مثال: اسٹیٹ کے اندازہ کے لیے سینسر فیوژن
import numpy as np
from scipy.spatial.transform import Rotation as R

class SensorFusion:
    def __init__(self):
        self.imu_data = None
        self.camera_data = None
        self.lidar_data = None
        self.joint_encoders = None

        # کیلمین فلٹر پیرامیٹرز
        self.state = np.zeros(13)  # [position, orientation, velocity, angular_velocity]
        self.covariance = np.eye(13) * 100  # ابتدائی عدم یقینی

    def update_from_imu(self, linear_accel, angular_vel, dt):
        """IMU ڈیٹا سے اسٹیٹ کا اندازہ اپ ڈیٹ کریں"""
        # IMU پیمائشوں کو انٹیگریٹ کریں
        # یہ ایک سادہ مثال ہے - حقیقی نفاذ کیلمین فلٹرنگ استعمال کرے گا
        pass

    def update_from_camera(self, visual_features):
        """بصری خصوصیات سے اسٹیٹ کا اندازہ اپ ڈیٹ کریں"""
        # پوزیشن کی اصلاح کے لیے بصری خصوصیات استعمال کریں
        pass

    def update_from_lidar(self, range_measurements):
        """لیڈار ڈیٹا سے اسٹیٹ کا اندازہ اپ ڈیٹ کریں"""
        # پوزیشن کی تصدیق کے لیے رینج ڈیٹا استعمال کریں
        pass

    def get_fused_state(self):
        """ضم کردہ اسٹیٹ کا اندازہ لوٹائیں"""
        return self.state
```

## موٹر سسٹم اور کنٹرول

### ایکچوایٹر کی اقسام

فزیکل ای آئی سسٹم مختلف ایکچوایٹر اقسام استعمال کرتے ہیں:

1. **روٹری ایکچوایٹرز**: سرو موٹر، سٹیپر موٹر
2. **لینیئر ایکچوایٹرز**: پنومیٹک، ہائیڈرولک، یا الیکٹرک لینیئر ایکچوایٹرز
3. **سافٹ ایکچوایٹرز**: پنومیٹک نیٹ ورکس، شیپ میموری الائے

### کنٹرول آرکیٹیکچر
```python
# مثال: ہائرارچیکل موٹر کنٹرول سسٹم
class MotorController:
    def __init__(self, joint_count):
        self.joint_count = joint_count
        self.joint_positions = np.zeros(joint_count)
        self.joint_velocities = np.zeros(joint_count)
        self.joint_efforts = np.zeros(joint_count)

        # ہر جوئنٹ کے لیے PID کنٹرولرز
        self.pid_controllers = [PIDController() for _ in range(joint_count)]

    def compute_joint_commands(self, desired_positions, dt):
        """تمام جوئنٹس کے لیے کمانڈز کا حساب لگائیں"""
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
        """موجودہ جوئنٹ ویلیوائز کے ساتھ اندرونی اسٹیٹ کو اپ ڈیٹ کریں"""
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

## سینسروموٹر کوآرڈینیشن پیٹرنز

### ریفلیکسیو بیہیویئرز

سادہ سینسروموٹر کوآرڈینیشن پیٹرنز زیادہ پیچیدہ رویوں کی بنیاد بنتے ہیں:

```python
# مثال: ریفلیکسیو بیہیویئرز
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
        """گرمی سے بچاؤ کے لیے ریفلیکس"""
        if max(temperatures) > self.safety_thresholds['temperature_limit']:
            # موٹر کی طاقت کم کریں
            return {'power_reduction': 0.5}
        else:
            return None

    def current_limit_reflex(self, currents):
        """موٹر کے نقصان سے بچنے کے لیے ریفلیکس"""
        if max(currents) > self.safety_thresholds['current_limit']:
            # ہنگامی اسٹاپ
            return {'emergency_stop': True}
        else:
            return None
```

### ریتھمک پیٹرنز

بہت سے بائیولوجیکل سسٹم لوکوموشن اور دیگر رویوں کے لیے ریتھمک پیٹرنز استعمال کرتے ہیں:

```python
# مثال: ریتھمک موشن کے لیے سینٹرل پیٹرن جنریٹر
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
        # ریتھمک آؤٹ پٹ جنریٹ کریں (مثلاً، چلنے کے لیے)
        left_leg = self.amplitude * math.sin(self.phase)
        right_leg = self.amplitude * math.sin(self.phase + math.pi)  # فیز سے باہر
        return {'left_leg': left_leg, 'right_leg': right_leg}
```

## ROS2 نفاذ: سینسروموٹر انٹیگریشن

یہاں ROS2 کا استعمال کرتے ہوئے سینسروموٹر انٹیگریشن کی مکمل مثال ہے:

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

        # سبسکرائبز
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

        # کنٹرول کمپوننٹس
        self.motor_controller = MotorController(12)  # 12 جوئنٹس کی مثال
        self.reflexive_behaviors = ReflexiveBehaviors()
        self.cpg = CentralPatternGenerator(frequency=0.5)
        self.cv_bridge = CvBridge()

        # کنٹرول ٹائمر
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

        # بیہیویئر اسٹیٹ
        self.current_behavior = 'idle'
        self.behavior_params = {}

    def joint_state_callback(self, msg):
        """جوئنٹ اسٹیٹ اپ ڈیٹس کو ہینڈل کریں"""
        self.joint_states = msg
        self.motor_controller.update_joint_states(
            np.array(msg.position),
            np.array(msg.velocity)
        )

    def laser_callback(self, msg):
        """لیزر اسکین ڈیٹا ہینڈل کریں"""
        self.laser_data = msg

    def imu_callback(self, msg):
        """IMU ڈیٹا ہینڈل کریں"""
        self.imu_data = msg

    def image_callback(self, msg):
        """کیمرہ امیج ڈیٹا ہینڈل کریں"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_data = cv_image
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def control_loop(self):
        """اصل سینسروموٹر انٹیگریشن لوپ"""
        if not all([self.joint_states, self.laser_data, self.imu_data]):
            return

        # 1. حسی ان پٹ کو پروسیس کریں
        sensor_data = self.process_sensors()

        # 2. تحفظ کے لیے ریفلیکسیو بیہیویئرز لاگو کریں
        reflex_action = self.apply_reflexive_behaviors(sensor_data)
        if reflex_action:
            self.execute_action(reflex_action)
            return  # تحفظ کے ریفلیکسز کو ترجیح دیں

        # 3. اسٹیٹ کی بنیاد پر بیہیویئر کا انتخاب کریں اور انجام دیں
        action = self.select_behavior(sensor_data)
        self.execute_action(action)

        # 4. تحفظ کی حیثیت اپ ڈیٹ کریں
        self.safety_pub.publish(Bool(data=True))

    def process_sensors(self):
        """تمام سینسر ڈیٹا کو ایک متحد نمائندگی میں پروسیس کریں"""
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
        """فوری تحفظ کے جوابات کے لیے ریفلیکسیو بیہیویئرز لاگو کریں"""
        # رکاوٹ سے بچنے کا ریفلیکس
        if len(sensor_data['laser_ranges']) > 0:
            min_distance = min([r for r in sensor_data['laser_ranges'] if r > 0], default=float('inf'))
            if min_distance < 0.3:  # 30cm تحفظ کی دوری
                self.get_logger().warn(f'Collision imminent: {min_distance:.2f}m')
                return {'linear_vel': 0.0, 'angular_vel': 0.0}

        # توازن ریفلیکس
        if sensor_data['imu_orientation']:
            # چیک کریں کہ کیا روبوٹ بہت زیادہ جھک رہا ہے
            orientation = sensor_data['imu_orientation']
            # سادہ چیک - حقیقت میں مناسب کوویٹریئن ریاضی استعمال کی جائے گی
            if abs(orientation.z) > 0.5:  # بہت جھکا ہوا
                self.get_logger().warn('Balance at risk - applying correction')
                return {'linear_vel': 0.0, 'angular_vel': 0.5}  # درست کرنے کی کوشش کریں

        return None

    def select_behavior(self, sensor_data):
        """موجودہ حالت اور سینسر ڈیٹا کی بنیاد پر بیہیویئر منتخب کریں"""
        # سینسر ڈیٹا کی بنیاد پر سادہ بیہیویئر سلیکشن
        if len(sensor_data['laser_ranges']) > 0:
            front_clear = all(r > 1.0 for r in sensor_data['laser_ranges'][300:600] if r > 0)

            if front_clear:
                return self.go_forward_behavior()
            else:
                return self.avoid_obstacle_behavior(sensor_data['laser_ranges'])
        else:
            return {'linear_vel': 0.0, 'angular_vel': 0.0}

    def go_forward_behavior(self):
        """سادہ فارورڈ موومنٹ بیہیویئر"""
        return {'linear_vel': 0.3, 'angular_vel': 0.0}

    def avoid_obstacle_behavior(self, laser_ranges):
        """رکاوٹ سے بچنے کا بیہیویئر"""
        # صاف ترین سمت تلاش کریں
        left_clear = sum(r > 1.0 for r in laser_ranges[0:180] if r > 0)
        right_clear = sum(r > 1.0 for r in laser_ranges[540:720] if r > 0)

        if left_clear > right_clear:
            return {'linear_vel': 0.0, 'angular_vel': 0.3}  # بائیں مڑیں
        else:
            return {'linear_vel': 0.0, 'angular_vel': -0.3}  # دائیں مڑیں

    def execute_action(self, action):
        """منتخب کردہ ایکشن انجام دیں"""
        if 'linear_vel' in action and 'angular_vel' in action:
            cmd = Twist()
            cmd.linear.x = action['linear_vel']
            cmd.angular.z = action['angular_vel']
            self.cmd_vel_pub.publish(cmd)

    def extract_image_features(self):
        """کیمرہ امیج سے متعلقہ خصوصیات نکالیں"""
        # سادہ فیچر ایکسٹریکشن کی مثال
        gray = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # ممکنہ اشیاء کے طور پر کنٹورز تلاش کریں
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

## اعلیٰ سینسروموٹر انٹیگریشن

### پریڈکٹو پروسیسنگ

اعلیٰ فزیکل ای آئی سسٹم ایکشنز کے نتائج کی توقع کرنے کے لیے پریڈکٹو ماڈلز استعمال کرتے ہیں:

```python
# مثال: پریڈکٹو سینسروموٹر ماڈل
class PredictiveModel:
    def __init__(self):
        self.sensor_history = []
        self.action_history = []
        self.prediction_model = None

    def update_model(self, current_sensor, action_taken):
        """نئے ڈیٹا کے ساتھ پریڈکٹو ماڈل کو اپ ڈیٹ کریں"""
        self.sensor_history.append(current_sensor)
        self.action_history.append(action_taken)

        # اندرونی پریڈکشن ماڈل کو اپ ڈیٹ کریں
        # (عمل میں، یہ مشین لرننگ کنیکس استعمال کرے گا)
        pass

    def predict_sensor_state(self, action_sequence):
        """ایکشن سیکوئنس کی بنیاد پر مستقبل کی سینسر اسٹیٹس کی توقع کریں"""
        # نتیجہ حاصل کرنے کے لیے اندرونی ماڈل استعمال کریں
        predicted_state = self.internal_prediction(action_sequence)
        return predicted_state

    def internal_prediction(self, action_sequence):
        """اندرونی پریڈکشن مکینزم"""
        # حالیہ تاریخ کی بنیاد پر سادہ پریڈکشن
        return action_sequence[-1] if action_sequence else None
```

### ایڈاپٹو سینسروموٹر کوآرڈینیشن

تجربے کی بنیاد پر اپنے سینسروموٹر کوآرڈینیشن کو ایڈاپٹ کرنے والے سسٹم:

```python
# مثال: ایڈاپٹو سینسروموٹر کوآرڈینیشن
class AdaptiveSensorimotorSystem:
    def __init__(self):
        self.sensory_weights = np.ones(10)  # ہر سینسر ماڈلٹی کے لیے وزن
        self.motor_mapping = np.eye(6)     # سینسر اسپیس سے موٹر اسپیس کا نقشہ
        self.performance_history = []

    def adapt_coordination(self, sensory_input, motor_output, performance_feedback):
        """کارکردگی کی بنیاد پر سینسروموٹر کوآرڈینیشن کو ایڈاپٹ کریں"""
        # سینسرز کے وزن کو اپ ڈیٹ کریں جن کا زیادہ فائدہ ہوا
        self.update_sensory_weights(sensory_input, performance_feedback)

        # مؤثرتا کی بنیاد پر موٹر نقشہ اپ ڈیٹ کریں
        self.update_motor_mapping(sensory_input, motor_output, performance_feedback)

        # مستقبل کی ایڈاپٹیشن کے لیے کارکردگی ذخیرہ کریں
        self.performance_history.append(performance_feedback)

    def update_sensory_weights(self, sensory_input, performance):
        """مختلف حسی ماڈلٹیز کے لیے وزن اپ ڈیٹ کریں"""
        # ان سینسرز کا وزن بڑھائیں جنہوں نے اچھی کارکردگی میں حصہ ڈالا
        # ان سینسرز کا وزن کم کریں جو کم مفید تھے
        pass

    def update_motor_mapping(self, sensory_input, motor_output, performance):
        """سینسری ان پٹ سے موٹر آؤٹ پٹ کا نقشہ اپ ڈیٹ کریں"""
        # کارکردگی کی بنیاد پر ٹرانسفارمیشن میٹرکس ایڈجسٹ کریں
        pass
```

## لیب: سینسروموٹر کوآرڈینیشن کا نفاذ

اس لیب میں، آپ ایک سادہ سینسروموٹر کوآرڈینیشن سسٹم نافذ کریں گے:

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

        # سبسکرائبز
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )

        # اسٹیٹ ویری ایبلز
        self.scan_data = None
        self.imu_data = None
        self.last_command_time = self.get_clock().now()

        # لیب پیرامیٹرز
        self.lab_state = 'exploration'  # exploration, obstacle_avoidance, balance
        self.exploration_pattern = 'random_walk'
        self.balance_threshold = 0.2

        # کنٹرول ٹائمر
        self.control_timer = self.create_timer(0.1, self.control_callback)

    def scan_callback(self, msg):
        """لیزر اسکین ڈیٹا ہینڈل کریں"""
        self.scan_data = msg

    def imu_callback(self, msg):
        """توازن کے لیے IMU ڈیٹا ہینڈل کریں"""
        self.imu_data = msg

    def control_callback(self):
        """اصل کنٹرول کال بیک جو سینسروموٹر کوآرڈینیشن نافذ کرتا ہے"""
        if not self.scan_data or not self.imu_data:
            return

        # پہلے توازن چیک کریں (تحفظ کی ترجیح)
        if self.check_balance():
            # ہنگامی توازن کی اصلاح
            cmd = Twist()
            cmd.angular.z = 0.5  # اورینٹیشن درست کریں
            self.cmd_pub.publish(cmd)
            self.status_pub.publish(String(data='BALANCE_CORRECTION'))
            return

        # موجودہ لیب اسٹیٹ انجام دیں
        if self.lab_state == 'exploration':
            command = self.exploration_behavior()
        elif self.lab_state == 'obstacle_avoidance':
            command = self.obstacle_avoidance_behavior()
        else:
            command = Twist()  # روکیں

        self.cmd_pub.publish(command)

        # سینسر ڈیٹا کی بنیاد پر لیب اسٹیٹ اپ ڈیٹ کریں
        self.update_lab_state()

    def check_balance(self):
        """چیک کریں کہ کیا روبوٹ محفوظ حد سے زیادہ جھک رہا ہے"""
        if self.imu_data:
            # IMU اورینٹیشن کا استعمال کرتے ہوئے سادہ توازن چیک
            # عمل میں، مناسب کوویٹریئن ریاضی استعمال کی جائے گی
            orientation = self.imu_data.orientation
            tilt_magnitude = abs(orientation.x) + abs(orientation.y)
            return tilt_magnitude > self.balance_threshold
        return False

    def exploration_behavior(self):
        """سینسروموٹر کوآرڈینیشن کا استعمال کرتے ہوئے تلاش کا بیہیویئر نافذ کریں"""
        cmd = Twist()

        # لیزر ڈیٹا کا استعمال کریں تلاش کی رہنمائی کے لیے
        if self.scan_data:
            # رکاوٹوں کے لیے سامنے چیک کریں
            front_ranges = self.scan_data.ranges[300:600]  # سامنے 60 ڈگری
            front_clear = all(r > 1.0 for r in front_ranges if r > 0)

            if front_clear:
                cmd.linear.x = 0.3  # آگے بڑھیں
            else:
                # رکاوٹوں سے بچیں
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
        """رکاوٹ سے بچنے کا بیہیویئر نافذ کریں"""
        cmd = Twist()

        if self.scan_data:
            # زیادہ ترقی یافتہ رکاوٹ سے بچنے کا طریقہ
            ranges = self.scan_data.ranges
            min_distance = min([r for r in ranges if r > 0], default=float('inf'))

            if min_distance > 1.0:  # محفوظ دوری
                cmd.linear.x = 0.3
            elif min_distance > 0.5:  # قریب آ رہا ہے
                cmd.linear.x = 0.1
                cmd.angular.z = 0.2  # مڑنا شروع کریں
            else:  # بہت قریب
                cmd.linear.x = 0.0
                cmd.angular.z = 0.5  # تیزی سے مڑیں

        return cmd

    def update_lab_state(self):
        """سینسر ڈیٹا اور کارکردگی کی بنیاد پر لیب اسٹیٹ اپ ڈیٹ کریں"""
        if self.scan_data:
            # مختلف ماحولوں کا سامنا کرتے ہوئے اسٹیٹ تبدیل کریں
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

## مشق: اپنا سینسروموٹر پیٹرن ڈیزائن کریں

ایک مخصوص کام (مثلاً، کوئی چیز اٹھانا، میز میں نیویگیٹ کرنا، شخص کا پیچھا کرنا) پر غور کریں اور ایک سینسروموٹر کوآرڈینیشن پیٹرن ڈیزائن کریں جو روبوٹ کو اس کام کو مؤثر طریقے سے انجام دینے کے قابل بنائے۔ غور کریں:

1. اس کام کے لیے کون سے سینسرز سب سے اہم ہوں گے؟
2. کون سے موٹر پیٹرنز کی ضرورت ہوگی؟
3. آپ حسی ان پٹ کو موٹر آؤٹ پٹ کے ساتھ کیسے منسق کریں گے؟
4. تحفظ کے لیے کون سے ریفلیکسیو بیہیویئرز اہم ہوں گے؟
5. نظام تجربے کی بنیاد پر کیسے ایڈاپٹ ہوگا؟

## خلاصہ

سینسروموٹر انٹیگریشن امبدڈ انٹیلی جنس کی بنیاد ہے، جو روبوٹس کو مسلسل ادراک-کارروائی لوپس کے ذریعے فزیکل دنیا کے ساتھ تعامل کرنے کے قابل بناتا ہے۔ کلیدی اجزاء میں شامل ہیں:

- متعدد حسی ماڈلٹیز جو ماحولیاتی معلومات فراہم کرتے ہیں
- موٹر سسٹم جو منسق کردہ ایکشنز انجام دیتے ہیں
- فوری جوابات کے لیے حقیقی وقت کی پروسیسنگ
- تحفظ اور استحکام کے لیے ریفلیکسیو بیہیویئرز
- ایکشن نتائج کی توقع کے لیے پریڈکٹو ماڈلز
- تجربے کے ساتھ بہتر ہونے والا ایڈاپٹو کوآرڈینیشن

ROS2 کے ذریعے ان اجزاء کا انضمام ایسے ترقی یافتہ فزیکل ای آئی سسٹم کی ترقی کو فعال کرتا ہے جو حقیقی دنیا کے ماحول میں مؤثر طریقے سے کام کر سکتے ہیں۔ ماحول کے ساتھ تعامل کے ذریعے سیکھنے اور ایڈاپٹ کرنے والے روبوٹس کو تیار کرنے کے لیے سینسروموٹر انٹیگریشن کو سمجھنا اہم ہے۔

اگلی میں، ہم ادراک-کارروائی لوپس کو مزید تفصیل سے تلاش کریں گے اور یہ دیکھیں گے کہ سادہ قواعد سے پیچیدہ رویے کیسے ابھرتے ہیں۔