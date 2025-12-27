---
sidebar_position: 3
---

# انسان نما روبوٹکس کی موجودہ حالت

## تعارف

انسان نما روبوٹکس روبوٹکس میں سب سے مہنگے اور مبہم حدود میں سے ایک کی نمائندگی کرتا ہے، جو انسانوں کی طرح نظر آنے والے اور انسانوں کے ساتھ قدرتی طور پر بات چیت کرنے والے روبوٹس تیار کرنے کا مقصد رکھتا ہے۔ یہ شعبہ میکانیکل انجینئرنگ، مصنوعی ذہانت، کمپیوٹر وژن، اور انسان-روبوٹ بات چیت کی ترقیات کو جوڑتا ہے۔

## تاریخی پس منظر

انسان نما روبوٹس کی ترقی کئی کلیدی ادوار میں ہوئی ہے:

### ابتدائی میکانیکل آٹومیٹا (18ویں-19ویں صدی)
- انسانی حرکات کی نقل کرنے کے لیے ڈیزائن کردہ میکانیکل اعداد
- متعینہ حرکات تک محدود
- بنیادی طور پر تفریح کے مقاصد کے لیے

### پہلے پروگرام کرنا قابل انسان نما (1960-1980)
- WABOT-1 (ویسیڈا یونیورسٹی، 1972): پہلا مکمل سکیل کا ذہیم انسان نما
- بنیادی حسی-موٹر کی صلاحیات متعارف کرائی گئیں
- محدود خودمختاری اور موبائلیٹی

### جدید انسان نما دور (1990-حالیہ)
- ہونڈا کا ASIMO (2000): دو پائوں والی چلنے اور انسانی بات چیت کا آغاز کیا
- سونی کا QRIO: جدید خودکار رویے
- موبائلیٹی، دستیابی، اور ذہانت میں مسلسل ترقی

## آج کے اہم انسان نما روبوٹس

### ASIMO (ہونڈا)
- **صلاحیات**: دو پائوں والی چلنا، دوڑنا، سیڑھیاں چڑھنا
- **سینسر**: متعدد کیمرے، فورس سینسر، الٹرا سونک سینسر
- **مصنوعی ذہانت کی خصوصیات**: پیش گوئی کی حرکت، رکاوٹ سے بچنا
- **محدودیات**: محدود دستیابی، کنٹرول شدہ ماحول کا کام

### Atlas (بوسٹن ڈائینامکس)
- **صلاحیات**: متحرک چلنا، دوڑنا، بیک فلپس، ہیرا پھیری
- **سینسر**: لیڈار، اسٹیریو وژن، اندرونی حسی سینسر
- **مصنوعی ذہانت کی خصوصیات**: متحرک توازن، پیچیدہ حرکت کی منصوبہ بندی
- **محدودیات**: کیبل سے منسلک بجلی کی فراہمی، پیچیدہ کنٹرول سسٹم

### Pepper (سافٹ بینک روبوٹکس)
- **صلاحیات**: انسانی جذبات کی پہچان، گفتگو
- **سینسر**: کیمرے، مائیکروفون، چھونے کے سینسر
- **مصنوعی ذہانت کی خصوصیات**: جذب کی پہچان، قدرتی زبان کی پروسیسنگ
- **محدودیات**: محدود موبائلیٹی، بنیادی طور پر اوپری جسم کی بات چیت

### Sophia (ہینسن روبوٹکس)
- **صلاحیات**: چہرے کے اظہار، گفتگو
- **سینسر**: چہرہ کی پہچان کے لیے کیمرے
- **مصنوعی ذہانت کی خصوصیات**: قدرتی زبان کی پروسیسنگ، چہرہ کی پہچان
- **محدودیات**: بنیادی طور پر نمائش کے لیے، محدود کارکردگی

## کلیدی ٹیکنالوجیز

### ایکچوایشن سسٹم
جدید انسان نما روبوٹس مختلف ایکچوایشن ٹیکنالوجیز کا استعمال کرتے ہیں:

```python
# مثال: انسان نما روبوٹ کے لیے جوائنٹ کنٹرول
class HumanoidJointController:
    def __init__(self, joint_name, joint_type):
        self.joint_name = joint_name
        self.joint_type = joint_type  # revolute, prismatic, etc.
        self.position = 0.0
        self.velocity = 0.0
        self.effort = 0.0

    def control_loop(self, target_position, dt):
        """
        ایک جوائنٹ کے لیے کنٹرول لوپ
        پوزیشن ٹریکنگ کے لیے PID کنٹرول نافذ کرتا ہے
        """
        error = target_position - self.position
        self.velocity = error / dt  # سادہ ڈیریویٹیو تقرب

        # PID کنٹرول پیرامیٹر
        Kp = 10.0  # تناسب کا گیان
        Ki = 0.1   # انٹیگرل گیان
        Kd = 0.5   # ڈیریویٹیو گیان

        # کنٹرول کوشش کا حساب لگائیں
        effort = Kp * error + Ki * self.integral_error + Kd * self.velocity

        # جوائنٹ پر کوشش لاگو کریں (حقیقی نظام میں، یہ ہارڈ ویئر کے ساتھ بات چیت کرے گا)
        self.effort = effort
        self.integrate_dynamics(dt)

    def integrate_dynamics(self, dt):
        """لاگو کوشش کی بنیاد پر جوائنٹ کی حالت کو اپ ڈیٹ کریں"""
        # سادہ ڈائینامکس انضمام
        self.position += self.velocity * dt
        self.integral_error += (self.position - self.target_position) * dt
```

### حسی سسٹم
انسان نما روبوٹس متعدد حسی ماڈلز کو ضم کرتے ہیں:

1. **وژن سسٹم**: چیزوں کی پہچان اور منظر کی سمجھ کے لیے کیمرے
2. **ٹیکٹائل سینسر**: ہیرا پھیری کے لیے فورس/ٹورک سینسر
3. **انرٹیل میزورمنٹ یونٹس (IMU)**: توازن اور سمت کے لیے
4. **مائیکروفون**: تقریر کی پہچان اور آواز کی مقامیت کے لیے

### کنٹرول سسٹم
اعلی درجے کے کنٹرول سسٹم انسان نما روبوٹس کو توازن برقرار رکھنے اور پیچیدہ حرکات انجام دینے کے قابل بناتے ہیں:

```python
# مثال: دو پائوں والے روبوٹ کے لیے توازن کنٹرول
import numpy as np

class BalanceController:
    def __init__(self, robot_mass, com_height):
        self.robot_mass = robot_mass
        self.com_height = com_height
        self.gravity = 9.81
        self.com_position = np.array([0.0, 0.0, com_height])  # مرکز ثقل
        self.com_velocity = np.array([0.0, 0.0, 0.0])

    def compute_zmp(self, com_pos, com_vel, com_acc):
        """
        توازن کنٹرول کے لیے صفر مومینٹ پوائنٹ (ZMP) کا حساب لگائیں
        ZMP = [x, y] جہاں کل مومینٹ صفر ہے
        """
        z_com = self.com_height
        g = self.gravity

        x_zmp = com_pos[0] - (z_com / g) * com_acc[0]
        y_zmp = com_pos[1] - (z_com / g) * com_acc[1]

        return np.array([x_zmp, y_zmp, 0.0])

    def balance_control(self, desired_com_pos, current_com_pos, dt):
        """
        پاؤں کی پوزیشنز کو ایڈجسٹ کر کے روبوٹ کا توازن کنٹرول کریں
        """
        # مرکز ثقل کی پوزیشن میں خامی کا حساب لگائیں
        com_error = desired_com_pos - current_com_pos

        # توازن کے لیے سادہ PD کنٹرول
        Kp = 10.0
        Kd = 2.0

        # خامی کو درست کرنے کے لیے ضروری تیزی
        desired_acc = Kp * com_error + Kd * (com_error - self.prev_error) / dt

        # ضروری ZMP حاصل کرنے کے لیے کمپیوٹ کریں
        zmp_ref = self.compute_zmp(current_com_pos, self.com_velocity, desired_acc)

        self.prev_error = com_error
        return zmp_ref
```

## ROS2 نافذ کاری: انسان نما روبوٹ انٹرفیس

یہاں دیکھیں کہ ROS2 کا استعمال کرتے ہوئے انسان نما روبوٹ کے ساتھ کیسے بات چیت کی جاتی ہے:

```python
# humanoid_robot_interface.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np

class HumanoidRobotInterface(Node):
    def __init__(self):
        super().__init__('humanoid_robot_interface')

        # روبوٹ کے مختلف پہلوؤں کے لیے پبلشرز
        self.joint_trajectory_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory', 10
        )
        self.base_cmd_pub = self.create_publisher(
            Twist, '/base_controller/cmd_vel', 10
        )
        self.status_pub = self.create_publisher(
            String, '/robot_status', 10
        )

        # روبوٹ فیڈ بیک کے لیے سبسکرائبرز
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # روبوٹ کنفیگریشن
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint', 'left_wrist_joint',
            'right_shoulder_joint', 'right_elbow_joint', 'right_wrist_joint',
            'head_yaw_joint', 'head_pitch_joint'
        ]

        self.current_joint_positions = {name: 0.0 for name in self.joint_names}
        self.robot_state = 'idle'

    def joint_state_callback(self, msg):
        """موجودہ جوائنٹ پوزیشنز کو اپ ڈیٹ کریں"""
        for i, name in enumerate(msg.name):
            if name in self.current_joint_positions:
                self.current_joint_positions[name] = msg.position[i]

    def move_to_pose(self, joint_positions, duration=5.0):
        """روبوٹ جوائنٹس کو مخصوص پوزیشنز میں لے جائیں"""
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = list(joint_positions.keys())

        point = JointTrajectoryPoint()
        point.positions = list(joint_positions.values())
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)

        trajectory_msg.points.append(point)
        self.joint_trajectory_pub.publish(trajectory_msg)

    def walk_forward(self, distance=1.0, speed=0.2):
        """روبوٹ کو آگے چلنے کا حکم دیں"""
        cmd = Twist()
        cmd.linear.x = speed
        self.base_cmd_pub.publish(cmd)

        # سفر کرنے کے لیے ضروری وقت کا حساب لگائیں
        travel_time = distance / speed
        self.get_logger().info(f'Commanding robot to walk {distance}m in {travel_time}s')

    def perform_greeting(self):
        """ایک سلام کا اشارہ کریں"""
        # سلام کے اشارے کے لیے جوائنٹ پوزیشنز کی وضاحت کریں
        greeting_pose = {
            'right_shoulder_joint': 1.5,  # دائیں بازو اٹھائیں
            'right_elbow_joint': -1.0,
            'right_wrist_joint': 0.5,
            'head_yaw_joint': 0.0,
            'head_pitch_joint': -0.2  # تھوڑا نیچے دیکھیں
        }

        # دیگر جوائنٹس کے لیے موجودہ پوزیشنز حاصل کریں
        current_positions = self.current_joint_positions.copy()
        current_positions.update(greeting_pose)

        self.move_to_pose(current_positions, duration=2.0)
        self.get_logger().info('Performing greeting gesture')

    def check_balance(self):
        """IMU ڈیٹا کا استعمال کرتے ہوئے روبوٹ کا توازن چیک کریں"""
        # ایک حقیقی نافذ کاری میں، یہ IMU ڈیٹا کا استعمال کرے گا
        # تاکہ یہ تعین کیا جا سکے کہ روبوٹ متوازن ہے یا نہیں
        balance_ok = True  # جگہ کا نمونہ

        if balance_ok:
            self.robot_state = 'balanced'
            self.status_pub.publish(String(data='BALANCED'))
        else:
            self.robot_state = 'unbalanced'
            self.status_pub.publish(String(data='UNBALANCED - ADJUSTING'))

def main(args=None):
    rclpy.init(args=args)
    robot_interface = HumanoidRobotInterface()

    # مثال کا استعمال
    robot_interface.check_balance()
    robot_interface.perform_greeting()

    try:
        rclpy.spin(robot_interface)
    except KeyboardInterrupt:
        pass
    finally:
        robot_interface.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## موجودہ چیلنجز

### میکانیکل چیلنجز
1. **توانائی کی کارکردگی**: انسان نما روبوٹس قابل ذکر بجلی کا استعمال کرتے ہیں
2. **دیمک**: پیچیدہ نظام کو پہننے اور ناکام ہونے کا خطرہ
3. **وزن کی تقسیم**: استحکام اور موبائلیٹی کے لیے وزن کا توازن
4. **دستیابی**: انسان کی طرح ہیرا پھیری کی صلاحیات حاصل کرنا

### کنٹرول چیلنجز
1. **توازن**: حرکت کے دوران استحکام برقرار رکھنا
2. **ہم آہنگی**: متعدد ڈگریوں کو فعال کرنا
3. **حقوقی وقت کی پروسیسنگ**: سخت وقت کی پابندیوں کو پورا کرنا
4. **مطابقت**: غیر متوقع صورتحال کے مطابق ایڈجسٹ کرنا

### مصنوعی ذہانت چیلنجز
1. **ادراک**: پیچیدہ حقیقی دنیا کے ماحول کو سمجھنا
2. **سیکھنا**: بات چیت کے ذریعہ مہارت حاصل کرنا
3. **سماجی بات چیت**: قدرتی انسان-روبوٹ رابطہ
4. **خودمختاری**: مسلسل انسانی نگرانی کے بغیر کام کرنا

## ایپلی کیشنز اور استعمال کے معاملات

### تحقیق اور ترقی
- انسان کی طرح چلنے کے مطالعے کے لیے پلیٹ فارم
- نئے کنٹرول الگورتھم کی جانچ
- انسان-روبوٹ بات چیت کی تحقیق

### صحت کی دیکھ بھال
- بزرگ دیکھ بھال کی مدد
- جسمانی علاج کی حمایت
- دواؤں کے ورزش

### سروس انڈسٹریز
- ہوٹلوں اور مالز میں کسٹمر سروس
- میوزیم میں گائیڈ روبوٹس
- وصولی اور معلومات کی خدمات

### تفریح
- تھیم پارک کے علاقوں میں
- تعاملی کارکردگی
- تعلیمی مظاہرے

## مستقبل کی سمتیں

### ٹیکنالوجیکل ترقیات
1. **نرم روبوٹکس**: محفوظ بات چیت کے لیے مطابقت پذیر مواد کا استعمال
2. **بائیو-الہام دہ ڈیزائن**: بائیولوجیکل نظام سے سیکھنا
3. **اعلی درجے کے مواد**: ہلکا، مضبوط، زیادہ کارآمد اجزاء
4. **نیورومورفک کمپیوٹنگ**: دماغ-الہام دہ پروسیسنگ سسٹم

### مصنوعی ذہانت کے ساتھ انضمام
1. **بڑے زبان کے ماڈل**: قدرتی گفتگو کی صلاحیات
2. **ریفورسمنٹ لرننگ**: مشق کے ذریعہ مہارت حاصل کرنا
3. **ملٹی موڈل مصنوعی ذہانت**: وژن، زبان، اور عمل کا انضمام
4. **جسمانی سیکھنا**: جسمانی بات چیت کے ذریعہ سیکھنا

## لیب: انسان نما روبوٹ سیمولیشن

اس لیب میں، ہم Gazebo کا استعمال کرتے ہوئے ایک سادہ انسان نما روبوٹ کی سیمولیشن تیار کریں گے:

```python
# lab_humanoid_simulation.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import math
import time

class HumanoidSimulationLab(Node):
    def __init__(self):
        super().__init__('humanoid_simulation_lab')

        # پبلشرز
        self.joint_traj_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory', 10
        )
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/cmd_vel', 10
        )

        # مسلسل اعمال کے لیے ٹائمر
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.phase = 0

    def timer_callback(self):
        """مرکزی سیمولیشن لوپ"""
        self.phase += 1

        if self.phase % 100 == 0:  # ہر 10 سیکنڈ (100 * 0.1s)
            self.perform_action_sequence()

        # چلنے کی حرکت کی تفریح
        self.perform_oscillating_motion()

    def perform_action_sequence(self):
        """اعمال کی ایک ترتیب کریں"""
        action = self.phase // 100 % 4  # 4 اعمال کے ذریعے چکر

        if action == 0:
            self.get_logger().info("Action: Wave hello")
            self.wave_hello()
        elif action == 1:
            self.get_logger().info("Action: Look around")
            self.look_around()
        elif action == 2:
            self.get_logger().info("Action: Step forward")
            self.step_forward()
        elif action == 3:
            self.get_logger().info("Action: Bow")
            self.bow()

    def wave_hello(self):
        """دائیں بازو کے ساتھ ہیلو کریں"""
        trajectory = JointTrajectory()
        trajectory.joint_names = ['right_shoulder_joint', 'right_elbow_joint']

        # ہیلو حرکت کے اشارے
        points = []

        # اشارہ 1: غیر جانبدار پوزیشن
        p1 = JointTrajectoryPoint()
        p1.positions = [0.0, 0.0]
        p1.time_from_start.sec = 1
        points.append(p1)

        # اشارہ 2: بازو اٹھائیں
        p2 = JointTrajectoryPoint()
        p2.positions = [1.0, -0.5]
        p2.time_from_start.sec = 2
        points.append(p2)

        # اشارہ 3: ہیلو کریں
        p3 = JointTrajectoryPoint()
        p3.positions = [1.2, -0.7]
        p3.time_from_start.sec = 3
        points.append(p3)

        # اشارہ 4: غیر جانبدار پر واپس
        p4 = JointTrajectoryPoint()
        p4.positions = [0.0, 0.0]
        p4.time_from_start.sec = 4
        points.append(p4)

        trajectory.points = points
        self.joint_traj_pub.publish(trajectory)

    def look_around(self):
        """دیکھنے کے لیے سر ہلائیں"""
        trajectory = JointTrajectory()
        trajectory.joint_names = ['head_yaw_joint', 'head_pitch_joint']

        points = []

        # مرکز
        p1 = JointTrajectoryPoint()
        p1.positions = [0.0, 0.0]
        p1.time_from_start.sec = 1
        points.append(p1)

        # بائیں دیکھیں
        p2 = JointTrajectoryPoint()
        p2.positions = [0.5, 0.0]
        p2.time_from_start.sec = 2
        points.append(p2)

        # دائیں دیکھیں
        p3 = JointTrajectoryPoint()
        p3.positions = [-0.5, 0.0]
        p3.time_from_start.sec = 3
        points.append(p3)

        # اوپر دیکھیں
        p4 = JointTrajectoryPoint()
        p4.positions = [0.0, -0.3]
        p4.time_from_start.sec = 4
        points.append(p4)

        # مرکز پر واپس
        p5 = JointTrajectoryPoint()
        p5.positions = [0.0, 0.0]
        p5.time_from_start.sec = 5
        points.append(p5)

        trajectory.points = points
        self.joint_traj_pub.publish(trajectory)

    def step_forward(self):
        """روبوٹ کو آگے بڑھنے کا حکم دیں"""
        cmd = Twist()
        cmd.linear.x = 0.3  # آگے بڑھیں
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

        # ایک لمحے کے بعد رکیں
        time.sleep(1.0)
        cmd.linear.x = 0.0
        self.cmd_vel_pub.publish(cmd)

    def bow(self):
        """ایک جھکاؤ کا اشارہ کریں"""
        trajectory = JointTrajectory()
        trajectory.joint_names = ['head_pitch_joint']

        points = []

        # غیر جانبدار
        p1 = JointTrajectoryPoint()
        p1.positions = [0.0]
        p1.time_from_start.sec = 1
        points.append(p1)

        # نیچے جھکائیں
        p2 = JointTrajectoryPoint()
        p2.positions = [0.5]  # نیچے دیکھیں
        p2.time_from_start.sec = 2
        points.append(p2)

        # غیر جانبدار پر واپس
        p3 = JointTrajectoryPoint()
        p3.positions = [0.0]
        p3.time_from_start.sec = 3
        points.append(p3)

        trajectory.points = points
        self.joint_traj_pub.publish(trajectory)

    def perform_oscillating_motion(self):
        """سادہ اوسیلیٹنگ حرکت کریں تاکہ سانس/آگہی کی تفریح کی جا سکے"""
        trajectory = JointTrajectory()
        trajectory.joint_names = ['torso_joint']  # اگر دستیاب ہو

        points = []
        p = JointTrajectoryPoint()
        # سادہ اوسیلیشن سائنز لہر کا استعمال کرتے ہوئے
        oscillation = 0.05 * math.sin(self.phase * 0.1)  # چھوٹا امپلی ٹیوڈ
        p.positions = [oscillation]
        p.time_from_start.sec = 0
        p.time_from_start.nanosec = 100000000  # 0.1 سیکنڈ
        points.append(p)

        trajectory.points = points
        self.joint_traj_pub.publish(trajectory)

def main(args=None):
    rclpy.init(args=args)
    lab = HumanoidSimulationLab()

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

1. آپ کے انسان نما روبوٹ کا بنیادی اطلاق کیا ہوگا؟
2. اسے کون سے مخصوص جوائنٹس اور ڈگریاں ضرورت ہوں گی؟
3. اس کے کام کے لیے کون سے سینسرز ضروری ہوں گے؟
4. آپ کلیدی چیلنجز (توانائی، توازن، وغیرہ) کو کیسے حل کریں گے؟
5. موجودہ روبوٹس کے مقابلے میں اس کی منفرد صلاحیات کیا ہوں گی؟

## خلاصہ

انسان نما روبوٹکس کے شعبے نے قابل ذکر ترقی کی ہے، روبوٹس جیسے ASIMO، Atlas، اور Pepper حیران کن صلاحیات کا مظاہرہ کر رہے ہیں۔ تاہم، توانائی کی کارکردگی، توازن، دستیابی، اور قدرتی بات چیت کے لحاظ سے اب بھی قابل ذکر چیلنجز موجود ہیں۔

جدید انسان نما روبوٹس اعلی درجے کے میکانیکل سسٹم، پیچیدہ کنٹرول الگورتھم، اور مصنوعی ذہانت کی صلاحیات کو ضم کرتے ہیں۔ ROS2 کا استعمال معیاری انٹرفیسز اور ترقی کے ڈھانچے کو فعال کرتا ہے۔

مستقبل کی ترقیات کا امکان نرم روبوٹکس، بائیو-الہام دہ ڈیزائن، اور مصنوعی ذہانت کے نظام کے ساتھ سخت انضمام پر مرکوز ہوگا، جو ہمیں واقعی خودمختار اور قابل روبوٹس کے قریب لاتا ہے۔

اگلے سبق میں، ہم انسان نما روبوٹکس کی تحقیق میں کلیدی چیلنجز اور مواقع کا جائزہ لیں گے۔