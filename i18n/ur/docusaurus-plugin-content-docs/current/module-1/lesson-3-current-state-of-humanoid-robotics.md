---
sidebar_position: 3
---

# ہیومنوائڈ روبوٹکس کی موجودہ حالت

## تعارف

ہیومنوائڈ روبوٹکس روبوٹکس کی سب سے طموں میں سے ایک کی نمائندگی کرتا ہے، جو انسانوں کی شبیہ رکھنے اور قدرتی طور پر ان کے ساتھ تعامل کرنے والے روبوٹس تیار کرنے کا مقصد رکھتا ہے۔ یہ شعبہ میکینیکل انجینئرنگ، مصنوعی ذہانت، کمپیوٹر وژن، اور انسان-روبوٹ تعامل میں ترقیات کو جوڑتا ہے۔

## تاریخی پس منظر

ہیومنوائڈ روبوٹس کی ترقی کئی کلیدی ادوار کے ذریعے ہوتی گئی ہے:

### ابتدائی مکینیکل آٹومیٹا (18ویں-19ویں صدی)
- انسانی اعمال کو نقل کرنے کے لیے ڈیزائن کردہ مکینیکل اعداد و شمار
- متعین حرکات تک محدود
- بنیادی طور پر تفریح کے مقاصد کے لیے

### پہلے پروگرام ایبل ہیومنوائڈ (1960-1980)
- WABOT-1 (ویسیڈا یونیورسٹی، 1972): پہلا فل سکیل انٹیلی جنس ہیومنوائڈ
- بنیادی سینسرو-موٹر کی صلاحیتوں کا متعارف
- محدود خودمختاری اور موبائیلیٹی

### جدید ہیومنوائڈ دور (1990-حالیہ)
- ہونڈا کا ASIMO (2000): ڈوپیڈل چلنے اور انسانی تعامل کا راستہ دکھایا
- سونی کا QRIO: جدید خودکار رویہ
- موبائیلیٹی، چستی، اور انٹیلی جنس میں مسلسل ترقی

## آج کے نمایاں ہیومنوائڈ روبوٹس

### ASIMO (ہونڈا)
- **صلاحیات**: ڈوپیڈل چلنا، دوڑنا، سیڑھیاں چڑھنا
- **سینسرز**: متعدد کیمرے، فورس سینسرز، الٹرا سونک سینسرز
- **ای آئی خصوصیات**: پیش گوئی کی حرکت، رکاوٹوں سے بچنا
- **حدیں**: محدود چستی، کنٹرول شدہ ماحول کا کام

### ایٹلس (بوسٹن ڈائی نیمکس)
- **صلاحیات**: ڈائی نیمک چلنا، دوڑنا، بیک فلپس، مینیپولیشن
- **سینسرز**: لیڈار، اسٹیریو وژن، پروپریوسیپٹو سینسرز
- **ای آئی خصوصیات**: ڈائی نیمک توازن، پیچیدہ حرکت کی منصوبہ بندی
- **حدیں**: ٹیتھرڈ پاور سپلائی، پیچیدہ کنٹرول سسٹم

### پیپر (سافٹ بینک روبوٹکس)
- **صلاحیات**: انسانی جذبات کی شناخت، گفتگو
- **سینسرز**: کیمرے، مائیکروفونز، ٹیکٹائل سینسرز
- **ای آئی خصوصیات**: جذبہ کا پتہ لگانا، قدرتی زبان کی پروسیسنگ
- **حدیں**: محدود موبائیلیٹی، بنیادی طور پر اپر بڈی انٹر ایکشن

### سو فیا (ہینسن روبوٹکس)
- **صلاحیات**: چہرے کے اظہار، گفتگو
- **سینسرز**: چہرہ کے پتہ لگانے کے لیے کیمرے
- **ای آئی خصوصیات**: قدرتی زبان کی پروسیسنگ، چہرہ کی شناخت
- **حدیں**: بنیادی طور پر ڈیمو کے لیے، محدود فعالیت

## کلیدی ٹیکنالوجیز

### ایکچوایشن سسٹم
جدید ہیومنوائڈ روبوٹس مختلف ایکچوایشن ٹیکنالوجیز کا استعمال کرتے ہیں:

```python
# مثال: ہیومنوائڈ روبوٹ کے لیے جوئنٹ کنٹرول
class HumanoidJointController:
    def __init__(self, joint_name, joint_type):
        self.joint_name = joint_name
        self.joint_type = joint_type  # revolute, prismatic, etc.
        self.position = 0.0
        self.velocity = 0.0
        self.effort = 0.0

    def control_loop(self, target_position, dt):
        """
        ایک جوئنٹ کے لیے کنٹرول لوپ
        PID کنٹرول کو پوزیشن ٹریکنگ کے لیے نافذ کرتا ہے
        """
        error = target_position - self.position
        self.velocity = error / dt  # سادہ ڈیریویٹیو تقریب

        # PID کنٹرول پیرامیٹرز
        Kp = 10.0  # تناسب کا گین
        Ki = 0.1   # انٹیگرل گین
        Kd = 0.5   # ڈیریویٹیو گین

        # کنٹرول ایفروت کا حساب لگائیں
        effort = Kp * error + Ki * self.integral_error + Kd * self.velocity

        # جوئنٹ پر ایفروت لاگو کریں (حقیقی سسٹم میں، یہ ہارڈویئر کے ساتھ انٹرفیس ہوگا)
        self.effort = effort
        self.integrate_dynamics(dt)

    def integrate_dynamics(self, dt):
        """لاگو کردہ ایفروت کے مطابق جوئنٹ کی حالت کو اپ ڈیٹ کریں"""
        # سادہ ڈائی نیمکس انٹیگریشن
        self.position += self.velocity * dt
        self.integral_error += (self.position - self.target_position) * dt
```

### حسی سسٹم
ہیومنوائڈ روبوٹس متعدد حسی ماڈلٹیز کو ضم کرتے ہیں:

1. **وژن سسٹم**: اشیاء کی شناخت اور منظر کی سمجھ کے لیے کیمرے
2. **ٹیکٹائل سینسرز**: مینیپولیشن کے لیے فورس/ٹورک سینسرز
3. **انرٹیل میزورمینٹ یونٹس (IMU)**: توازن اور سمت کے لیے
4. **مائیکروفونز**: گفتگو کی شناخت اور آواز کی مقامیت کے لیے

### کنٹرول سسٹم
جدید کنٹرول سسٹم ہیومنوائڈ روبوٹس کو توازن برقرار رکھنے اور پیچیدہ حرکات انجام دینے کے قابل بناتے ہیں:

```python
# مثال: بائی پیڈل روبوٹ کے لیے توازن کنٹرول
import numpy as np

class BalanceController:
    def __init__(self, robot_mass, com_height):
        self.robot_mass = robot_mass
        self.com_height = com_height
        self.gravity = 9.81
        self.com_position = np.array([0.0, 0.0, com_height])  # مرکزِ کم
        self.com_velocity = np.array([0.0, 0.0, 0.0])

    def compute_zmp(self, com_pos, com_vel, com_acc):
        """
        توازن کنٹرول کے لیے زیرو مومینٹ پوائنٹ (ZMP) کا حساب لگائیں
        ZMP = [x, y] جہاں نیٹ مومینٹ صفر ہے
        """
        z_com = self.com_height
        g = self.gravity

        x_zmp = com_pos[0] - (z_com / g) * com_acc[0]
        y_zmp = com_pos[1] - (z_com / g) * com_acc[1]

        return np.array([x_zmp, y_zmp, 0.0])

    def balance_control(self, desired_com_pos, current_com_pos, dt):
        """
        فٹ کی پوزیشنز کو ایڈجسٹ کرکے روبوٹ کا توازن کنٹرول کریں
        """
        # مرکزِ کم کی پوزیشن میں خرابی کا حساب لگائیں
        com_error = desired_com_pos - current_com_pos

        # توازن کے لیے سادہ PD کنٹرول
        Kp = 10.0
        Kd = 2.0

        # خرابی کو درست کرنے کے لیے مطلوبہ ایکسلریشن
        desired_acc = Kp * com_error + Kd * (com_error - self.prev_error) / dt

        # مطلوبہ ایکسلریشن حاصل کرنے کے لیے ZMP کا حساب لگائیں
        zmp_ref = self.compute_zmp(current_com_pos, self.com_velocity, desired_acc)

        self.prev_error = com_error
        return zmp_ref
```

## ROS2 نفاذ: ہیومنوائڈ روبوٹ انٹرفیس

یہاں دیکھیں کہ ROS2 کا استعمال کرتے ہوئے ہیومنوائڈ روبوٹ کے ساتھ کیسے انٹرفیس کیا جائے:

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

        # روبوٹ کی فیڈ بیک کے لیے سبسکرائبز
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # روبوٹ کی کنفیگریشن
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
        """موجودہ جوئنٹ پوزیشنز کو اپ ڈیٹ کریں"""
        for i, name in enumerate(msg.name):
            if name in self.current_joint_positions:
                self.current_joint_positions[name] = msg.position[i]

    def move_to_pose(self, joint_positions, duration=5.0):
        """روبوٹ جوئنٹس کو مخصوص پوزیشنز پر منتقل کریں"""
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
        """ایک سلام کا اظہار کریں"""
        # سلام کے اظہار کے لیے جوئنٹ پوزیشنز کی وضاحت کریں
        greeting_pose = {
            'right_shoulder_joint': 1.5,  # دائیں بازو اٹھائیں
            'right_elbow_joint': -1.0,
            'right_wrist_joint': 0.5,
            'head_yaw_joint': 0.0,
            'head_pitch_joint': -0.2  # ذرا نیچے دیکھیں
        }

        # دیگر جوئنٹس کے لیے موجودہ پوزیشنز حاصل کریں
        current_positions = self.current_joint_positions.copy()
        current_positions.update(greeting_pose)

        self.move_to_pose(current_positions, duration=2.0)
        self.get_logger().info('Performing greeting gesture')

    def check_balance(self):
        """IMU ڈیٹا کا استعمال کرتے ہوئے روبوٹ کا توازن چیک کریں"""
        # ایک حقیقی نفاذ میں، یہ IMU ڈیٹا کا استعمال کرے گا
        # توازن کا تعین کرنے کے لیے
        balance_ok = True  # جگہ کا نام

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

### مکینیکل چیلنجز
1. **توانائی کی کارکردگی**: ہیومنوائڈ روبوٹس کو قابلِ تحسین طاقت کی ضرورت ہوتی ہے
2. **مہارتوں کا پن**: پیچیدہ میکانزم ٹوٹنے اور ناکام ہونے کے قابل
3. **وزن کی تقسیم**: استحکام اور موبائیلیٹی کے لیے وزن کا توازن
4. **چستی**: انسانی طرز کی مینیپولیشن کی صلاحیتوں کا حصول

### کنٹرول چیلنجز
1. **توازن**: حرکت کے دوران استحکام برقرار رکھنا
2. ** coordination**: متعدد ڈگریز آف فریڈم کو مطابقت دینا
3. **ریل ٹائم پروسیسنگ**: سخت ٹائم کنٹرول کو پورا کرنا
4. **مطابقت**: غیر متوقع صورتحال کے لیے ایڈجسٹ کرنا

### ای آئی چیلنجز
1. **ادراک**: پیچیدہ حقیقی دنیا کے ماحول کو سمجھنا
2. **سیکھنا**: تعامل کے ذریعے مہارتوں کا حصول
3. **سماجی تعامل**: قدرتی انسان-روبوٹ رابطہ
4. **خودمختاری**: مسلسل انسانی نگرانی کے بغیر کام کرنا

## اطلاقیات اور استعمال کے معاملات

### تحقیق اور ترقی
- انسانی طرز کی لوموکشن کا مطالعہ کرنے کے لیے پلیٹ فارم
- نئے کنٹرول الگوردمز کی جانچ
- انسان-روبوٹ تعامل کی تحقیق

### ہیلتھ کیئر
- بزرگ دیکھ بھال کی مدد
- جسمانی تھراپی کی معاونت
- ری ہیبیلیٹیشن ورزشیں

### سروس انڈسٹریز
- ہوٹلوں اور مالز میں کسٹمر سروس
- میوزیم میں گائیڈ روبوٹس
- وصول اور معلومات کی خدمات

### تفریح
- تھیم پارک اٹریکشنز
- انٹرایکٹو پرفارمنسز
- تعلیمی ڈیموسٹریشنز

## مستقبل کی سمتیں

### ٹیکنالوجی کی ترقیات
1. **سافٹ روبوٹکس**: محفوظ تعامل کے لیے مطیع میٹریلز کا استعمال
2. **بائیو-انسپائرڈ ڈیزائن**: بائیولوجیکل سسٹم سے سیکھنا
3. **جدید میٹریلز**: ہلکے، مضبوط، زیادہ کارآمد اجزاء
4. **نیورومورفک کمپوٹنگ**: دماغ سے متاثرہ پروسیسنگ سسٹم

### ای آئی کے ساتھ انضمام
1. **بڑے زبان کے ماڈلز**: قدرتی گفتگو کی صلاحیتیں
2. **ری اینفورسمنٹ لرننگ**: مشق کے ذریعے مہارت کا حصول
3. **مलٹی موڈل ای آئی**: وژن، زبان، اور ایکشن کا انضمام
4. **امبدڈ لرننگ**: جسمانی تعامل کے ذریعے سیکھنا

## لیب: ہیومنوائڈ روبوٹ سیمولیشن

اس لیب میں، ہم Gazebo کا استعمال کرتے ہوئے ہیومنوائڈ روبوٹ کی ایک سادہ سیمولیشن تیار کریں گے:

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

        # مسلسل ایکشنز کے لیے ٹائمر
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.phase = 0

    def timer_callback(self):
        """مرکزی سیمولیشن لوپ"""
        self.phase += 1

        if self.phase % 100 == 0:  # ہر 10 سیکنڈ (100 * 0.1s)
            self.perform_action_sequence()

        # چلنے کی حرکت کی شبیہ سازی
        self.perform_oscillating_motion()

    def perform_action_sequence(self):
        """ایکسرسائز کی ترتیب انجام دیں"""
        action = self.phase // 100 % 4  # 4 ایکسرسائز کے ذریعے سائیکل

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
        """دائیں ہاتھ سے ہلاﺅ"""
        trajectory = JointTrajectory()
        trajectory.joint_names = ['right_shoulder_joint', 'right_elbow_joint']

        # ہلاﺅنے کی حرکت کے پوائنٹس
        points = []

        # پوائنٹ 1: خاموشی کی پوزیشن
        p1 = JointTrajectoryPoint()
        p1.positions = [0.0, 0.0]
        p1.time_from_start.sec = 1
        points.append(p1)

        # پوائنٹ 2: ہاتھ اٹھائیں
        p2 = JointTrajectoryPoint()
        p2.positions = [1.0, -0.5]
        p2.time_from_start.sec = 2
        points.append(p2)

        # پوائنٹ 3: ہلاﺅ
        p3 = JointTrajectoryPoint()
        p3.positions = [1.2, -0.7]
        p3.time_from_start.sec = 3
        points.append(p3)

        # پوائنٹ 4: خاموشی پر واپسی
        p4 = JointTrajectoryPoint()
        p4.positions = [0.0, 0.0]
        p4.time_from_start.sec = 4
        points.append(p4)

        trajectory.points = points
        self.joint_traj_pub.publish(trajectory)

    def look_around(self):
        """چاروں طرف دیکھنے کے لیے سر ہلائیں"""
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

        # مرکز پر واپسی
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
        """ایک سلام کا اظہار کریں"""
        trajectory = JointTrajectory()
        trajectory.joint_names = ['head_pitch_joint']

        points = []

        # خاموشی
        p1 = JointTrajectoryPoint()
        p1.positions = [0.0]
        p1.time_from_start.sec = 1
        points.append(p1)

        # نیچے جھکیں
        p2 = JointTrajectoryPoint()
        p2.positions = [0.5]  # نیچے دیکھیں
        p2.time_from_start.sec = 2
        points.append(p2)

        # خاموشی پر واپسی
        p3 = JointTrajectoryPoint()
        p3.positions = [0.0]
        p3.time_from_start.sec = 3
        points.append(p3)

        trajectory.points = points
        self.joint_traj_pub.publish(trajectory)

    def perform_oscillating_motion(self):
        """سونے/آگاہی کی شبیہ سازی کے لیے معمولی جھولنے والی حرکت انجام دیں"""
        trajectory = JointTrajectory()
        trajectory.joint_names = ['torso_joint']  # اگر دستیاب ہو

        points = []
        p = JointTrajectoryPoint()
        # سائین لہر کا استعمال کرتے ہوئے معمولی جھول
        oscillation = 0.05 * math.sin(self.phase * 0.1)  # چھوٹا امپلی ٹوڈ
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

## مشق: اپنے ہیومنوائڈ روبوٹ کا ڈیزائن کریں

مندرجہ ذیل ڈیزائن چیلنج پر غور کریں:

1. آپ کے ہیومنوائڈ روبوٹ کی بنیادی اطلاقیت کیا ہوگی؟
2. اسے کون سے مخصوص جوئنٹس اور ڈگریز آف فریڈم کی ضرورت ہوں گے؟
3. اس کے کام کے لیے کون سے سینسرز ضروری ہوں گے؟
4. آپ کلیدی چیلنجوں (توانائی، توازن، وغیرہ) کا سامنا کیسے کریں گے؟
5. موجودہ روبوٹس کے مقابلے میں اس کی منفرد صلاحیات کیا ہوں گی؟

## خلاصہ

ہیومنوائڈ روبوٹکس کے شعبے نے قابلِ تحسین پیشرفت کی ہے، جیسے ASIMO، ایٹلس، اور پیپر کے ساتھ جو متاثر کن صلاحیات کا مظاہرہ کرتے ہیں۔ تاہم، توانائی کی کارکردگی، توازن، چستی، اور قدرتی تعامل کے لحاظ سے اب بھی قابلِ تحسین چیلنج باقی ہیں۔

جدید ہیومنوائڈ روبوٹس جدید مکینیکل سسٹم، ماہرانہ کنٹرول الگوردمز، اور ای آئی کی صلاحیتوں کو ضم کرتے ہیں۔ ROS2 کا استعمال معیاری انٹرفیسز اور ترقی کے فریم ورکس کے قابل بناتا ہے۔

مستقبل کی ترقیات کا امکان ہے کہ وہ سافٹ روبوٹکس، بائیو-انسپائرڈ ڈیزائن، اور ای آئی سسٹم کے ساتھ سخت انضمام پر توجہ مرکوز کریں گی، ہمیں واقعی خودمختار اور قابل روبوٹس کے قریب لاتی ہے۔

اگلی میں، ہم ہیومنوائڈ روبوٹکس کی تحقیق میں کلیدی چیلنجوں اور مواقع کا جائزہ لیں گے۔