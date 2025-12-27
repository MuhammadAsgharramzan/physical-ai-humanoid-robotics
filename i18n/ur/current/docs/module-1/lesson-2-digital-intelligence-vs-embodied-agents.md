---
sidebar_position: 2
---

# ڈیجیٹل ذہانت بمقابلہ جسمانی ایجنٹس

## ڈیجیٹل-جسمانی تقسیم

روایتی مصنوعی ذہانت کے نظام بنیادی طور پر ڈیجیٹل شعبوں میں کام کرتے ہیں، جسمانی پابندیوں کے بغیر معلومات کو عمل کرتے ہیں۔ اس کے برعکس، جسمانی ایجنٹس جسمانی ماحول میں موجود ہوتے ہیں، جہاں ان کی شکل اور حقیقی دنیا کے ساتھ بات چیت ان کی ذہانت کو بنیادی طور پر متاثر کرتی ہے۔

## ڈیجیٹل ذہانت: مطلق سرزمین

ڈیجیٹل ذہانت کے نظام کنٹرول شدہ، مطلق ماحول میں کام کرتے ہیں:

- **مکمل معلومات**: نظام کی حالت کا مکمل علم
- **متعینہ آپریشنز**: دی گئی ان پٹ کے لیے قابل پیش گوئی کے نتائج
- **جسمانی پابندیاں نہیں**: کوئی مٹر، گریویٹی، یا میٹریل کی حدود نہیں
- **سادہ ماڈل**: حقیقت کی مطلق نمائندگی

### ڈیجیٹل مصنوعی ذہانت کی خصوصیات

1. **سمبولک پروسیسنگ**: مطلق علامات اور نمائندگیوں کا استعمال
2. **رول-بیسڈ ریزننگ**: باقاعدہ علم پر لاگو منطق
3. **پیٹرن ریکوگنیشن**: ڈیجیٹل ڈیٹا میں پیٹرن کی شناخت
4. **بہترین بنانا**: اچھی طرح سے وضاحت شدہ خلا میں ریاضی کا بہترین بنانا

```python
# مثال: راستہ کی منصوبہ بندی کے لیے ڈیجیٹل مصنوعی ذہانت کا نقطہ نظر
def digital_path_planner(goal_position, obstacles):
    """
    روایتی ڈیجیٹل نقطہ نظر برائے راستہ کی منصوبہ بندی
    ماحول کا مکمل علم تصور کرتا ہے
    """
    # خلا کی مطلق نمائندگی تیار کریں
    graph = create_graph_from_abstract_map(obstacles)

    # الگوردم (مثلاً A*) کو مطلق خلا میں لاگو کریں
    path = a_star_algorithm(graph, start_pos, goal_position)

    return path  # راستہ کے اشاریوں کی مطلق ترتیب
```

## جسمانی ایجنٹس: جسمانی حقیقت میں ذہانت

جسمانی ایجنٹس کو جسمانی حقیقت کی پیچیدگیوں کو سنبھالنا ہوتا ہے:

- **جزوی معلومات**: ماحول سے محدود حسی ان پٹ
- **سٹوکاسٹک آپریشنز**: جسمانی بات چیت کی وجہ سے غیر یقینی نتائج
- **جسمانی پابندیاں**: جسمانیات کے قوانین اور مواد کی خصوصیات کے تابع
- **حقوقی وقت کی پروسیسنگ**: متحرک ماحولیاتی تبدیلیوں کا جواب دینا

### جسمانی ذہانت کی خصوصیات

1. **سینسریموٹر انضمام**: ادراک اور عمل کے درمیان تنگ تعلق
2. **نمودار ہونے والے رویے**: سادہ اصولوں سے پیچیدہ رویے
3. **مطابقت کا سیکھنا**: جسمانی بات چیت کے ذریعہ سیکھنا
4. **morphological computation**: جسمانی شکل کا حساب کو سنبھالنا

```python
# مثال: راستہ کی منصوبہ بندی کے لیے جسمانی نقطہ نظر
class EmbodiedPathPlanner:
    def __init__(self, robot_model):
        self.robot = robot_model
        self.sensors = robot_model.sensors
        self.actuators = robot_model.actuators

    def plan_path(self, goal_position):
        """
        جسمانی نقطہ نظر: حقیقی وقت کے حسی ان پٹ کی بنیاد پر منصوبہ بندی
        اور ایجنٹ کی جسمانی صلاحیات
        """
        while not self.reached_goal(goal_position):
            # ماحول کا احساس
            sensory_data = self.sensors.get_data()

            # جسمانی پابندیوں کے مطابق ردعمل
            action = self.select_action_based_on_embodiment(
                sensory_data, goal_position
            )

            # عمل کریں اور نتائج دیکھیں
            self.actuators.execute(action)

            # تجربے سے سیکھیں
            self.adapt_behavior_from_experience()
```

## اہم فروق

### معلومات کی پروسیسنگ

| ڈیجیٹل ذہانت | جسمانی ایجنٹس |
|----------------|-----------------|
| مطلق علامات کو عمل کرتا ہے | حسی سگنلز کو عمل کرتا ہے |
| مکمل حالت کا علم | جزوی حالت کا تخمینہ |
| بیچ پروسیسنگ | مسلسل حقیقی وقت کی پروسیسنگ |
| سٹیٹک ماحول ماڈل | متحرک ماحول ماڈل |

### سیکھنے کے طریقے

| ڈیجیٹل ذہانت | جسمانی ایجنٹس |
|----------------|-----------------|
| ڈیٹا سیٹس سے نگرانی کے ساتھ سیکھنا | بات چیت کے ذریعہ سیکھنا |
| آف لائن تربیت | آن لائن سیکھنا |
| جامع ماڈل | مخصوص رویے |
| علامتی علم کا انتقال | جسمانی مہارت حاصل کرنا |

### مسئلہ حل کرنے کے طریقے

ڈیجیٹل مصنوعی ذہانت عام طور پر استعمال کرتا ہے:
- علامتی سوچ بچار
- ریاضی کا بہترین بنانا
- تلاش الگورتھم
- علم کی نمائندگی

جسمانی ایجنٹس اکثر استعمال کرتے ہیں:
- ردعمل کے رویے
- نمودار ہونے والی حکمت عملیاں
- morphological computation
- تقسیم شدہ کنٹرول

## مورا ویک متناقض

مورا ویک متناقض ڈیجیٹل اور جسمانی ذہانت کے درمیان ایک بنیادی فرق کو اجاگر کرتا ہے:

> "کمپیوٹر کو انٹیلی جنس ٹیسٹس یا چیکرز کھیلنے میں بالغ سطح کی کارکردگی دکھانے میں نسبتاً آسانی سے کیا جا سکتا ہے، اور ایک سالہ انسانی بچے کے ہنر دینا مشکل یا ناممکن ہے۔"

یہ متناقض یہ ثابت کرتا ہے کہ:
- بلند سطحی تجزیہ کرنا ڈیجیٹل طور پر نقل کرنا آسان ہے
- کم سطحی حسی موٹر کے ہنر نافذ کرنا انتہائی چیلنجنگ ہے
- جسمانی بات چیت کو بہت سے نظاموں کے مربوط ہونے کی ضرورت ہوتی ہے

## morphological computation

جسمانی ایجنٹس کا ایک کلیدی فائدہ morphological computation ہے - یہ خیال کہ ایجنٹ کی جسمانی شکل اس کی حساب کی صلاحیات میں حصہ ڈالتی ہے۔

### morphological computation کی مثالیں

1. **پیسیو ڈائینامک واکنگ**: روبوٹس جو صرف گریویٹی اور جسمانی ڈائینامکس کا استعمال کر کے چلتے ہیں
2. **کمپلائنس مکینزم**: ساختیں جو کنٹرول کے لیے لچک کا استعمال کرتی ہیں
3. **جسمانی سوچ بچار**: کلیہ عمل جو باڈی-ماحول کی بات چیت سے نمودار ہوتے ہیں

```python
# مثال: پیسیو ڈائینامک واکر
class PassiveWalker:
    def __init__(self):
        # جسمانی ڈیزائن چلنے کے قابل بناتا ہے بغیر فعال کنٹرول کے
        self.leg_length = 0.5  # میٹر
        self.mass_distribution = self.calculate_optimal_mass()
        self.foot_design = self.design_passive_foot()

    def walk_down_slope(self, slope_angle):
        """
        چلنا جسمانی ڈیزائن اور ماحولیاتی بات چیت سے نمودار ہوتا ہے
        کم فعال کنٹرول کی ضرورت ہے
        """
        # جسمانی شکل قدرتی طور پر چلنے کی حرکت پیدا کرتی ہے
        # جب مناسب ڈھلوان پر رکھا جاتا ہے
        return self.emergent_walking_pattern(slope_angle)
```

## سیمولیشن بمقابلہ حقیقت کا فرق

ڈیجیٹل سے جسمانی ذہانت کی طرف منتقلی کو "حقیقت کا فرق" کا سامنا ہے:

- **سیمولیشن کی درستی**: ڈیجیٹل ماڈل تمام جسمانی پیچیدگیوں کو ظاہر نہیں کر سکتے
- **ٹرانسفر لرننگ**: سیمولیشن میں سیکھی گئی مہارتیں حقیقت میں منتقل نہیں ہو سکتیں
- **حسی فروق**: حقیقی سینسرز میں شور، تاخیر، اور حدود ہوتی ہیں
- **ایکچویٹر کی حدود**: حقیقی ایکچویٹر میں تاخیر، توانائی کی حدود، اور پہننے کی حدود ہوتی ہیں

## ROS2 نافذ کاری: ڈیجیٹل بمقابلہ جسمانی نقطہ نظر

آئیے دیکھیں کہ ہم دونوں نقطہ نظر کا استعمال کرتے ہوئے ایک سادہ نیویگیشن کام کیسے نافذ کر سکتے ہیں:

### ڈیجیٹل نقطہ نظر (صرف سیمولیشن)

```python
# digital_navigation.py
import numpy as np

def simulate_navigation(start, goal, obstacles):
    """صرف ڈیجیٹل سیمولیشن نقطہ نظر"""
    # ماحول کا مکمل علم
    environment_map = create_perfect_map(obstacles)

    # A* کا استعمال کرتے ہوئے بہترین راستہ کی منصوبہ بندی کریں
    path = a_star(start, goal, environment_map)

    # عمل کی تفریح (کوئی حقیقی دنیا کی عدم یقینی نہیں)
    execution_log = []
    for waypoint in path:
        execution_log.append(f"Moving to {waypoint}")

    return execution_log, path
```

### جسمانی نقطہ نظر (حقیقی روبوٹ)

```python
# embodied_navigation.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math

class EmbodiedNavigator(Node):
    def __init__(self):
        super().__init__('embodied_navigator')

        # حقیقی حسی ان پٹ کے لیے سبسکرائبرز
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        # حرکت کے حکم کے لیے پبلشر
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # روبوٹ کی حالت
        self.current_pose = None
        self.scan_data = None
        self.goal = None

        # نیویگیشن پیرامیٹر
        self.linear_speed = 0.3  # میٹر/سیکنڈ
        self.angular_speed = 0.5  # ریڈین/سیکنڈ
        self.safe_distance = 0.5  # میٹر

    def scan_callback(self, msg):
        self.scan_data = msg.ranges

        # حقیقی سینسر ڈیٹا کو نویس، تاخیر، اور حدود کے ساتھ عمل کریں
        if self.goal and self.current_pose:
            self.navigate_towards_goal()

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

        # حقیقی پوزیشن کا استعمال کریں بجائے مکمل سیمولیشن کے
        if self.goal and self.scan_data:
            self.navigate_towards_goal()

    def navigate_towards_goal(self):
        """جسمانی نیویگیشن کے ساتھ حقیقی دنیا کی پابندیاں"""
        if not self.scan_data or not self.current_pose:
            return

        msg = Twist()

        # راستے میں رکاوٹوں کی جانچ کریں (حقیقی سینسر ڈیٹا)
        min_distance = min([r for r in self.scan_data if r > 0], default=float('inf'))

        if min_distance < self.safe_distance:
            # ٹکر سے بچنے کے لیے رکیں یا مڑیں
            msg.linear.x = 0.0
            msg.angular.z = self.angular_speed
        else:
            # جسمانی پابندیوں کو مدنظر رکھتے ہوئے مقصد کی طرف جائیں
            goal_direction = self.calculate_goal_direction()
            msg.linear.x = self.linear_speed * goal_direction.linear_factor
            msg.angular.z = self.angular_speed * goal_direction.angular_factor

        self.cmd_vel_pub.publish(msg)

    def calculate_goal_direction(self):
        """حقیقی پوزیشن اور مقصد کی بنیاد پر حرکت کا حساب لگائیں"""
        # یہ حساب کو روبوٹ کی جسمانی پابندیوں کا خیال رکھنا چاہیے
        # فقط مطلق کوآرڈینیٹس کے بجائے
        pass

def main(args=None):
    rclpy.init(args=args)
    navigator = EmbodiedNavigator()

    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## لیب: ڈیجیٹل بمقابلہ جسمانی نقطہ نظر کا موازنہ

اس لیب میں، ہم ایک ہی نیویگیشن کام کے لیے دونوں نقطہ نظر نافذ کریں گے اور ان کی کارکردگی کا موازنہ کریں گے:

```python
# lab_comparison.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import time

class ComparisonLab(Node):
    def __init__(self):
        super().__init__('comparison_lab')

        # سبسکرائبرز
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        # پبلشرز
        self.digital_cmd_pub = self.create_publisher(
            Twist, '/digital_cmd_vel', 10
        )
        self.embodied_cmd_pub = self.create_publisher(
            Twist, '/embodied_cmd_vel', 10
        )
        self.analysis_pub = self.create_publisher(
            String, '/comparison_analysis', 10
        )

        self.scan_data = None
        self.last_comparison_time = time.time()

    def scan_callback(self, msg):
        self.scan_data = msg.ranges
        self.compare_approaches()

    def digital_approach(self):
        """سیمولیٹڈ ڈیجیٹل نقطہ نظر"""
        # ایک حقیقی نافذ کاری میں یہ مکمل معلومات کا استعمال کرے گا
        # اس مثال کے لیے، ہم ڈیجیٹل نقطہ نظر کی تفریح کریں گے
        cmd = Twist()
        cmd.linear.x = 0.5  # ہمیشہ "مکمل" سیمولیشن میں آگے بڑھیں
        return cmd

    def embodied_approach(self):
        """حقیقی جسمانی نقطہ نظر"""
        if not self.scan_data:
            return Twist()

        cmd = Twist()

        # حقیقی سینسر ڈیٹا کی پروسیسنگ کے ساتھ حدود
        min_dist = min([r for r in self.scan_data if r > 0], default=float('inf'))

        if min_dist < 0.6:  # حقیقی رکاوٹ کا پتہ لگانا
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5
        else:
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0

        return cmd

    def compare_approaches(self):
        """دو نقطہ نظر کا موازنہ کریں"""
        if not self.scan_data or time.time() - self.last_comparison_time < 2.0:
            return

        digital_cmd = self.digital_approach()
        embodied_cmd = self.embodied_approach()

        # فروق کا تجزیہ
        analysis = f"Digital: linear={digital_cmd.linear.x}, angular={digital_cmd.angular.z} | "
        analysis += f"Embodied: linear={embodied_cmd.linear.x}, angular={embodied_cmd.angular.z}"

        self.analysis_pub.publish(String(data=analysis))
        self.last_comparison_time = time.time()

def main(args=None):
    rclpy.init(args=args)
    lab = ComparisonLab()

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

## مشق: اپنے روبوٹ کا تجزیہ کریں

ایک روبوٹ کا خیال کریں جو آپ ڈیزائن کر سکتے ہیں:

1. ڈیجیٹل سیمولیشن اور حقیقی روبوٹ کے درمیان کلیدی فروق کیا ہوں گے؟
2. جسمانی شکل اس کی صلاحیات کو کیسے متاثر کرے گی؟
3. اس کے ڈیزائن سے morphological computation کیا نمودار ہو سکتی ہے؟
4. آپ سیمولیشن-سے-حقیقت کے فرق کو کیسے پُٹریں گے؟

## خلاصہ

ڈیجیٹل ذہانت اور جسمانی ایجنٹس کے درمیان فرق جسمانی مصنوعی ذہانت کو سمجھنے کے لیے بنیادی ہے۔ جبکہ ڈیجیٹل مصنوعی ذہانت کنٹرول شدہ، مطلق ماحول میں کام کرتی ہے، جسمانی ایجنٹس جسمانی حقیقت کی پیچیدگیوں کو سنبھالنے کے لیے مجبور ہیں۔ اس کے لیے مسئلہ حل کرنے، سیکھنے، اور نظام ڈیزائن کے لیے مختلف نقطہ نظر کی ضرورت ہوتی ہے۔

کلیدی بصیرت یہ ہے کہ جسمانی اظہار صرف ایک پابندی نہیں بلکہ ایک وسیلہ ہے جو morphological computation، حسی موٹر انضمام، اور حقیقی دنیا کے سیکھنے کے مواقع کے ذریعہ ذہانت کو بڑھا سکتا ہے۔

اگلے سبق میں، ہم انسان نما روبوٹکس کی موجودہ حالت کا جائزہ لیں گے اور دیکھیں گے کہ یہ اصول حقیقی نظاموں میں کیسے لاگو ہوتے ہیں۔