---
sidebar_position: 2
---

# ڈیجیٹل انٹیلی جنس بمقابلہ امبدڈ ایجنٹس

## ڈیجیٹل-فزیکل تقسیم

روایتی ای آئی سسٹم بنیادی طور پر ڈیجیٹل ڈومینز میں کام کرتے ہیں، جسمانی پابندیوں کے بغیر مجرد معلومات کو پروسیس کرتے ہیں۔ اس کے برعکس، امبدڈ ایجنٹس جسمانی ماحولوں کے اندر موجود ہوتے ہیں، جہاں ان کی شکل اور حقیقی دنیا کے ساتھ تعاملات بنیادی طور پر ان کی انٹیلی جنس کو متاثر کرتے ہیں۔

## ڈیجیٹل انٹیلی جنس: مجرد سرزمین

ڈیجیٹل انٹیلی جنس سسٹم کنٹرول شدہ، مجرد ماحولوں میں کام کرتے ہیں:

- **کامل معلومات**: سسٹم کی حالت کا مکمل علم
- ** determinant آپریشنز**: دی گئی ان پٹس کے لیے قابل پیش گوئی نتائج
- **جسمانی پابندیاں نہیں**: کوئی مٹھی، گریویٹی، یا میٹریل کی حدود نہیں
- **سادہ ماڈلز**: حقیقت کی مجرد نمائندگیاں

### ڈیجیٹل ای آئی کی خصوصیات

1. **سمبولک پروسیسنگ**: مجرد علامتوں اور نمائندگیوں کا مینیپولیشن
2. **رول-بیسڈ ریزننگ**: رسمی علم پر لاگو کردہ منطق
3. **پیٹرن ریکوگنیشن**: ڈیجیٹل ڈیٹا میں پیٹرنز کی شناخت
4. **آپٹیمائزیشن**: خوبصورت طور پر وضاحت شدہ خالص میں ریاضیاتی آپٹیمائزیشن

```python
# مثال: ڈیجیٹل ای آئی کا راستہ منصوبہ بندی کا نقطہ نظر
def digital_path_planner(goal_position, obstacles):
    """
    راستہ منصوبہ بندی کا روایتی ڈیجیٹل نقطہ نظر
    ماحول کے بارے میں مکمل علم کا فرض کرتا ہے
    """
    # خالص کی مجرد نمائندگی تیار کریں
    graph = create_graph_from_abstract_map(obstacles)

    # الگوردم (جیسے، A*) کو مجرد خالص میں لاگو کریں
    path = a_star_algorithm(graph, start_pos, goal_position)

    return path  # ویز پوائنٹس کی مجرد ترتیب
```

## امبدڈ ایجنٹس: جسمانی حقیقت میں انٹیلی جنس

امبدڈ ایجنٹس جسمانی حقیقت کی پیچیدگیوں کو نیویگیٹ کرنا چاہیے:

- **جزوی معلومات**: ماحول سے محدود حسی ان پٹ
- **سٹوکاسٹک آپریشنز**: جسمانی تعاملات کی وجہ سے غیر یقینی نتائج
- **جسمانی پابندیاں**: طبیعی قوانین اور میٹریل کی خصوصیات کے تابع
- **ریل ٹائم پروسیسنگ**: متحرک ماحولیاتی تبدیلیوں کا جواب دینا

### امبدڈ انٹیلی جنس کی خصوصیات

1. **سینسروموٹر انٹیگریشن**: ادراک اور کارروائی کے درمیان سخت ربط
2. **ایمرجینٹ بیہیویئرز**: سادہ قواعد سے ابھرنے والے پیچیدہ رویے
3. **ایڈاپٹیو لرننگ**: جسمانی تعامل کے ذریعے سیکھنا
4. **مورفولوجیکل کمپوٹیشن**: جسمانی شکل کمپوٹیشن میں حصہ ڈالتی ہے

```python
# مثال: راستہ منصوبہ بندی کا امبدڈ نقطہ نظر
class EmbodiedPathPlanner:
    def __init__(self, robot_model):
        self.robot = robot_model
        self.sensors = robot_model.sensors
        self.actuators = robot_model.actuators

    def plan_path(self, goal_position):
        """
        امبدڈ نقطہ نظر: حقیقی وقت کے حسی ان پٹ کی بنیاد پر منصوبہ بندی
        اور ایجنٹ کی جسمانی صلاحیتوں کے مطابق
        """
        while not self.reached_goal(goal_position):
            # ماحول کا ادراک کریں
            sensory_data = self.sensors.get_data()

            # جسمانی پابندیوں کے مطابق ردعمل دیں
            action = self.select_action_based_on_embodiment(
                sensory_data, goal_position
            )

            # کارروائی انجام دیں اور نتائج مشاہدہ کریں
            self.actuators.execute(action)

            # تعامل سے سیکھیں
            self.adapt_behavior_from_experience()
```

## کلیدی فروق

### معلومات کی پروسیسنگ

| ڈیجیٹل انٹیلی جنس | امبدڈ ایجنٹس |
|---------------------|-----------------|
| مجرد علامتوں کو پروسیس کرتا ہے | حسی سگنلز کو پروسیس کرتا ہے |
| مکمل ریاست کا علم | جزوی ریاست کا تخمینہ |
| بیچ پروسیسنگ | مسلسل حقیقی وقت کی پروسیسنگ |
| سٹیٹک ماحول کا ماڈل | متحرک ماحول کا ماڈل |

### سیکھنے کے میکنزم

| ڈیجیٹل انٹیلی جنس | امبدڈ ایجنٹس |
|---------------------|-----------------|
| ڈیٹا سیٹس سے نگرانی شدہ سیکھنا | تعامل کے ذریعے سیکھنا |
| آف لائن تربیت | آن لائن سیکھنا |
| جنرلائزڈ ماڈلز | مخصوص رویے |
| علامتی علم کا منتقل | جسمانی مہارت کا حصول |

### مسئلہ حل کرنے کے نقطہ نظر

ڈیجیٹل ای آئی عام طور پر استعمال کرتا ہے:
- علامتی استدلال
- ریاضیاتی آپٹیمائزیشن
- تلاش الگوردمز
- علم کی نمائندگی

امبدڈ ایجنٹس اکثر استعمال کرتے ہیں:
- ری ایکٹو بیہیویئرز
- ایمرجینٹ حکمت عمل
- مورفولوجیکل کمپوٹیشن
- تقسیم کردہ کنٹرول

## مورا ویک پیراڈوکس

مورا ویک پیراڈوکس ڈیجیٹل اور امبدڈ انٹیلی جنس کے درمیان ایک بنیادی فرق کو اجاگر کرتا ہے:

> "کمپیوٹرز کو انٹیلی جنس ٹیسٹس میں بالغ سطح کی کارکردگی یا چیکرز کھیلنے میں دکھانے میں نسبتاً آسان ہے، اور انہیں ایک سالہ انسانی بچے کی مہارتوں کو دینا مشکل یا ناممکن ہے۔"

یہ پیراڈوکس ظاہر کرتا ہے کہ:
- ہائی لیول استدلال کو ڈیجیٹل طور پر نقل کرنا آسان ہے
- کم لیول سینسروموٹر کی مہارتوں کو نافذ کرنا انتہائی چیلنجنگ ہے
- جسمانی تعامل کو بہت سے سسٹم کی مسلسل ضمیت کی ضرورت ہوتی ہے

## مورفولوجیکل کمپوٹیشن

امبدڈ ایجنٹس کا ایک کلیدی فائدہ مورفولوجیکل کمپوٹیشن ہے - خیال کہ ایجنٹ کی جسمانی شکل اس کی کمپوٹیشنل صلاحیتوں میں حصہ ڈالتی ہے۔

### مورفولوجیکل کمپوٹیشن کی مثالیں

1. **پیسیو ڈائی نیمک وانکنگ**: روبوٹس جو صرف گریویٹی اور جسمانی ڈائی نیمکس کا استعمال کرتے ہوئے چلتے ہیں
2. **کمپلائیںٹ میکنزمز**: ایسے ڈھانچے جو کنٹرول کے لیے لچک کا استعمال کرتے ہیں
3. **امبدڈ کوگنیشن**: وہ کوگنیٹو عمل جو بدن-ماحول کے تعامل سے ابھرتے ہیں

```python
# مثال: پیسیو ڈائی نیمک واکر
class PassiveWalker:
    def __init__(self):
        # جسمانی ڈیزائن چلنے کے قابل بناتا ہے بغیر کسی فعال کنٹرول کے
        self.leg_length = 0.5  # میٹر
        self.mass_distribution = self.calculate_optimal_mass()
        self.foot_design = self.design_passive_foot()

    def walk_down_slope(self, slope_angle):
        """
        چلنے کا عمل جسمانی ڈیزائن اور ماحولیاتی تعامل سے ابھرتا ہے
        کم فعال کنٹرول کی ضرورت ہے
        """
        # جسمانی شکل قدرتی طور پر چلنے کا عمل پیدا کرتی ہے
        # جب مناسب ڈھلوان پر رکھا جاتا ہے
        return self.emergent_walking_pattern(slope_angle)
```

## سیمولیشن بمقابلہ حقیقت کا فرق

ڈیجیٹل سے امبدڈ انٹیلی جنس کی منتقلی کو "ریئلٹی گیپ" کا سامنا ہے:

- **سیمولیشن کی درستی**: ڈیجیٹل ماڈلز تمام جسمانی پیچیدگیوں کو نہیں پکڑ سکتے
- **ٹرانسفر لرننگ**: سیمولیشن میں سیکھی گئی مہارتیں حقیقت میں منتقل نہیں ہو سکتیں
- **حسی فروق**: حقیقی سینسرز میں شور، دیر، اور حدود ہوتی ہیں
- **ایکچوایٹر کی حدود**: حقیقی ایکچوایٹر میں تاخیر، طاقت کی حدود، اور پہناؤ ہوتا ہے

## ROS2 نفاذ: ڈیجیٹل بمقابلہ امبدڈ نقطہ نظر

آئیے دیکھیں کہ ہم دونوں نقطہ نظر کا استعمال کرتے ہوئے ایک سادہ نیویگیشن کام کیسے نافذ کر سکتے ہیں:

### ڈیجیٹل نقطہ نظر (صرف سیمولیشن)

```python
# digital_navigation.py
import numpy as np

def simulate_navigation(start, goal, obstacles):
    """صرف ڈیجیٹل سیمولیشن نقطہ نظر"""
    # ماحول کا مکمل علم
    environment_map = create_perfect_map(obstacles)

    # A* کا استعمال کرتے ہوئے بہترین راستہ منصوبہ بندی کریں
    path = a_star(start, goal, environment_map)

    # عمل کی شبیہ سازی (کوئی حقیقی دنیا کے عدم یقین نہیں)
    execution_log = []
    for waypoint in path:
        execution_log.append(f"Moving to {waypoint}")

    return execution_log, path
```

### امبدڈ نقطہ نظر (حقیقی روبوٹ)

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

        # حقیقی حسی ان پٹ کے لیے سبسکرائبز
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

        # نیویگیشن پیرامیٹرز
        self.linear_speed = 0.3  # میٹر/سیکنڈ
        self.angular_speed = 0.5  # ریڈین/سیکنڈ
        self.safe_distance = 0.5  # میٹر

    def scan_callback(self, msg):
        self.scan_data = msg.ranges

        # حقیقی سینسر ڈیٹا کو شور اور حدود کے ساتھ پروسیس کریں
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

        # راستے میں رکاوٹوں کی جانچ (حقیقی سینسر ڈیٹا)
        min_distance = min([r for r in self.scan_data if r > 0], default=float('inf'))

        if min_distance < self.safe_distance:
            # رکنے یا موڑنے کے تصادم سے بچنے کے لیے
            msg.linear.x = 0.0
            msg.angular.z = self.angular_speed
        else:
            # جسمانی پابندیوں کو مدنظر رکھتے ہوئے مقصد کی طرف بڑھیں
            goal_direction = self.calculate_goal_direction()
            msg.linear.x = self.linear_speed * goal_direction.linear_factor
            msg.angular.z = self.angular_speed * goal_direction.angular_factor

        self.cmd_vel_pub.publish(msg)

    def calculate_goal_direction(self):
        """حقیقی پوزیشن اور مقصد کی بنیاد پر حرکت کا حساب لگائیں"""
        # یہ حساب جسمانی پابندیوں کو مدنظر رکھنا چاہیے
        # صرف مجرد کوآرڈینیٹس کے بجائے حقیقی روبوٹ کا
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

## لیب: ڈیجیٹل بمقابلہ امبدڈ نقطہ نظر کا موازنہ

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

        # سبسکرائبز
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
        """شبیہ سازی شدہ ڈیجیٹل نقطہ نظر"""
        # اس میں ایک حقیقی نفاذ میں مکمل معلومات کا استعمال ہوگا
        # اس مثال کے لیے، ہم ڈیجیٹل نقطہ نظر کی شبیہ سازی کریں گے
        cmd = Twist()
        cmd.linear.x = 0.5  # ہمیشہ "مکمل" سیمولیشن میں آگے بڑھیں
        return cmd

    def embodied_approach(self):
        """حقیقی امبدڈ نقطہ نظر"""
        if not self.scan_data:
            return Twist()

        cmd = Twist()

        # حقیقی سینسر ڈیٹا کی پروسیسنگ میں حدود
        min_dist = min([r for r in self.scan_data if r > 0], default=float('inf'))

        if min_dist < 0.6:  # حقیقی رکاوٹ کا پتہ لگانا
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5
        else:
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0

        return cmd

    def compare_approaches(self):
        """دونوں نقطہ نظر کا موازنہ کریں"""
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

ایک روبوٹ پر غور کریں جس کا آپ ڈیزائن کر سکتے ہیں:

1. ڈیجیٹل سیمولیشن اور حقیقی روبوٹ کے درمیان کلیدی فروق کیا ہوں گے؟
2. جسمانی شکل اس کی صلاحیتوں کو کیسے متاثر کرے گی؟
3. اس کے ڈیزائن سے کون سی مورفولوجیکل کمپوٹیشن ابھر سکتی ہے؟
4. آپ سیمولیشن سے حقیقت کے فرق کو کیسے پُر کریں گے؟

## خلاصہ

ڈیجیٹل انٹیلی جنس اور امبدڈ ایجنٹس کے درمیان فرق فزیکل ای آئی کو سمجھنے کے لیے بنیادی ہے۔ جبکہ ڈیجیٹل ای آئی مجرد، کنٹرول شدہ ماحولوں میں کام کرتا ہے، امبدڈ ایجنٹس جسمانی حقیقت کی پیچیدگیوں کو نیویگیٹ کرنا چاہیے۔ اس کے لیے مسئلہ حل کرنے، سیکھنے، اور سسٹم ڈیزائن کے لیے مختلف نقطہ نظر کی ضرورت ہوتی ہے۔

کلیدی بصیرت یہ ہے کہ امبدمنٹ محض ایک پابندی نہیں ہے بلکہ ایک وسیلہ ہے جو مورفولوجیکل کمپوٹیشن، سینسروموٹر انٹیگریشن، اور حقیقی دنیا کے سیکھنے کے مواقع کے ذریعے انٹیلی جنس کو بہتر بنا سکتا ہے۔

اگلی میں، ہم ہیومنوائڈ روبوٹکس کی موجودہ حالت کا جائزہ لیں گے اور دیکھیں گے کہ یہ اصول حقیقی سسٹم میں کیسے لاگو ہوتے ہیں۔