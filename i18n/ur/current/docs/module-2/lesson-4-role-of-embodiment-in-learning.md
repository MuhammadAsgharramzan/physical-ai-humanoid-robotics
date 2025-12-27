---
sidebar_position: 4
---

# سیکھنے اور ذہانت میں جسمانیت کا کردار

## تعارف

جسمانیت سیکھنے اور ذہانت میں ایک اہم کردار ادا کرتی ہے، جو بنیادی طور پر اس طرح سے شکل دیتی ہے کہ جسمانی مصنوعی ذہانت کے نظام نظریات حاصل کرتے ہیں اور صلاحیات کو ترقی دیتے ہیں۔ روایتی مصنوعی ذہانت کے نظاموں کے برعکس جو مطلق ڈیٹا سے سیکھتے ہیں، جسمانی ایجنٹس اپنے جسمانی ماحول کے ساتھ براہ راست بات چیت کے ذریعے سیکھتے ہیں، جس کے نتیجے میں زیادہ مضبوط اور موافق ذہانت پیدا ہوتی ہے۔

## جسمانیت کی فرضیت

جسمانیت کی فرضیت یہ تجویز کرتی ہے کہ ایجنٹ کی جسمانی شکل اور حسی موٹر کی صلاحیات اس کی کلیہ ترقی کو نمایاں طور پر متاثر کرتی ہیں۔ یہ ذہانت کے صرف کمپیوٹیشنل ہونے کے کلاسیکی نقطہ نظر کو چیلنج کرتی ہے، اس کے بجائے جسم، دماغ، اور ماحول کے درمیان تنگ ربط کو زیادہ اہمیت دیتی ہے۔

### جسمانی سوچ بچار کے کلیدی اصول

1. **جسمانیت کی پابندی**: جسمانی شکل مخصوص قسم کی سوچ بچار کو محدود کرتی ہے اور فعال کرتی ہے
2. **ماحولیاتی ربط**: کلیہ عمل ماحولیاتی بات چیت کے ساتھ گہرائی سے جڑے ہوتے ہیں
3. **تقسیم شدہ کمپیوٹیشن**: کمپیوٹیشن دماغ، جسم، اور ماحول میں تقسیم ہے
4. **متحرک بات چیت**: ذہانت مسلسل متحرک بات چیت سے نمودار ہوتی ہے

```python
# مثال: ماحولیاتی بات چیت کے ذریعے جسمانی سیکھنا
class EmbodiedLearner:
    def __init__(self, body_properties, sensor_config, actuator_config):
        self.body_properties = body_properties  # جسمانی پابندیاں اور صلاحیات
        self.sensors = sensor_config
        self.actuators = actuator_config
        self.experience_buffer = []
        self.learning_model = self.initialize_learning_model()

    def interact_with_environment(self, environment_state):
        """ایجنٹ ماحول کے ساتھ اپنی جسمانی صلاحیات کے مطابق بات چیت کرتا ہے"""
        # حسی ادراک کے ذریعے
        sensory_input = self.sense(environment_state)

        # جسمانی شکل کی پابندی کے تحت عمل منتخب کریں
        action = self.select_action(sensory_input, environment_state)

        # ماحول کے ساتھ جسمانی بات چیت
        environment_response = self.execute_action(action, environment_state)

        # بات چیت سے سیکھیں
        self.learn_from_interaction(sensory_input, action, environment_response)

        return environment_response

    def sense(self, environment_state):
        """جسمانی حسی ترتیب کی پابندی کے تحت حسی پروسیسنگ"""
        # ایجنٹ صرف وہی ادراک کر سکتا ہے جس کی اس کے سینسرز اجازت دیتے ہیں
        sensed_data = {}
        for sensor_type, config in self.sensors.items():
            sensed_data[sensor_type] = self.process_sensor_data(
                environment_state, config
            )
        return sensed_data

    def select_action(self, sensory_input, environment_state):
        """جسمانی صلاحیات کی پابندی کے تحت عمل منتخب کریں"""
        # اعمال ایجنٹ کی جسمانی شکل کے ساتھ مطابقت رکھنا چاہیے
        possible_actions = self.get_possible_actions()

        # سیکھنے کے ماڈل کے ذریعے عمل منتخب کریں
        selected_action = self.learning_model.select_action(
            sensory_input, possible_actions, environment_state
        )

        # یقینی بنائیں کہ عمل جسمانی طور پر ممکن ہے
        if self.is_action_feasible(selected_action):
            return selected_action
        else:
            # جسمانی طور پر ممکن عمل کے لیے واپسی
            return self.get_feasible_fallback_action(selected_action)

    def get_possible_actions(self):
        """ایجنٹ کی جسمانی ترتیب کے لحاظ سے ممکنہ اعمال حاصل کریں"""
        # یہ ایجنٹ کی جسمانی شکل کے ذریعے طے ہوتا ہے
        actions = []

        # مثال کے طور پر، اگر ایجنٹ کے پاس ٹانگیں ہیں، تو وہ چل سکتا ہے
        if 'legs' in self.body_properties:
            actions.extend(['walk_forward', 'walk_backward', 'turn_left', 'turn_right'])

        # اگر اس کے پاس ہاتھ ہیں، تو یہ ہیرا پھیری کر سکتا ہے
        if 'arms' in self.body_properties:
            actions.extend(['reach', 'grasp', 'manipulate'])

        # اگر اس کے پاس تقریر کی صلاحیت ہے
        if 'speech' in self.body_properties:
            actions.append('speak')

        return actions

    def is_action_feasible(self, action):
        """چیک کریں کہ کیا عمل جسمانی طور پر ممکن ہے"""
        # نفاذ ایجنٹ کی مخصوص جسمانی ترتیب پر منحصر ہے
        return True  # مثال کے لیے سادہ بنایا گیا

    def execute_action(self, action, environment_state):
        """عمل انجام دیں اور ماحولیاتی جواب دیکھیں"""
        # عمل کا جسمانی انجام
        result = self.actuators.execute(action)

        # ماحولیاتی جواب
        new_state = environment_state.apply_action(action, self.body_properties)

        return new_state

    def learn_from_interaction(self, sensory_input, action, environment_response):
        """بات چیت کی بنیاد پر سیکھنے کا ماڈل اپ ڈیٹ کریں"""
        experience = {
            'sensory_input': sensory_input,
            'action': action,
            'environment_response': environment_response,
            'outcome': self.evaluate_outcome(action, environment_response)
        }

        self.experience_buffer.append(experience)

        # نئے تجربے کے ساتھ سیکھنے کا ماڈل اپ ڈیٹ کریں
        self.learning_model.update(experience)
```

## جسمانی سیکھنے کی اقسام

### 1. جسمانی مشق کے ذریعے موٹر سیکھنا

موٹر مہارتوں کو دہرائی گئی جسمانی مشق کے ذریعے سیکھا جاتا ہے، جہاں جسم کی جسمانی خصوصیات سیکھنے کے عمل کو شکل دیتی ہیں:

```python
# مثال: جسمانی مشق کے ذریعے موٹر سیکھنا
class MotorLearningSystem:
    def __init__(self, robot_dynamics_model):
        self.dynamics_model = robot_dynamics_model
        self.motor_primitives = {}  # سیکھے گئے حرکت کے نمونے
        self.practice_sessions = []  # مشق کے ایام کی تاریخ
        self.performance_metrics = {}

    def practice_movement(self, movement_pattern, repetitions=10):
        """کارکردگی میں بہتری کے لیے حرکت کے نمونے کی مشق کریں"""
        session_results = []

        for i in range(repetitions):
            # موجودہ مہارت کی سطح کے ساتھ حرکت کو انجام دیں
            execution_result = self.execute_movement(movement_pattern)

            # کارکردگی کا جائزہ لیں
            performance_score = self.evaluate_performance(execution_result)

            # نتائج محفوظ کریں
            session_results.append({
                'attempt': i,
                'result': execution_result,
                'score': performance_score,
                'errors': execution_result.get('errors', [])
            })

            # نتائج کی بنیاد پر موٹر پرائمری اپ ڈیٹ کریں
            self.update_motor_primitive(movement_pattern, execution_result)

        # طویل مدتی سیکھنے کے لیے ایام محفوظ کریں
        self.practice_sessions.append({
            'movement': movement_pattern,
            'results': session_results,
            'improvement': self.calculate_improvement(session_results)
        })

        return session_results

    def execute_movement(self, pattern):
        """جسمانی جسم کا استعمال کرتے ہوئے حرکت کو انجام دیں"""
        # جسمانی انجام کے لیے ڈائنامکس ماڈل استعمال کریں
        execution = self.dynamics_model.execute_pattern(pattern)

        # جسمانی پابندیاں اور شور شامل کریں
        execution['success'] = self.is_successful_execution(execution)
        execution['energy_consumed'] = self.calculate_energy_consumption(execution)

        return execution

    def update_motor_primitive(self, pattern, result):
        """نجام کے نتائج کی بنیاد پر موٹر پرائمری اپ ڈیٹ کریں"""
        if pattern not in self.motor_primitives:
            self.motor_primitives[pattern] = {
                'parameters': {},
                'success_rate': 0.0,
                'efficiency': 1.0
            }

        # کامیاب انجام کی بنیاد پر پیرامیٹر اپ ڈیٹ کریں
        if result['success']:
            self.motor_primitives[pattern]['success_rate'] += 0.1
        else:
            self.motor_primitives[pattern]['success_rate'] -= 0.05

        # 0 اور 1 کے درمیان محدود کریں
        self.motor_primitives[pattern]['success_rate'] = max(
            0.0, min(1.0, self.motor_primitives[pattern]['success_rate'])
        )

    def is_successful_execution(self, execution):
        """تعین کریں کہ کیا حرکت کا انجام کامیاب تھا"""
        # جسمانی پابندیوں اور مقاصد کی بنیاد پر
        return execution.get('completed', False) and execution.get('energy_efficient', True)

    def calculate_energy_consumption(self, execution):
        """جسمانی ڈائنامکس کی بنیاد پر توانائی کی کھپت کا حساب لگائیں"""
        # توانائی کا حساب لگانے کے لیے جسمانی ماڈل استعمال کریں
        return execution.get('effort', 0.0)
```

### 2. حسی موٹر تجربے کے ذریعے ادراکی سیکھنا

ادراکی صلاحیات حسی اور موٹر اعمال کے درمیان بات چیت کے ذریعے ترقی پاتی ہیں:

```python
# مثال: حسی موٹر تجربے کے ذریعے ادراکی سیکھنا
class SensorimotorPerceptualLearner:
    def __init__(self, sensor_config, motor_config):
        self.sensors = sensor_config
        self.motors = motor_config
        self.perceptual_models = {}
        self.sensorimotor_correlations = {}

    def explore_environment(self, exploration_strategy):
        """حسی موٹر من coordination کا استعمال کرتے ہوئے ماحول کا جائزہ لیں"""
        exploration_results = []

        for action in exploration_strategy:
            # تلاشیاتی عمل انجام دیں
            motor_state = self.execute_exploratory_action(action)

            # حسی نتائج کا مشاہدہ کریں
            sensory_state = self.get_sensory_feedback(action, motor_state)

            # حسی موٹر کے تعلقات سیکھیں
            self.update_sensorimotor_model(action, sensory_state)

            exploration_results.append({
                'action': action,
                'motor_state': motor_state,
                'sensory_state': sensory_state
            })

        return exploration_results

    def execute_exploratory_action(self, action):
        """تلاشیاتی مقصد کے لیے عمل انجام دیں"""
        # مثال کے طور پر: چیز کو چھونے کے لیے ہاتھ لائیں، بہتر دیکھنے کے لیے سر گھمائیں
        return self.motors.execute(action)

    def get_sensory_feedback(self, action, motor_state):
        """تلاشیاتی عمل سے حسی فیڈ بیک حاصل کریں"""
        # موجودہ سینسر ریڈنگز کو عمل کے سیاق کے ساتھ جوڑیں
        sensory_data = {}
        for sensor_type in self.sensors:
            sensory_data[sensor_type] = self.sensors[sensor_type].read()

        # حسی موٹر سیکھنے کے لیے عمل کا سیاق شامل کریں
        return {
            'raw_sensory': sensory_data,
            'action_context': action,
            'motor_context': motor_state
        }

    def update_sensorimotor_model(self, action, sensory_state):
        """حسی موٹر کے تعلقات کے ماڈل کو اپ ڈیٹ کریں"""
        # سیکھیں کہ اعمال حسی ان پٹ کو کیسے متاثر کرتے ہیں
        key = self.create_sensorimotor_key(action, sensory_state)

        if key not in self.sensorimotor_correlations:
            self.sensorimotor_correlations[key] = {
                'frequency': 0,
                'consistency': 0.0,
                'predictive_value': 0.0
            }

        self.sensorimotor_correlations[key]['frequency'] += 1
```

### 3. جسمانی جذب کے ذریعے تصوراتی سیکھنا

مطلق تصورات جسمانی تجربے میں جڑے ہوتے ہیں:

```python
# مثال: جسمانی جذب کے ذریعے تصورات سیکھنا
class PhysicalConceptLearner:
    def __init__(self):
        self.concept_representations = {}
        self.physical_experiences = []

    def learn_concept_from_experience(self, concept_name, physical_experience):
        """جسمانی بات چیت کے تجربے سے تصور سیکھیں"""
        # جسمانی تجربہ محفوظ کریں
        self.physical_experiences.append({
            'concept': concept_name,
            'experience': physical_experience,
            'context': physical_experience.get('environment_context', {})
        })

        # تصور کی نمائندگی اپ ڈیٹ کریں
        if concept_name not in self.concept_representations:
            self.concept_representations[concept_name] = {
                'instances': [],
                'prototypes': None,
                'relations': {}
            }

        # تصور میں نیا نمونہ شامل کریں
        self.concept_representations[concept_name]['instances'].append(
            physical_experience
        )

        # تمام نمونوں کی بنیاد پر پروٹو ٹائپ اپ ڈیٹ کریں
        self.update_concept_prototype(concept_name)

    def update_concept_prototype(self, concept_name):
        """تمام نمونوں کی بنیاد پر پروٹو ٹائپ نمائندگی اپ ڈیٹ کریں"""
        instances = self.concept_representations[concept_name]['instances']

        if len(instances) == 0:
            return

        # کلیدی خصوصیات کے اوسط کے طور پر پروٹو ٹائپ کا حساب لگائیں
        prototype = {}
        for key in instances[0].keys():
            if isinstance(instances[0][key], (int, float)):
                # عددی اقدار کا اوسط
                prototype[key] = sum(instance[key] for instance in instances) / len(instances)
            else:
                # زمرہ وار اقدار کے لیے، سب سے عام تلاش کریں
                values = [instance[key] for instance in instances]
                prototype[key] = max(set(values), key=values.count)

        self.concept_representations[concept_name]['prototypes'] = prototype

    def ground_abstract_reasoning(self, abstract_query):
        """جسمانی تجربے میں مطلق استدلال کو جذب کریں"""
        # مطلق تصورات کو جسمانی تجربوں سے منسلک کریں
        grounded_query = self.map_to_physical_experiences(abstract_query)

        # جسمانی مثالوں کا استعمال کرتے ہوئے استدلال کریں
        reasoning_result = self.physical_analogy_reasoning(grounded_query)

        # مطلق ڈومین میں واپس جائیں
        abstract_result = self.map_to_abstract_result(reasoning_result)

        return abstract_result

    def map_to_physical_experiences(self, abstract_query):
        """مطلق استفسار کو متعلقہ جسمانی تجربوں سے منسلک کریں"""
        # مطلق تصور کے لیے متعلقہ جسمانی تجربے تلاش کریں
        relevant_experiences = []
        for exp in self.physical_experiences:
            if self.is_relevant_experience(exp, abstract_query):
                relevant_experiences.append(exp)

        return {
            'query': abstract_query,
            'relevant_experiences': relevant_experiences
        }

    def is_relevant_experience(self, experience, query):
        """تعین کریں کہ کیا تجربہ استفسار کے لیے متعلقہ ہے"""
        # سادہ متعلقہ چیک
        return True  # جگہ کا نمونہ
```

## ROS2 نفاذ: جسمانی سیکھنے کا سسٹم

یہاں ایک جامع ROS2 نفاذ ہے جو جسمانی سیکھنے کو ظاہر کرتا ہے:

```python
# embodied_learning_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, LaserScan, Image
from geometry_msgs.msg import Twist, Point, Pose
from std_msgs.msg import String, Float32, Bool
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
import numpy as np
import cv2
import math
from collections import deque
import pickle

class EmbodiedLearningNode(Node):
    def __init__(self):
        super().__init__('embodied_learning')

        # پبلشرز
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.learning_status_pub = self.create_publisher(String, '/learning_status', 10)
        self.experience_pub = self.create_publisher(String, '/experience_log', 10)

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
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        # سسٹم کمپوننٹس
        self.cv_bridge = CvBridge()
        self.motor_learner = MotorLearningSystem(None)  # اصل ڈائنامکس ماڈل استعمال کرے گا
        self.perceptual_learner = SensorimotorPerceptualLearner({}, {})
        self.concept_learner = PhysicalConceptLearner()

        # ڈیٹا اسٹوریج
        self.joint_states = None
        self.imu_data = None
        self.laser_data = None
        self.image_data = None

        # سیکھنے کے اجزاء
        self.experience_buffer = deque(maxlen=1000)
        self.learning_enabled = True
        self.exploration_phase = True

        # سیکھنے کے پیرامیٹر
        self.exploration_rate = 0.3
        self.learning_rate = 0.1
        self.experience_threshold = 100

        # کنٹرول لوپ
        self.control_timer = self.create_timer(0.05, self.learning_loop)

        # سیکھنے کی حالت
        self.learning_state = {
            'total_experiences': 0,
            'successful_interactions': 0,
            'learning_progress': 0.0
        }

    def joint_state_callback(self, msg):
        """جوائنٹ حالت کی تازہ کاریوں کو سنبھالیں"""
        self.joint_states = msg

    def imu_callback(self, msg):
        """IMU ڈیٹا کو سنبھالیں"""
        self.imu_data = msg

    def laser_callback(self, msg):
        """لیزر اسکین ڈیٹا کو سنبھالیں"""
        self.laser_data = msg

    def image_callback(self, msg):
        """کیمرہ امیج ڈیٹا کو سنبھالیں"""
        try:
            self.image_data = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Image callback error: {e}')

    def learning_loop(self):
        """جسمانی سیکھنے کا اصل لوپ نافذ کریں"""
        if not all([self.joint_states, self.imu_data, self.laser_data]):
            return

        # 1. ادراک کا مرحلہ
        perceptual_state = self.process_perception()

        # 2. تجربہ جمع کرنے کا مرحلہ
        experience = self.collect_experience(perceptual_state)

        # 3. سیکھنے کا مرحلہ
        if self.learning_enabled:
            self.update_learning_models(experience)

        # 4. رویہ منتخب کرنے کا مرحلہ
        behavior = self.select_behavior_based_on_learning()

        # 5. عمل انجام دینے کا مرحلہ
        action = self.execute_behavior(behavior)

        # 6. جائزہ کا مرحلہ
        self.evaluate_interaction(action, perceptual_state)

        # 7. حالت کی رپورٹنگ
        self.report_learning_status()

    def process_perception(self):
        """تمام سینسر ڈیٹا کو ادراک کی حالت میں عمل کریں"""
        perceptual_state = {
            'proprioception': self.get_proprioceptive_state(),
            'exteroception': self.get_exteroceptive_state(),
            'spatial_awareness': self.get_spatial_awareness(),
            'balance_state': self.get_balance_state()
        }

        return perceptual_state

    def get_proprioceptive_state(self):
        """پروپریوسیف سینسرز سے اندرونی حالت حاصل کریں"""
        if self.joint_states:
            return {
                'joint_positions': np.array(self.joint_states.position),
                'joint_velocities': np.array(self.joint_states.velocity),
                'joint_efforts': np.array(self.joint_states.effort),
                'body_configuration': self.get_body_configuration()
            }
        return {}

    def get_exteroceptive_state(self):
        """ایکسٹروپریوسیف سینسرز سے بیرونی حالت حاصل کریں"""
        exteroception = {}

        if self.laser_data:
            exteroception['obstacles'] = self.analyze_obstacles()
            exteroception['environment_layout'] = self.create_environment_map()

        if self.image_data is not None:
            exteroception['visual_features'] = self.extract_visual_features()

        if self.imu_data:
            exteroception['inertial_state'] = {
                'orientation': self.imu_data.orientation,
                'angular_velocity': self.imu_data.angular_velocity,
                'linear_acceleration': self.imu_data.linear_acceleration
            }

        return exteroception

    def get_spatial_awareness(self):
        """سینسر ڈیٹا سے جگہ کا ادراک تیار کریں"""
        spatial_info = {}

        if self.laser_data:
            # سادہ جگہ کی نمائندگی تیار کریں
            ranges = np.array(self.laser_data.ranges)
            angles = np.linspace(self.laser_data.angle_min, self.laser_data.angle_max, len(ranges))

            # کارٹیزین کوآرڈینیٹس میں تبدیل کریں
            x_points = ranges * np.cos(angles)
            y_points = ranges * np.sin(angles)

            spatial_info['obstacle_points'] = np.column_stack([x_points, y_points])
            spatial_info['free_space_estimate'] = self.estimate_free_space(ranges)

        return spatial_info

    def get_balance_state(self):
        """موجودہ توازن کی حالت حاصل کریں"""
        if self.imu_data:
            # سادہ توازن کا حساب
            orientation = self.imu_data.orientation
            tilt_magnitude = abs(orientation.x) + abs(orientation.y)
            balance_score = max(0.0, 1.0 - tilt_magnitude * 5)
            return balance_score

        return 1.0

    def get_body_configuration(self):
        """موجودہ جسم کی ترتیب حاصل کریں"""
        if self.joint_states:
            # جوائنٹ پوزیشنز کی بنیاد پر جسم کی ترتیب کی نمائندگی کریں
            config = {}
            for i, name in enumerate(self.joint_states.name):
                config[name] = self.joint_states.position[i]
            return config
        return {}

    def analyze_obstacles(self):
        """لیزر اسکینر سے رکاوٹ ڈیٹا کا تجزیہ کریں"""
        if self.laser_data:
            ranges = np.array(self.laser_data.ranges)
            valid_ranges = ranges[(ranges > 0) & (ranges < self.laser_data.range_max)]

            if len(valid_ranges) > 0:
                return {
                    'closest_distance': min(valid_ranges),
                    'obstacle_density': len(valid_ranges) / len(ranges),
                    'directional_analysis': self.analyze_directional_obstacles(ranges)
                }

        return {'closest_distance': float('inf'), 'obstacle_density': 0.0, 'directional_analysis': {}}

    def analyze_directional_obstacles(self, ranges):
        """مختلف سمت میں رکاوٹوں کا تجزیہ کریں"""
        sector_size = len(ranges) // 8  # 8 سیکٹرز میں تقسیم کریں
        analysis = {}

        for i in range(8):
            start_idx = i * sector_size
            end_idx = min((i + 1) * sector_size, len(ranges))
            sector_ranges = ranges[start_idx:end_idx]

            valid_sector = sector_ranges[(sector_ranges > 0) & (sector_ranges < max(ranges))]
            if len(valid_sector) > 0:
                analysis[f'sector_{i}'] = {
                    'min_distance': min(valid_sector),
                    'avg_distance': np.mean(valid_sector),
                    'obstacle_count': len(valid_sector)
                }
            else:
                analysis[f'sector_{i}'] = {
                    'min_distance': float('inf'),
                    'avg_distance': float('inf'),
                    'obstacle_count': 0
                }

        return analysis

    def create_environment_map(self):
        """سادہ ماحول کی نمائندگی تیار کریں"""
        if self.laser_data:
            ranges = np.array(self.laser_data.ranges)
            angles = np.linspace(self.laser_data.angle_min, self.laser_data.angle_max, len(ranges))

            # جائز رینج فلٹر کریں
            valid_mask = (ranges > 0) & (ranges < self.laser_data.range_max)
            valid_angles = angles[valid_mask]
            valid_ranges = ranges[valid_mask]

            # روبوٹ کے مطابق کارٹیزین کوآرڈینیٹس میں تبدیل کریں
            x_coords = valid_ranges * np.cos(valid_angles)
            y_coords = valid_ranges * np.sin(valid_angles)

            return np.column_stack([x_coords, y_coords])

        return np.array([])

    def extract_visual_features(self):
        """کیمرہ امیج سے خصوصیات نکالیں"""
        if self.image_data is not None:
            # سادہ خصوصیات نکالنے کا عمل
            gray = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2GRAY)

            # کنارہ کا پتہ لگانا
            edges = cv2.Canny(gray, 50, 150)

            # کنٹور تلاش کریں
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # خصوصیات نکالیں
            features = {
                'edge_density': np.sum(edges) / edges.size,
                'contour_count': len(contours),
                'largest_contour_area': max([cv2.contourArea(c) for c in contours]) if contours else 0,
                'horizontal_symmetry': self.calculate_horizontal_symmetry(gray)
            }

            return features

        return {}

    def calculate_horizontal_symmetry(self, image):
        """امیج کی افقی توازن کا حساب لگائیں"""
        height, width = image.shape
        left_half = image[:, :width//2]
        right_half = image[:, width//2:width - (width % 2)]
        right_half_flipped = cv2.flip(right_half, 1)

        if left_half.shape == right_half_flipped.shape:
            diff = cv2.absdiff(left_half, right_half_flipped)
            symmetry_score = 1.0 - (np.mean(diff) / 255.0)  # Normalize to [0,1]
            return symmetry_score
        return 0.0

    def estimate_free_space(self, ranges):
        """ماحول میں کھلی جگہ کی مقدار کا تخمینہ لگائیں"""
        valid_ranges = ranges[(ranges > 0) & (ranges < max(ranges))]
        if len(valid_ranges) > 0:
            free_ratio = len(valid_ranges) / len(ranges)
            avg_distance = np.mean(valid_ranges)
            return free_ratio * avg_distance
        return 0.0

    def collect_experience(self, perceptual_state):
        """موجودہ ادراک کی حالت سے تجربہ جمع کریں"""
        experience = {
            'timestamp': self.get_clock().now().nanoseconds,
            'perceptual_state': perceptual_state,
            'motor_state': self.get_current_motor_state(),
            'action_taken': self.get_recent_action(),
            'environment_context': self.get_environment_context(),
            'outcome': self.get_recent_outcome()
        }

        # تجربہ بفر میں شامل کریں
        self.experience_buffer.append(experience)
        self.learning_state['total_experiences'] += 1

        # لاگنگ کے لیے تجربہ شائع کریں
        self.experience_pub.publish(
            String(data=f"Experience collected at {experience['timestamp']}")
        )

        return experience

    def get_current_motor_state(self):
        """موجودہ موٹر/ایکچوایٹر کی حالت حاصل کریں"""
        if self.joint_states:
            return {
                'joint_commands': list(self.joint_states.position),  # آخری کمانڈ کی گئی پوزیشنز
                'actual_positions': list(self.joint_states.position),
                'velocities': list(self.joint_states.velocity)
            }
        return {}

    def get_recent_action(self):
        """حال ہی میں لیا گیا عمل حاصل کریں"""
        # ایک حقیقی نفاذ میں، یہ آخری عمل کو ٹریک کرے گا
        return {'type': 'unknown', 'parameters': {}}

    def get_environment_context(self):
        """موجودہ ماحولیاتی سیاق حاصل کریں"""
        return {
            'obstacle_density': self.get_obstacle_density(),
            'space_openness': self.get_space_openness(),
            'lighting_conditions': self.estimate_lighting_conditions()
        }

    def get_obstacle_density(self):
        """موجودہ رکاوٹ کی کثافت حاصل کریں"""
        if self.laser_data:
            ranges = np.array(self.laser_data.ranges)
            valid_ranges = ranges[(ranges > 0) & (ranges < self.laser_data.range_max)]
            return len(valid_ranges) / len(ranges)
        return 0.0

    def get_space_openness(self):
        """جگہ کی کھلی ہونے کا تخمینہ حاصل کریں"""
        if self.laser_data:
            ranges = np.array(self.laser_data.ranges)
            valid_ranges = ranges[(ranges > 0) & (ranges < self.laser_data.range_max)]
            if len(valid_ranges) > 0:
                return np.mean(valid_ranges) / self.laser_data.range_max
        return 0.0

    def estimate_lighting_conditions(self):
        """کیمرہ سے روشنی کی حالت کا تخمینہ لگائیں"""
        if self.image_data is not None:
            gray = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            return mean_brightness / 255.0  # [0, 1] تک معمول کریں
        return 0.5  # ڈیفالٹ طور پر درمیانی روشنی

    def get_recent_outcome(self):
        """حالیہ بات چیت کا نتیجہ حاصل کریں"""
        # سادہ نتیجہ کا جائزہ
        return {
            'success': True,
            'efficiency': 0.8,
            'safety': 1.0,
            'learning_potential': 0.5
        }

    def update_learning_models(self, experience):
        """نئے تجربے کی بنیاد پر سیکھنے کے ماڈلز کو اپ ڈیٹ کریں"""
        # موٹر سیکھنے کا ماڈل اپ ڈیٹ کریں
        self.update_motor_learning(experience)

        # ادراکی سیکھنے کا ماڈل اپ ڈیٹ کریں
        self.update_perceptual_learning(experience)

        # تصوراتی سیکھنے کا ماڈل اپ ڈیٹ کریں
        self.update_conceptual_learning(experience)

        # مجموعی سیکھنے کی پیشرفت اپ ڈیٹ کریں
        self.update_learning_progress()

    def update_motor_learning(self, experience):
        """تجربے کی بنیاد پر موٹر سیکھنا اپ ڈیٹ کریں"""
        # مثال: توازن اور کارکردگی کی بنیاد پر چلنے کے نمونے کو بہتر بنائیں
        balance_score = experience['perceptual_state']['balance_state']
        if balance_score > 0.7:  # اچھا توازن برقرار رکھا
            self.learning_state['successful_interactions'] += 1

    def update_perceptual_learning(self, experience):
        """تجربے کی بنیاد پر ادراکی سیکھنا اپ ڈیٹ کریں"""
        # مثال: بصری خصوصیات کو جگہ کے لےآؤٹ کے ساتھ منسلک کرنا سیکھیں
        visual_features = experience['perceptual_state']['exteroception'].get('visual_features', {})
        spatial_awareness = experience['perceptual_state']['spatial_awareness']

        # حسی موٹر تعلقات کو اپ ڈیٹ کریں
        self.perceptual_learner.update_sensorimotor_model(
            'visual_processing',
            {'visual': visual_features, 'spatial': spatial_awareness}
        )

    def update_conceptual_learning(self, experience):
        """تجربے کی بنیاد پر تصوراتی سیکھنا اپ ڈیٹ کریں"""
        # مثال: "کھلی جگہ" یا "تنگ راستہ" جیسے تصورات سیکھیں
        env_context = experience['environment_context']

        if env_context['space_openness'] > 0.7:
            self.concept_learner.learn_concept_from_experience(
                'open_space',
                {'openness': env_context['space_openness'], 'obstacle_density': env_context['obstacle_density']}
            )
        elif env_context['obstacle_density'] > 0.5:
            self.concept_learner.learn_concept_from_experience(
                'obstacle_rich',
                {'openness': env_context['space_openness'], 'obstacle_density': env_context['obstacle_density']}
            )

    def update_learning_progress(self):
        """مجموعی سیکھنے کی پیشرفت میٹرک اپ ڈیٹ کریں"""
        if self.learning_state['total_experiences'] > 0:
            self.learning_state['learning_progress'] = (
                self.learning_state['successful_interactions'] /
                self.learning_state['total_experiences']
            )

    def select_behavior_based_on_learning(self):
        """جمع کردہ سیکھنے کی بنیاد پر رویہ منتخب کریں"""
        if len(self.experience_buffer) < 10:
            # ابتدائی تلاش کا مرحلہ
            return self.exploration_behavior()
        else:
            # سیکھے گئے ماڈلز کا استعمال کرتے ہوئے رویہ منتخب کریں
            return self.learned_behavior()

    def exploration_behavior(self):
        """تلاش کے مرحلے کے لیے رویہ"""
        # متنوع تجربات جمع کرنے کے لیے زیادہ تلاش کی شرح
        if np.random.random() < self.exploration_rate:
            # بے ترتیب تلاشیاتی عمل
            return {
                'type': 'exploration',
                'action': self.get_random_exploratory_action(),
                'intention': 'gather_diverse_experiences'
            }
        else:
            # سادہ بنیادی رویہ
            return {
                'type': 'baseline',
                'action': self.get_baseline_action(),
                'intention': 'maintain_stability'
            }

    def get_random_exploratory_action(self):
        """بے ترتیب تلاشیاتی عمل حاصل کریں"""
        actions = [
            {'linear': 0.2, 'angular': 0.0},  # آگے بڑھیں
            {'linear': 0.0, 'angular': 0.3},  # بائیں مڑیں
            {'linear': 0.0, 'angular': -0.3}, # دائیں مڑیں
            {'linear': -0.1, 'angular': 0.0}, # پیچھے جائیں
            {'linear': 0.1, 'angular': 0.5}   # بائیں مڑیں
        ]
        return np.random.choice(actions)

    def get_baseline_action(self):
        """بنیادی مستحکم عمل حاصل کریں"""
        return {'linear': 0.1, 'angular': 0.0}  # معمولی فارورڈ موشن

    def learned_behavior(self):
        """جمع کردہ سیکھنے کی بنیاد پر رویہ"""
        # سیکھے گئے ماڈلز کا استعمال کرتے ہوئے بہترین عمل منتخب کریں
        current_state = self.get_current_state_representation()

        # اس مثال کے لیے، سیکھنے کی بنیاد پر سادہ ہیورسٹکس استعمال کریں
        if self.learning_state['learning_progress'] > 0.8:
            # اچھی طرح سے سیکھا ہوا سسٹم زیادہ ترقی یافتہ اعمال کر سکتا ہے
            return {
                'type': 'sophisticated',
                'action': self.get_sophisticated_action(),
                'intention': 'apply_learned_skills'
            }
        else:
            # سیکھتے ہوئے دوران محتاط رویہ
            return {
                'type': 'conservative',
                'action': self.get_conservative_action(),
                'intention': 'continue_learning_safely'
            }

    def get_current_state_representation(self):
        """سیکھنے کے ماڈلز کے لیے موجودہ حالت کی نمائندگی حاصل کریں"""
        return {
            'balance': self.get_balance_state(),
            'obstacles': self.analyze_obstacles(),
            'space': self.get_space_openness()
        }

    def get_sophisticated_action(self):
        """سیکھنے کی بنیاد پر ترقی یافتہ عمل حاصل کریں"""
        # ترقی یافتہ رویہ کے لیے سیکھے گئے ماڈلز کا استعمال کریں
        obstacles = self.analyze_obstacles()

        if obstacles['closest_distance'] < 0.5:
            # ترقی یافتہ رکاوٹ سے بچاؤ
            return self.get_clever_avoidance_action(obstacles)
        else:
            # مؤثر راستہ فالو کرنا
            return {'linear': 0.3, 'angular': 0.0}

    def get_clever_avoidance_action(self, obstacles):
        """چالاک رکاوٹ سے بچاؤ کا عمل حاصل کریں"""
        # سمتی رکاوٹ ڈیٹا کا تجزیہ کریں
        directional = obstacles['directional_analysis']

        # سب سے صاف سمت تلاش کریں
        best_direction = min(
            directional.items(),
            key=lambda x: x[1]['min_distance']
        )

        # صاف سمت کی طرف مڑیں
        sector_idx = int(best_direction[0].split('_')[1])
        turn_amount = (sector_idx - 4) * 0.1  # مرکز سیکٹر 4 ہے

        return {'linear': 0.1, 'angular': turn_amount}

    def get_conservative_action(self):
        """سیکھتے ہوئے دوران محتاط عمل حاصل کریں"""
        # محفوظ، محتاط رویہ
        obstacles = self.analyze_obstacles()

        if obstacles['closest_distance'] < 0.8:
            return {'linear': 0.0, 'angular': 0.2}  # دور مڑیں
        else:
            return {'linear': 0.1, 'angular': 0.0}  # آہستہ آگے

    def execute_behavior(self, behavior):
        """منتخب کردہ رویہ انجام دیں"""
        action = behavior['action']

        # عمل کو ROS پیغام میں تبدیل کریں
        cmd = Twist()
        cmd.linear.x = action.get('linear', 0.0)
        cmd.angular.z = action.get('angular', 0.0)

        # کمانڈ شائع کریں
        self.cmd_vel_pub.publish(cmd)

        return cmd

    def evaluate_interaction(self, action, perceptual_state):
        """بات چیت کے نتیجے کا جائزہ لیں"""
        # یہ جانچیں کہ عمل نے اس کا مقصد کتنا پورا کیا
        balance_after_action = perceptual_state['balance_state']

        # کامیابی کے میٹرکس اپ ڈیٹ کریں
        if balance_after_action > 0.7:  # اچھا توازن برقرار رکھا
            self.learning_state['successful_interactions'] += 1

    def report_learning_status(self):
        """موجودہ سیکھنے کی حالت کی رپورٹ کریں"""
        status_msg = String()
        status_msg.data = (
            f"Experiences: {self.learning_state['total_experiences']}, "
            f"Success Rate: {self.learning_state['learning_progress']:.2f}, "
            f"Concepts Learned: {len(self.concept_learner.concept_representations)}"
        )

        self.learning_status_pub.publish(status_msg)

    def save_learning_state(self, filepath):
        """فائل میں موجودہ سیکھنے کی حالت محفوظ کریں"""
        learning_data = {
            'experience_buffer': list(self.experience_buffer),
            'learning_state': self.learning_state,
            'concept_representations': self.concept_learner.concept_representations,
            'physical_experiences': self.concept_learner.physical_experiences
        }

        with open(filepath, 'wb') as f:
            pickle.dump(learning_data, f)

    def load_learning_state(self, filepath):
        """فائل سے سیکھنے کی حالت لوڈ کریں"""
        with open(filepath, 'rb') as f:
            learning_data = pickle.load(f)

        self.experience_buffer = deque(learning_data['experience_buffer'], maxlen=1000)
        self.learning_state = learning_data['learning_state']
        self.concept_learner.concept_representations = learning_data['concept_representations']
        self.concept_learner.physical_experiences = learning_data['physical_experiences']

def main(args=None):
    rclpy.init(args=args)
    learning_node = EmbodiedLearningNode()

    try:
        rclpy.spin(learning_node)
    except KeyboardInterrupt:
        # بند کرنے سے پہلے سیکھنے کی حالت محفوظ کریں
        learning_node.save_learning_state('/tmp/embodied_learning_state.pkl')
        pass
    finally:
        learning_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## مورفولوجیکل کمپیوٹیشن اور سیکھنا

جسمانیت مورفولوجیکل کمپیوٹیشن کو فعال کرتی ہے، جہاں جسمانی شکل کمپیوٹیشن میں حصہ ڈالتی ہے:

```python
# مثال: سیکھنے میں مورفولوجیکل کمپیوٹیشن
class MorphologicalComputationLearner:
    def __init__(self, body_mechanics):
        self.body_mechanics = body_mechanics
        self.morphological_advantages = {}
        self.passive_dynamics = {}

    def discover_morphological_computation(self, task_environment):
        """یہ دریافت کریں کہ جسمانیت کو کمپیوٹیشن کے لیے کیسے استعمال کیا جا سکتا ہے"""
        # یہ تلاش کریں کہ جسم کی خصوصیات محسوساتی مسائل کو کیسے حل کر سکتی ہیں
        morphological_solutions = []

        for body_part, properties in self.body_mechanics.items():
            if self.can_use_for_computation(body_part, properties, task_environment):
                solution = self.derive_morphological_solution(
                    body_part, properties, task_environment
                )
                morphological_solutions.append(solution)

        return morphological_solutions

    def can_use_for_computation(self, body_part, properties, environment):
        """چیک کریں کہ کیا جسم کا حصہ کمپیوٹیشنل مقاصد کے لیے استعمال ہو سکتا ہے"""
        # مثال: چیز کی ہیرا پھیری کے لیے مطیع ہاتھ
        # مثال: توانائی کی کارکردگی کے لیے پینڈولم جیسی ٹانگیں
        # مثال: ماحولیاتی پتہ لگانے کے لیے م passive sensors
        return properties.get('compliance', 0) > 0.5 or properties.get('sensitivity', 0) > 0.5

    def derive_morphological_solution(self, body_part, properties, environment):
        """ایک حل ڈھونڈیں جو جسمانیت کو کمپیوٹیشن کے لیے استعمال کرے"""
        if properties.get('compliance', 0) > 0.5:
            # مطیع کو کنٹرول کے لیے استعمال کریں
            return {
                'type': 'compliant_manipulation',
                'body_part': body_part,
                'principle': 'use_mechanical_compliance_for_control',
                'application': 'grasping_unknown_objects'
            }
        elif properties.get('sensitivity', 0) > 0.5:
            # حساسیت کو ادراک کے لیے استعمال کریں
            return {
                'type': 'sensitive_perception',
                'body_part': body_part,
                'principle': 'use_mechanical_sensitivity_for_sensing',
                'application': 'texture_recognition'
            }
        else:
            return None

    def exploit_passive_dynamics(self, movement_task):
        """کارکردگی کے لیے جسم کی م passive ڈائنامکس کا استعمال کریں"""
        # کمپیوٹیشنل بوجھ کو کم کرنے کے لیے جسم کی قدرتی ڈائنامکس استعمال کریں
        passive_solution = self.body_mechanics.calculate_passive_dynamics(
            movement_task.goal
        )

        # صرف ان حصوں کے لیے فعال کنٹرول کا حساب کریں جن کی ضرورت ہو
        active_control = self.compute_active_control(
            movement_task, passive_solution
        )

        return {
            'passive_component': passive_solution,
            'active_component': active_control,
            'efficiency_gain': self.calculate_efficiency_gain(
                passive_solution, active_control
            )
        }

    def calculate_efficiency_gain(self, passive, active):
        """passive ڈائنامکس کا استعمال کرتے ہوئے کارکردگی کا فائدہ کا حساب لگائیں"""
        # مکمل فعال بمقابلہ passive-active نقطہ نظر کی توانائی کی کھپت کا موازنہ کریں
        return 0.3  # جگہ کا نمونہ قیمت
```

## لیب: جسمانی سیکھنے کا نفاذ

اس لیب میں، آپ ایک جسمانی سیکھنے کا سسٹم نافذ کریں گے:

```python
# lab_embodied_learning.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32
import numpy as np

class EmbodiedLearningLab(Node):
    def __init__(self):
        super().__init__('embodied_learning_lab')

        # پبلشرز
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/lab_status', 10)
        self.learning_pub = self.create_publisher(Float32, '/learning_progress', 10)

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

        # سیکھنے کے اجزاء
        self.experience_log = []
        self.learning_enabled = True
        self.exploration_phase = True

        # سیکھنے کے پیرامیٹر
        self.exploration_rate = 0.3
        self.success_threshold = 0.7

        # لیب کی حالت
        self.learning_state = {
            'total_interactions': 0,
            'successful_interactions': 0,
            'learning_progress': 0.0,
            'current_strategy': 'exploration'
        }

        # کنٹرول لوپ
        self.control_timer = self.create_timer(0.1, self.learning_control_loop)

    def joint_callback(self, msg):
        """جوائنٹ حالت ڈیٹا کو سنبھالیں"""
        self.joint_data = msg

    def imu_callback(self, msg):
        """IMU ڈیٹا کو سنبھالیں"""
        self.imu_data = msg

    def scan_callback(self, msg):
        """لیزر اسکین ڈیٹا کو سنبھالیں"""
        self.scan_data = msg

    def learning_control_loop(self):
        """اصل سیکھنے کا کنٹرول لوپ"""
        if not all([self.joint_data, self.imu_data, self.scan_data]):
            return

        # 1. ادراک: موجودہ صورتحال کا جائزہ لیں
        situation_assessment = self.assess_situation()

        # 2. سیکھنا: اندرونی ماڈلز کو اپ ڈیٹ کریں
        if self.learning_enabled:
            self.update_internal_models(situation_assessment)

        # 3. رویہ: سیکھنے کی بنیاد پر رویہ منتخب کریں
        behavior = self.select_behavior(situation_assessment)

        # 4. عمل: رویہ انجام دیں
        action = self.execute_behavior(behavior)

        # 5. جائزہ: نتیجہ کا جائزہ لیں
        outcome = self.assess_outcome(action, situation_assessment)

        # 6. موافقت: نتیجے کی بنیاد پر رویہ کو موافق بنائیں
        self.adapt_behavior(outcome)

        # 7. رپورٹنگ: سیکھنے کی حالت شائع کریں
        self.publish_learning_status()

    def assess_situation(self):
        """تمام سینسرز کا استعمال کرتے ہوئے موجودہ صورتحال کا جائزہ لیں"""
        situation = {
            'safety_level': self.assess_safety(),
            'navigation_state': self.assess_navigation_state(),
            'balance_state': self.assess_balance(),
            'environment_complexity': self.assess_environment_complexity()
        }
        return situation

    def assess_safety(self):
        """موجودہ سلامتی کی سطح کا جائزہ لیں"""
        if self.scan_data:
            min_distance = min([r for r in self.scan_data.ranges if r > 0], default=float('inf'))
            return min_distance  # زیادہ محفوظ ہے
        return float('inf')

    def assess_navigation_state(self):
        """نیویگیشن کی صورتحال کا جائزہ لیں"""
        if self.scan_data:
            front_clear = all(r > 1.0 for r in self.scan_data.ranges[300:600] if r > 0)
            left_clear = all(r > 0.8 for r in self.scan_data.ranges[0:180] if r > 0)
            right_clear = all(r > 0.8 for r in self.scan_data.ranges[540:720] if r > 0)

            return {
                'front_clear': front_clear,
                'left_clear': left_clear,
                'right_clear': right_clear,
                'path_options': sum([front_clear, left_clear, right_clear])
            }
        return {'front_clear': True, 'left_clear': True, 'right_clear': True, 'path_options': 3}

    def assess_balance(self):
        """IMU کا استعمال کرتے ہوئے موجودہ توازن کا جائزہ لیں"""
        if self.imu_data:
            # سادہ توازن کا جائزہ
            orientation = self.imu_data.orientation
            tilt = abs(orientation.x) + abs(orientation.y)
            balance_score = max(0.0, 1.0 - tilt * 3)  # 1.0 = مکمل طور پر متوازن
            return balance_score
        return 1.0

    def assess_environment_complexity(self):
        """موجودہ ماحول کی پیچیدگی کا جائزہ لیں"""
        if self.scan_data:
            ranges = np.array(self.scan_data.ranges)
            valid_ranges = ranges[(ranges > 0) & (ranges < self.scan_data.range_max)]

            if len(valid_ranges) > 0:
                # فاصلوں میں تغیر کی بنیاد پر پیچیدگی
                distance_variance = np.var(valid_ranges)
                obstacle_density = len(valid_ranges) / len(ranges)

                return distance_variance * obstacle_density
        return 0.0

    def update_internal_models(self, situation):
        """ situation کی بنیاد پر اندرونی سیکھنے کے ماڈلز کو اپ ڈیٹ کریں"""
        # موجودہ situation کو تجربہ لاگ میں شامل کریں
        experience = {
            'situation': situation,
            'timestamp': self.get_clock().now().nanoseconds,
            'action_taken': self.get_recent_action(),
            'outcome_assessed': self.assess_recent_outcome()
        }

        self.experience_log.append(experience)

        # سیکھنے کی پیشرفت کو اپ ڈیٹ کریں
        self.learning_state['total_interactions'] += 1
        if self.assess_recent_outcome()['success']:
            self.learning_state['successful_interactions'] += 1

    def get_recent_action(self):
        """حال ہی میں لیا گیا عمل حاصل کریں"""
        # عمل میں، یہ آخری عمل کو ٹریک کرے گا
        return {'type': 'unknown', 'parameters': {}}

    def assess_recent_outcome(self):
        """حالیہ عمل کا نتیجہ کا جائزہ لیں"""
        # سادہ نتیجہ کا جائزہ
        safety = self.assess_safety()
        balance = self.assess_balance()

        success = safety > 0.5 and balance > 0.7
        efficiency = 0.8  # جگہ کا نمونہ

        return {'success': success, 'efficiency': efficiency}

    def select_behavior(self, situation):
        """جمع کردہ سیکھنے کی بنیاد پر رویہ منتخب کریں"""
        # سیکھنے کی پیشرفت کے مطابق تلاش اور استعمال کے درمیان سوئچ کریں
        learning_progress = self.calculate_learning_progress()

        if learning_progress < 0.3 or self.exploration_phase:
            # ابتدائی سیکھنے کا مرحلہ - زیادہ تلاش کریں
            return self.exploration_behavior(situation)
        else:
            # بعد کا مرحلہ - سیکھے گئے علم کا استعمال کریں
            return self.exploitation_behavior(situation)

    def exploration_behavior(self, situation):
        """تلاش کے مرحلے کے لیے رویہ"""
        # نئے اعمال کی کوشش کا زیادہ موقع
        if np.random.random() < self.exploration_rate:
            # ایک بے ترتیب تلاشیاتی عمل کی کوشش کریں
            return {
                'type': 'exploratory',
                'action': self.get_exploratory_action(situation),
                'exploration_value': 1.0
            }
        else:
            # سیکھے گئے علم کا استعمال کریں
            return {
                'type': 'learned',
                'action': self.get_learned_action(situation),
                'exploration_value': 0.3
            }

    def exploitation_behavior(self, situation):
        """استعمال کے مرحلے کے لیے رویہ"""
        # زیادہ تر وقت سیکھے گئے رویہ کا استعمال کریں
        if np.random.random() < 0.9:  # 90% وقت سیکھا ہوا رویہ استعمال کریں
            return {
                'type': 'learned',
                'action': self.get_learned_action(situation),
                'exploration_value': 0.1
            }
        else:
            # کبھی کبھار تلاش کرنا جاری رکھیں
            return {
                'type': 'exploratory',
                'action': self.get_exploratory_action(situation),
                'exploration_value': 0.8
            }

    def get_exploratory_action(self, situation):
        """ایک تلاشیاتی عمل حاصل کریں"""
        # ماحول کے بارے میں سیکھنے کے لیے مختلف اعمال کی کوشش کریں
        actions = [
            {'linear': 0.2, 'angular': 0.0},  # فارورڈ
            {'linear': 0.0, 'angular': 0.4},  # بائیں مڑیں
            {'linear': 0.0, 'angular': -0.4}, # دائیں مڑیں
            {'linear': 0.1, 'angular': 0.2},  # وکر
        ]
        return np.random.choice(actions)

    def get_learned_action(self, situation):
        """سیکھے گئے علم کی بنیاد پر عمل حاصل کریں"""
        # situation کے جائزے کا استعمال کرتے ہوئے مناسب عمل منتخب کریں
        if situation['safety_level'] < 0.5:  # رکاوٹوں سے بہت قریب
            # رکاوٹوں سے بچیں
            if situation['navigation_state']['left_clear']:
                return {'linear': 0.0, 'angular': 0.3}
            elif situation['navigation_state']['right_clear']:
                return {'linear': 0.0, 'angular': -0.3}
            else:
                return {'linear': -0.1, 'angular': 0.0}  # بیک اپ
        elif situation['navigation_state']['front_clear']:
            # راستہ صاف ہونے پر آگے بڑھیں
            return {'linear': 0.3, 'angular': 0.0}
        else:
            # صاف راستہ تلاش کرنے کے لیے ڈیفالٹ طور پر مڑیں
            return {'linear': 0.0, 'angular': 0.2}

    def execute_behavior(self, behavior):
        """منتخب کردہ رویہ انجام دیں"""
        action = behavior['action']

        # ٹویسٹ پیغام بنائیں
        cmd = Twist()
        cmd.linear.x = action.get('linear', 0.0)
        cmd.angular.z = action.get('angular', 0.0)

        # کمانڈ شائع کریں
        self.cmd_pub.publish(cmd)

        return cmd

    def assess_outcome(self, action, situation):
        """عمل کا نتیجہ کا جائزہ لیں"""
        new_situation = self.assess_situation()

        # situation میں بہتری کا جائزہ لیں
        safety_improved = new_situation['safety_level'] > situation['safety_level']
        balance_maintained = new_situation['balance_state'] > 0.7

        outcome = {
            'action': action,
            'previous_situation': situation,
            'new_situation': new_situation,
            'safety_improved': safety_improved,
            'balance_maintained': balance_maintained,
            'success': safety_improved and balance_maintained,
            'learning_value': self.calculate_learning_value(action, situation, new_situation)
        }

        return outcome

    def calculate_learning_value(self, action, prev_situation, new_situation):
        """سیکھنے کے لیے اس تجربے کی کتنی قیمت ہے کا حساب لگائیں"""
        # نوآوری اور نتیجے کی بنیاد پر قیمت
        novelty = self.calculate_novelty(action, prev_situation)
        outcome_success = 1.0 if (new_situation['safety_level'] > prev_situation['safety_level']
                                and new_situation['balance_state'] > 0.7) else 0.5

        return novelty * outcome_success

    def calculate_novelty(self, action, situation):
        """یہ تعین کریں کہ یہ تجربہ کتنا نیا ہے"""
        # پچھلے تجربات کے ساتھ موازنہ کریں
        if len(self.experience_log) < 10:
            return 1.0  # شروع کرنے پر زیادہ نوآوری

        # سادہ نوآوری کا حساب
        return 0.5  # جگہ کا نمونہ

    def adapt_behavior(self, outcome):
        """نتیجے کی بنیاد پر رویہ کو موافق بنائیں"""
        # کامیابی کی بنیاد پر تلاش کی شرح کو اپ ڈیٹ کریں
        if outcome['success']:
            # چیزیں اچھی چل رہی ہوں تو تلاش کم کریں
            self.exploration_rate = max(0.1, self.exploration_rate * 0.99)
        else:
            # مشکل میں تلاش بڑھائیں
            self.exploration_rate = min(0.8, self.exploration_rate * 1.01)

        # کامیابی کی شرح کے مطابق حکمت عمل کو اپ ڈیٹ کریں
        success_rate = self.calculate_success_rate()
        if success_rate > self.success_threshold:
            self.learning_state['current_strategy'] = 'exploitation'
            self.exploration_phase = False
        else:
            self.learning_state['current_strategy'] = 'exploration'
            self.exploration_phase = True

    def calculate_success_rate(self):
        """مجموعی کامیابی کی شرح کا حساب لگائیں"""
        if self.learning_state['total_interactions'] > 0:
            return (self.learning_state['successful_interactions'] /
                   self.learning_state['total_interactions'])
        return 0.0

    def calculate_learning_progress(self):
        """مجموعی سیکھنے کی پیشرفت کا حساب لگائیں"""
        return self.calculate_success_rate()

    def publish_learning_status(self):
        """موجودہ سیکھنے کی حالت شائع کریں"""
        progress = self.calculate_learning_progress()

        # پیشرفت شائع کریں
        progress_msg = Float32()
        progress_msg.data = progress
        self.learning_pub.publish(progress_msg)

        # حالت شائع کریں
        status_msg = String()
        status_msg.data = (
            f"Progress: {progress:.2f}, Strategy: {self.learning_state['current_strategy']}, "
            f"Exploration: {self.exploration_rate:.2f}"
        )
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    lab = EmbodiedLearningLab()

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

## مشق: اپنا جسمانی سیکھنے کا سسٹم ڈیزائن کریں

مندرجہ ذیل ڈیزائن چیلنج پر غور کریں:

1. آپ کون سی مخصوص مہارت یا صلاحیت اپنے جسمانی ایجنٹ کو سیکھنے دینا چاہتے ہیں؟
2. ایجنٹ کی جسمانی شکل سیکھنے کے عمل کو کیسے متاثر کرے گی؟
3. اس مہارت کو سیکھنے کے لیے کون سے حسی موٹر تجربات سب سے قیمتی ہوں گے؟
4. سیکھنے کے دوران تلاش اور استعمال کے درمیان توازن کیسے قائم کریں گے؟
5. سیکھنے کی پیشرفت کا جائزہ لینے کے لیے آپ کون سے میٹرکس استعمال کریں گے؟
6. نظام تجربے کی بنیاد پر اپنی سیکھنے کی حکمت عمل کو کیسے موافق بنائے گا؟

## خلاصہ

سیکھنے اور ذہانت میں جسمانیت ایک بنیادی کردار ادا کرتی ہے:

- **جسمانی پابندیاں فراہم کرنا**: جسم کی شکل اور صلاحیات یہ تعین کرتی ہیں کہ کیا سیکھا جا سکتا ہے
- **حسی موٹر سیکھنے کو فعال کرنا**: مہارتوں کو ماحول کے ساتھ جسمانی بات چیت کے ذریعے ترقی دی جاتی ہے
- **جذب شدہ تصورات کو فعال کرنا**: مطلق تصورات جسمانی تجربے میں جڑے ہوتے ہیں
- **مورفولوجیکل کمپیوٹیشن کو فعال کرنا**: جسمانی خصوصیات کمپیوٹیشنل کاموں میں حصہ ڈال سکتی ہیں
- **سیکھنے کے مواقع پیدا کرنا**: جسمانی بات چیت مکمل، متعدد ماڈلز کے تجربات پیدا کرتی ہے

جسمانیت کے ذریعے ادراک، عمل، اور سیکھنے کا انضمام روبوٹس کو مضبوط، موافق ذہانت ترقی دینے کے قابل بناتا ہے جو حقیقی دنیا کے ماحول کی پیچیدگیوں کا سامنا کر سکے۔ ان اصولوں کو سمجھنا مؤثر جسمانی مصنوعی ذہانت کے نظام ترقی دینے کے لیے اہم ہے۔

اگلے ماڈیول میں، ہم روبوٹکس کے لیے مخصوص مصنوعی ذہانت کی تکنیکوں کا جائزہ لیں گے، بشمول کمپیوٹر وژن، مشین لرننگ، اور راستہ منصوبہ بندی الگورتھم جو جسمانی ذہانت کا فائدہ اٹھاتے ہیںں۔