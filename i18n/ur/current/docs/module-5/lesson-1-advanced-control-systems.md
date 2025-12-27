---
sidebar_position: 1
---

# روبوٹکس کے لیے اعلی درجے کے کنٹرول سسٹم

## تعارف

اعلی درجے کے کنٹرول سسٹم پیچیدہ جسمانی مصنوعی ذہانت کے نظاموں کو تخلیق کرنے کے لیے ضروری ہیں جو متحرک ماحول میں مؤثر طریقے سے کام کر سکیں۔ یہ نظام بنیادی پوزیشن یا رفتار کنٹرول سے کہیں آگے جاتے ہیں تاکہ پیچیدہ رویے، موافق جوابات، اور عدم یقینی کے مقابلے میں مضبوط کارکردگی کو فعال کیا جا سکے۔ یہ سبق روبوٹکس اطلاقات کے لیے خصوصی طور پر ڈیزائن کردہ اعلی درجے کی کنٹرول تکنیکوں کا جائزہ لیتا ہے۔

## روبوٹکس میں کلاسیکل کنٹرول تھیوری

### PID کنٹرول اور اس کی اقسام

PID (پروپورشل-انٹیگرل-ڈیریویٹو) کنٹرول روبوٹکس میں اب بھی بنیادی ہے:

```python
# مثال: روبوٹکس کے لیے اعلی درجے کا PID کنٹرولر
class AdvancedPIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, dt=0.01):
        self.kp = kp  # تناسب کا گیان
        self.ki = ki  # انٹیگرل گیان
        self.kd = kd  # ڈیریویٹو گیان
        self.dt = dt  # وقت کا قدم

        # اندرونی حالت
        self.error_sum = 0.0
        self.error_prev = 0.0
        self.derivative_filtered = 0.0
        self.integral_limit = 10.0  # اینٹی-وینڈ اپ کی حد

        # ڈیریویٹو فلٹر کے پیرامیٹر
        self.alpha = 0.1  # ڈیریویٹو ٹرم کے لیے فلٹر کوائف
        self.setpoint = 0.0
        self.measurement = 0.0

    def update(self, setpoint, measurement):
        """نئے سیٹ پوائنٹ اور پیمائش کے ساتھ PID کنٹرولر کو اپ ڈیٹ کریں"""
        self.setpoint = setpoint
        self.measurement = measurement

        # خامی کا حساب لگائیں
        error = setpoint - measurement

        # تناسب ٹرم
        p_term = self.kp * error

        # انٹیگرل ٹرم اینٹی-وینڈ اپ کے ساتھ
        self.error_sum += error * self.dt
        # اینٹی-وینڈ اپ: انٹیگرل ٹرم کو محدود کریں
        self.error_sum = max(-self.integral_limit, min(self.integral_limit, self.error_sum))
        i_term = self.ki * self.error_sum

        # ڈیریویٹو ٹرم فلٹر کے ساتھ (شور کم کرنا)
        raw_derivative = (error - self.error_prev) / self.dt
        # ڈیریویٹو کے لیے پہلے درجے کا کم فلٹر
        self.derivative_filtered = (
            self.alpha * raw_derivative +
            (1 - self.alpha) * self.derivative_filtered
        )
        d_term = self.kd * self.derivative_filtered

        # آؤٹ پٹ کا حساب لگائیں
        output = p_term + i_term + d_term

        # اگلی تکرار کے لیے خامی کو محفوظ کریں
        self.error_prev = error

        return output

    def tune_pid(self, method='ziegler_nichols'):
        """ مختلف طریقوں کا استعمال کرتے ہوئے PID پیرامیٹر ٹیون کریں"""
        if method == 'ziegler_nichols':
            # زیگلر-نچلس ٹیوننگ طریقہ (سادہ)
            ku = 2.0  # اعلی گیان (تجربہ کار طور پر تعین کیا جائے گا)
            tu = 1.0  # جھمکن کا دور (تجربہ کار طور پر تعین کیا جائے گا)

            self.kp = 0.6 * ku
            self.ki = 1.2 * ku / tu
            self.kd = 0.075 * ku * tu

        elif method == 'cruise_control':
            # کروز کنٹرول مخصوص ٹیوننگ
            self.kp = 0.8
            self.ki = 0.1
            self.kd = 0.05

    def reset(self):
        """اندرونی حالت ری سیٹ کریں"""
        self.error_sum = 0.0
        self.error_prev = 0.0
        self.derivative_filtered = 0.0
```

### کیسکیڈ کنٹرول سسٹم

پیچیدہ روبوٹک نظاموں کے لیے، کیسکیڈ کنٹرول بہتر کارکردگی فراہم کرتا ہے:

```python
# مثال: روبوٹ جوائنٹ کنٹرول کے لیے کیسکیڈ کنٹرول
class CascadeController:
    def __init__(self):
        # بیرونی لوپ: پوزیشن کنٹرول
        self.position_controller = AdvancedPIDController(kp=2.0, ki=0.1, kd=0.05)

        # اندرونی لوپ: رفتار کنٹرول
        self.velocity_controller = AdvancedPIDController(kp=1.5, ki=0.2, kd=0.02)

        # اندرونی ترین لوپ: کرنٹ/ٹورک کنٹرول
        self.current_controller = AdvancedPIDController(kp=1.0, ki=0.3, kd=0.01)

        self.dt = 0.01  # کنٹرول لوپ وقت کا قدم

    def compute_control(self, desired_position, current_position,
                       current_velocity, current_current):
        """کیسکیڈ ڈھانچے کے ذریعے کنٹرول کمپیوٹ کریں"""

        # پوزیشن لوپ: رفتار کمانڈ جنریٹ کریں
        velocity_command = self.position_controller.update(
            desired_position, current_position
        )

        # رفتار لوپ: کرنٹ/ٹورک کمانڈ جنریٹ کریں
        current_command = self.velocity_controller.update(
            velocity_command, current_velocity
        )

        # کرنٹ لوپ: حتمی کنٹرول آؤٹ پٹ جنریٹ کریں
        control_output = self.current_controller.update(
            current_command, current_current
        )

        return {
            'position_error': desired_position - current_position,
            'velocity_command': velocity_command,
            'current_command': current_command,
            'final_output': control_output
        }

    def update_gains_adaptively(self, performance_metrics):
        """کارکردگی کی بنیاد پر گیانز کو موافق بنائیں"""
        # مثال: ٹریکنگ خامی کی بنیاد پر گیانز ایڈجسٹ کریں
        position_error = performance_metrics.get('position_error', 0.0)
        velocity_error = performance_metrics.get('velocity_error', 0.0)

        # اگر خامی بڑی ہے تو گیانز بڑھائیں
        if abs(position_error) > 0.1:  # حد
            self.position_controller.kp *= 1.05
            self.position_controller.ki *= 1.05
        else:
            # اگر خامی چھوٹی ہے تو گیانز کم کریں (جھمکن کم کریں)
            self.position_controller.kp *= 0.99
            self.position_controller.ki *= 0.99

        # اسی طرح رفتار کنٹرولر کے لیے
        if abs(velocity_error) > 0.05:
            self.velocity_controller.kp *= 1.05
            self.velocity_controller.ki *= 1.05
```

## جدید کنٹرول تکنیکیں

### ماڈل پریڈکٹو کنٹرول (MPC)

MPC کنٹرول کے لیے خاص طور پر طاقتور ہے جو پابندیوں اور متعدد اہداف کے ساتھ ہو:

```python
# مثال: روبوٹکس کے لیے ماڈل پریڈکٹو کنٹرول
import numpy as np
from scipy.optimize import minimize
import cvxpy as cp

class ModelPredictiveController:
    def __init__(self, state_dim, control_dim, prediction_horizon=10):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.N = prediction_horizon  # پیش گوئی کا افق

        # سسٹم میٹرکسز (خصوصی روبوٹ کے لیے شناخت کی جائے گی)
        self.A = np.eye(state_dim)  # حالت کی منتقلی میٹرکس
        self.B = np.zeros((state_dim, control_dim))  # کنٹرول ان پٹ میٹرکس
        self.C = np.eye(state_dim)  # آؤٹ پٹ میٹرکس

        # قیمت کی میٹرکسز
        self.Q = np.eye(state_dim)  # حالت کی قیمت میٹرکس
        self.R = np.eye(control_dim)  # کنٹرول قیمت میٹرکس
        self.Qf = np.eye(state_dim)  # ٹرمنل قیمت میٹرکس

        # پابندیاں
        self.u_min = -1.0  # کم از کم کنٹرول ان پٹ
        self.u_max = 1.0   # زیادہ سے زیادہ کنٹرول ان پٹ
        self.x_min = -np.inf  # حالت کی پابندیاں
        self.x_max = np.inf

    def setup_optimization_problem(self, x0, x_ref):
        """MPC آپٹیمائزیشن مسئلہ سیٹ اپ کریں"""
        # فیصلہ کن متغیرات: پیش گوئی کے افق پر کنٹرول ان پٹس
        U = cp.Variable((self.N, self.control_dim))

        # پیش گوئی کے افق پر حالت کے متغیرات
        X = cp.Variable((self.N + 1, self.state_dim))

        # ہدف کا فنکشن: حالت اور کنٹرول قیمتوں کو کم کریں
        cost = 0

        # چلتی ہوئی قیمتیں
        for k in range(self.N):
            cost += cp.quad_form(X[k] - x_ref, self.Q)
            cost += cp.quad_form(U[k], self.R)

        # ٹرمنل قیمت
        cost += cp.quad_form(X[self.N] - x_ref, self.Qf)

        # پابندیاں
        constraints = []

        # ابتدائی حالت کی پابندی
        constraints.append(X[0] == x0)

        # سسٹم ڈائنیمکس کی پابندیاں
        for k in range(self.N):
            constraints.append(X[k+1] == self.A @ X[k] + self.B @ U[k])

        # کنٹرول ان پٹ کی پابندیاں
        for k in range(self.N):
            constraints.append(U[k] >= self.u_min)
            constraints.append(U[k] <= self.u_max)

        # حالت کی پابندیاں (اگر قابل اطلاق ہو)
        for k in range(self.N + 1):
            constraints.append(X[k] >= self.x_min)
            constraints.append(X[k] <= self.x_max)

        # آپٹیمائزیشن مسئلہ تشکیل دیں
        problem = cp.Problem(cp.Minimize(cost), constraints)

        return problem, U, X

    def compute_control(self, current_state, reference_trajectory):
        """MPC کا استعمال کرتے ہوئے بہترین کنٹرول کمپیوٹ کریں"""
        # سادگی کے لیے، واحد حوالہ نکتہ استعمال کریں
        # عمل میں، ٹریجکٹری استعمال کریں
        x_ref = reference_trajectory[0] if len(reference_trajectory) > 0 else np.zeros(self.state_dim)

        # آپٹیمائزیشن سیٹ اپ کریں
        problem, U, X = self.setup_optimization_problem(current_state, x_ref)

        # آپٹیمائزیشن مسئلہ حل کریں
        try:
            problem.solve(solver=cp.ECOS, verbose=False)

            if problem.status == cp.OPTIMAL:
                # پہلا کنٹرول ان پٹ لوٹائیں (ریسیڈنگ ہورائزون)
                optimal_control = U.value[0] if U.value is not None else np.zeros(self.control_dim)
                return optimal_control, True
            else:
                # آپٹیمائزیشن ناکام ہونے پر سادہ کنٹرول پر واپس جائیں
                return self.fallback_control(current_state, x_ref), False

        except Exception as e:
            self.get_logger().error(f'MPC optimization failed: {e}')
            return self.fallback_control(current_state, x_ref), False

    def fallback_control(self, current_state, reference_state):
        """MPC ناکام ہونے پر فیل بیک کنٹرول لا
        # سادہ حالت فیڈ بیک کنٹرول
        K = np.eye(self.control_dim)  # مناسب طریقے سے ڈیزائن کیا جائے گا
        error = reference_state - current_state[:self.control_dim]  # سادہ
        return K @ error

    def update_model_matrices(self, new_A, new_B):
        """موافق MPC کے لیے سسٹم میٹرکسز اپ ڈیٹ کریں"""
        self.A = new_A
        self.B = new_B
```

### موافق کنٹرول سسٹم

موافق کنٹرول سسٹم کا سلوک کی بنیاد پر پیرامیٹر ایڈجسٹ کرتا ہے:

```python
# مثال: ماڈل ریفرنس موافق کنٹرول (MRAC)
class ModelReferenceAdaptiveController:
    def __init__(self, reference_model_params, initial_controller_params):
        self.reference_model = self.initialize_reference_model(reference_model_params)
        self.controller_params = initial_controller_params.copy()
        self.adaptation_rate = 0.01  # سیکھنے کی شرح
        self.param_bounds = {'min': -10.0, 'max': 10.0}  # پیرامیٹر کی حدیں

        # موافقت کی حالت
        self.error_history = []
        self.parameter_history = []

    def initialize_reference_model(self, params):
        """مطلوبہ سلوک کے لیے حوالہ ماڈل شروع کریں"""
        # حوالہ ماڈل: مطلوبہ سسٹم ڈائنیمکس
        # مثال: دوسرے درجے کا سسٹم
        wn = params.get('natural_frequency', 1.0)
        zeta = params.get('damping_ratio', 0.7)

        # دوسرے درجے کا سسٹم: s^2 + 2*zeta*wn*s + wn^2
        return {
            'omega_n': wn,
            'zeta': zeta,
            'denominator': [1.0, 2*zeta*wn, wn**2]
        }

    def compute_control(self, reference_input, actual_output, reference_output):
        """موافقت کے ساتھ کنٹرول کمپیوٹ کریں"""
        # ٹریکنگ خامی کا حساب لگائیں
        tracking_error = reference_output - actual_output

        # خامی کی بنیاد پر پیرامیٹر کو اپ ڈیٹ کریں
        self.update_parameters(tracking_error, actual_output)

        # اپ ڈیٹ کی گئی پیرامیٹر کا استعمال کرتے ہوئے کنٹرول کمپیوٹ کریں
        control_signal = self.compute_adaptive_control(
            reference_input, actual_output, tracking_error
        )

        return control_signal

    def update_parameters(self, error, output):
        """خامی کی بنیاد پر کنٹرولر پیرامیٹر اپ ڈیٹ کریں"""
        # گریڈینٹ ڈیسینٹ اڈاپٹیشن لا
        # سادہ مثال: تناسب کا گیان ایڈجسٹ کریں
        if len(self.error_history) > 1:
            # سسٹم ماڈل کی بنیاد پر خامی کا گریڈینٹ استعمال کریں
            param_gradient = self.compute_param_gradient(error, output)

            # پیرامیٹر اپ ڈیٹ کریں
            for param_name, current_val in self.controller_params.items():
                gradient_val = param_gradient.get(param_name, 0.0)
                new_val = current_val - self.adaptation_rate * gradient_val

                # حدیں لاگو کریں
                new_val = max(self.param_bounds['min'],
                             min(self.param_bounds['max'], new_val))

                self.controller_params[param_name] = new_val

        # تاریخ میں محفوظ کریں
        self.error_history.append(error)
        self.parameter_history.append(self.controller_params.copy())

    def compute_param_gradient(self, error, output):
        """پیرامیٹر کے لحاظ سے خامی کا گریڈینٹ کمپیوٹ کریں"""
        # یہ سسٹم ماڈل کی بنیاد پر کمپیوٹ کیا جائے گا
        # اس مثال کے لیے، سادہ گریڈینٹس استعمال کریں
        gradients = {}

        # مثال گریڈینٹس (سسٹم ماڈل سے حاصل کیا جائے گا)
        if 'kp' in self.controller_params:
            gradients['kp'] = -error * output  # سادہ گریڈینٹ

        if 'ki' in self.controller_params:
            gradients['ki'] = -error  # سادہ گریڈینٹ

        return gradients

    def compute_adaptive_control(self, reference, output, error):
        """موافق پیرامیٹر کا استعمال کرتے ہوئے کنٹرول کمپیوٹ کریں"""
        # موجودہ موافق پیرامیٹر استعمال کریں
        kp = self.controller_params.get('kp', 1.0)
        ki = self.controller_params.get('ki', 0.0)

        # موافق گیانز کے ساتھ PID جیسا کنٹرول
        control = kp * error

        # انٹیگرل ٹرم شامل کریں اگر دستیاب ہو
        if len(self.error_history) > 0:
            integral_error = sum(self.error_history) * 0.01  # dt تقرب
            control += ki * integral_error

        return control

    def reset_adaptation(self):
        """ابتدائی پیرامیٹر پر اڈاپٹیشن ری سیٹ کریں"""
        self.error_history.clear()
        self.parameter_history.clear()
```

## مضبوط کنٹرول سسٹم

### H-انفنٹی کنٹرول

H-انفنٹی کنٹرول ماڈل عدم یقینی کے لیے مضبوطی فراہم کرتا ہے:

```python
# مثال: H-انفنٹی کنٹرول فریم ورک
class HInfinityController:
    def __init__(self, system_order, uncertainty_bound=0.1):
        self.system_order = system_order
        self.uncertainty_bound = uncertainty_bound

        # کنٹرولر میٹرکسز (synthesized کیا جائے گا)
        self.K = np.zeros((system_order, system_order))  # حالت فیڈ بیک گیان
        self.L = np.zeros((system_order, system_order))  # آبزروور گیان
        self.P = np.eye(system_order)  # رکّاٹی مساوات کا حل

        # کارکردگی کے وزن کے میٹرکسز
        self.W_performance = np.eye(system_order)
        self.W_control = 0.1 * np.eye(system_order)
        self.W_uncertainty = np.eye(system_order)

        self.gamma = 1.0  # کارکردگی کی حد

    def synthesize_controller(self, nominal_system):
        """نامیکل سسٹم کے لیے H-انفنٹی کنٹرولر synthesis کریں"""
        # یہ H-انفنٹی synthesis مسئلہ حل کرے گا
        # اس مثال کے لیے، سادہ نتیجہ لوٹائیں
        A, B, C, D = nominal_system

        # سادہ ڈیزائن: مضبوطی کے لیے پول پلیسمنٹ
        desired_poles = self.compute_robust_poles(A, self.gamma)

        # حالت فیڈ بیک گیان K ڈیزائن کریں
        self.K = self.place_poles(A, B, desired_poles)

        # آبزروور گیان L ڈیزائن کریں
        self.L = self.place_poles(A.T, C.T, desired_poles).T

        return self.K, self.L

    def compute_robust_poles(self, A, gamma):
        """مضبوط کارکردگی کے لیے پولز کمپیوٹ کریں"""
        # یہ H-انفنٹی آپٹیمائزیشن میں شامل ہوگا
        # اس مثال کے لیے، سادہ مضبوط پول پلیسمنٹ استعمال کریں
        eigenvals = np.linalg.eigvals(A)

        # مضبوطی کے لیے پولز کو شفٹ کریں
        robust_poles = []
        for ev in eigenvals:
            if np.real(ev) >= 0:  # غیر مستحکم پولز
                robust_poles.append(-abs(ev) - 0.5)  # مستحکم کریں مارجن کے ساتھ
            else:  # مستحکم پولز
                robust_poles.append(ev * 0.8)  # زیادہ مستحکم کریں

        return robust_poles

    def place_poles(self, A, B, poles):
        """مطلوبہ مقامات پر پولز رکھیں"""
        # سنگل ان پٹ کے کیس کے لیے اکرمن کا فارمولا استعمال کریں
        # ملٹی ان پٹ کے لیے، زیادہ ترقی یافتہ طریقہ استعمال کریں گے
        n = A.shape[0]

        # کنٹرول ایبلٹی میٹرکس
        C = np.zeros((n, n))
        C[:, 0] = B.flatten()
        for i in range(1, n):
            C[:, i] = (A @ C[:, i-1]).flatten()

        # کنٹرول ایبلٹی چیک کریں
        if np.linalg.matrix_rank(C) < n:
            raise ValueError("System is not controllable")

        # کریکٹرسٹک پولی نومیل کے کوائف
        poly_coeffs = np.poly(poles)

        # اکرمن کا فارمولا
        K = np.zeros(n)
        K[-1] = 1.0
        for i in range(n-1):
            K[i] = -poly_coeffs[n-i]

        # اصل کوآرڈینیٹس میں ٹرانسفارم کریں
        K = K @ np.linalg.inv(C)

        return K.reshape(1, -1)

    def compute_robust_control(self, state_estimate, reference, disturbance_estimate=0):
        """مضبوط کنٹرول ایکشن کمپیوٹ کریں"""
        # حالت فیڈ بیک حوالہ ٹریکنگ کے ساتھ
        error = reference - state_estimate
        control = -self.K @ state_estimate + self.K @ reference

        # خلل رد کریں
        control -= self.K @ disturbance_estimate

        return control.flatten()

    def update_for_uncertainty(self, measured_uncertainty):
        """ماپی گئی عدم یقینی کے لیے کنٹرولر اپ ڈیٹ کریں"""
        # ماپی گئی عدم یقینی کی بنیاد پر کارکردگی کی حد ایڈجسٹ کریں
        if measured_uncertainty > self.uncertainty_bound:
            # کارکردگی کم کر کے مضبوطی بڑھائیں
            self.gamma *= 1.1
        else:
            # بہتر کارکردگی کی اجازت دیں
            self.gamma *= 0.95

        # کنٹرولر دوبارہ synthesis کریں
        # عمل میں، میٹرکسز کو معمولی طور پر اپ ڈیٹ کریں گے
```

## ROS2 نفاذ: اعلی درجے کے کنٹرول سسٹم

یہاں اعلی درجے کے کنٹرول سسٹم کا جامع ROS2 نفاذ ہے:

```python
# advanced_control_systems.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import String, Float32
from builtin_interfaces.msg import Time
import numpy as np
from scipy import linalg
import control  # Python Control Systems Library

class AdvancedControlSystem(Node):
    def __init__(self):
        super().__init__('advanced_control_system')

        # پبلشرز
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.control_status_pub = self.create_publisher(String, '/control_status', 10)
        self.performance_pub = self.create_publisher(Float32, '/control_performance', 10)

        # سبسکرائبرز
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )

        # کنٹرول سسٹم کمپوننٹس
        self.pid_controllers = {}
        self.mpc_controller = ModelPredictiveController(state_dim=6, control_dim=2)
        self.adaptive_controller = ModelReferenceAdaptiveController(
            {'natural_frequency': 1.0, 'damping_ratio': 0.7},
            {'kp': 1.0, 'ki': 0.1}
        )
        self.h_inf_controller = HInfinityController(system_order=4)

        # روبوٹ کی حالت
        self.joint_states = None
        self.imu_data = None
        self.current_pose = None
        self.desired_trajectory = []

        # کنٹرول پیرامیٹر
        self.control_frequency = 100.0  # Hz
        self.dt = 1.0 / self.control_frequency
        self.control_mode = 'pid'  # pid, mpc, adaptive, robust

        # جوائنٹ نامز اور ابتدائی سیٹ اپ
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.initialize_controllers()

        # کارکردگی کی نگرانی
        self.performance_metrics = {
            'tracking_error': 0.0,
            'control_effort': 0.0,
            'stability_margin': 0.0
        }

        # کنٹرول ٹائمر
        self.control_timer = self.create_timer(1.0/self.control_frequency, self.control_loop)

    def initialize_controllers(self):
        """ہر جوائنٹ کے لیے PID کنٹرولرز شروع کریں"""
        for joint_name in self.joint_names:
            self.pid_controllers[joint_name] = AdvancedPIDController(
                kp=2.0, ki=0.1, kd=0.05, dt=self.dt
            )

    def joint_state_callback(self, msg):
        """جوائنٹ حالت کی تازہ کاریوں کو سنبھالیں"""
        self.joint_states = msg

    def imu_callback(self, msg):
        """جھکاؤ کنٹرول کے لیے IMU ڈیٹا کو سنبھالیں"""
        self.imu_data = msg
        # IMU ڈیٹا سے موجودہ پوزیشن اپ ڈیٹ کریں
        self.current_pose = self.imu_to_pose(msg)

    def imu_to_pose(self, imu_msg):
        """IMU ڈیٹا کو پوزیشن کے اندازے میں تبدیل کریں"""
        pose = Pose()
        pose.orientation = imu_msg.orientation
        # پوزیشن دیگر ذرائع سے آئے گی (اودومیٹر، وغیرہ)
        return pose

    def control_loop(self):
        """کئی کنٹرول حکمت عمل کے ساتھ اصل کنٹرول لوپ"""
        if not self.joint_states:
            return

        # موجودہ حالت حاصل کریں
        current_positions = dict(zip(self.joint_states.name, self.joint_states.position))
        current_velocities = dict(zip(self.joint_states.name, self.joint_states.velocity))

        # کنٹرول حکمت عمل منتخب کریں
        if self.control_mode == 'pid':
            commands = self.pid_control(current_positions, current_velocities)
        elif self.control_mode == 'mpc':
            commands = self.mpc_control(current_positions, current_velocities)
        elif self.control_mode == 'adaptive':
            commands = self.adaptive_control(current_positions, current_velocities)
        elif self.control_mode == 'robust':
            commands = self.robust_control(current_positions, current_velocities)
        else:
            commands = self.fallback_control(current_positions)

        # کمانڈز انجام دیں
        self.execute_commands(commands)

        # کارکردگی کی نگرانی کریں
        self.monitor_performance(current_positions, commands)

        # حالت شائع کریں
        self.publish_control_status()

    def pid_control(self, current_positions, current_velocities):
        """PID-مبنی جوائنٹ کنٹرول"""
        commands = JointState()
        commands.name = self.joint_names
        commands.position = []
        commands.velocity = []
        commands.effort = []

        # سادہ ٹریجکٹری فالو کرنا
        desired_positions = self.get_desired_positions()

        for joint_name in self.joint_names:
            if joint_name in current_positions and joint_name in desired_positions:
                # PID کنٹرولر اپ ڈیٹ کریں
                control_output = self.pid_controllers[joint_name].update(
                    desired_positions[joint_name],
                    current_positions[joint_name]
                )

                commands.position.append(desired_positions[joint_name])
                commands.velocity.append(control_output)
                commands.effort.append(control_output)  # سادہ میپنگ
            else:
                commands.position.append(0.0)
                commands.velocity.append(0.0)
                commands.effort.append(0.0)

        return commands

    def mpc_control(self, current_positions, current_velocities):
        """MPC-مبنی کنٹرول"""
        # حالت ویکٹر تیار کریں
        state = self.pack_state_vector(current_positions, current_velocities)

        # حوالہ ٹریجکٹری حاصل کریں
        reference_trajectory = self.get_reference_trajectory()

        # بہترین کنٹرول کمپیوٹ کریں
        optimal_control, success = self.mpc_controller.compute_control(
            state, reference_trajectory
        )

        # جوائنٹ کمانڈز میں تبدیل کریں
        commands = self.convert_control_to_commands(optimal_control)
        return commands

    def adaptive_control(self, current_positions, current_velocities):
        """سسٹم کی شناخت کی بنیاد پر موافق کنٹرول"""
        # موجودہ حالت حاصل کریں
        current_state = self.pack_state_vector(current_positions, current_velocities)

        # حوالہ حالت حاصل کریں
        reference_state = self.get_reference_state()

        # موافق کنٹرول کمپیوٹ کریں
        control_signal = self.adaptive_controller.compute_control(
            reference_state, current_state, reference_state
        )

        # کمانڈز میں تبدیل کریں
        commands = self.convert_control_to_commands(control_signal)
        return commands

    def robust_control(self, current_positions, current_velocities):
        """مضبوط H-انفنٹی کنٹرول"""
        # حالت ویکٹر پیک کریں
        state = self.pack_state_vector(current_positions, current_velocities)

        # حوالہ حاصل کریں
        reference = self.get_reference_state()

        # مضبوط کنٹرول کمپیوٹ کریں
        control_signal = self.h_inf_controller.compute_robust_control(
            state, reference
        )

        # کمانڈز میں تبدیل کریں
        commands = self.convert_control_to_commands(control_signal)
        return commands

    def fallback_control(self, current_positions):
        """اعلی درجے کی تکنیکوں کے ناکام ہونے پر فیل بیک کنٹرول"""
        commands = JointState()
        commands.name = list(current_positions.keys())
        commands.position = list(current_positions.values())
        commands.velocity = [0.0] * len(current_positions)
        commands.effort = [0.0] * len(current_positions)
        return commands

    def pack_state_vector(self, positions, velocities):
        """پوزیشنز اور ویلوسیٹیز کو حالت ویکٹر میں پیک کریں"""
        state = np.zeros(2 * len(self.joint_names))

        for i, joint_name in enumerate(self.joint_names):
            state[i] = positions.get(joint_name, 0.0)  # پہلے نصف میں پوزیشنز
            state[i + len(self.joint_names)] = velocities.get(joint_name, 0.0)  # دوسرے نصف میں ویلوسیٹیز

        return state

    def get_desired_positions(self):
        """مطلوبہ جوائنٹ پوزیشنز حاصل کریں (ٹریجکٹری پلانر سے)"""
        # یہ ٹریجکٹری پلانر کے ساتھ بات چیت کرے گا
        # فی الحال، سادہ سائنوسوڈل ٹریجکٹری لوٹائیں
        import time
        t = time.time()
        desired = {}
        for i, joint_name in enumerate(self.joint_names):
            desired[joint_name] = 0.5 * np.sin(0.5 * t + i * np.pi / 3)
        return desired

    def get_reference_trajectory(self):
        """MPC کے لیے حوالہ ٹریجکٹری حاصل کریں"""
        # سادہ حوالہ ٹریجکٹری جنریٹ کریں
        reference_trajectory = []
        for k in range(self.mpc_controller.N):
            state = np.zeros(self.mpc_controller.state_dim)
            # سادہ گولائی ٹریجکٹری
            state[0] = 0.5 * np.cos(0.1 * k)  # x پوزیشن
            state[1] = 0.5 * np.sin(0.1 * k)  # y پوزیشن
            reference_trajectory.append(state)
        return reference_trajectory

    def get_reference_state(self):
        """موجودہ حوالہ حالت حاصل کریں"""
        return np.zeros(2 * len(self.joint_names))  # جگہ کا نمونہ

    def convert_control_to_commands(self, control_signal):
        """کنٹرول سگنل کو ROS کمانڈ میں تبدیل کریں"""
        commands = JointState()
        commands.name = self.joint_names

        # کنٹرول سگنل کو جوائنٹ کمانڈز میں میپ کریں
        # یہ روبوٹ کنیمیٹکس کے مطابق ہوگا
        for i, joint_name in enumerate(self.joint_names):
            if i < len(control_signal):
                commands.position.append(control_signal[i])
                commands.velocity.append(control_signal[i])
                commands.effort.append(control_signal[i])
            else:
                commands.position.append(0.0)
                commands.velocity.append(0.0)
                commands.effort.append(0.0)

        return commands

    def execute_commands(self, commands):
        """کمپیوٹ کی گئی کنٹرول کمانڈز انجام دیں"""
        self.joint_cmd_pub.publish(commands)

    def monitor_performance(self, current_positions, commands):
        """کنٹرول کارکردگی کی میٹرکسز کی نگرانی کریں"""
        # ٹریکنگ خامی کا حساب لگائیں
        desired_positions = self.get_desired_positions()
        tracking_errors = []
        for joint_name in self.joint_names:
            if joint_name in current_positions and joint_name in desired_positions:
                error = abs(desired_positions[joint_name] - current_positions[joint_name])
                tracking_errors.append(error)

        self.performance_metrics['tracking_error'] = np.mean(tracking_errors) if tracking_errors else 0.0

        # کنٹرول کوشش کا حساب لگائیں
        control_efforts = [abs(effort) for effort in commands.effort]
        self.performance_metrics['control_effort'] = np.mean(control_efforts) if control_efforts else 0.0

        # کارکردگی کی میٹرکسز شائع کریں
        perf_msg = Float32()
        perf_msg.data = self.performance_metrics['tracking_error']
        self.performance_pub.publish(perf_msg)

    def publish_control_status(self):
        """کنٹرول سسٹم کی حالت شائع کریں"""
        status_msg = String()
        status_msg.data = (
            f"Mode: {self.control_mode}, "
            f"Error: {self.performance_metrics['tracking_error']:.3f}, "
            f"Effort: {self.performance_metrics['control_effort']:.3f}, "
            f"Joints: {len(self.joint_names)}"
        )
        self.control_status_pub.publish(status_msg)

    def switch_control_mode(self, new_mode):
        """کنٹرول موڈز کے درمیان سوئچ کریں"""
        if new_mode in ['pid', 'mpc', 'adaptive', 'robust']:
            self.control_mode = new_mode
            self.get_logger().info(f'Switched to {new_mode} control mode')

class SlidingModeController:
    """مضبوط کارکردگی کے لیے سلائیڈنگ موڈ کنٹرول"""
    def __init__(self, state_dim, control_dim, sliding_surface_params=None):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.lambda_ = 1.0  # سلائیڈنگ سطح کا پیرامیٹر
        self.kappa = 0.1   # باؤنڈری لیئر کی چوڑائی
        self.rho = 10.0    # سوئچنگ گیان

        if sliding_surface_params:
            self.surface_params = sliding_surface_params
        else:
            # ڈیفالٹ سلائیڈنگ سطح: s = λe + ė
            self.surface_params = {'lambda': self.lambda_}

    def compute_control(self, state, reference, dt):
        """سلائیڈنگ موڈ کنٹرول کمپیوٹ کریں"""
        # ٹریکنگ خامی
        error = reference - state[:self.state_dim//2]  # فرض کریں state = [positions, velocities]
        error_derivative = state[self.state_dim//2:]   # ویلوسیٹیز

        # سلائیڈنگ سطح
        s = self.surface_params['lambda'] * error + error_derivative

        # مساوی کنٹرول (نامیکل سسٹم)
        u_eq = self.compute_equivalent_control(state, reference)

        # سوئچنگ کنٹرول
        u_sw = self.compute_switching_control(s)

        # کل کنٹرول
        u_total = u_eq + u_sw

        return u_total

    def compute_equivalent_control(self, state, reference):
        """مساوی کنٹرول کمپیوٹ کریں (نامیکل سسٹم)"""
        # اس مثال کے لیے، سادہ PD کنٹرول استعمال کریں
        error = reference - state[:self.state_dim//2]
        error_derivative = state[self.state_dim//2:]
        return 2.0 * error + 1.0 * error_derivative

    def compute_switching_control(self, s):
        """سوئچنگ کنٹرول کمپیوٹ کریں"""
        # چیٹر کو کم کرنے کے لیے باؤنڈری لیئر
        saturation = np.tanh(s / self.kappa)
        return -self.rho * saturation

class FuzzyLogicController:
    """غیر خطی نظام کے لیے فuzzy لوگک کنٹرولر"""
    def __init__(self):
        # فuzzy سیٹس اور رولز کی وضاحت کریں
        self.fuzzy_rules = [
            {'input': {'error': 'negative', 'derivative': 'negative'}, 'output': 'negative_high'},
            {'input': {'error': 'negative', 'derivative': 'zero'}, 'output': 'negative_medium'},
            {'input': {'error': 'negative', 'derivative': 'positive'}, 'output': 'negative_low'},
            {'input': {'error': 'zero', 'derivative': 'negative'}, 'output': 'negative_medium'},
            {'input': {'error': 'zero', 'derivative': 'zero'}, 'output': 'zero'},
            {'input': {'error': 'zero', 'derivative': 'positive'}, 'output': 'positive_medium'},
            {'input': {'error': 'positive', 'derivative': 'negative'}, 'output': 'positive_low'},
            {'input': {'error': 'positive', 'derivative': 'zero'}, 'output': 'positive_medium'},
            {'input': {'error': 'positive', 'derivative': 'positive'}, 'output': 'positive_high'}
        ]

    def fuzzify(self, error, error_derivative):
        """کریسپ ان پٹس کو فuzzy ویلیوز میں تبدیل کریں"""
        membership = {
            'error': self.get_membership(error, 'error'),
            'derivative': self.get_membership(error_derivative, 'derivative')
        }
        return membership

    def get_membership(self, value, var_type):
        """ایک متغیر کے لیے ممبرشپ ویلیوز حاصل کریں"""
        if var_type == 'error':
            return {
                'negative': self.triangle_membership(value, -2, -1, 0),
                'zero': self.triangle_membership(value, -1, 0, 1),
                'positive': self.triangle_membership(value, 0, 1, 2)
            }
        elif var_type == 'derivative':
            return {
                'negative': self.triangle_membership(value, -1, -0.5, 0),
                'zero': self.triangle_membership(value, -0.5, 0, 0.5),
                'positive': self.triangle_membership(value, 0, 0.5, 1)
            }

    def triangle_membership(self, x, a, b, c):
        """مثلثی ممبرشپ فنکشن"""
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:  # b < x < c
            return (c - x) / (c - b)

    def infer(self, memberships):
        """فuzzy رولز لاگو کر کے فuzzy آؤٹ پٹ حاصل کریں"""
        output_membership = {}
        for rule in self.fuzzy_rules:
            # فائرنگ اسٹرینتھ حاصل کریں
            error_fuzz = memberships['error'][rule['input']['error']]
            deriv_fuzz = memberships['derivative'][rule['input']['derivative']]
            firing_strength = min(error_fuzz, deriv_fuzz)

            # آؤٹ پٹ پر لاگو کریں
            output_type = rule['output']
            if output_type not in output_membership:
                output_membership[output_type] = 0.0
            output_membership[output_type] = max(output_membership[output_type], firing_strength)

        return output_membership

    def defuzzify(self, output_membership):
        """فuzzy آؤٹ پٹ کو کریسپ کنٹرول ویلیو میں تبدیل کریں"""
        # مرکز کی گریویٹی کا طریقہ
        numerator = 0.0
        denominator = 0.0

        # آؤٹ پٹ ممبرشپ فنکشنز کی وضاحت کریں
        output_centers = {
            'negative_high': -3.0,
            'negative_medium': -2.0,
            'negative_low': -1.0,
            'zero': 0.0,
            'positive_low': 1.0,
            'positive_medium': 2.0,
            'positive_high': 3.0
        }

        for output_type, membership_value in output_membership.items():
            if output_type in output_centers:
                numerator += output_centers[output_type] * membership_value
                denominator += membership_value

        return numerator / denominator if denominator != 0.0 else 0.0

    def compute_control(self, error, error_derivative):
        """فuzzy کنٹرول آؤٹ پٹ کمپیوٹ کریں"""
        memberships = self.fuzzify(error, error_derivative)
        fuzzy_output = self.infer(memberships)
        crisp_output = self.defuzzify(fuzzy_output)
        return crisp_output

def main(args=None):
    rclpy.init(args=args)
    control_system = AdvancedControlSystem()

    try:
        rclpy.spin(control_system)
    except KeyboardInterrupt:
        pass
    finally:
        control_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## غیر لکیری کنٹرول تکنیکیں

### فیڈ بیک لکیریکرن

زیادہ غیر لکیری روبوٹک نظاموں کے لیے:

```python
# مثال: روبوٹ مینیپولیٹر کے لیے فیڈ بیک لکیریکرن
class FeedbackLinearizationController:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.nominal_controller = AdvancedPIDController()

    def feedback_linearize(self, q, dq, ddq_desired):
        """روبوٹ ڈائنیمکس پر فیڈ بیک لکیریکرن لاگو کریں"""
        # روبوٹ ڈائنیمکس: M(q)ddq + C(q,dq)dq + g(q) = τ
        M = self.robot_model.mass_matrix(q)
        C = self.robot_model.coriolis_matrix(q, dq)
        g = self.robot_model.gravity_vector(q)

        # مطلوبہ ایکسلریشن حاصل کرنے کے لیے مطلوبہ ٹورک
        tau = M @ ddq_desired + C @ dq + g

        return tau

    def compute_control(self, q, dq, q_desired, dq_desired, ddq_desired):
        """فیڈ بیک لکیریکرن کا استعمال کرتے ہوئے کنٹرول کمپیوٹ کریں"""
        # سب سے پہل، اصل کوآرڈینیٹس میں خامی کمپیوٹ کریں
        q_error = q_desired - q
        dq_error = dq_desired - dq

        # نامیکل کنٹرولر کو مطلوبہ ایکسلریشن حاصل کرنے کے لیے لاگو کریں
        ddq_command = self.nominal_controller.update(q_desired, q)

        # فیڈ بیک لکیریکرن لاگو کریں
        tau = self.feedback_linearize(q, dq, ddq_command)

        return tau
```

### بیک اسٹیپنگ کنٹرول

کیسکیڈ غیر لکیری نظاموں کے لیے منظم ڈیزائن:

```python
# مثال: بیک اسٹیپنگ کنٹرولر
class BacksteppingController:
    def __init__(self):
        self.stabilizing_functions = []
        self.control_gains = []

    def design_for_system(self, system_order):
        """دی گئی ترتیب کے نظام کے لیے بیک اسٹیپنگ کنٹرولر ڈیزائن کریں"""
        for i in range(system_order):
            # i-th سب سسٹم کے لیے مستحکم کن فنکشن ڈیزائن کریں
            self.stabilizing_functions.append(self.design_stabilizing_function(i))
            self.control_gains.append(1.0)  # ابتدائی گیانز

    def design_stabilizing_function(self, step):
        """بیک اسٹیپنگ قدم کے لیے مستحکم کن فنکشن ڈیزائن کریں"""
        # یہ منظم بیک اسٹیپنگ ڈیزائن نافذ کرے گا
        # ہر قدم کے لیے، ایک ورچوئل کنٹرول ڈیزائن کریں جو سب سسٹم کو مستحکم کرے
        def stabilizing_func(states_up_to_i, desired_states_up_to_i):
            # قدم i کے لیے ورچوئل کنٹرول
            error = desired_states_up_to_i[-1] - states_up_to_i[-1]
            virtual_control = -self.control_gains[step] * error
            return virtual_control
        return stabilizing_func

    def compute_control(self, full_state, desired_trajectory):
        """بیک اسٹیپنگ ڈیزائن کا استعمال کرتے ہوئے کنٹرول کمپیوٹ کریں"""
        # بازگشتی بیک اسٹیپنگ الگورتھم نافذ کریں
        # یہ ہر سب سسٹم کے لیے ورچوئل کنٹرولز کمپیوٹ کرے گا
        # اور آخر میں اصل کنٹرول ان پٹ
        control = 0.0

        # سادہ نفاذ
        for i in range(len(full_state)):
            if i < len(self.stabilizing_functions):
                # اس قدم کے لیے ورچوئل کنٹرول کمپیوٹ کریں
                current_states = full_state[:i+1]
                desired_states = desired_trajectory[:i+1]
                virtual_control = self.stabilizing_functions[i](
                    current_states, desired_states
                )
                control = virtual_control

        return control
```

## لیب: اعلی درجے کے کنٹرول سسٹم نافذ کرنا

اس لیب میں، آپ اعلی درجے کا کنٹرول سسٹم نافذ کریں گے:

```python
# lab_advanced_control.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32
import numpy as np

class AdvancedControlLab(Node):
    def __init__(self):
        super().__init__('advanced_control_lab')

        # پبلشرز
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.performance_pub = self.create_publisher(Float32, '/control_performance', 10)
        self.status_pub = self.create_publisher(String, '/control_status', 10)

        # سبسکرائبرز
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10
        )

        # کنٹرول کمپوننٹس
        self.cascade_controller = CascadeController()
        self.sliding_controller = SlidingModeController(state_dim=4, control_dim=2)
        self.fuzzy_controller = FuzzyLogicController()

        # ڈیٹا اسٹوریج
        self.joint_data = None
        self.control_mode = 'cascade'  # cascade, sliding, fuzzy

        # کنٹرول پیرامیٹر
        self.control_frequency = 50.0
        self.performance_history = []

        # کنٹرول ٹائمر
        self.control_timer = self.create_timer(1.0/self.control_frequency, self.control_loop)

    def joint_callback(self, msg):
        """جوائنٹ حالت ڈیٹا کو سنبھالیں"""
        self.joint_data = msg

    def control_loop(self):
        """اصل کنٹرول لوپ"""
        if self.joint_data is None:
            return

        # موجودہ حالت حاصل کریں
        current_state = np.array(self.joint_data.position + self.joint_data.velocity)

        # مطلوبہ حالت حاصل کریں
        desired_state = self.get_desired_state()

        # موڈ کی بنیاد پر کنٹرول کمپیوٹ کریں
        if self.control_mode == 'cascade':
            control_output = self.cascade_control(current_state, desired_state)
        elif self.control_mode == 'sliding':
            control_output = self.sliding_control(current_state, desired_state)
        elif self.control_mode == 'fuzzy':
            control_output = self.fuzzy_control(current_state, desired_state)
        else:
            control_output = np.zeros(len(current_state)//2)  # صفر کنٹرول

        # کنٹرول انجام دیں
        self.execute_control(control_output)

        # کارکردگی کی نگرانی کریں
        self.monitor_performance(current_state, desired_state, control_output)

        # حالت شائع کریں
        self.publish_status()

    def get_desired_state(self):
        """مطلوبہ حالت ٹریجکٹری حاصل کریں"""
        import time
        t = time.time()
        # سادہ سائنوسوڈل ٹریجکٹری
        desired_positions = [0.5 * np.sin(0.5 * t + i * np.pi/4) for i in range(6)]
        desired_velocities = [0.5 * 0.5 * np.cos(0.5 * t + i * np.pi/4) for i in range(6)]
        return np.array(desired_positions + desired_velocities)

    def cascade_control(self, current_state, desired_state):
        """کیسکیڈ کنٹرول نفاذ"""
        # سادہ کیسکیڈ کنٹرول
        position_error = desired_state[:6] - current_state[:6]
        velocity_error = desired_state[6:] - current_state[6:]

        # پوزیشن کنٹرولر
        velocity_command = position_error * 2.0  # سادہ تناسب

        # رفتار کنٹرولر
        control_output = (velocity_command - current_state[6:]) * 1.5  # سادہ تناسب

        return control_output

    def sliding_control(self, current_state, desired_state):
        """سلائیڈنگ موڈ کنٹرول نفاذ"""
        # خامی کا حساب لگائیں
        error = desired_state[:6] - current_state[:6]
        error_deriv = desired_state[6:] - current_state[6:]

        # سلائیڈنگ سطح
        s = error + error_deriv

        # کنٹرول لا
        control_output = -5.0 * np.sign(s)  # سادہ سلائیڈنگ کنٹرول

        return control_output

    def fuzzy_control(self, current_state, desired_state):
        """فuzzy لوگک کنٹرول نفاذ"""
        # مثال کے طور پر پہلے جوائنٹ کے لیے خامیاں کا حساب لگائیں
        error = desired_state[0] - current_state[0]
        error_deriv = desired_state[6] - current_state[6]

        # پہلے جوائنٹ کے لیے فuzzy کنٹرول
        fuzzy_output = self.fuzzy_controller.compute_control(error, error_deriv)

        # تمام جوائنٹس پر لاگو کریں (سادہ)
        control_output = np.array([fuzzy_output] * 6)

        return control_output

    def execute_control(self, control_output):
        """کمپیوٹ کی گئی کنٹرول انجام دیں"""
        # جوائنٹ کمانڈ تیار کریں
        joint_cmd = JointState()
        joint_cmd.name = [f'joint_{i}' for i in range(len(control_output))]
        joint_cmd.effort = control_output.tolist()
        joint_cmd.position = [0.0] * len(control_output)  # پوزیشن کنٹرول کے ذریعے اپ ڈیٹ کیا جائے گا
        joint_cmd.velocity = control_output.tolist()

        self.joint_cmd_pub.publish(joint_cmd)

    def monitor_performance(self, current_state, desired_state, control_output):
        """کنٹرول کارکردگی کی نگرانی کریں"""
        # ٹریکنگ خامی کا حساب لگائیں
        position_error = desired_state[:6] - current_state[:6]
        tracking_error = np.mean(np.abs(position_error))

        # کنٹرول کوشش کا حساب لگائیں
        control_effort = np.mean(np.abs(control_output))

        # تاریخ میں محفوظ کریں
        self.performance_history.append({
            'error': tracking_error,
            'effort': control_effort,
            'timestamp': self.get_clock().now()
        })

        # تاریخ کو قابل انتظام رکھیں
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

        # کارکردگی میٹرک شائع کریں
        perf_msg = Float32()
        perf_msg.data = tracking_error
        self.performance_pub.publish(perf_msg)

    def publish_status(self):
        """کنٹرول حالت شائع کریں"""
        status_msg = String()
        if self.performance_history:
            avg_error = np.mean([p['error'] for p in self.performance_history[-10:]])
        else:
            avg_error = 0.0

        status_msg.data = (
            f"Mode: {self.control_mode}, "
            f"Avg Error: {avg_error:.3f}, "
            f"Joints: {len(self.joint_data.position) if self.joint_data else 0}"
        )
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    lab = AdvancedControlLab()

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

## مشق: اپنا اعلی درجے کا کنٹرول سسٹم ڈیزائن کریں

مندرجہ ذیل ڈیزائن چیلنج پر غور کریں:

1. آپ کون سے روبوٹ سسٹم کو کنٹرول کر رہے ہیں (موبائل روبوٹ، مینیپولیٹر، انسان نما روبوٹ)?
2. آپ کے سسٹم کی کلیدی ڈائنیمک خصوصیات کیا ہیں?
3. کون سی اعلی درجے کی کنٹرول تکنیک مناسب ہے (PID اقسام، MPC، موافق، مضبوط، غیر لکیری)?
4. آپ کا کنٹرولر کون سی پابندیاں سنبھالنا چاہیے?
5. آپ ماڈل عدم یقینی کا سامنا کیسے کریں گے?
6. کون سی کارکردگی کی میٹرکسز سب سے اہم ہیں?
7. آپ استحکام اور حفاظت کو کیسے یقینی بنائیں گے?
8. آپ کون سی تجرباتی توثیق کریں گے?

## خلاصہ

اعلی درجے کے کنٹرول سسٹم پیچیدہ روبوٹکس اطلاقات کے لیے ضروری ہیں، جو روبوٹس کو پیچیدہ، متحرک ماحول میں مؤثر طریقے سے کام کرنے کے قابل بناتے ہیں۔ کلیدی تصورات میں شامل ہیں:

- **PID اور کیسکیڈ کنٹرول**: بہت سے روبوٹک کنٹرول سسٹم کی بنیاد
- **ماڈل پریڈکٹو کنٹرول**: پابندیوں اور متعدد اہداف کی بہینئریشن کو سنبھالتا ہے
- **موافق کنٹرول**: سسٹم کے سلوک کی بنیاد پر پیرامیٹر ایڈجسٹ کرتا ہے
- **مضبوط کنٹرول**: عدم یقینی کے باوجود کارکردگی برقرار رکھتا ہے
- **غیر لکیری کنٹرول**: روبوٹک نظام میں ذاتی غیر لکیریت کو سنبھالتا ہے
- **فuzzy لوگک کنٹرول**: غیر درست یا عدم یقینی کی معلومات سے نمٹتا ہے

ROS2 میں ان اعلی درجے کی کنٹرول تکنیکوں کا انضمام پیچیدہ روبوٹک نظام کی ترقی کو فعال کرتا ہے جو پیچیدہ کاموں کو زیادہ کارکردگی اور قابل اعتمادی کے ساتھ سنبھال سکتے ہیں۔ ان تصورات کو سمجھنا حقیقی دنیا کے منظار ناموں میں مؤثر طریقے سے کام کرنے والے روبوٹس ترقی دینے کے لیے ضروری ہے۔

اگلے سبق میں، ہم متحرک بات چیت کے نمونوں اور یہ دیکھنے کا جائزہ لیں گے کہ روبوٹ ماحولیاتی حالات اور صارف کی ضروریات کی تبدیلی کے مطابق اپنا سلوک کیسے ایڈجسٹ کر سکتے ہیں۔