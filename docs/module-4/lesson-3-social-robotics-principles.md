---
sidebar_position: 3
---

# Social Robotics Principles

## Introduction

Social robotics focuses on creating robots that can interact naturally and effectively with humans in social contexts. Unlike task-focused robots, social robots must understand human social cues, exhibit appropriate social behaviors, and build relationships with users. This lesson explores the principles, design considerations, and implementation techniques for creating socially intelligent robots.

## Fundamentals of Social Robotics

### Social Cues and Signals

Social robots must recognize and respond to various human social cues:

```python
# Example: Social cue recognition system
class SocialCueRecognizer:
    def __init__(self):
        self.eye_contact_threshold = 0.7  # Confidence threshold for eye contact
        self.proximity_thresholds = {
            'intimate': 0.5,    # meters
            'personal': 1.2,    # meters
            'social': 3.6,      # meters
            'public': 7.6       # meters
        }
        self.gesture_mappings = {
            'wave': 'greeting_acknowledged',
            'pointing': 'attention_requested',
            'head_nod': 'agreement',
            'head_shake': 'disagreement',
            'open_hand': 'offer',
            'closed_fist': 'attention'
        }

    def recognize_eye_contact(self, face_position, camera_center, confidence):
        """Recognize eye contact based on face position"""
        if confidence < self.eye_contact_threshold:
            return False

        # Check if face is centered (indicating attention)
        face_x, face_y = face_position
        center_x, center_y = camera_center

        # Calculate distance from center
        distance = ((face_x - center_x)**2 + (face_y - center_y)**2)**0.5

        # Eye contact is likely if face is reasonably centered
        return distance < 100  # pixels threshold

    def recognize_proximity(self, distance_to_human):
        """Recognize social proximity zone"""
        if distance_to_human <= self.proximity_thresholds['intimate']:
            return 'intimate'
        elif distance_to_human <= self.proximity_thresholds['personal']:
            return 'personal'
        elif distance_to_human <= self.proximity_thresholds['social']:
            return 'social'
        else:
            return 'public'

    def recognize_gesture(self, gesture_type, confidence):
        """Recognize and interpret human gestures"""
        if confidence > 0.7 and gesture_type in self.gesture_mappings:
            return self.gesture_mappings[gesture_type]
        return 'unknown'

    def interpret_social_context(self, cues):
        """Interpret social context from multiple cues"""
        context = {
            'attention_level': self.calculate_attention_level(cues),
            'interaction_readiness': self.calculate_interaction_readiness(cues),
            'social_distance': self.recognize_proximity(cues.get('distance', 2.0)),
            'engagement_type': self.determine_engagement_type(cues)
        }
        return context

    def calculate_attention_level(self, cues):
        """Calculate human attention level based on cues"""
        attention_score = 0.0

        if cues.get('eye_contact', False):
            attention_score += 0.4
        if cues.get('facing_direction', 0) < 0.3:  # Facing robot
            attention_score += 0.3
        if cues.get('gesture_confidence', 0) > 0.7:
            attention_score += 0.3

        return min(1.0, attention_score)

    def calculate_interaction_readiness(self, cues):
        """Calculate if human is ready for interaction"""
        readiness_score = 0.0

        if self.calculate_attention_level(cues) > 0.5:
            readiness_score += 0.4
        if cues.get('proximity', 'public') in ['personal', 'social']:
            readiness_score += 0.3
        if cues.get('open_posture', False):
            readiness_score += 0.3

        return min(1.0, readiness_score)

    def determine_engagement_type(self, cues):
        """Determine type of social engagement"""
        if cues.get('eye_contact', False) and cues.get('gesture') == 'wave':
            return 'greeting'
        elif cues.get('proximity', 'public') == 'personal' and cues.get('speaking', False):
            return 'conversation'
        elif cues.get('pointing', False):
            return 'request_for_help'
        else:
            return 'passive_observation'
```

### Theory of Mind in Social Robots

Social robots should model human mental states and intentions:

```python
# Example: Theory of mind implementation
class TheoryOfMind:
    def __init__(self):
        self.belief_model = {}
        self.desire_model = {}
        self.intention_model = {}
        self.user_personality_model = {}

    def update_belief_model(self, user_id, observed_action, environment_state):
        """Update belief model about user's beliefs"""
        if user_id not in self.belief_model:
            self.belief_model[user_id] = {}

        # Update based on observed actions and environment
        belief_update = self.infer_belief_from_action(
            observed_action, environment_state
        )
        self.belief_model[user_id].update(belief_update)

    def infer_belief_from_action(self, action, environment):
        """Infer user's beliefs from their actions"""
        # Example: If user looks at an object, they likely believe it exists
        if action['type'] == 'gaze' and action['target'] in environment['objects']:
            return {action['target']: {'exists': True, 'location': environment['objects'][action['target']]['location']}}
        return {}

    def update_desire_model(self, user_id, user_input, observed_context):
        """Update desire model based on user input and context"""
        if user_id not in self.desire_model:
            self.desire_model[user_id] = {}

        # Extract desires from natural language
        desires = self.extract_desires_from_input(user_input)
        self.desire_model[user_id].update(desires)

    def extract_desires_from_input(self, user_input):
        """Extract desires from user input"""
        desires = {}
        input_lower = user_input.lower()

        # Simple keyword-based extraction (in practice, use more sophisticated NLP)
        if 'want' in input_lower or 'need' in input_lower:
            desires['goal'] = input_lower
        if 'help' in input_lower:
            desires['assistance_needed'] = True
        if 'tired' in input_lower:
            desires['rest_needed'] = True

        return desires

    def predict_user_intention(self, user_id, current_context):
        """Predict user's intention based on models"""
        belief_state = self.belief_model.get(user_id, {})
        desire_state = self.desire_model.get(user_id, {})
        context = current_context

        # Predict intention based on beliefs, desires, and context
        predicted_intention = self.reason_intention(belief_state, desire_state, context)
        return predicted_intention

    def reason_intention(self, beliefs, desires, context):
        """Reason about user's intention"""
        # This would implement more sophisticated reasoning
        # based on belief-desire-intention (BDI) model
        if desires.get('assistance_needed'):
            return 'requesting_help'
        elif desires.get('rest_needed'):
            return 'seeking_comfort'
        else:
            return 'exploring_environment'
```

## Social Interaction Patterns

### Proxemics and Personal Space

Understanding and respecting human personal space:

```python
# Example: Proxemics management system
class ProxemicsManager:
    def __init__(self):
        self.personal_space = {
            'intimate': 0.45,    # meters
            'personal': 1.2,     # meters
            'social': 3.6,       # meters
            'public': 12.0       # meters
        }
        self.cultural_adaptations = {
            'default': {'personal': 1.2, 'social': 3.6},
            'latin_american': {'personal': 0.9, 'social': 2.0},
            'middle_eastern': {'personal': 1.5, 'social': 4.0},
            'asian': {'personal': 1.5, 'social': 4.5}
        }
        self.current_culture = 'default'

    def calculate_comfortable_distance(self, interaction_type):
        """Calculate comfortable distance for different interaction types"""
        if interaction_type == 'greeting':
            return self.personal_space['social'] * 0.8  # Slightly closer for greeting
        elif interaction_type == 'conversation':
            return self.personal_space['personal']
        elif interaction_type == 'collaboration':
            return self.personal_space['intimate'] * 1.5  # Closer for collaboration
        elif interaction_type == 'presentation':
            return self.personal_space['public'] * 0.5  # Farther for presentation
        else:
            return self.personal_space['social']

    def adjust_for_cultural_norms(self, culture):
        """Adjust personal space based on cultural norms"""
        if culture in self.cultural_adaptations:
            self.current_culture = culture
            self.personal_space['personal'] = self.cultural_adaptations[culture]['personal']
            self.personal_space['social'] = self.cultural_adaptations[culture]['social']

    def evaluate_proximity_comfort(self, current_distance, interaction_type):
        """Evaluate if current distance is comfortable"""
        comfortable_distance = self.calculate_comfortable_distance(interaction_type)

        # Calculate comfort score (0-1, where 1 is most comfortable)
        if current_distance < comfortable_distance * 0.5:
            # Too close - uncomfortable
            comfort_score = max(0, 1 - (comfortable_distance * 0.5 - current_distance) / (comfortable_distance * 0.5))
        elif current_distance > comfortable_distance * 2:
            # Too far - impersonal
            comfort_score = max(0, 1 - (current_distance - comfortable_distance * 2) / (comfortable_distance * 2))
        else:
            # Within comfortable range
            comfort_score = 1.0

        return comfort_score

    def suggest_distance_adjustment(self, current_distance, interaction_type):
        """Suggest distance adjustment for optimal interaction"""
        target_distance = self.calculate_comfortable_distance(interaction_type)

        if current_distance < target_distance * 0.8:
            return 'move_back', target_distance * 0.8 - current_distance
        elif current_distance > target_distance * 1.2:
            return 'move_closer', current_distance - target_distance * 1.2
        else:
            return 'maintain_distance', 0.0
```

### Turn-Taking and Conversation Flow

Managing natural conversation flow:

```python
# Example: Conversation turn-taking system
class ConversationManager:
    def __init__(self):
        self.current_speaker = 'human'  # 'human' or 'robot'
        self.turn_history = []
        self.silence_threshold = 1.5  # seconds before robot takes turn
        self.speech_termination_threshold = 0.5  # seconds of silence to consider speech ended
        self.backchannel_probabilities = {
            'mm-hmm': 0.3,
            'uh-huh': 0.3,
            'I see': 0.2,
            'right': 0.2
        }

    def process_speech_input(self, speech_data, timestamp):
        """Process incoming speech and manage turn-taking"""
        if self.current_speaker == 'human':
            # Continue human turn
            self.extend_human_turn(speech_data, timestamp)
        else:
            # Interrupt robot turn to let human speak
            self.yield_to_human(speech_data, timestamp)

    def extend_human_turn(self, speech_data, timestamp):
        """Extend human's turn as they continue speaking"""
        if self.turn_history and self.turn_history[-1]['speaker'] == 'human':
            self.turn_history[-1]['end_time'] = timestamp
            self.turn_history[-1]['content'] += speech_data
        else:
            self.turn_history.append({
                'speaker': 'human',
                'start_time': timestamp,
                'end_time': timestamp,
                'content': speech_data
            })

    def yield_to_human(self, speech_data, timestamp):
        """Yield turn to human speaker"""
        # End robot turn
        if (self.turn_history and
            self.turn_history[-1]['speaker'] == 'robot' and
            timestamp - self.turn_history[-1]['end_time'] > 0.1):

            # Add robot's partial response to history
            self.turn_history.append({
                'speaker': 'human',
                'start_time': timestamp,
                'end_time': timestamp,
                'content': speech_data
            })

        self.current_speaker = 'human'

    def manage_robot_turn(self, understanding_result):
        """Manage robot's turn in conversation"""
        if self.should_robot_speak():
            response = self.generate_response(understanding_result)
            self.current_speaker = 'robot'

            self.turn_history.append({
                'speaker': 'robot',
                'start_time': self.get_current_time(),
                'end_time': None,
                'content': response,
                'response_type': self.determine_response_type(understanding_result)
            })

            return response
        return None

    def should_robot_speak(self):
        """Determine if robot should take the turn"""
        if not self.turn_history:
            return True  # Robot can initiate conversation

        last_turn = self.turn_history[-1]
        time_since_last = self.get_current_time() - last_turn['end_time']

        if last_turn['speaker'] == 'human':
            # Human finished speaking, check if it's time to respond
            return time_since_last >= self.silence_threshold
        else:
            # Robot finished speaking, wait for human response
            return False

    def generate_response(self, understanding_result):
        """Generate appropriate response based on understanding"""
        intent = understanding_result.get('intent', 'unknown')

        if intent == 'greeting':
            return "Hello! How can I help you today?"
        elif intent == 'question':
            return self.generate_question_response(understanding_result)
        elif intent == 'command':
            return self.generate_command_acknowledgment(understanding_result)
        else:
            return "I'm listening. What would you like to talk about?"

    def generate_question_response(self, understanding_result):
        """Generate response to questions"""
        question_type = understanding_result.get('question_type', 'unknown')

        if question_type == 'yes_no':
            return "That's an interesting question. I think it depends on the situation."
        elif question_type == 'wh':
            return "I'd be happy to help you with that. Could you tell me more?"
        else:
            return "I understand you're asking me something. Could you repeat it?"

    def generate_command_acknowledgment(self, understanding_result):
        """Generate acknowledgment for commands"""
        command = understanding_result.get('command', 'unknown')
        return f"I'll work on {command} for you right away."

    def determine_response_type(self, understanding_result):
        """Determine type of response to generate"""
        intent = understanding_result.get('intent', 'unknown')

        response_types = {
            'greeting': 'acknowledgment',
            'question': 'informative',
            'command': 'action_acknowledgment',
            'statement': 'backchannel',
            'unknown': 'clarification'
        }

        return response_types.get(intent, 'clarification')

    def generate_backchannel(self):
        """Generate backchannel responses to show active listening"""
        import random
        backchannels = list(self.backchannel_probabilities.keys())
        weights = list(self.backchannel_probabilities.values())

        return random.choices(backchannels, weights=weights)[0]

    def get_current_time(self):
        """Get current timestamp"""
        import time
        return time.time()
```

## Emotional Intelligence in Social Robots

### Emotion Recognition and Response

Social robots should recognize and appropriately respond to human emotions:

```python
# Example: Emotion recognition and response system
class EmotionRecognitionSystem:
    def __init__(self):
        self.emotion_categories = {
            'happy': {'valence': 0.8, 'arousal': 0.6, 'expression': ['smile', 'laugh', 'bright_eyes']},
            'sad': {'valence': -0.7, 'arousal': 0.2, 'expression': ['frown', 'tears', 'downcast_eyes']},
            'angry': {'valence': -0.9, 'arousal': 0.9, 'expression': ['scowl', 'furrowed_brow', 'tight_lips']},
            'fearful': {'valence': -0.6, 'arousal': 0.8, 'expression': ['wide_eyes', 'open_mouth', 'tense_posture']},
            'surprised': {'valence': 0.3, 'arousal': 0.9, 'expression': ['wide_eyes', 'open_mouth', 'raised_eyebrows']},
            'neutral': {'valence': 0.0, 'arousal': 0.3, 'expression': ['relaxed', 'normal_posture']}
        }
        self.emotion_threshold = 0.6  # Confidence threshold for emotion recognition
        self.empathy_responses = self.define_empathy_responses()

    def define_empathy_responses(self):
        """Define empathetic responses for different emotions"""
        return {
            'happy': [
                "You seem happy! I'm glad to see you're in a good mood.",
                "Your smile is infectious! What's making you so happy?",
                "I love seeing people smile. You brighten up the room!"
            ],
            'sad': [
                "You seem a bit down. Is there anything I can do to help?",
                "I'm sorry you're feeling sad. I'm here if you need to talk.",
                "I can see you're having a tough time. Remember, it's okay to not be okay."
            ],
            'angry': [
                "I can sense you're upset. Let's take a moment to breathe.",
                "I understand you're frustrated. How can I help resolve this?",
                "It seems like something is bothering you. I'm here to listen."
            ],
            'fearful': [
                "You seem worried. I'm here to keep you safe.",
                "I can see you're anxious. Let me help you feel more secure.",
                "You look concerned. Is there something specific that's worrying you?"
            ],
            'surprised': [
                "Wow, something seems to have caught you off guard!",
                "You look surprised! Is everything okay?",
                "That seemed unexpected! How can I help?"
            ]
        }

    def recognize_emotion(self, facial_features, vocal_features, behavioral_features):
        """Recognize emotion from multiple modalities"""
        emotion_scores = {}

        # Analyze facial expressions
        face_emotion = self.analyze_facial_emotion(facial_features)
        for emotion, score in face_emotion.items():
            emotion_scores[emotion] = emotion_scores.get(emotion, 0) + score * 0.5

        # Analyze vocal patterns
        voice_emotion = self.analyze_vocal_emotion(vocal_features)
        for emotion, score in voice_emotion.items():
            emotion_scores[emotion] = emotion_scores.get(emotion, 0) + score * 0.3

        # Analyze behavioral patterns
        behavior_emotion = self.analyze_behavioral_emotion(behavioral_features)
        for emotion, score in behavior_emotion.items():
            emotion_scores[emotion] = emotion_scores.get(emotion, 0) + score * 0.2

        # Normalize scores
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            normalized_scores = {k: v/total_score for k, v in emotion_scores.items()}
        else:
            normalized_scores = {'neutral': 1.0}

        return normalized_scores

    def analyze_facial_emotion(self, features):
        """Analyze facial features for emotion"""
        # This would use computer vision to detect facial expressions
        # For this example, we'll return placeholder scores
        return {
            'happy': 0.1,
            'sad': 0.1,
            'angry': 0.1,
            'fearful': 0.1,
            'surprised': 0.1,
            'neutral': 0.5
        }

    def analyze_vocal_emotion(self, features):
        """Analyze vocal features for emotion"""
        # This would analyze pitch, volume, speech patterns
        # For this example, we'll return placeholder scores
        return {
            'happy': 0.1,
            'sad': 0.1,
            'angry': 0.1,
            'fearful': 0.1,
            'surprised': 0.1,
            'neutral': 0.5
        }

    def analyze_behavioral_emotion(self, features):
        """Analyze behavioral features for emotion"""
        # This would analyze posture, movement patterns, etc.
        # For this example, we'll return placeholder scores
        return {
            'happy': 0.1,
            'sad': 0.1,
            'angry': 0.1,
            'fearful': 0.1,
            'surprised': 0.1,
            'neutral': 0.5
        }

    def generate_empathetic_response(self, recognized_emotions):
        """Generate empathetic response based on recognized emotions"""
        # Find the most prominent emotion
        dominant_emotion = max(recognized_emotions.items(), key=lambda x: x[1])

        if dominant_emotion[1] > self.emotion_threshold:
            emotion = dominant_emotion[0]
            responses = self.empathy_responses.get(emotion, ["I notice you might be feeling something."])
            import random
            return random.choice(responses)
        else:
            return "You seem to be in a balanced mood. How can I assist you?"

    def adapt_behavior_to_emotion(self, user_emotion):
        """Adapt robot behavior based on user's emotion"""
        behavior_modifications = {
            'happy': {
                'speed': 'faster',
                'energy': 'higher',
                'tone': 'upbeat',
                'gestures': 'more_expressive'
            },
            'sad': {
                'speed': 'slower',
                'energy': 'softer',
                'tone': 'gentle',
                'gestures': 'restrained'
            },
            'angry': {
                'speed': 'calm',
                'energy': 'neutral',
                'tone': 'soothing',
                'gestures': 'non-threatening'
            },
            'fearful': {
                'speed': 'reassuring',
                'energy': 'calming',
                'tone': 'supportive',
                'gestures': 'protective'
            }
        }

        return behavior_modifications.get(user_emotion, {
            'speed': 'normal',
            'energy': 'normal',
            'tone': 'neutral',
            'gestures': 'standard'
        })
```

## ROS2 Implementation: Social Robotics System

Here's a comprehensive ROS2 implementation of social robotics principles:

```python
# social_robotics_system.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, Pose, Point
from std_msgs.msg import String, Bool, Float32
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
import numpy as np
import cv2
from collections import deque

class SocialRoboticsSystem(Node):
    def __init__(self):
        super().__init__('social_robotics_system')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.speech_pub = self.create_publisher(String, '/tts_input', 10)
        self.social_status_pub = self.create_publisher(String, '/social_status', 10)
        self.emotion_pub = self.create_publisher(String, '/recognized_emotion', 10)
        self.gesture_pub = self.create_publisher(String, '/recognized_gesture', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.voice_sub = self.create_subscription(
            String, '/speech_to_text', self.voice_callback, 10
        )

        # System components
        self.cv_bridge = CvBridge()
        self.social_cue_recognizer = SocialCueRecognizer()
        self.theory_of_mind = TheoryOfMind()
        self.proxemics_manager = ProxemicsManager()
        self.conversation_manager = ConversationManager()
        self.emotion_system = EmotionRecognitionSystem()

        # Data storage
        self.image_data = None
        self.scan_data = None
        self.voice_data = None
        self.face_positions = []
        self.human_distances = []

        # Social state
        self.social_engagement = 'idle'
        self.attention_focus = None
        self.engagement_history = deque(maxlen=50)
        self.user_models = {}

        # Social parameters
        self.engagement_timeout = 10.0  # seconds
        self.social_learning_enabled = True

        # Control loop
        self.social_timer = self.create_timer(0.1, self.social_control_loop)

    def image_callback(self, msg):
        """Handle camera image for social cue recognition"""
        try:
            self.image_data = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Image callback error: {e}')

    def scan_callback(self, msg):
        """Handle laser scan for proximity detection"""
        self.scan_data = msg

    def voice_callback(self, msg):
        """Handle voice input for conversation"""
        self.voice_data = msg.data

    def social_control_loop(self):
        """Main social control loop"""
        # Process social cues from all sensors
        social_cues = self.process_social_cues()

        # Update social state based on cues
        self.update_social_state(social_cues)

        # Generate appropriate social responses
        self.generate_social_response(social_cues)

        # Publish social status
        self.publish_social_status(social_cues)

    def process_social_cues(self):
        """Process all social cues from sensors"""
        cues = {}

        # Process visual cues (faces, gestures, eye contact)
        if self.image_data is not None:
            visual_cues = self.process_visual_cues()
            cues.update(visual_cues)

        # Process proximity cues
        if self.scan_data is not None:
            proximity_cues = self.process_proximity_cues()
            cues.update(proximity_cues)

        # Process voice cues
        if self.voice_data is not None:
            voice_cues = self.process_voice_cues()
            cues.update(voice_cues)
            self.voice_data = None  # Clear after processing

        return cues

    def process_visual_cues(self):
        """Process visual social cues"""
        cues = {}

        # Detect faces
        gray = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            # Get the largest face (closest person)
            largest_face = max(faces, key=lambda f: f[2] * f[3])  # width * height
            x, y, w, h = largest_face
            face_center = (x + w//2, y + h//2)
            image_center = (self.image_data.shape[1]//2, self.image_data.shape[0]//2)

            # Check for eye contact
            eye_contact = self.social_cue_recognizer.recognize_eye_contact(
                face_center, image_center, 0.8  # confidence placeholder
            )

            cues['face_detected'] = True
            cues['face_position'] = face_center
            cues['eye_contact'] = eye_contact
            cues['attention_level'] = 1.0 if eye_contact else 0.5

            # Store for user modeling
            self.face_positions.append(face_center)
        else:
            cues['face_detected'] = False
            cues['attention_level'] = 0.0

        return cues

    def process_proximity_cues(self):
        """Process proximity-based social cues"""
        cues = {}

        # Find closest human (assuming humans are detected as obstacles at human height)
        if self.scan_data:
            valid_ranges = [r for r in self.scan_data.ranges if 0 < r < self.scan_data.range_max]
            if valid_ranges:
                closest_distance = min(valid_ranges)
                cues['distance'] = closest_distance
                cues['proximity_zone'] = self.social_cue_recognizer.recognize_proximity(closest_distance)

                # Store for proxemics management
                self.human_distances.append(closest_distance)

                # Adjust position based on proxemics
                adjustment_type, adjustment_distance = self.proxemics_manager.suggest_distance_adjustment(
                    closest_distance, 'conversation'
                )

                if adjustment_type == 'move_back':
                    cmd = Twist()
                    cmd.linear.x = -0.1  # Move away gently
                    self.cmd_vel_pub.publish(cmd)
                elif adjustment_type == 'move_closer':
                    cmd = Twist()
                    cmd.linear.x = 0.05  # Move closer gently
                    self.cmd_vel_pub.publish(cmd)

        return cues

    def process_voice_cues(self):
        """Process voice-based social cues"""
        cues = {}

        if self.voice_data:
            # Simple analysis of voice input
            cues['speaking'] = True
            cues['voice_content'] = self.voice_data
            cues['conversation_initiated'] = True

            # Update conversation manager
            understanding_result = self.basic_nlp_understanding(self.voice_data)
            response = self.conversation_manager.manage_robot_turn(understanding_result)

            if response:
                self.speech_pub.publish(String(data=response))

        return cues

    def basic_nlp_understanding(self, text):
        """Basic NLP understanding for conversation"""
        text_lower = text.lower()
        result = {'intent': 'unknown', 'question_type': 'unknown'}

        # Simple intent detection
        if any(word in text_lower for word in ['hello', 'hi', 'hey']):
            result['intent'] = 'greeting'
        elif any(word in text_lower for word in ['how are you', 'how do you do']):
            result['intent'] = 'greeting_response'
        elif '?' in text:
            result['intent'] = 'question'
            if any(word in text_lower for word in ['what', 'where', 'when', 'who', 'why', 'how']):
                result['question_type'] = 'wh'
            elif any(word in text_lower for word in ['can', 'could', 'will', 'would', 'is', 'are', 'do', 'does']):
                result['question_type'] = 'yes_no'
        elif any(word in text_lower for word in ['please', 'can you', 'could you']):
            result['intent'] = 'request'
        else:
            result['intent'] = 'statement'

        return result

    def update_social_state(self, cues):
        """Update social state based on cues"""
        # Update engagement level
        attention_level = cues.get('attention_level', 0.0)
        proximity_zone = cues.get('proximity_zone', 'public')

        if attention_level > 0.5 and proximity_zone in ['personal', 'social']:
            self.social_engagement = 'engaged'
            self.attention_focus = cues.get('face_position')
        elif attention_level > 0.2:
            self.social_engagement = 'aware'
        else:
            self.social_engagement = 'idle'

        # Update engagement history
        self.engagement_history.append({
            'timestamp': self.get_clock().now(),
            'state': self.social_engagement,
            'cues': cues
        })

        # Update user model if face detected
        if cues.get('face_detected') and self.social_learning_enabled:
            face_pos = cues.get('face_position')
            user_id = self.identify_user(face_pos)
            self.update_user_model(user_id, cues)

    def identify_user(self, face_position):
        """Identify or assign user ID based on face position"""
        # Simple user identification based on face position
        # In practice, this would use face recognition
        if not self.user_models:
            user_id = 'user_0'
        else:
            # Find closest existing user model
            closest_user = min(self.user_models.keys(),
                             key=lambda u: np.linalg.norm(
                                 np.array(face_position) - np.array(self.user_models[u].get('last_seen_position', [0, 0])
                             ))
            if np.linalg.norm(np.array(face_position) - np.array(self.user_models[closest_user].get('last_seen_position', [0, 0]))) < 50:  # pixels
                user_id = closest_user
            else:
                user_id = f'user_{len(self.user_models)}'

        if user_id not in self.user_models:
            self.user_models[user_id] = {
                'first_seen': self.get_clock().now(),
                'interaction_count': 0,
                'preferences': {}
            }

        self.user_models[user_id]['last_seen_position'] = face_position
        self.user_models[user_id]['last_seen_time'] = self.get_clock().now()
        self.user_models[user_id]['interaction_count'] += 1

        return user_id

    def update_user_model(self, user_id, cues):
        """Update model of user based on interaction cues"""
        if user_id in self.user_models:
            # Update user preferences based on interaction
            if cues.get('proximity_zone') == 'intimate':
                self.user_models[user_id]['preferences']['close_interaction'] = True
            if cues.get('eye_contact'):
                self.user_models[user_id]['preferences']['direct_engagement'] = True

    def generate_social_response(self, cues):
        """Generate appropriate social response based on cues"""
        response_type = self.determine_response_type(cues)

        if response_type == 'greeting':
            self.generate_greeting_response(cues)
        elif response_type == 'acknowledgment':
            self.generate_acknowledgment_response(cues)
        elif response_type == 'empathy':
            self.generate_empathy_response(cues)
        elif response_type == 'engagement':
            self.generate_engagement_response(cues)

    def determine_response_type(self, cues):
        """Determine appropriate response type based on cues"""
        if cues.get('face_detected') and not cues.get('speaking'):
            if cues.get('eye_contact') and cues.get('attention_level', 0) > 0.7:
                return 'greeting'
            else:
                return 'acknowledgment'
        elif cues.get('speaking'):
            return 'engagement'
        else:
            return 'idle'

    def generate_greeting_response(self, cues):
        """Generate greeting response"""
        greeting_options = [
            "Hello! It's nice to meet you.",
            "Hi there! How can I help you today?",
            "Good day! I'm happy to see you.",
            "Hello! I've been waiting to meet you."
        ]

        import random
        greeting = random.choice(greeting_options)

        # Add gesture if appropriate
        cmd = Twist()
        cmd.angular.z = 0.2  # Gentle turn to face person
        self.cmd_vel_pub.publish(cmd)

        self.speech_pub.publish(String(data=greeting))

    def generate_acknowledgment_response(self, cues):
        """Generate acknowledgment response"""
        if cues.get('eye_contact'):
            acknowledgment = self.conversation_manager.generate_backchannel()
            self.speech_pub.publish(String(data=acknowledgment))

    def generate_empathy_response(self, cues):
        """Generate empathetic response based on emotional cues"""
        # This would integrate with emotion recognition system
        pass

    def generate_engagement_response(self, cues):
        """Generate engagement response to conversation"""
        # Handled by conversation manager
        pass

    def publish_social_status(self, cues):
        """Publish current social status"""
        status_msg = String()
        status_msg.data = (
            f"Engagement: {self.social_engagement}, "
            f"Users: {len(self.user_models)}, "
            f"Attention: {cues.get('attention_level', 0):.1f}, "
            f"Proximity: {cues.get('proximity_zone', 'unknown')}"
        )
        self.social_status_pub.publish(status_msg)

class SocialLearningSystem:
    """Learn and adapt social behaviors based on interaction"""
    def __init__(self):
        self.interaction_outcomes = []
        self.behavior_preferences = {}
        self.cultural_adaptation = None

    def record_interaction_outcome(self, interaction_data, outcome):
        """Record outcome of social interaction"""
        self.interaction_outcomes.append({
            'data': interaction_data,
            'outcome': outcome,
            'timestamp': Time()
        })

        # Update behavior preferences based on outcome
        self.update_behavior_preferences(interaction_data, outcome)

    def update_behavior_preferences(self, interaction_data, outcome):
        """Update preferences based on interaction outcome"""
        # Positive outcomes reinforce behaviors
        if outcome == 'positive':
            behavior = interaction_data.get('behavior', 'unknown')
            if behavior not in self.behavior_preferences:
                self.behavior_preferences[behavior] = 0
            self.behavior_preferences[behavior] += 1
        elif outcome == 'negative':
            behavior = interaction_data.get('behavior', 'unknown')
            if behavior in self.behavior_preferences:
                self.behavior_preferences[behavior] = max(0, self.behavior_preferences[behavior] - 1)

    def adapt_to_user_feedback(self, user_feedback):
        """Adapt behavior based on user feedback"""
        feedback_type = user_feedback.get('type', 'neutral')
        intensity = user_feedback.get('intensity', 0.5)

        if feedback_type == 'positive':
            # Increase use of successful behaviors
            pass
        elif feedback_type == 'negative':
            # Decrease use of unsuccessful behaviors
            pass
        elif feedback_type == 'correction':
            # Learn from corrections
            pass

    def get_adapted_behavior(self, context):
        """Get behavior adapted to current context and learned preferences"""
        # Select behavior based on learned preferences and current context
        preferred_behaviors = sorted(
            self.behavior_preferences.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return most preferred behavior that fits context
        for behavior, score in preferred_behaviors:
            if self.behavior_fits_context(behavior, context):
                return behavior

        return 'default_behavior'

    def behavior_fits_context(self, behavior, context):
        """Check if behavior fits current context"""
        # Implement context checking logic
        return True

class SocialNormsCompliance:
    """Ensure robot behavior complies with social norms"""
    def __init__(self):
        self.norms_database = self.load_social_norms()
        self.context_sensitivity = True

    def load_social_norms(self):
        """Load database of social norms"""
        return {
            'greeting_norms': {
                'formal': ['good morning', 'good afternoon', 'good evening'],
                'informal': ['hi', 'hello', 'hey']
            },
            'personal_space': {
                'professional': 1.2,
                'friendly': 0.9,
                'intimate': 0.4
            },
            'cultural_norms': {
                'japanese': {'bow_greeting': True, 'direct_eye_contact': False},
                'middle_eastern': {'respectful_distance': True, 'formal_greeting': True}
            }
        }

    def evaluate_behavior_appropriateness(self, behavior, context):
        """Evaluate if behavior is appropriate for context"""
        # Check against social norms database
        is_appropriate = True
        violations = []

        # Example: check personal space violation
        if (behavior.get('action') == 'approach' and
            context.get('distance') < self.norms_database['personal_space'].get(context.get('relationship', 'friendly'), 0.9)):
            is_appropriate = False
            violations.append('personal_space_violation')

        return is_appropriate, violations

    def adjust_behavior_for_norms(self, behavior, context):
        """Adjust behavior to comply with social norms"""
        adjusted_behavior = behavior.copy()

        # Apply cultural adaptations
        if context.get('cultural_context'):
            cultural_norms = self.norms_database['cultural_norms'].get(context['cultural_context'], {})
            for norm, value in cultural_norms.items():
                if norm == 'bow_greeting' and value:
                    adjusted_behavior['greeting_style'] = 'bow'
                elif norm == 'direct_eye_contact' and not value:
                    adjusted_behavior['eye_contact_duration'] = 'moderate'

        return adjusted_behavior

def main(args=None):
    rclpy.init(args=args)
    social_system = SocialRoboticsSystem()

    try:
        rclpy.spin(social_system)
    except KeyboardInterrupt:
        pass
    finally:
        social_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Social Robotics Concepts

### Social Presence and Engagement

Creating a sense of social presence in robots:

```python
# Example: Social presence and engagement system
class SocialPresenceSystem:
    def __init__(self):
        self.presence_indicators = {
            'attentiveness': 0.0,
            'responsiveness': 0.0,
            'expressiveness': 0.0,
            'warmth': 0.0
        }
        self.engagement_strategies = {
            'gaze_shifting': self.gaze_shifting_behavior,
            'micro_expressions': self.micro_expression_behavior,
            'rhythmic_movement': self.rhythmic_movement_behavior,
            'vocal_variations': self.vocal_variation_behavior
        }

    def update_presence_indicators(self, interaction_data):
        """Update presence indicators based on interaction"""
        # Update attentiveness based on attention tracking
        self.presence_indicators['attentiveness'] = self.calculate_attentiveness(interaction_data)

        # Update responsiveness based on response times
        self.presence_indicators['responsiveness'] = self.calculate_responsiveness(interaction_data)

        # Update expressiveness based on facial/body expressions
        self.presence_indicators['expressiveness'] = self.calculate_expressiveness(interaction_data)

        # Update warmth based on social cues
        self.presence_indicators['warmth'] = self.calculate_warmth(interaction_data)

    def calculate_attentiveness(self, data):
        """Calculate attentiveness score"""
        # Based on eye contact, response to name, etc.
        score = 0.5  # base score
        if data.get('eye_contact', False):
            score += 0.3
        if data.get('response_to_attention', False):
            score += 0.2
        return min(1.0, score)

    def calculate_responsiveness(self, data):
        """Calculate responsiveness score"""
        # Based on response time and quality
        response_time = data.get('response_time', 2.0)  # seconds
        if response_time < 1.0:
            return 0.9
        elif response_time < 2.0:
            return 0.7
        elif response_time < 3.0:
            return 0.5
        else:
            return 0.2

    def calculate_expressiveness(self, data):
        """Calculate expressiveness score"""
        # Based on facial expressions, gestures, etc.
        return 0.6  # placeholder

    def calculate_warmth(self, data):
        """Calculate warmth score"""
        # Based on tone, friendliness, etc.
        return 0.7  # placeholder

    def generate_engaging_behavior(self):
        """Generate behavior that enhances social presence"""
        behaviors = []

        # Add gaze shifting to appear more lifelike
        behaviors.append(self.engagement_strategies['gaze_shifting']())

        # Add micro-expressions for expressiveness
        behaviors.append(self.engagement_strategies['micro_expressions']())

        # Add rhythmic movement to appear more natural
        behaviors.append(self.engagement_strategies['rhythmic_movement']())

        return behaviors

    def gaze_shifting_behavior(self):
        """Generate natural gaze shifting pattern"""
        # Simulate human-like gaze patterns
        gaze_targets = ['face', 'hands', 'objects', 'environment']
        import random
        target = random.choice(gaze_targets)
        duration = random.uniform(0.5, 2.0)
        return {'type': 'gaze_shift', 'target': target, 'duration': duration}

    def micro_expression_behavior(self):
        """Generate subtle facial expressions"""
        expressions = ['slight_smile', 'eyebrow_raise', 'head_tilt', 'eye_widen']
        import random
        expr = random.choice(expressions)
        intensity = random.uniform(0.1, 0.3)
        return {'type': 'micro_expression', 'expression': expr, 'intensity': intensity}

    def rhythmic_movement_behavior(self):
        """Generate subtle rhythmic movements"""
        movements = [
            {'type': 'breathing', 'rate': 0.2, 'amplitude': 0.01},
            {'type': 'posture_shift', 'rate': 0.05, 'amplitude': 0.02},
            {'type': 'head_micromovement', 'rate': 0.1, 'amplitude': 0.005}
        ]
        return movements

    def vocal_variation_behavior(self):
        """Generate vocal variations for natural speech"""
        variations = {
            'pitch_modulation': 0.1,  # 10% variation
            'tempo_changes': 0.05,    # 5% tempo change
            'pause_patterns': [0.2, 0.5, 1.0]  # Common pause durations
        }
        return variations
```

### Cultural Adaptation in Social Robots

Adapting social behaviors to different cultural contexts:

```python
# Example: Cultural adaptation system
class CulturalAdaptationSystem:
    def __init__(self):
        self.cultural_profiles = {
            'individualistic': {
                'personal_space': 1.2,
                'eye_contact_norms': 'high',
                'greeting_style': 'handshake',
                'communication_style': 'direct'
            },
            'collectivistic': {
                'personal_space': 0.9,
                'eye_contact_norms': 'moderate',
                'greeting_style': 'bow',
                'communication_style': 'indirect'
            },
            'high_context': {
                'nonverbal_cues': 'very_important',
                'silence_tolerance': 'high',
                'relationship_building': 'essential'
            },
            'low_context': {
                'verbal_clarity': 'very_important',
                'direct_communication': 'preferred',
                'task_focus': 'priority'
            }
        }
        self.user_cultural_model = {}
        self.adaptation_history = []

    def detect_cultural_background(self, user_data):
        """Detect user's cultural background from interaction patterns"""
        # Analyze interaction patterns, communication style, etc.
        detected_culture = 'individualistic'  # placeholder
        confidence = 0.7  # placeholder

        return detected_culture, confidence

    def adapt_behavior_to_culture(self, user_id, cultural_profile):
        """Adapt robot behavior to match cultural expectations"""
        if user_id not in self.user_cultural_model:
            self.user_cultural_model[user_id] = {}

        # Adjust personal space
        if 'personal_space' in cultural_profile:
            self.update_personal_space(cultural_profile['personal_space'])

        # Adjust eye contact behavior
        if 'eye_contact_norms' in cultural_profile:
            self.update_eye_contact_behavior(cultural_profile['eye_contact_norms'])

        # Adjust greeting style
        if 'greeting_style' in cultural_profile:
            self.update_greeting_style(cultural_profile['greeting_style'])

        # Adjust communication style
        if 'communication_style' in cultural_profile:
            self.update_communication_style(cultural_profile['communication_style'])

        # Record adaptation
        self.adaptation_history.append({
            'user_id': user_id,
            'cultural_profile': cultural_profile,
            'timestamp': Time()
        })

    def update_personal_space(self, distance):
        """Update preferred personal space distance"""
        # This would interface with proxemics manager
        pass

    def update_eye_contact_behavior(self, norms):
        """Update eye contact behavior based on cultural norms"""
        if norms == 'high':
            self.eye_contact_duration = 3.0  # seconds
            self.eye_contact_frequency = 0.8  # 80% of interaction time
        elif norms == 'moderate':
            self.eye_contact_duration = 1.5
            self.eye_contact_frequency = 0.5
        else:  # low
            self.eye_contact_duration = 0.5
            self.eye_contact_frequency = 0.2

    def update_greeting_style(self, style):
        """Update greeting style"""
        greeting_styles = {
            'handshake': {'gesture': 'handshake', 'distance': 1.0},
            'bow': {'gesture': 'bow', 'distance': 1.5},
            'wave': {'gesture': 'wave', 'distance': 2.0}
        }
        self.current_greeting_style = greeting_styles.get(style, greeting_styles['wave'])

    def update_communication_style(self, style):
        """Update communication style"""
        if style == 'direct':
            self.communication_patterns = {
                'clarity': 'high',
                'indirectness': 'low',
                'context_dependency': 'low'
            }
        else:  # indirect
            self.communication_patterns = {
                'clarity': 'moderate',
                'indirectness': 'high',
                'context_dependency': 'high'
            }

    def learn_from_cultural_feedback(self, user_id, feedback):
        """Learn from cultural feedback to improve adaptation"""
        # Adjust cultural profile based on user feedback
        if user_id in self.user_cultural_model:
            current_profile = self.user_cultural_model[user_id]
            # Update profile based on feedback
            pass
```

## Lab: Implementing Social Robot Behaviors

In this lab, you'll implement social robot behaviors:

```python
# lab_social_robotics.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import numpy as np
import cv2

class SocialRoboticsLab(Node):
    def __init__(self):
        super().__init__('social_robotics_lab')

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.speech_pub = self.create_publisher(String, '/tts_input', 10)
        self.social_pub = self.create_publisher(String, '/social_behavior', 10)
        self.status_pub = self.create_publisher(String, '/social_status', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.voice_sub = self.create_subscription(
            String, '/speech_commands', self.voice_callback, 10
        )

        # System components
        self.cv_bridge = CvBridge()
        self.image_data = None
        self.scan_data = None
        self.voice_command = None

        # Social state
        self.social_engagement = 'idle'
        self.human_detected = False
        self.human_distance = float('inf')
        self.social_behavior = 'passive'
        self.interaction_history = []

        # Face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Control loop
        self.control_timer = self.create_timer(0.1, self.social_control_loop)

    def image_callback(self, msg):
        """Handle camera image for social interaction"""
        try:
            self.image_data = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Image callback error: {e}')

    def scan_callback(self, msg):
        """Handle laser scan for proximity detection"""
        self.scan_data = msg

    def voice_callback(self, msg):
        """Handle voice commands"""
        self.voice_command = msg.data

    def social_control_loop(self):
        """Main social control loop"""
        # Process social cues
        cues = self.process_social_cues()

        # Update social state
        self.update_social_state(cues)

        # Generate appropriate social behavior
        self.generate_social_behavior(cues)

        # Publish status
        self.publish_status()

        # Clear processed commands
        if self.voice_command:
            self.voice_command = None

    def process_social_cues(self):
        """Process social cues from sensors"""
        cues = {}

        # Process visual cues (face detection)
        if self.image_data is not None:
            gray = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                # Human detected
                cues['human_detected'] = True
                x, y, w, h = faces[0]  # Use first detected face
                cues['face_position'] = (x + w//2, y + h//2)  # Center of face
                cues['face_size'] = w * h  # Size indicates distance
            else:
                cues['human_detected'] = False

        # Process proximity cues
        if self.scan_data is not None:
            valid_ranges = [r for r in self.scan_data.ranges if 0 < r < self.scan_data.range_max]
            if valid_ranges:
                closest_distance = min(valid_ranges)
                cues['closest_distance'] = closest_distance

        return cues

    def update_social_state(self, cues):
        """Update social state based on cues"""
        # Update human detection
        self.human_detected = cues.get('human_detected', False)

        # Update distance
        if 'closest_distance' in cues:
            self.human_distance = cues['closest_distance']

        # Determine engagement level
        if self.human_detected and self.human_distance < 2.0:  # Within 2 meters
            if self.human_distance < 1.0:  # Very close
                self.social_engagement = 'engaged'
            else:  # At social distance
                self.social_engagement = 'aware'
        else:
            self.social_engagement = 'idle'

        # Update social behavior based on engagement
        if self.social_engagement == 'engaged':
            self.social_behavior = 'interactive'
        elif self.social_engagement == 'aware':
            self.social_behavior = 'attentive'
        else:
            self.social_behavior = 'passive'

    def generate_social_behavior(self, cues):
        """Generate appropriate social behavior"""
        if self.social_behavior == 'interactive':
            self.perform_interactive_behavior(cues)
        elif self.social_behavior == 'attentive':
            self.perform_attentive_behavior(cues)
        elif self.social_behavior == 'passive':
            self.perform_passive_behavior(cues)

    def perform_interactive_behavior(self, cues):
        """Perform interactive social behavior"""
        # Face the person
        if 'face_position' in cues:
            face_x, face_y = cues['face_position']
            image_width = self.image_data.shape[1] if self.image_data is not None else 640
            center_x = image_width // 2

            # Calculate direction to turn
            error = face_x - center_x
            turn_speed = 0.002 * error  # Proportional control

            cmd = Twist()
            cmd.angular.z = max(-0.5, min(0.5, turn_speed))
            self.cmd_pub.publish(cmd)

        # Greet if not recently greeted
        if not self.recently_greeted():
            greeting = "Hello! I'm happy to see you."
            self.speech_pub.publish(String(data=greeting))
            self.log_interaction('greeting')

    def perform_attentive_behavior(self, cues):
        """Perform attentive social behavior"""
        # Gentle movement to show attention
        cmd = Twist()
        cmd.angular.z = 0.1 * np.sin(self.get_clock().now().nanoseconds / 1e9)  # Slow oscillation
        self.cmd_pub.publish(cmd)

        # Periodic acknowledgment
        if self.should_acknowledge():
            acknowledgment = "I see you there."
            self.speech_pub.publish(String(data=acknowledgment))
            self.log_interaction('acknowledgment')

    def perform_passive_behavior(self, cues):
        """Perform passive social behavior"""
        # Stop moving
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

        # Remain quiet unless spoken to
        if self.voice_command and 'hello' in self.voice_command.lower():
            response = "Hello! How can I help you?"
            self.speech_pub.publish(String(data=response))
            self.log_interaction('response_to_greeting')

    def recently_greeted(self):
        """Check if recently greeted"""
        recent_greetings = [i for i in self.interaction_history
                          if i['type'] == 'greeting'
                          and (self.get_clock().now().nanoseconds - i['timestamp'].nanoseconds) < 30e9]  # 30 seconds
        return len(recent_greetings) > 0

    def should_acknowledge(self):
        """Check if should acknowledge presence"""
        import time
        # Acknowledge every 10 seconds if engaged
        if not hasattr(self, 'last_acknowledgment'):
            self.last_acknowledgment = time.time()

        if time.time() - self.last_acknowledgment > 10:
            self.last_acknowledgment = time.time()
            return True
        return False

    def log_interaction(self, interaction_type):
        """Log social interaction"""
        self.interaction_history.append({
            'type': interaction_type,
            'timestamp': self.get_clock().now(),
            'engagement': self.social_engagement
        })

        # Keep history manageable
        if len(self.interaction_history) > 20:
            self.interaction_history = self.interaction_history[-20:]

    def publish_status(self):
        """Publish social status"""
        status_msg = String()
        status_msg.data = (
            f"Engagement: {self.social_engagement}, "
            f"Behavior: {self.social_behavior}, "
            f"Human: {self.human_detected}, "
            f"Distance: {self.human_distance:.2f}m, "
            f"Interactions: {len(self.interaction_history)}"
        )
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    lab = SocialRoboticsLab()

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

## Exercise: Design Your Own Social Robot

Consider the following design challenge:

1. What social context will your robot operate in (home, office, hospital, etc.)?
2. What types of social interactions are most important in that context?
3. How will your robot recognize and respond to social cues?
4. What cultural considerations need to be addressed?
5. How will the robot maintain appropriate social distance and personal space?
6. How will the robot adapt its behavior based on user feedback?
7. What measures will ensure the robot's behavior remains socially appropriate?

## Summary

Social robotics involves creating robots that can interact naturally and appropriately with humans in social contexts. Key concepts include:

- **Social Cues Recognition**: Understanding eye contact, gestures, facial expressions, and proximity
- **Theory of Mind**: Modeling human beliefs, desires, and intentions
- **Proxemics**: Managing personal space and social distance appropriately
- **Conversation Management**: Handling turn-taking and natural dialogue flow
- **Emotional Intelligence**: Recognizing and responding to human emotions
- **Cultural Adaptation**: Adjusting behavior for different cultural contexts
- **Social Learning**: Adapting behavior based on interaction outcomes

The integration of these social robotics principles in ROS2 enables the development of robots that can interact naturally and effectively with humans. Understanding these concepts is crucial for developing robots that can be accepted and useful in social environments.

In the next lesson, we'll explore ethical considerations in human-robot interaction and the responsible development of social robotics systems.