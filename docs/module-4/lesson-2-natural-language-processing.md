---
sidebar_position: 2
---

# Natural Language Processing for Robot Interaction

## Introduction

Natural Language Processing (NLP) enables robots to understand and respond to human language, making interaction more intuitive and natural. In Physical AI systems, NLP must handle real-world contexts, spatial references, and embodied actions. This lesson explores NLP techniques specifically designed for robot interaction, including speech recognition, language understanding, and conversational interfaces.

## Fundamentals of Robot NLP

### Key Challenges in Robot NLP

Robot NLP faces unique challenges compared to traditional NLP applications:

1. **Embodied Context**: Language is grounded in physical environment and robot capabilities
2. **Real-time Processing**: Responses must be generated quickly for natural interaction
3. **Multimodal Integration**: Language must be integrated with perception and action
4. **Spatial References**: Understanding spatial language like "over there" or "to the left"
5. **Actionable Understanding**: Converting language into executable robot commands

```python
# Example: Robot NLP architecture
class RobotNLPSystem:
    def __init__(self):
        self.speech_recognizer = SpeechRecognizer()
        self.language_understanding = LanguageUnderstanding()
        self.spatial_reasoner = SpatialReasoner()
        self.action_generator = ActionGenerator()
        self.response_generator = ResponseGenerator()

    def process_utterance(self, audio_input):
        """Process an utterance from speech to action"""
        # 1. Speech recognition
        text = self.speech_recognizer.recognize(audio_input)

        # 2. Language understanding
        intent, entities = self.language_understanding.parse(text)

        # 3. Spatial reasoning
        resolved_entities = self.spatial_reasoner.resolve_spatial_references(
            entities, self.get_robot_context()
        )

        # 4. Action generation
        action = self.action_generator.generate_action(intent, resolved_entities)

        # 5. Response generation
        response = self.response_generator.generate_response(action)

        return {
            'text': text,
            'intent': intent,
            'entities': resolved_entities,
            'action': action,
            'response': response
        }
```

### Spatial Language Understanding

Robots must understand spatial references that are relative to their current context:

```python
# Example: Spatial reference resolution
class SpatialReasoner:
    def __init__(self):
        self.robot_pose = [0, 0, 0]  # x, y, theta
        self.object_map = {}  # Object positions in world frame
        self.spatial_keywords = {
            'left': -1, 'right': 1,
            'front': 1, 'back': -1,
            'near': 1, 'far': 10,
            'here': 0, 'there': 5
        }

    def resolve_spatial_references(self, entities, context):
        """Resolve spatial references in entities using robot context"""
        resolved_entities = {}

        for entity_type, entity_value in entities.items():
            if entity_type in ['location', 'object', 'direction']:
                resolved_entities[entity_type] = self.resolve_entity(
                    entity_value, context
                )
            else:
                resolved_entities[entity_type] = entity_value

        return resolved_entities

    def resolve_entity(self, entity_value, context):
        """Resolve a single entity with spatial context"""
        if isinstance(entity_value, str):
            # Handle spatial keywords
            if entity_value in self.spatial_keywords:
                return self.resolve_spatial_keyword(entity_value, context)
            elif 'relative' in entity_value:
                return self.resolve_relative_reference(entity_value, context)
            else:
                # Look up in object map
                return self.object_map.get(entity_value, entity_value)
        else:
            return entity_value

    def resolve_spatial_keyword(self, keyword, context):
        """Resolve spatial keyword to coordinates"""
        if keyword in ['left', 'right']:
            # Calculate position to left/right of robot
            angle_offset = np.pi/2 if keyword == 'left' else -np.pi/2
            target_angle = self.robot_pose[2] + angle_offset
            distance = 1.0  # Default distance
            x = self.robot_pose[0] + distance * np.cos(target_angle)
            y = self.robot_pose[1] + distance * np.sin(target_angle)
            return [x, y]
        elif keyword in ['front', 'back']:
            # Calculate position in front/behind robot
            distance = 1.0 if keyword == 'front' else -1.0
            x = self.robot_pose[0] + distance * np.cos(self.robot_pose[2])
            y = self.robot_pose[1] + distance * np.sin(self.robot_pose[2])
            return [x, y]
        else:
            return self.robot_pose[:2]  # Default to current position

    def resolve_relative_reference(self, reference, context):
        """Resolve relative spatial reference"""
        # Parse reference like "the object to my left"
        # This would involve more complex parsing
        return self.robot_pose[:2]
```

## Speech Recognition for Robotics

### Real-time Speech Recognition

Robots need real-time speech recognition that can handle environmental noise and interruptions:

```python
# Example: Real-time speech recognition for robotics
import speech_recognition as sr
import pyaudio
import threading
import queue

class RealTimeSpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.listening = False
        self.energy_threshold = 300  # Adjust for ambient noise

        # Configure recognizer
        self.recognizer.energy_threshold = self.energy_threshold
        self.recognizer.dynamic_energy_threshold = True

    def start_listening(self):
        """Start real-time listening"""
        self.listening = True
        self.recognizer.listen_in_background(
            self.microphone, self.audio_callback, phrase_time_limit=5
        )

    def audio_callback(self, recognizer, audio):
        """Callback for audio input"""
        if self.listening:
            self.audio_queue.put(audio)

    def process_audio(self):
        """Process audio and return recognized text"""
        try:
            audio = self.audio_queue.get(timeout=0.1)
            # Use Google Speech Recognition or other service
            try:
                text = self.recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                return None
            except sr.RequestError as e:
                print(f"Speech recognition error: {e}")
                return None
        except queue.Empty:
            return None

    def stop_listening(self):
        """Stop listening"""
        self.listening = False

# Example: Keyword spotting for robot activation
class KeywordSpotter:
    def __init__(self, keywords=['robot', 'hey robot', 'attention']):
        self.keywords = keywords
        self.activation_keywords = keywords

    def detect_keyword(self, text):
        """Detect if text contains activation keyword"""
        text_lower = text.lower()
        for keyword in self.activation_keywords:
            if keyword in text_lower:
                return True, keyword
        return False, None
```

### Noise-Robust Recognition

Robots operate in noisy environments, requiring robust speech recognition:

```python
# Example: Noise-robust speech processing
import numpy as np
from scipy import signal

class NoiseRobustRecognizer:
    def __init__(self):
        self.noise_profile = None
        self.is_calibrated = False

    def calibrate_noise_profile(self, noise_audio):
        """Calibrate noise profile from background noise"""
        # Extract noise characteristics
        self.noise_profile = self.extract_noise_features(noise_audio)
        self.is_calibrated = True

    def extract_noise_features(self, audio_data):
        """Extract features for noise characterization"""
        # Use spectral subtraction or other noise reduction techniques
        # This is a simplified example
        return np.mean(np.abs(audio_data))

    def denoise_audio(self, audio_data):
        """Apply noise reduction to audio"""
        if not self.is_calibrated:
            return audio_data

        # Simple spectral subtraction (in practice, use more sophisticated methods)
        # This is a placeholder implementation
        return audio_data

    def preprocess_for_recognition(self, audio_data):
        """Preprocess audio for better recognition"""
        # Denoise
        denoised = self.denoise_audio(audio_data)

        # Apply voice activity detection
        vad_result = self.voice_activity_detection(denoised)

        # Return only speech segments
        return vad_result

    def voice_activity_detection(self, audio_data):
        """Detect voice activity in audio"""
        # Simple energy-based VAD
        frame_size = 1024
        energy_threshold = 0.01

        frames = [audio_data[i:i+frame_size] for i in range(0, len(audio_data), frame_size)]
        speech_frames = []

        for frame in frames:
            energy = np.mean(np.abs(frame))
            if energy > energy_threshold:
                speech_frames.extend(frame)

        return np.array(speech_frames)
```

## Language Understanding and Intent Recognition

### Intent Classification

Classifying user intents for robot action:

```python
# Example: Intent classification for robot commands
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

class IntentClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True, stop_words='english')),
            ('classifier', MultinomialNB())
        ])
        self.is_trained = False
        self.intent_labels = {
            0: 'move_forward',
            1: 'move_backward',
            2: 'turn_left',
            3: 'turn_right',
            4: 'stop',
            5: 'navigate_to',
            6: 'pick_up',
            7: 'put_down',
            8: 'greet',
            9: 'follow',
            10: 'find_object',
            11: 'bring_object'
        }
        self.training_data = []
        self.training_labels = []

    def add_training_example(self, text, intent):
        """Add training example for intent classification"""
        self.training_data.append(text)
        self.training_labels.append(intent)

    def train_classifier(self):
        """Train the intent classifier"""
        if len(self.training_data) < 10:
            print("Not enough training data")
            return False

        # Convert string labels to numeric labels
        label_to_num = {v: k for k, v in self.intent_labels.items()}
        numeric_labels = [label_to_num[label] for label in self.training_labels]

        self.pipeline.fit(self.training_data, numeric_labels)
        self.is_trained = True
        return True

    def classify_intent(self, text):
        """Classify intent from text"""
        if not self.is_trained:
            return 'unknown', 0.0

        prediction = self.pipeline.predict([text])[0]
        probabilities = self.pipeline.predict_proba([text])[0]
        confidence = max(probabilities)

        intent = self.intent_labels.get(prediction, 'unknown')
        return intent, confidence

    def get_training_examples(self):
        """Get example training phrases for each intent"""
        examples = {
            'move_forward': [
                'move forward', 'go forward', 'move ahead', 'go straight',
                'move straight', 'go straight ahead', 'move to the front'
            ],
            'move_backward': [
                'move backward', 'go backward', 'move back', 'go back',
                'reverse', 'move backwards', 'go backwards'
            ],
            'turn_left': [
                'turn left', 'go left', 'turn to the left', 'make a left',
                'pivot left', 'rotate left', 'turn leftward'
            ],
            'turn_right': [
                'turn right', 'go right', 'turn to the right', 'make a right',
                'pivot right', 'rotate right', 'turn rightward'
            ],
            'stop': [
                'stop', 'halt', 'freeze', 'stand still', 'don\'t move',
                'cease movement', 'pause'
            ],
            'navigate_to': [
                'go to the kitchen', 'navigate to the bedroom', 'move to the office',
                'go to', 'navigate to', 'move to', 'go over to', 'head to'
            ],
            'pick_up': [
                'pick up the cup', 'grab the book', 'take the object',
                'pick up', 'grab', 'take', 'lift', 'collect'
            ],
            'put_down': [
                'put down the cup', 'place the book', 'set down',
                'put down', 'place', 'set', 'drop', 'release'
            ],
            'greet': [
                'hello', 'hi', 'hey', 'greetings', 'good morning',
                'good afternoon', 'good evening', 'how are you'
            ],
            'follow': [
                'follow me', 'come with me', 'accompany me', 'follow',
                'come after me', 'walk with me', 'accompany'
            ],
            'find_object': [
                'find the cup', 'locate the book', 'where is the phone',
                'find', 'locate', 'where is', 'search for', 'look for'
            ],
            'bring_object': [
                'bring me the cup', 'get me the book', 'bring the phone',
                'bring', 'get', 'fetch', 'carry', 'deliver'
            ]
        }
        return examples

# Example: Advanced intent classification with context
class ContextualIntentClassifier:
    def __init__(self):
        self.base_classifier = IntentClassifier()
        self.context_aware_rules = self.define_context_rules()

    def define_context_rules(self):
        """Define rules for context-aware intent classification"""
        return {
            'in_kitchen': {
                'cup': 'fetch_object',
                'water': 'fetch_object',
                'food': 'fetch_object'
            },
            'in_bedroom': {
                'pillow': 'fetch_object',
                'blanket': 'fetch_object',
                'book': 'fetch_object'
            }
        }

    def classify_with_context(self, text, context):
        """Classify intent considering current context"""
        base_intent, confidence = self.base_classifier.classify_intent(text)

        # Apply context-aware rules
        if context and 'location' in context:
            location = context['location']
            if location in self.context_aware_rules:
                # Check if text contains location-specific objects
                for obj, action in self.context_aware_rules[location].items():
                    if obj in text.lower():
                        return action, max(confidence, 0.8)  # High confidence for context match

        return base_intent, confidence
```

### Named Entity Recognition

Identifying objects, locations, and other entities in user commands:

```python
# Example: Named Entity Recognition for robotics
class EntityRecognizer:
    def __init__(self):
        self.object_entities = [
            'cup', 'bottle', 'book', 'phone', 'keys', 'wallet', 'toy',
            'plate', 'fork', 'spoon', 'knife', 'glass', 'bowl', 'box',
            'chair', 'table', 'sofa', 'bed', 'lamp', 'computer'
        ]
        self.location_entities = [
            'kitchen', 'bedroom', 'living room', 'bathroom', 'office',
            'dining room', 'hallway', 'garage', 'garden', 'entrance'
        ]
        self.person_entities = [
            'me', 'you', 'him', 'her', 'them', 'person', 'man', 'woman',
            'child', 'mom', 'dad', 'son', 'daughter', 'friend'
        ]
        self.action_entities = [
            'pick up', 'put down', 'bring', 'take', 'grab', 'lift',
            'carry', 'deliver', 'fetch', 'get', 'move', 'go to'
        ]

    def extract_entities(self, text):
        """Extract named entities from text"""
        entities = {
            'objects': [],
            'locations': [],
            'people': [],
            'actions': [],
            'quantities': [],
            'spatial': []
        }

        text_lower = text.lower()

        # Extract object entities
        for obj in self.object_entities:
            if obj in text_lower:
                entities['objects'].append(obj)

        # Extract location entities
        for loc in self.location_entities:
            if loc in text_lower:
                entities['locations'].append(loc)

        # Extract person entities
        for person in self.person_entities:
            if person in text_lower:
                entities['people'].append(person)

        # Extract action entities (multi-word)
        for action in self.action_entities:
            if action in text_lower:
                entities['actions'].append(action)

        # Extract quantities and spatial references
        entities['quantities'] = self.extract_quantities(text)
        entities['spatial'] = self.extract_spatial_references(text)

        return entities

    def extract_quantities(self, text):
        """Extract quantity information"""
        quantities = []
        words = text.lower().split()

        number_words = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }

        for i, word in enumerate(words):
            if word in number_words:
                quantities.append(number_words[word])
            elif word.isdigit():
                quantities.append(int(word))

        return quantities

    def extract_spatial_references(self, text):
        """Extract spatial references"""
        spatial_refs = []
        words = text.lower().split()

        spatial_keywords = [
            'left', 'right', 'front', 'back', 'behind', 'in front of',
            'next to', 'near', 'far', 'close to', 'away from', 'between'
        ]

        for word in words:
            if word in spatial_keywords:
                spatial_refs.append(word)

        return spatial_refs

    def resolve_coreferences(self, entities, context):
        """Resolve coreferences like 'it', 'that', 'there'"""
        resolved_entities = entities.copy()

        # This would implement more sophisticated coreference resolution
        # based on context and previous conversation
        return resolved_entities
```

## ROS2 Implementation: NLP for Robot Interaction

Here's a comprehensive ROS2 implementation of NLP for robot interaction:

```python
# nlp_robot_interaction.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import AudioData
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import String, Bool, Float32
from builtin_interfaces.msg import Time
from tf2_ros import TransformListener, Buffer
import numpy as np
import speech_recognition as sr
import threading
import queue

class NLPRobotInterface(Node):
    def __init__(self):
        super().__init__('nlp_robot_interface')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.speech_pub = self.create_publisher(String, '/tts_input', 10)
        self.nlp_status_pub = self.create_publisher(String, '/nlp_status', 10)
        self.intent_pub = self.create_publisher(String, '/recognized_intent', 10)

        # Subscribers
        self.audio_sub = self.create_subscription(
            AudioData, '/audio', self.audio_callback, 10
        )

        # TF listener for spatial context
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # NLP components
        self.speech_recognizer = RealTimeSpeechRecognizer()
        self.intent_classifier = IntentClassifier()
        self.entity_recognizer = EntityRecognizer()
        self.spatial_reasoner = SpatialReasoner()
        self.action_generator = ActionGenerator()
        self.response_generator = ResponseGenerator()

        # Data storage
        self.audio_buffer = queue.Queue()
        self.current_text = ""
        self.current_intent = "unknown"
        self.current_entities = {}
        self.robot_pose = None

        # NLP state
        self.nlp_enabled = True
        self.listening_state = "inactive"  # inactive, listening, processing
        self.conversation_context = []

        # Initialize intent classifier with training data
        self.initialize_intent_classifier()

        # Control parameters
        self.processing_frequency = 10.0  # Hz
        self.silence_timeout = 2.0  # seconds

        # Timers
        self.processing_timer = self.create_timer(1.0/self.processing_frequency, self.process_speech)

    def initialize_intent_classifier(self):
        """Initialize intent classifier with training data"""
        examples = self.intent_classifier.get_training_examples()

        for intent, phrases in examples.items():
            for phrase in phrases:
                self.intent_classifier.add_training_example(phrase, intent)

        self.intent_classifier.train_classifier()

    def audio_callback(self, msg):
        """Handle audio input"""
        if self.nlp_enabled and self.listening_state == "inactive":
            self.audio_buffer.put(msg.data)
            self.listening_state = "listening"

    def process_speech(self):
        """Process speech input and generate responses"""
        if not self.audio_buffer.empty() and self.listening_state == "listening":
            try:
                audio_data = self.audio_buffer.get_nowait()
                self.listening_state = "processing"

                # Process audio to text
                text = self.speech_to_text(audio_data)

                if text:
                    self.current_text = text
                    self.process_text(text)

                self.listening_state = "inactive"
            except queue.Empty:
                pass

        # Publish status
        self.publish_nlp_status()

    def speech_to_text(self, audio_data):
        """Convert audio to text"""
        # This is a simplified version - in practice, you'd use proper audio processing
        # For this example, we'll return a placeholder
        return "move forward"  # Placeholder - implement real STT

    def process_text(self, text):
        """Process text input through NLP pipeline"""
        self.get_logger().info(f"Processing text: {text}")

        # 1. Intent classification
        intent, confidence = self.intent_classifier.classify_intent(text)
        self.current_intent = intent

        # 2. Entity recognition
        entities = self.entity_recognizer.extract_entities(text)
        self.current_entities = entities

        # 3. Spatial reasoning
        context = self.get_robot_context()
        resolved_entities = self.spatial_reasoner.resolve_spatial_references(
            entities, context
        )

        # 4. Action generation
        action = self.action_generator.generate_action(intent, resolved_entities)

        # 5. Execute action
        self.execute_action(action)

        # 6. Generate response
        response = self.response_generator.generate_response(action, intent)
        self.speech_pub.publish(String(data=response))

        # 7. Update conversation context
        self.conversation_context.append({
            'text': text,
            'intent': intent,
            'entities': resolved_entities,
            'action': action,
            'response': response,
            'timestamp': self.get_clock().now()
        })

        # Publish intent
        self.intent_pub.publish(String(data=f"{intent} ({confidence:.2f})"))

    def get_robot_context(self):
        """Get current robot context including pose and environment"""
        context = {
            'robot_pose': self.get_robot_pose(),
            'current_room': self.get_current_room(),
            'detected_objects': self.get_detected_objects(),
            'previous_interactions': self.conversation_context[-5:]  # Last 5 interactions
        }
        return context

    def get_robot_pose(self):
        """Get current robot pose from TF"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time()
            )
            return [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.rotation.z  # Simplified
            ]
        except:
            return [0, 0, 0]  # Default pose

    def get_current_room(self):
        """Get current room/area (would use localization system)"""
        # In practice, this would interface with a room detection system
        return "unknown"

    def get_detected_objects(self):
        """Get currently detected objects (would interface with perception system)"""
        # In practice, this would come from object detection system
        return []

    def execute_action(self, action):
        """Execute the generated action"""
        if action['type'] == 'navigation':
            cmd = Twist()
            cmd.linear.x = action.get('linear_speed', 0.0)
            cmd.angular.z = action.get('angular_speed', 0.0)
            self.cmd_vel_pub.publish(cmd)
        elif action['type'] == 'manipulation':
            # This would interface with manipulation system
            pass
        elif action['type'] == 'communication':
            # This would interface with speech system
            pass

    def publish_nlp_status(self):
        """Publish NLP system status"""
        status_msg = String()
        status_msg.data = (
            f"State: {self.listening_state}, "
            f"Intent: {self.current_intent}, "
            f"Entities: {len(self.current_entities.get('objects', []))} objects"
        )
        self.nlp_status_pub.publish(status_msg)

class ActionGenerator:
    """Generate robot actions from intents and entities"""
    def __init__(self):
        self.action_templates = {
            'move_forward': {'type': 'navigation', 'linear_speed': 0.3, 'angular_speed': 0.0},
            'move_backward': {'type': 'navigation', 'linear_speed': -0.2, 'angular_speed': 0.0},
            'turn_left': {'type': 'navigation', 'linear_speed': 0.0, 'angular_speed': 0.5},
            'turn_right': {'type': 'navigation', 'linear_speed': 0.0, 'angular_speed': -0.5},
            'stop': {'type': 'navigation', 'linear_speed': 0.0, 'angular_speed': 0.0},
            'greet': {'type': 'communication', 'message': 'Hello! How can I help you?'},
            'follow': {'type': 'navigation', 'follow_mode': True}
        }

    def generate_action(self, intent, entities):
        """Generate action from intent and entities"""
        if intent in self.action_templates:
            action = self.action_templates[intent].copy()

            # Customize action based on entities
            if intent == 'navigate_to' and entities.get('locations'):
                action['type'] = 'navigation'
                action['target_location'] = entities['locations'][0]
                action['linear_speed'] = 0.3
                action['angular_speed'] = 0.0

            elif intent == 'pick_up' and entities.get('objects'):
                action['type'] = 'manipulation'
                action['object_to_pick'] = entities['objects'][0]

            elif intent == 'find_object' and entities.get('objects'):
                action['type'] = 'navigation'
                action['search_object'] = entities['objects'][0]

            return action

        return {'type': 'unknown', 'intent': intent, 'entities': entities}

class ResponseGenerator:
    """Generate natural language responses"""
    def __init__(self):
        self.response_templates = {
            'move_forward': "Okay, moving forward.",
            'move_backward': "Okay, moving backward.",
            'turn_left': "Turning left.",
            'turn_right': "Turning right.",
            'stop': "Stopping.",
            'navigate_to': "On my way to the {location}.",
            'pick_up': "Picking up the {object}.",
            'put_down': "Putting down the {object}.",
            'greet': "Hello! How can I assist you today?",
            'follow': "I'll follow you now.",
            'find_object': "I'll look for the {object} for you.",
            'bring_object': "I'll bring the {object} to you.",
            'unknown': "I'm sorry, I didn't understand that command.",
            'error': "I encountered an error processing your request."
        }

    def generate_response(self, action, intent):
        """Generate response based on action and intent"""
        if intent in self.response_templates:
            template = self.response_templates[intent]

            # Fill in template variables
            if '{location}' in template and action.get('target_location'):
                return template.format(location=action['target_location'])
            elif '{object}' in template and action.get('object_to_pick'):
                return template.format(object=action['object_to_pick'])
            elif '{object}' in template and action.get('search_object'):
                return template.format(object=action['search_object'])
            else:
                return template

        return self.response_templates.get('unknown', "I didn't understand.")

class ConversationalManager:
    """Manage conversational context and follow-up responses"""
    def __init__(self):
        self.conversation_history = []
        self.current_topic = None
        self.user_context = {}

    def update_conversation(self, user_input, robot_response, action_taken):
        """Update conversation with new interaction"""
        interaction = {
            'user_input': user_input,
            'robot_response': robot_response,
            'action_taken': action_taken,
            'timestamp': Time()
        }
        self.conversation_history.append(interaction)

        # Limit history size
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def handle_follow_up(self, user_input):
        """Handle follow-up questions based on conversation context"""
        if not self.conversation_history:
            return None

        # Check if user input refers to previous interaction
        if self.refers_to_previous_action(user_input):
            # Handle follow-up to previous action
            return self.generate_follow_up_response(user_input)

        return None

    def refers_to_previous_action(self, user_input):
        """Check if input refers to previous action"""
        pronouns = ['it', 'that', 'this', 'there']
        return any(pronoun in user_input.lower() for pronoun in pronouns)

    def generate_follow_up_response(self, user_input):
        """Generate response to follow-up question"""
        # This would implement more sophisticated follow-up handling
        # based on conversation context
        return "I'm ready for your next command."

def main(args=None):
    rclpy.init(args=args)
    nlp_interface = NLPRobotInterface()

    try:
        rclpy.spin(nlp_interface)
    except KeyboardInterrupt:
        pass
    finally:
        nlp_interface.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced NLP Techniques for Robotics

### Semantic Parsing

Converting natural language to formal representations that robots can execute:

```python
# Example: Semantic parser for robot commands
class SemanticParser:
    def __init__(self):
        self.grammar_rules = self.define_grammar()
        self.semantic_templates = self.define_semantic_templates()

    def define_grammar(self):
        """Define grammar rules for robot commands"""
        return {
            'S': ['NP VP'],  # Sentence: Noun Phrase + Verb Phrase
            'NP': ['Det N', 'Det Adj N', 'Pronoun'],  # Noun Phrase
            'VP': ['V', 'V NP', 'V PP', 'V NP PP'],  # Verb Phrase
            'PP': ['P NP'],  # Prepositional Phrase
            'Det': ['the', 'a', 'an'],  # Determiner
            'N': ['robot', 'cup', 'kitchen', 'bedroom', 'object'],  # Noun
            'V': ['go', 'move', 'pick', 'bring', 'find'],  # Verb
            'P': ['to', 'from', 'in', 'on', 'at'],  # Preposition
            'Adj': ['red', 'blue', 'big', 'small'],  # Adjective
            'Pronoun': ['it', 'that', 'there', 'me']  # Pronoun
        }

    def define_semantic_templates(self):
        """Define semantic templates for command interpretation"""
        return {
            'go_to_location': {
                'pattern': ['go', 'to', 'location'],
                'semantic': lambda entities: {
                    'action': 'navigate',
                    'target': entities.get('location'),
                    'method': 'path_planning'
                }
            },
            'pick_up_object': {
                'pattern': ['pick', 'up', 'object'],
                'semantic': lambda entities: {
                    'action': 'manipulate',
                    'operation': 'grasp',
                    'target': entities.get('object')
                }
            },
            'bring_object_to_person': {
                'pattern': ['bring', 'object', 'to', 'person'],
                'semantic': lambda entities: {
                    'action': 'delivery',
                    'object': entities.get('object'),
                    'destination': entities.get('person'),
                    'method': 'fetch_and_carry'
                }
            }
        }

    def parse_sentence(self, sentence):
        """Parse sentence and generate semantic representation"""
        tokens = sentence.lower().split()

        # Simple pattern matching (in practice, use more sophisticated parsing)
        entities = self.extract_entities(sentence)

        # Determine semantic meaning
        semantic_meaning = self.match_semantic_template(tokens, entities)

        return semantic_meaning

    def extract_entities(self, sentence):
        """Extract entities from sentence"""
        # Use the entity recognizer from earlier
        entity_recognizer = EntityRecognizer()
        return entity_recognizer.extract_entities(sentence)

    def match_semantic_template(self, tokens, entities):
        """Match tokens to semantic templates"""
        # Simple keyword-based matching
        text = ' '.join(tokens)

        if any(word in text for word in ['go to', 'move to', 'navigate to']):
            return {
                'action': 'navigate',
                'target': entities.get('locations', [None])[0] if entities.get('locations') else None
            }
        elif any(word in text for word in ['pick up', 'grasp', 'take']):
            return {
                'action': 'manipulate',
                'operation': 'grasp',
                'target': entities.get('objects', [None])[0] if entities.get('objects') else None
            }
        elif any(word in text for word in ['bring', 'deliver', 'carry']):
            return {
                'action': 'delivery',
                'object': entities.get('objects', [None])[0] if entities.get('objects') else None,
                'destination': entities.get('locations', [None])[0] if entities.get('locations') else None
            }

        return {'action': 'unknown', 'entities': entities}

class GroundedLanguageUnderstanding:
    """Ground language understanding in robot's physical context"""
    def __init__(self):
        self.perception_system = None  # Would connect to robot's perception
        self.action_system = None     # Would connect to robot's action system
        self.world_model = {}         # Robot's model of the world

    def ground_language_in_context(self, semantic_meaning, context):
        """Ground semantic meaning in current context"""
        grounded_meaning = semantic_meaning.copy()

        # Resolve ambiguous references using context
        if grounded_meaning.get('target') == 'it' or grounded_meaning.get('target') == 'that':
            # Resolve using most recently mentioned object
            last_object = self.get_last_mentioned_object(context)
            if last_object:
                grounded_meaning['target'] = last_object

        # Verify target exists in world model
        if grounded_meaning.get('target'):
            resolved_target = self.resolve_target(grounded_meaning['target'], context)
            grounded_meaning['target'] = resolved_target

        # Check if action is feasible in current context
        feasible = self.is_action_feasible(grounded_meaning, context)
        grounded_meaning['feasible'] = feasible

        return grounded_meaning

    def get_last_mentioned_object(self, context):
        """Get the last mentioned object from context"""
        # This would access conversation history
        return "unknown_object"

    def resolve_target(self, target, context):
        """Resolve target using world model and perception"""
        # Look up target in world model
        if target in self.world_model:
            return self.world_model[target]

        # If not in world model, try to perceive it
        # This would interface with perception system
        return target

    def is_action_feasible(self, action, context):
        """Check if action is feasible in current context"""
        # Check robot's current state and capabilities
        # Check environmental constraints
        # Check safety constraints
        return True  # Simplified
```

### Dialogue Management

Managing multi-turn conversations with robots:

```python
# Example: Dialogue manager for robot interaction
class DialogueManager:
    def __init__(self):
        self.dialogue_state = {}
        self.conversation_history = []
        self.system_initiative = False
        self.user_initiative = True
        self.confidence_threshold = 0.7

    def process_user_input(self, user_input):
        """Process user input and generate system response"""
        # Parse user input
        parsed_input = self.parse_user_input(user_input)

        # Update dialogue state
        self.update_dialogue_state(parsed_input)

        # Generate system response
        response = self.generate_response(parsed_input)

        # Update conversation history
        self.conversation_history.append({
            'user': user_input,
            'system': response,
            'parsed': parsed_input,
            'timestamp': self.get_timestamp()
        })

        return response

    def parse_user_input(self, user_input):
        """Parse user input into structured representation"""
        # Use NLP components to parse input
        intent_classifier = IntentClassifier()
        entity_recognizer = EntityRecognizer()

        intent, confidence = intent_classifier.classify_intent(user_input)
        entities = entity_recognizer.extract_entities(user_input)

        return {
            'text': user_input,
            'intent': intent,
            'entities': entities,
            'confidence': confidence
        }

    def update_dialogue_state(self, parsed_input):
        """Update dialogue state based on user input"""
        # Update current topic
        if parsed_input['intent'] != 'unknown':
            self.dialogue_state['current_topic'] = parsed_input['intent']

        # Update entity references
        if parsed_input['entities']['objects']:
            self.dialogue_state['last_object'] = parsed_input['entities']['objects'][-1]

        if parsed_input['entities']['locations']:
            self.dialogue_state['last_location'] = parsed_input['entities']['locations'][-1]

    def generate_response(self, parsed_input):
        """Generate appropriate system response"""
        if parsed_input['confidence'] < self.confidence_threshold:
            return self.generate_uncertainty_response(parsed_input)

        intent = parsed_input['intent']

        if intent == 'greet':
            return self.generate_greeting_response()
        elif intent == 'navigate_to':
            return self.generate_navigation_response(parsed_input)
        elif intent == 'pick_up':
            return self.generate_manipulation_response(parsed_input)
        else:
            return self.generate_general_response(parsed_input)

    def generate_uncertainty_response(self, parsed_input):
        """Generate response when system is uncertain"""
        return "I'm not sure I understood that correctly. Could you please rephrase your request?"

    def generate_greeting_response(self):
        """Generate greeting response"""
        return "Hello! I'm ready to help. What would you like me to do?"

    def generate_navigation_response(self, parsed_input):
        """Generate response for navigation requests"""
        location = parsed_input['entities'].get('locations', ['destination'])[0]
        return f"Okay, I'll navigate to the {location} for you."

    def generate_manipulation_response(self, parsed_input):
        """Generate response for manipulation requests"""
        obj = parsed_input['entities'].get('objects', ['object'])[0]
        return f"I'll pick up the {obj} for you."

    def generate_general_response(self, parsed_input):
        """Generate general response"""
        return "I understand. I'll work on that for you."

    def get_timestamp(self):
        """Get current timestamp"""
        import time
        return time.time()

class MultiModalDialogueSystem:
    """Integrate speech with other modalities for richer interaction"""
    def __init__(self):
        self.speech_system = NLPRobotInterface(None)
        self.gesture_system = None  # Would connect to gesture recognition
        self.vision_system = None   # Would connect to computer vision
        self.fusion_system = self.initialize_fusion_system()

    def initialize_fusion_system(self):
        """Initialize multimodal fusion system"""
        return {
            'confidence_weights': {
                'speech': 0.6,
                'gesture': 0.3,
                'vision': 0.1
            },
            'conflict_resolution': self.resolve_modality_conflicts
        }

    def process_multimodal_input(self, speech_input, gesture_input, vision_input):
        """Process input from multiple modalities"""
        # Process each modality
        speech_result = self.process_speech(speech_input) if speech_input else None
        gesture_result = self.process_gesture(gesture_input) if gesture_input else None
        vision_result = self.process_vision(vision_input) if vision_input else None

        # Fuse results based on confidence
        fused_result = self.fuse_modalities(speech_result, gesture_result, vision_result)

        return fused_result

    def process_speech(self, speech_input):
        """Process speech input"""
        # Use existing NLP pipeline
        return self.speech_system.process_text(speech_input)

    def process_gesture(self, gesture_input):
        """Process gesture input"""
        # Would interface with gesture recognition system
        return {'modality': 'gesture', 'gesture': gesture_input}

    def process_vision(self, vision_input):
        """Process vision input"""
        # Would interface with computer vision system
        return {'modality': 'vision', 'content': vision_input}

    def fuse_modalities(self, speech_result, gesture_result, vision_result):
        """Fuse information from multiple modalities"""
        results = [r for r in [speech_result, gesture_result, vision_result] if r is not None]

        if not results:
            return {'action': 'no_input', 'confidence': 0.0}

        # Weighted combination based on modality reliability
        fused_result = {}
        total_weight = 0

        for i, result in enumerate(results):
            modality = result.get('modality', f'modality_{i}')
            weight = self.fusion_system['confidence_weights'].get(modality, 0.5)

            # Combine results (simplified)
            if 'intent' in result:
                fused_result['intent'] = fused_result.get('intent', []) + [result['intent']]
            if 'entities' in result:
                fused_result['entities'] = fused_result.get('entities', {}).update(result['entities'])

            total_weight += weight

        fused_result['confidence'] = total_weight / len(results) if results else 0.0
        return fused_result

    def resolve_modality_conflicts(self, results):
        """Resolve conflicts between different modalities"""
        # This would implement conflict resolution strategies
        # based on context, reliability, and user preferences
        pass
```

## Lab: Implementing NLP Robot Interface

In this lab, you'll implement a natural language processing system for robot interaction:

```python
# lab_nlp_robot_interface.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
import numpy as np

class NLPInterfaceLab(Node):
    def __init__(self):
        super().__init__('nlp_interface_lab')

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.response_pub = self.create_publisher(String, '/nlp_response', 10)
        self.status_pub = self.create_publisher(String, '/nlp_status', 10)

        # Subscribers
        self.command_sub = self.create_subscription(
            String, '/nlp_commands', self.command_callback, 10
        )

        # NLP components
        self.intent_classifier = IntentClassifier()
        self.entity_recognizer = EntityRecognizer()
        self.response_generator = ResponseGenerator()

        # Initialize with training data
        self.initialize_components()

        # State
        self.current_command = ""
        self.last_intent = "unknown"
        self.command_history = []

        # Control loop
        self.control_timer = self.create_timer(0.1, self.nlp_control_loop)

    def initialize_components(self):
        """Initialize NLP components with training data"""
        examples = self.intent_classifier.get_training_examples()

        for intent, phrases in examples.items():
            for phrase in phrases:
                self.intent_classifier.add_training_example(phrase, intent)

        self.intent_classifier.train_classifier()

    def command_callback(self, msg):
        """Handle incoming voice commands"""
        self.current_command = msg.data

    def nlp_control_loop(self):
        """Main NLP control loop"""
        if self.current_command:
            # Process the command
            self.process_command(self.current_command)

            # Clear command after processing
            self.current_command = ""

        # Publish status
        self.publish_status()

    def process_command(self, command):
        """Process a voice command through NLP pipeline"""
        self.get_logger().info(f"Processing command: {command}")

        # 1. Intent classification
        intent, confidence = self.intent_classifier.classify_intent(command)
        self.last_intent = intent

        # 2. Entity recognition
        entities = self.entity_recognizer.extract_entities(command)

        # 3. Action execution
        action_result = self.execute_action_based_on_intent(intent, entities, command)

        # 4. Generate response
        response = self.response_generator.generate_response(action_result, intent)
        self.response_pub.publish(String(data=response))

        # 5. Update history
        self.command_history.append({
            'command': command,
            'intent': intent,
            'entities': entities,
            'response': response,
            'confidence': confidence
        })

        # Limit history size
        if len(self.command_history) > 10:
            self.command_history = self.command_history[-10:]

        self.get_logger().info(f"Intent: {intent}, Confidence: {confidence:.2f}")

    def execute_action_based_on_intent(self, intent, entities, original_command):
        """Execute action based on recognized intent"""
        cmd = Twist()

        if intent == 'move_forward':
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0
        elif intent == 'move_backward':
            cmd.linear.x = -0.2
            cmd.angular.z = 0.0
        elif intent == 'turn_left':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5
        elif intent == 'turn_right':
            cmd.linear.x = 0.0
            cmd.angular.z = -0.5
        elif intent == 'stop':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        elif intent == 'greet':
            # Robot greeting - just respond, don't move
            pass
        elif intent == 'navigate_to':
            # In a real system, this would start navigation to a location
            # For this lab, just move forward as a placeholder
            cmd.linear.x = 0.2
        else:
            # Unknown intent - don't move
            pass

        # Publish command if it's a movement command
        if any([cmd.linear.x, cmd.angular.z]):
            self.cmd_pub.publish(cmd)

        return {
            'intent': intent,
            'entities': entities,
            'command_executed': cmd.linear.x != 0 or cmd.angular.z != 0,
            'original_command': original_command
        }

    def publish_status(self):
        """Publish NLP system status"""
        status_msg = String()
        status_msg.data = (
            f"Last Intent: {self.last_intent}, "
            f"History: {len(self.command_history)}, "
            f"Commands: {', '.join([c['intent'] for c in self.command_history[-3:]])}"
        )
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    lab = NLPInterfaceLab()

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

## Exercise: Design Your Own NLP Robot System

Consider the following design challenge:

1. What specific robot tasks will your NLP system handle?
2. What vocabulary and command structures are most natural for your users?
3. How will you handle ambiguous or unclear user commands?
4. What spatial and contextual references need to be understood?
5. How will the system maintain conversation context across multiple turns?
6. What feedback mechanisms will help users understand system responses?
7. How will you handle errors or failed command interpretations?

## Summary

Natural Language Processing is essential for intuitive robot interaction, enabling users to communicate with robots using everyday language. Key concepts include:

- **Speech Recognition**: Converting spoken language to text in real-time
- **Intent Classification**: Understanding user intentions from language
- **Entity Recognition**: Identifying objects, locations, and other entities
- **Spatial Reasoning**: Grounding language in physical context
- **Dialogue Management**: Handling multi-turn conversations
- **Multimodal Integration**: Combining speech with other interaction modalities

The integration of these NLP techniques in ROS2 enables the development of sophisticated conversational interfaces that make robots more accessible and easier to use. Understanding these concepts is crucial for developing robots that can interact naturally with humans through language.

In the next lesson, we'll explore social robotics principles and how robots can interact naturally with humans in social contexts.