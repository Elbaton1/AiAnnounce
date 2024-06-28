import cv2
import pytesseract
from deepface import DeepFace
from deep_sort_realtime.deepsort_tracker import DeepSort
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from google.cloud import texttospeech
import os
import random

# Initialize Google TTS client
def init_tts():
    client = texttospeech.TextToSpeechClient()
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Wavenet-D"
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.0
    )
    return client, voice, audio_config

# Function to generate and play commentary
def generate_and_play_commentary(client, voice, audio_config, commentary):
    synthesis_input = texttospeech.SynthesisInput(text=commentary)
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    with open("commentary.mp3", "wb") as out:
        out.write(response.audio_content)
    os.system("mpg123 commentary.mp3")  # Use a media player to play the commentary

# Function to recognize jersey number
def recognize_jersey_number(img, bounding_box):
    x, y, w, h = bounding_box
    player_img = img[y:y+h, x:x+w]
    number_text = pytesseract.image_to_string(player_img, config='--psm 6')
    return number_text.strip()

# Function to recognize face
def recognize_face(face_img, known_faces):
    result = DeepFace.find(img_path=face_img, db_path=known_faces)
    return result['identity'][0] if not result.empty else "Unknown"

# Initialize the DeepSort tracker
tracker = DeepSort(max_age=30, n_init=3)

# Function to update tracker with detections
def update_tracker(frame, detections):
    tracks = tracker.update_tracks(detections, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, str(track_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

# Initialize the pose estimator
model = 'mobilenet_thin'
w, h = model_wh('432x368')
e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

# Function to recognize actions based on pose
def recognize_action(humans, frame):
    for human in humans:
        # Extract key points and recognize actions
        # Implement your action recognition logic here
        pass

# Load YOLO model
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# Detect objects
def detect_objects(img, net, output_layers):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, confidences, class_ids, indexes

# Draw labels on detected objects
def draw_labels(boxes, confidences, class_ids, classes, indexes, img):
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), font, 1, color, 2)
    return img

# Generate commentary based on detected events
def generate_commentary(player_actions):
    commentary = ""
    action_phrases = {
        "receives_ball": [
            "{} receives the ball and takes it up the court with authority!",
            "Watch {} as he catches the pass and immediately scans the floor for his next move.",
            "{} grabs the ball and wastes no time, looking to create an opportunity.",
            "With a quick pass, {} receives the ball and is already eyeing the basket.",
            "{} pulls in the pass, and the offense is on the move!"
        ],
        "dribbling": [
            "Look at the handles on {}! He’s weaving through defenders like they’re standing still.",
            "{} is showcasing some exceptional ball control, moving with purpose.",
            "{} dances around the defense with precision dribbling.",
            "Watch how {} navigates through traffic with ease, keeping the ball on a string.",
            "{} is putting on a dribbling clinic out there, leaving defenders in his wake."
        ],
        "pass": [
            "{} delivers a pinpoint pass to {}, threading the needle through the defense.",
            "What vision! {} spots an open teammate and executes a flawless pass.",
            "{} with a beautiful assist, setting up {} for an easy basket.",
            "A spectacular no-look pass from {}! The defense never saw it coming.",
            "{} fires a crisp pass to {}, who is ready to make a move."
        ],
        "shoot_and_score": [
            "{} pulls up for the shot... and it's nothing but net! What a beauty!",
            "{} takes the jumper... it's good! Perfect execution.",
            "{} launches a three-pointer... and it's in! The crowd goes wild!",
            "With a quick release, {} nails the shot from downtown.",
            "{} shoots over the defender... and scores! Incredible!"
        ],
        "defense": [
            "{} is putting pressure on the opponent with tight defense.",
            "Look at {}, shadowing every move and denying any open looks.",
            "{} with a lockdown defense, forcing a tough shot.",
            "{} steps up and draws the charge! What a defensive play.",
            "{} is all over his man, making it difficult to advance."
        ],
        "rebound": [
            "{} grabs the rebound.",
            "{} snatches the ball off the board.",
            "{} gets the rebound and resets the play.",
            "{} fights for the rebound and comes out with the ball.",
            "{} is dominating the boards today, another great rebound."
        ],
        "steal": [
            "{} with the steal! He read that pass like a book.",
            "Quick hands from {}, disrupting the play and taking the ball.",
            "{} intercepts the pass and is off to the races!",
            "{} strips the ball away and creates a turnover.",
            "A brilliant defensive play by {}, stealing the ball cleanly."
        ],
        "block": [
            "{} blocks the shot! What a defensive play!",
            "{} denies the shot with a fantastic block.",
            "{} swats the ball away with authority.",
            "{} rises up and blocks the shot!",
            "An amazing rejection by {}, shutting down the offense."
        ],
        "foul": [
            "{} commits a foul.",
            "{} is called for a foul.",
            "{} fouls the opponent, and the play stops.",
            "A foul on {}, the referee blows the whistle.",
            "{} with an aggressive foul, and the game pauses."
        ]
    }

    for player, actions in player_actions.items():
        for action in actions:
            if action in action_phrases:
                commentary += random.choice(action_phrases[action]).format(player) + " "
            else:
                commentary += "{} performs {}. ".format(player, action)
    return commentary.strip()

# Integration function to update game state and generate commentary
def update_game_state_and_commentary(frame, detections, known_faces):
    # Update tracker
    tracked_frame = update_tracker(frame, detections)

    # Recognize players and actions
    commentary = ""
    player_actions = {}  # Dictionary to store actions for each player
    for detection in detections:
        bbox, label = detection
        # Recognize jersey number
        number = recognize_jersey_number(frame, bbox)
        # Assuming 'label' contains player name or identifier
        if label not in player_actions:
            player_actions[label] = []
        player_actions[label].append(number)  # Append jersey number

    # Generate commentary based on player actions
    commentary = generate_commentary(player_actions)

    return tracked_frame, commentary

# Example usage (replace with your actual implementation)
if __name__ == "__main__":
    # Example image or video frame
    frame = cv2.imread('example_frame.jpg')

    # Example detections format [(bounding_box, label), ...]
    detections = [([100, 100, 50, 50], "Player1"), ([200, 200, 50, 50], "Player2")]

    # Example known_faces path
    known_faces = 'path_to_known_faces'

    # Update game state and get commentary
    updated_frame, commentary = update_game_state_and_commentary(frame, detections, known_faces)
    print("Commentary:", commentary)

    # Display or further process the updated frame
    cv2.imshow('Updated Frame', updated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
