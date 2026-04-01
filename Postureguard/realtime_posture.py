import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import time
import winsound

# ── 1. LSTM Model ─────────────────────────────────────────
class PostureLSTM(nn.Module):
    def __init__(self, input_size=99, hidden_size=128, num_layers=2):
        super(PostureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out).squeeze()

# ── 2. Load model ─────────────────────────────────────────
model = PostureLSTM()
model.load_state_dict(torch.load('posture_model.pth'))
model.eval()
print("Model loaded!")

# ── 3. MediaPipe + webcam ─────────────────────────────────
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

SEQ_LEN = 30
frame_buffer = deque(maxlen=SEQ_LEN)

# ── 4. Alert state ────────────────────────────────────────
label_text = "Collecting frames..."
label_color = (255, 255, 0)
confidence = 0.0

bad_posture_start = None      # when bad posture started
alert_triggered = False       # has alert fired this session
ALERT_THRESHOLD = 5           # seconds before alert fires
last_beep_time = 0            # avoid beeping every frame
BEEP_COOLDOWN = 10            # seconds between beeps

print("PostureGuard running! Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks,
                               mp_pose.POSE_CONNECTIONS)

        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints += [lm.x, lm.y, lm.z]
        frame_buffer.append(keypoints)

        if len(frame_buffer) == SEQ_LEN:
            sequence = np.array(frame_buffer, dtype=np.float32)
            tensor = torch.tensor(sequence).unsqueeze(0)

            with torch.no_grad():
                output = model(tensor).item()

            confidence = output

            if output < 0.5:
                label_text = "GOOD POSTURE"
                label_color = (0, 255, 0)
                bad_posture_start = None
                alert_triggered = False
            else:
                label_text = "BAD POSTURE"
                label_color = (0, 0, 255)

                if bad_posture_start is None:
                    bad_posture_start = time.time()

    # ── 5. Alert logic ────────────────────────────────────
    alert_active = False
    bad_duration = 0

    if bad_posture_start is not None:
        bad_duration = time.time() - bad_posture_start

        if bad_duration >= ALERT_THRESHOLD:
            alert_active = True

            # Beep every BEEP_COOLDOWN seconds
            now = time.time()
            if now - last_beep_time > BEEP_COOLDOWN:
                winsound.Beep(1000, 500)
                last_beep_time = now

    # ── 6. Draw UI ────────────────────────────────────────
    # Top bar
    cv2.rectangle(frame, (0, 0), (640, 110), (0, 0, 0), -1)

    # Posture label
    cv2.putText(frame, label_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_color, 3)

    # Confidence
    conf_display = confidence if confidence >= 0.5 else 1 - confidence
    cv2.putText(frame, f"Confidence: {conf_display*100:.1f}%", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Bad posture timer
    if bad_posture_start is not None:
        cv2.putText(frame, f"Bad posture: {bad_duration:.1f}s", (400, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

    # Alert banner
    if alert_active:
        cv2.rectangle(frame, (0, 115), (640, 185), (0, 0, 200), -1)
        cv2.putText(frame, "! ALERT: Fix your posture now !", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Buffer progress
    cv2.putText(frame, f"Buffer: {len(frame_buffer)}/{SEQ_LEN}", (480, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    cv2.imshow("PostureGuard - Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("PostureGuard stopped.")