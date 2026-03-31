import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque

# ── 1. LSTM Model (same architecture as training) ─────────
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

# ── 2. Load trained model ─────────────────────────────────
model = PostureLSTM()
model.load_state_dict(torch.load('posture_model.pth'))
model.eval()
print("Model loaded successfully!")

# ── 3. MediaPipe setup ────────────────────────────────────
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# ── 4. Webcam setup ───────────────────────────────────────
cap = cv2.VideoCapture(0)
SEQ_LEN = 30
frame_buffer = deque(maxlen=SEQ_LEN)

label_text = "Collecting frames..."
label_color = (255, 255, 0)
confidence = 0.0

print("Starting real-time posture detection! Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks,
                               mp_pose.POSE_CONNECTIONS)

        # Extract keypoints
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints += [lm.x, lm.y, lm.z]

        frame_buffer.append(keypoints)

        # Once we have 30 frames, run the LSTM
        if len(frame_buffer) == SEQ_LEN:
            sequence = np.array(frame_buffer, dtype=np.float32)
            tensor = torch.tensor(sequence).unsqueeze(0)  # (1, 30, 99)

            with torch.no_grad():
                output = model(tensor).item()

            confidence = output

            if output < 0.5:
                label_text = "GOOD POSTURE"
                label_color = (0, 255, 0)   # green
            else:
                label_text = "BAD POSTURE"
                label_color = (0, 0, 255)   # red

    # ── 5. Display on screen ──────────────────────────────
    # Background box for text
    cv2.rectangle(frame, (0, 0), (640, 100), (0, 0, 0), -1)

    # Main label
    cv2.putText(frame, label_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_color, 3)

    # Confidence score
    conf_display = confidence if confidence >= 0.5 else 1 - confidence
    cv2.putText(frame, f"Confidence: {conf_display*100:.1f}%", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Frame buffer progress
    cv2.putText(frame, f"Buffer: {len(frame_buffer)}/{SEQ_LEN}", (480, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("PostureGuard - Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("PostureGuard stopped.")
