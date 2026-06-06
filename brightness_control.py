import cv2
import mediapipe as mp
import math
import screen_brightness_control as sbc

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, c = frame.shape

            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]

            x1 = int(thumb.x * w)
            y1 = int(thumb.y * h)

            x2 = int(index.x * w)
            y2 = int(index.y * h)

            cv2.circle(frame, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 10, (255, 0, 255), cv2.FILLED)

            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            distance = math.hypot(x2 - x1, y2 - y1)

            brightness = int((distance - 20) * 100 / (200 - 20))
            brightness = max(0, min(100, brightness))

            try:
                sbc.set_brightness(brightness)
            except:
                pass
            
            cv2.putText(
                frame,
                f"Brightness: {brightness}%",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("Brightness Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
