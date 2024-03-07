import cv2
import mediapipe as mp
import numpy as np
import pyfirmata

# Initialize PyFirmata for Arduino communication
port = "COM5"  # Change this to your Arduino port
board = pyfirmata.Arduino(port)
servo_pinX = board.get_pin('d:9:s')  # Pin 9 on Arduino for X-axis servo
servo_pinY = board.get_pin('d:10:s')  # Pin 10 on Arduino for Y-axis servo
servo_pinZ = board.get_pin('d:11:s')  # Pin 11 on Arduino for Z-axis servo

# Initialize Mediapipe Hands
mpHand = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHand.Hands(min_detection_confidence=0.8)

# Set up camera
cap = cv2.VideoCapture(0)
ws, hs = 1280, 720
cap.set(3, ws)
cap.set(4, hs)

# Calculate the distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

while cap.isOpened():
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    multiHandDetection = results.multi_hand_landmarks
    lmList = []

    if multiHandDetection:
        for id, lm in enumerate(multiHandDetection):
            mpDraw.draw_landmarks(img, lm, mpHand.HAND_CONNECTIONS,
                                  mpDraw.DrawingSpec(color=(0, 255, 255), thickness=4, circle_radius=7),
                                  mpDraw.DrawingSpec(color=(0, 0, 0), thickness=4))

        singleHandDetection = multiHandDetection[0]
        for lm in singleHandDetection.landmark:
            h, w, c = img.shape
            lm_x, lm_y = int(lm.x * w), int(lm.y * h)
            lmList.append([lm_x, lm_y])

        # Get the positions of the index and middle finger
        index_finger = lmList[8]
        middle_finger = lmList[12]

        # Calculate the distance between the fingers
        finger_distance = calculate_distance(index_finger, middle_finger)

        # Convert the finger distance to a servo angle (adjust the scaling factor as needed)
        servoX = int(np.interp(index_finger[0], [0, ws], [180, 0]))
        servoY = int(np.interp(index_finger[1], [0, hs], [0, 180]))
        servoZ = int(np.interp(finger_distance, [0, hs], [0, 180]))

        # Set the servo angles
        servo_pinX.write(servoX)
        servo_pinY.write(servoY)
        servo_pinZ.write(servoZ)

        # Print the servo angles
        print(f"Servo X Angle: {servoX}")
        print(f"Servo Y Angle: {servoY}")
        print(f"Servo Z Angle: {servoZ}")

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

