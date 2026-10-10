import cv2
import mediapipe as mp
import numpy as np
import math
import time


# ==========================================
# INITIALIZATION
# ==========================================

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


# ==========================================
# CALCULATE JOINT ANGLE
# ==========================================

def calculate_angle(a, b, c):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    radians = (
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(math.degrees(radians))

    if angle > 180:
        angle = 360 - angle

    return angle


# ==========================================
# CALCULATE TORSO ANGLE
# ==========================================

def calculate_torso_angle(shoulder, hip):
    dx = shoulder[0] - hip[0]
    dy = hip[1] - shoulder[1]

    return abs(math.degrees(math.atan2(dx, dy)))


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    print("\n====================================")
    print("      AI FITNESS FORM ANALYZER")
    print("====================================")

    print("\n1. Squat")
    print("2. Deadlift")

    choice = input("Choose exercise (1 or 2): ").strip()

    if choice == "1":
        exercise = "SQUAT"
    elif choice == "2":
        exercise = "DEADLIFT"
    else:
        print("Invalid choice.")
        return

    # Get body measurements
    try:
        torso_length = float(input("Enter torso length in cm: "))
        leg_length = float(input("Enter leg length in cm: "))

        if torso_length <= 0 or leg_length <= 0:
            print("Measurements must be positive.")
            return

    except ValueError:
        print("Enter valid numbers.")
        return

    torso_leg_ratio = torso_length / leg_length

    print("\nExercise:", exercise)
    print("Torso/leg ratio:", round(torso_leg_ratio, 2))
    print("Stand sideways to the camera.")
    print("Keep your entire body visible.")
    print("Press Q to quit.")

    # ==========================================
    # OPEN CAMERA
    # ==========================================

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        camera.release()
        print("ERROR: Could not open webcam.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "AI Fitness Form Analyzer"

    # ==========================================
    # SQUAT TRACKING VARIABLES
    # ==========================================

    squat_started = False
    lowest_knee_angle = 180
    max_torso_angle = 0

    squat_rating = ""
    rating_until = 0

    knee_history = []

    SQUAT_START_ANGLE = 140
    SQUAT_STANDING_ANGLE = 150

    # ==========================================
    # START POSE DETECTION
    # ==========================================

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:

            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            while True:

                success, frame = camera.read()

                if not success:
                    print("Could not read camera.")
                    break

                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                results = pose.process(rgb_frame)

                feedback = "Show your full body"

                knee_angle = None
                hip_angle = None
                torso_angle = None

                # ==================================
                # BODY LANDMARK DETECTION
                # ==================================

                if results.pose_landmarks:

                    mp_draw.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )

                    lm = results.pose_landmarks.landmark

                    shoulder = lm[
                        mp_pose.PoseLandmark.LEFT_SHOULDER.value
                    ]

                    hip = lm[
                        mp_pose.PoseLandmark.LEFT_HIP.value
                    ]

                    knee = lm[
                        mp_pose.PoseLandmark.LEFT_KNEE.value
                    ]

                    ankle = lm[
                        mp_pose.PoseLandmark.LEFT_ANKLE.value
                    ]

                    visibility = min(
                        shoulder.visibility,
                        hip.visibility,
                        knee.visibility,
                        ankle.visibility
                    )

                    if visibility >= 0.5:

                        shoulder_point = [shoulder.x, shoulder.y]
                        hip_point = [hip.x, hip.y]
                        knee_point = [knee.x, knee.y]
                        ankle_point = [ankle.x, ankle.y]

                        # Calculate joint angles
                        measured_knee_angle = calculate_angle(
                            hip_point,
                            knee_point,
                            ankle_point
                        )

                        hip_angle = calculate_angle(
                            shoulder_point,
                            hip_point,
                            knee_point
                        )

                        torso_angle = calculate_torso_angle(
                            shoulder_point,
                            hip_point
                        )

                        # Smooth knee angle
                        knee_history.append(measured_knee_angle)

                        if len(knee_history) > 5:
                            knee_history.pop(0)

                        knee_angle = float(np.mean(knee_history))

                        # ==================================
                        # SQUAT ANALYSIS
                        # ==================================

                        if exercise == "SQUAT":

                            # Start tracking when knees bend
                            if (
                                not squat_started
                                and knee_angle < SQUAT_START_ANGLE
                            ):

                                squat_started = True
                                lowest_knee_angle = knee_angle
                                max_torso_angle = torso_angle

                                squat_rating = ""

                            if squat_started:

                                lowest_knee_angle = min(
                                    lowest_knee_angle,
                                    knee_angle
                                )

                                max_torso_angle = max(
                                    max_torso_angle,
                                    torso_angle
                                )

                                if knee_angle > 120:
                                    feedback = "Descending - control movement"

                                elif knee_angle > 90:
                                    feedback = "Squat depth detected"

                                else:
                                    feedback = "Deep squat position"

                                # Complete squat when standing returns
                                if knee_angle >= SQUAT_STANDING_ANGLE:

                                    squat_started = False

                                    # Experimental rating rules
                                    if (
                                        lowest_knee_angle <= 100
                                        and max_torso_angle < 50
                                    ):
                                        squat_rating = "EXCELLENT"

                                    elif lowest_knee_angle <= 120:
                                        squat_rating = "GOOD"

                                    else:
                                        squat_rating = "BAD - CHECK FORM"

                                    rating_until = time.time() + 3

                                    feedback = "Squat completed!"

                            else:
                                feedback = "Standing - perform a squat"

                        # ==================================
                        # DEADLIFT ANALYSIS
                        # ==================================

                        elif exercise == "DEADLIFT":

                            if hip_angle > 155:
                                feedback = "Upright position detected"

                            elif hip_angle > 95:
                                feedback = "Hip hinge detected"

                            else:
                                feedback = "Deep hinge - check back position"

                            if (
                                torso_angle > 60
                                and hip_angle < 150
                            ):
                                feedback = "Check back position and hip hinge"

                    else:
                        feedback = "Keep shoulder, hip, knee and ankle visible"

                # ==================================
                # DISPLAY EXERCISE AND MEASUREMENTS
                # ==================================

                cv2.putText(
                    frame,
                    "Exercise: " + exercise,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Torso: {torso_length:.1f} cm",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Leg: {leg_length:.1f} cm",
                    (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Torso/leg ratio: {torso_leg_ratio:.2f}",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

                if knee_angle is not None:
                    cv2.putText(
                        frame,
                        f"Knee angle: {knee_angle:.1f} deg",
                        (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2
                    )

                if hip_angle is not None:
                    cv2.putText(
                        frame,
                        f"Hip angle: {hip_angle:.1f} deg",
                        (10, 165),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2
                    )

                if torso_angle is not None:
                    cv2.putText(
                        frame,
                        f"Torso angle: {torso_angle:.1f} deg",
                        (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2
                    )

                # ==================================
                # DISPLAY SQUAT RATING
                # ==================================

                if (
                    exercise == "SQUAT"
                    and time.time() < rating_until
                ):

                    cv2.rectangle(
                        frame,
                        (30, 220),
                        (610, 285),
                        (30, 30, 30),
                        -1
                    )

                    if squat_rating == "EXCELLENT":
                        rating_color = (0, 255, 0)

                    elif squat_rating == "GOOD":
                        rating_color = (0, 255, 255)

                    else:
                        rating_color = (0, 0, 255)

                    cv2.putText(
                        frame,
                        squat_rating,
                        (50, 263),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        rating_color,
                        2
                    )

                # ==================================
                # DISPLAY LIVE FEEDBACK
                # ==================================

                cv2.rectangle(
                    frame,
                    (0, frame.shape[0] - 50),
                    (frame.shape[1], frame.shape[0]),
                    (30, 30, 30),
                    -1
                )

                cv2.putText(
                    frame,
                    feedback,
                    (10, frame.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                # ==================================
                # SHOW CAMERA
                # ==================================

                cv2.imshow(window_name, frame)

                key = cv2.waitKey(30) & 0xFF

                if key == ord("q") or key == 27:
                    break

                try:
                    if cv2.getWindowProperty(
                        window_name,
                        cv2.WND_PROP_VISIBLE
                    ) < 1:
                        break
                except cv2.error:
                    break

    except Exception as error:
        print("Program error:", error)

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Program closed.")


if __name__ == "__main__":
    main()