"""
Estimate the angles of the head using a webcam.

The code is mostly based on:
https://towardsdatascience.com/head-pose-estimation-using-python-d165d3541600
"""
import time
from typing import List, Tuple, Optional

import cv2
import mediapipe as mp
import numpy as np


def main() -> None:
    """Execute the main routine."""
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    ball_x = None  # type: Optional[int]
    ball_radius = 10

    # Pixels / second
    ball_velocity = 300

    timestamp = time.time()

    cap = cv2.VideoCapture(0)
    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            # Flip the image horizontally for a later selfie-view display
            # Also convert the color space from BGR to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # To improve performance
            image.flags.writeable = False

            # Get the result
            results = face_mesh.process(image)

            # To improve performance
            image.flags.writeable = True

            # Convert the color space from RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            img_h, img_w, img_c = image.shape
            face_3d = []
            face_2d = []

            if ball_x is None:
                ball_x = img_w / 2

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                            x, y = int(lm.x * img_w), int(lm.y * img_h)

                            # Get the 2D Coordinates
                            face_2d.append((x, y))

                            # Get the 3D Coordinates
                            face_3d.append((x, y, lm.z))

                            # Convert it to the NumPy array
                    face_2d = np.array(face_2d, dtype=np.float64)

                    # Convert it to the NumPy array
                    face_3d = np.array(face_3d, dtype=np.float64)

                    # The camera matrix
                    focal_length = 1 * img_w

                    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                           [0, focal_length, img_w / 2],
                                           [0, 0, 1]])

                    # The Distance Matrix
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    # Solve PnP
                    success, rot_vec, trans_vec = cv2.solvePnP(
                        face_3d,
                        face_2d,
                        cam_matrix,
                        dist_matrix
                    )
                    if not success:
                        raise AssertionError("Expected Perspective-n-Point to succeed")

                    # Get rotational matrix
                    rmat, jac = cv2.Rodrigues(rot_vec)

                    # Get angles
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    # Get the y rotation degree
                    x = angles[0] * 360
                    y = angles[1] * 360

                    now = time.time()
                    time_delta = now - timestamp

                    # See where the user's head tilting
                    if y < -10:
                        text = f"{x:5.1f} {y:5.1f}: Looking Left"
                        ball_x = max(0, ball_x - ball_velocity * time_delta)
                    elif y > 10:
                        text = f"{x:5.1f} {y:5.1f} Looking Right"
                        ball_x = min(
                            img_w - ball_radius,
                            ball_x + ball_velocity * time_delta
                        )

                    elif x < -10:
                        text = f"{x:5.1f} {y:5.1f} Looking Down"
                    else:
                        text = f"{x:5.1f} {y:5.1f} Forward"

                    # Add the text on the image
                    cv2.putText(
                        image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2
                    )

                    # Draw the ball
                    cv2.circle(
                        image,
                        (int(ball_x), 30), ball_radius, (255, 255, 255), -1
                    )

                    timestamp = now

            cv2.imshow('Head Pose Estimation', image)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()


if __name__ == "__main__":
    main()
