"""Understand commands *via* the head movement."""
import enum
import threading
from typing import Optional, Set

import cv2
import mediapipe as mp
import numpy as np

import cactusss.command


class Pose(enum.Enum):
    """Model the head poses."""

    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class Recognizer:
    #: Set for all the head pose recognized
    active_poses: Set[Pose]

    #: Set if the video capture is opened
    capture_is_opened: bool

    head_image: Optional[cv2.Mat]

    _received_close: bool
    _thread: threading.Thread

    def __init__(self) -> None:
        """Initialize with the unset values."""
        self.active_poses = set()
        self._received_close = False
        self.capture_is_opened = False
        self.head_image = None

    def start(self) -> None:
        """Start the head pose recognition."""
        self._thread = threading.Thread(
            target=Recognizer._run_endless_loop, args=(self,)
        )
        self._thread.start()

    def __enter__(self) -> None:
        """Handle the initialization of the controller."""
        self.start()

    def close(self) -> None:
        """Stop the video capture."""
        self._received_close = True
        self._thread.join()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Handle the deconstruction."""
        self.close()

    def _run_endless_loop(self) -> None:
        """Run the endless loop and recognize the head pose."""
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        cap = cv2.VideoCapture(0)
        try:
            self.capture_is_opened = cap.isOpened()
            while not self._received_close:
                self.capture_is_opened = cap.isOpened()
                if not self.capture_is_opened:
                    break

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

                if results.multi_face_landmarks:
                    min_x = min_y = max_x = max_y = None  # type: Optional[int]

                    for face_landmarks in results.multi_face_landmarks:
                        for idx, lm in enumerate(face_landmarks.landmark):
                            x, y = int(lm.x * img_w), int(lm.y * img_h)

                            if min_x is None:
                                min_x = x
                                max_x = x
                                min_y = y
                                max_y = y
                            else:
                                min_x = min(x, min_x)
                                max_x = max(x, max_x)
                                min_y = min(y, min_y)
                                max_y = max(y, max_y)

                            if (
                                idx == 33
                                or idx == 263
                                or idx == 1
                                or idx == 61
                                or idx == 291
                                or idx == 199
                            ):
                                if idx == 1:
                                    nose_2d = (lm.x * img_w, lm.y * img_h)
                                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                                # Get the 2D Coordinates
                                face_2d.append((x, y))

                                # Get the 3D Coordinates
                                face_3d.append((x, y, lm.z))

                                # Convert it to the NumPy array

                        self.head_image = image[min_y : max_y + 1, min_x : max_x + 1]

                        face_2d = np.array(face_2d, dtype=np.float64)

                        # Convert it to the NumPy array
                        face_3d = np.array(face_3d, dtype=np.float64)

                        # The camera matrix
                        focal_length = 1 * img_w

                        cam_matrix = np.array(
                            [
                                [focal_length, 0, img_h / 2],
                                [0, focal_length, img_w / 2],
                                [0, 0, 1],
                            ]
                        )

                        # The Distance Matrix
                        dist_matrix = np.zeros((4, 1), dtype=np.float64)

                        # Solve PnP
                        success, rot_vec, trans_vec = cv2.solvePnP(
                            face_3d, face_2d, cam_matrix, dist_matrix
                        )
                        if not success:
                            raise AssertionError(
                                "Expected Perspective-n-Point to succeed"
                            )

                        # Get rotational matrix
                        rmat, jac = cv2.Rodrigues(rot_vec)

                        # Get angles
                        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                        # Get the rotation in degrees
                        x = angles[0] * 360
                        y = angles[1] * 360

                        active_poses = set()

                        # See where the user's head tilting
                        if y < -10:
                            active_poses.add(Pose.LEFT)
                        elif y > 10:
                            active_poses.add(Pose.RIGHT)

                        if x < -10:
                            active_poses.add(Pose.DOWN)
                        elif x > 10:
                            active_poses.add(Pose.UP)

                        self.active_poses = active_poses
        finally:
            cap.release()
