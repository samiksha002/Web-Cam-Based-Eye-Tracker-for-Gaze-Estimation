import cv2
import math
import mediapipe as mp
import numpy as np
import time
import pyautogui

RES_SCREEN = pyautogui.size()
SCREEN_WIDTH = 1800
SCREEN_HEIGHT = 1169

class GazeCalculator:
    def __init__(self, left_pupil, right_pupil, left_pupil_circumfrence, right_pupil_circumfrence,pitch,yaw,roll,nose,left_angle,right_angle,distance_left,distance_right):
        self.left_pupil = left_pupil
        self.right_pupil = right_pupil
        self.left_pupil_circumfrence = left_pupil_circumfrence
        self.right_pupil_circumfrence = right_pupil_circumfrence
        self.pitch=pitch
        self.yaw=yaw
        self.nose = nose
        self.roll=roll
        self.left_angle=left_angle
        self.right_angle=right_angle
        self.distance_left=distance_left
        self.distance_right=distance_right


    def pupil_coordinates_left(self):
        if self.left_pupil is not None:
            return self.left_pupil[0], self.left_pupil[1]
        else:
            # Handle the case where left_pupil is None
            return 0.0, 0.0
    def pupil_coordinates_right(self):
        if self.left_pupil is not None:
            return self.right_pupil[0],self.right_pupil[1]
        else:
            # Handle the case where left_pupil is None
            return 0.0, 0.0

class EyeTracker:
    def __init__(self):
        self.gaze_calculator = GazeCalculator(None,None,None,None,None,None,None,None,None,None,None,None)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.left_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.right_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_iris = [474, 475, 476, 477]
        self.left_iris = [469, 470, 471, 472]
        self.l_h_left = [33]
        self.l_h_right = [133]
        self.r_h_left = [362]
        self.r_h_right = [263]
        
        # Added: Pupil and Iris attributes
        self.left_pupil = None
        self.right_pupil = None
        self.left_pupil_circumfrence = None
        self.right_pupil_circumfrence = None
        self.nose = None
        
        # Added: Variables for Purkinje detection
        self.left_purkinje = None
        self.right_purkinje = None
        self.left_eye_bb = None  # Define the left eye bounding box
        self.right_eye_bb = None  # Define the right eye bounding box
        self.frame_gray = None  # Store the grayscale frame
        self.normalized_right_pupil_x=None
        self.normalized_right_pupil_y=None
        self.normalized_left_pupil_x=None
        self.normalized_left_pupil_y=None
        self.blink_detected=True
        self.brightness_level=None

    def euclidean_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        distance = math.sqrt((x2 - x1) * 2 + (y2 - y1) * 2)
        return distance

    def iris_position(self, iris_center, right_point, left_point):
        center_to_right_dist = self.euclidean_distance(iris_center, right_point)
        total_distance = self.euclidean_distance(right_point, left_point)
        ratio = center_to_right_dist / total_distance
        iris_position = ""
        if ratio <= 0.42:
            iris_position = "right"
        elif 0.42 < ratio <= 0.57:
            iris_position = "center"
        else:
            iris_position = "left"
        return iris_position, ratio

    def detect_pupil(self, eye_region):
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        gray_eye = cv2.GaussianBlur(gray_eye, (5, 5), 0)
        gray_eye = cv2.equalizeHist(gray_eye)

        _, threshold = cv2.threshold(gray_eye, 30, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest_contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                return True, (cx, cy)

        return False, None

    def calculate_angle(self, eye_center, webcam_center):
        # Calculate vector from webcam center to eye center
        eye_vector = np.array(eye_center) - np.array(webcam_center)

        # Calculate angle between eye vector and horizontal axis
        angle_radians = np.arctan2(eye_vector[1], eye_vector[0])
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees

    def update(self, frame):
        self.frame = frame
        self._analyze()
        
    def pupil_left_coordinates(self,normalized_left_pupil_x,normalized_left_pupil_y):
        self.normalized_left_pupil_x=normalized_left_pupil_x
        self.normalized_left_pupil_y=normalized_left_pupil_y
        return self.normalized_left_pupil_x,self.normalized_left_pupil_y

    def pupil_right_coordinates(self,right_pupil_coordinates_x,right_pupil_coordinates_y):
        self.normalized_right_pupil_x=right_pupil_coordinates_x
        self.normalized_right_pupil_y=right_pupil_coordinates_y
        return self.normalized_right_pupil_x,self.normalized_right_pupil_y
        
    def is_point_inside_ellipse(self, nose, chin, forehead, center, axes, angle):
        # Convert points to tuples if they are not
        nose = tuple(nose) if not isinstance(nose, tuple) else nose
        chin = tuple(chin) if not isinstance(chin, tuple) else chin
        forehead = tuple(forehead) if not isinstance(forehead, tuple) else forehead

        # Check if any of the points are inside the rotated ellipse
        is_nose_inside = self._is_single_point_inside_ellipse(nose, center, axes, angle)
        is_chin_inside = self._is_single_point_inside_ellipse(chin, center, axes, angle)
        is_forehead_inside = self._is_single_point_inside_ellipse(forehead, center, axes, angle)
        
        return is_nose_inside and is_chin_inside and is_forehead_inside

    def _is_single_point_inside_ellipse(self, point, center, axes, angle):
        x, y = point
        cx, cy = center
        a, b = axes

        # Translate the point to the ellipse's local coordinate system
        xp = np.cos(np.radians(angle)) * (x - cx) - np.sin(np.radians(angle)) * (y - cy)
        yp = np.sin(np.radians(angle)) * (x - cx) + np.cos(np.radians(angle)) * (y - cy)

        # Check if the point is inside the rotated ellipse equation
        return (xp / a) ** 2 + (yp / b) ** 2 <= 1


    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w,img_c = frame.shape
        rgb_frame.flags.writeable = False
        start = time.time()
        webcam_center = (frame.shape[1] // 2, frame.shape[0] // 2)
        with self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            results = face_mesh.process(rgb_frame)

        rgb_frame.flags.writeable = True

        face_2d = []
        face_3d = []
        left_x = 0
        left_y = 0
        right_x = 0
        right_y = 0
        nose_2d = []
        nose_3d =[]
        left_pupil_circumfrence = 0
        right_pupil_circumfrence = 0
        pitch = 0
        yaw = 0
        roll = 0
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the average brightness of the frame
        brightness = int(round(gray_frame.mean()))

        # Adjust screen brightness based on the frame brightness
        brightness_level = int((brightness / 255.0) * 100)  # Assuming screen brightness range is 0-100
        brightness_level = max(1, min(brightness_level, 100))
        self.brightness_level=brightness_level
            
        text=f"BRIGHTNESS LEVEL: {brightness_level}"
        cv2.putText(frame, text, (20,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,20,155),2)
        
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199 or idx == 152 or idx == 10:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        if idx == 152:
                            chin_2d = (lm.x * img_w, lm.y * img_h)
                        if idx == 10:
                            forehead_2d = (lm.x * img_w, lm.y * img_h)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        face_2d.append([x, y])
                        face_3d.append(([x, y, lm.z]))

                # Get 2d Coord
                face_2d = np.array(face_2d, dtype=np.float64)

                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])
                distortion_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

                # getting rotational of face
                rmat, jac = cv2.Rodrigues(rotation_vec)

                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                # here based on axis rot angle is calculated
                if y < -10:
                    text = "Looking Left"
                elif y > 10:
                    text = "Looking Right"
                elif x < -10:
                    text = "Looking Down"
                elif x > 10:
                    text = "Looking Up"
                else:
                    text = "Forward"

                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rotation_vec, translation_vec, cam_matrix,
                                                                 distortion_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                pitch= str(np.round(x,2))
                yaw= str(np.round(y,2))
                roll=str(np.round(z,2))

                cv2.line(frame, p1, p2, (255, 255, 255), 3)

                #cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                
                instructions = "Place your head inside the frame and stay still dont MOVE much"
                instructions_2 =  "PRESS 'c' to start calibration"
                cv2.putText(frame, instructions, (110, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)
                cv2.putText(frame, instructions_2, (183, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)
                
                # Get nose 2D coordinates
                nose_2d = (int(nose_2d[0]), int(nose_2d[1]))
                chin_2d = (int(chin_2d[0]), int(chin_2d[1]))
                forehead_2d = (int(forehead_2d[0]), int(forehead_2d[1]))
                
                # Draw oval in the middle of the frame
                oval_center = (img_w // 2, img_h // 2)
                oval_axes = (int(img_w // 4.45), int(img_h // 4.5))
                is_nose_inside_ellipse = self.is_point_inside_ellipse(nose_2d,chin_2d,forehead_2d, oval_center, oval_axes, 90)

                # Draw rotated oval in the middle of the frame with different color based on the condition
                oval_color = (0, 255, 0)  # Default color (green)
                if not is_nose_inside_ellipse:
                    oval_color = (0, 0, 255)  # Change color to red
                
                
                #cv2.ellipse(frame, oval_center, oval_axes, 90, 0, 360, oval_color, 2)
                rect1_width = oval_axes[0] // 3
                rect1_height = oval_axes[1] // 3
                rect2_width = oval_axes[0] // 3
                rect2_height = oval_axes[1] // 3

                upper_left_rect1 = (oval_center[0] - rect1_width-20, (oval_center[1] - oval_axes[1] // 2)+10)
                lower_right_rect1 = (oval_center[0]-20, (oval_center[1] - rect1_height // 2)+10)

                upper_left_rect2 = (oval_center[0]+20, (oval_center[1] - oval_axes[1] // 2)+10)
                lower_right_rect2 = (oval_center[0] + rect2_width+20, (oval_center[1] - rect2_height // 2)+10)

                # Draw rectangles
                #cv2.rectangle(frame, upper_left_rect1, lower_right_rect1, (255, 0, 0), 2)
                #cv2.rectangle(frame, upper_left_rect2, lower_right_rect2, (255, 0, 0), 2)

            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime

            #cv2.putText(frame, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Assuming you have only one face in the frame
            face_landmarks = results.multi_face_landmarks[0]

            # Extract coordinates of the eyes
            right_eye_coords = [(landmark.x, landmark.y) for landmark in face_landmarks.landmark[474:478]]
            left_eye_coords = [(landmark.x, landmark.y) for landmark in face_landmarks.landmark[469:473]]

            # Calculate coordinates of the pupils
            right_x = (right_eye_coords[0][0] + right_eye_coords[2][0]+right_eye_coords[1][0]+right_eye_coords[3][0]) / 4 * frame.shape[1]
            right_y = (right_eye_coords[0][1] + right_eye_coords[2][1]+right_eye_coords[1][1]+right_eye_coords[3][1]) / 4 * frame.shape[0]
            right_radius = abs((right_eye_coords[0][0] - right_eye_coords[2][0])/2)
            right_pupil_circumfrence =  2*3.14*right_radius
            
            left_x = (left_eye_coords[0][0] + left_eye_coords[2][0]) / 2 * frame.shape[1]
            left_y = (left_eye_coords[1][1] + left_eye_coords[3][1]) / 2 * frame.shape[0]
            left_radius = abs((left_eye_coords[0][0] - left_eye_coords[2][0])/2)
            left_pupil_circumfrence = 2*3.14*left_radius

            left_x1 = (left_eye_coords[0][0] + left_eye_coords[2][0]+left_eye_coords[1][0]+left_eye_coords[3][0]) / 4 * frame.shape[1]
            left_y1 = (left_eye_coords[1][1] + left_eye_coords[3][1]+left_eye_coords[2][1]+left_eye_coords[0][1]) / 4 * frame.shape[0]

            # Draw 'X' marker on the pupils
            cv2.drawMarker(frame, (int(right_x), int(right_y)), (255, 255, 0), markerType=cv2.MARKER_STAR,
                           markerSize=4,
                           thickness=1)
            '''cv2.drawMarker(frame, (int(left_x), int(left_y)), (255, 255, 0), markerType=cv2.MARKER_STAR,
                           markerSize=4,
                           thickness=1)'''
            
            cv2.drawMarker(frame, (int(left_x1), int(left_y1)), (255, 255, 0), markerType=cv2.MARKER_STAR,
                           markerSize=4,
                           thickness=1)
            

            d1=self.euclidean_distance((left_eye_coords[1][0],left_eye_coords[1][1]),(left_eye_coords[3][0],left_eye_coords[3][1]))
            d2=self.euclidean_distance((right_eye_coords[1][0],right_eye_coords[1][1]),(right_eye_coords[3][0],right_eye_coords[3][1]))
            
            
            
            if d1<0.24 and d2<0.235:
                cv2.putText(frame, 'blink detected', (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0, 255), 3)
                self.blink_detected=True
            else:
                self.blink_detected=False
                
                

        
        #leftpupil = (left_x, left_y)
        leftpupil=(left_x1,left_y1)
        rightpupil = (right_x, right_y)
        
        
        self.gaze_calculator.left_pupil = leftpupil
        self.gaze_calculator.right_pupil = rightpupil
        self.gaze_calculator.right_pupil_circumfrence = right_pupil_circumfrence
        self.gaze_calculator.left_pupil_circumfrence = left_pupil_circumfrence
        
        self.gaze_calculator.nose = nose_2d
        self.gaze_calculator.pitch=pitch
        self.gaze_calculator.yaw=yaw
        self.gaze_calculator.roll=roll

        left_eye_angle = self.calculate_angle(self.gaze_calculator.left_pupil, webcam_center)


        # Calculate angle between webcam and right eye
        right_eye_angle = self.calculate_angle(self.gaze_calculator.right_pupil, webcam_center)


        # Calculate distance between webcam and eyes (Euclidean distance)
        left_eye_distance = np.linalg.norm(np.array(self.gaze_calculator.left_pupil) - np.array(webcam_center))
        right_eye_distance = np.linalg.norm(np.array(self.gaze_calculator.right_pupil) - np.array(webcam_center))
        self.gaze_calculator.left_angle=left_eye_angle
        self.gaze_calculator.right_angle=right_eye_angle
        self.gaze_calculator.distance_left=left_eye_distance
        self.gaze_calculator.distance_right=right_eye_distance
        text_left = f"Left Eye Distance: {self.gaze_calculator.distance_left}"
        text_right = f"Right Eye Distance: {self.gaze_calculator.distance_right}"

        # Define the position for the text
        position_left = (10, 30)  # Example position for left eye distance
        position_right = (10, 60)  # Example position for right eye distance

        # Define the font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (50, 20, 155)  # White color
        thickness = 2

        # Assuming 'frame' is the variable holding your image/frame
        # Overlay text on the frame
        cv2.putText(frame, text_left, position_left, font, font_scale, font_color, thickness)
        cv2.putText(frame, text_right, position_right, font, font_scale, font_color, thickness)

        return frame


if __name__ == "__main__":
    eye_tracker = EyeTracker()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        processed_frame = eye_tracker.process_frame(frame)

        cv2.imshow("img", processed_frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
