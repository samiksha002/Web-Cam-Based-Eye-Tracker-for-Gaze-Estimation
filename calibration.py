import cv2
import pandas as pd
import numpy as np
import pyautogui
from sklearn.svm import SVR
from eye_tracker import EyeTracker
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import csv
import pygame
import time

from eye_tracker import EyeTracker
# Initialize Pygame
pygame.init()

# Load the sound file
pygame.mixer.init()
#sound = pygame.mixer.Sound("start-13691.mp3")


FRAME_WIDTH = 640
FRAME_HEIGHT = 480
RES_SCREEN = pyautogui.size()

# CSV file path
csv_file_path = "eye2.csv"
csv_file_path2='predicted_eye10.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["screen_coordinates", "left_x", "left_y", "right_x", "right_y", "nose_x", "nose_y"])

with open(csv_file_path2, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["screen_coordinates", "left_x", "left_y", "right_x", "right_y", "nose_x", "nose_y"])


class CalibrationSVR:
    def __init__(self, base_model=None, stacking_model=None):
        self.base_model = base_model
        self.stacking_model = stacking_model

    def update(self, data):
        screen_coordinates = np.empty((0, 2))
        all_pupil_coords = np.empty((0, 6))

        for screen_point, (left_pupil, right_pupil, nose) in data:
            if left_pupil is not None and right_pupil is not None and nose is not None:
                screen_coordinates = np.vstack((screen_coordinates, screen_point))
                all_pupil_coords = np.vstack((all_pupil_coords, left_pupil + right_pupil + nose))

            else:
                print("Missing data for screen point:", screen_point)

        calibration_data = pd.DataFrame({
            'left_x': all_pupil_coords[:, 0],
            'left_y': all_pupil_coords[:, 1],
            'right_x': all_pupil_coords[:, 2],
            'right_y': all_pupil_coords[:, 3],
            'nose_x': all_pupil_coords[:, 4],
            'nose_y': all_pupil_coords[:, 5]
        })

    
        df_screen_coordinates = pd.DataFrame(screen_coordinates, columns=['screen_x', 'screen_y'])

        self.base_model.fit(calibration_data, df_screen_coordinates)

        # Predict using the base model for stacking
        base_model_predictions = self.base_model.predict(calibration_data)

        # Fit the stacking model
        self.stacking_model.fit(base_model_predictions, screen_coordinates)

    def predict(self, real_time_data):
        # Predict using the base model (e.g., SVR)
        base_model_predictions = self.base_model.predict(real_time_data)

        # Predict using the stacking model
        final_predictions = self.stacking_model.predict(base_model_predictions)

        return final_predictions


def move_mouse(screen_coordinates):
    scaling_factor = 1
    x, y = screen_coordinates
    original_failsafe_setting = pyautogui.FAILSAFE
    pyautogui.FAILSAFE = False
    pyautogui.moveTo(x * scaling_factor, y * scaling_factor)
    pyautogui.FAILSAFE = original_failsafe_setting


def get_current_key(screen_coordinates, box_width, box_height, keyboard_layout):
    # Calculate the row and column index of the current key
    col_index = int(screen_coordinates[0] // box_width)
    row_index = int(screen_coordinates[1] // box_height)

    # Ensure the indices are within the valid range
    col_index = max(0, min(col_index, len(keyboard_layout[0]) - 1))
    row_index = max(0, min(row_index, len(keyboard_layout) - 1))

    # Get the current key based on the indices
    current_key = keyboard_layout[row_index][col_index]

    return current_key


recent_keys = []
all_stable_keys = []


def visualize_mouse_movement(screen, mouse_positions, screen_coordinates):
    screen.clean_1()
    global recent_keys, all_stable_keys

    # Draw grid lines
    keyboard_layout = [
        ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
        ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
        ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';'],
        ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/'],
        ['Del','Space','[',']','?','+','-','*','_','=']
          # Add Backspace button
    ]

    # Define box parameters
    rows, cols = len(keyboard_layout), len(keyboard_layout[0])
    box_color = (0, 0, 0)  # White color for boxes
    box_thickness = 2

    # Calculate box width and height
    box_width = screen.width // cols
    box_height = screen.height // rows


    current_key = get_current_key(screen_coordinates, box_width, box_height, keyboard_layout)
    recent_keys.append(current_key)
    stability_threshold = 5

    if len(recent_keys) >= stability_threshold:
        most_common_key = max(set(recent_keys), key=recent_keys.count)

        if recent_keys.count(most_common_key) == stability_threshold:
            if most_common_key == 'Del':
                # Perform backspace functionality
                if all_stable_keys:  # If there are stable keys recorded
                    all_stable_keys.pop()
            elif most_common_key == 'Space':
                all_stable_keys.append(' ')

            else:
                all_stable_keys.append(most_common_key)

                # Change color or make it blink for a few seconds
            for i in range(rows):
                for j in range(cols):
                    if keyboard_layout[i][j] == most_common_key:
                        x1 = j * box_width
                        y1 = i * box_height
                        x2 = (j + 1) * box_width
                        y2 = (i + 1) * box_height

                            # Draw blinking or color-changing box
                        for _ in range(5):  # Blinking for 5 times
                            cv2.rectangle(screen.screen, (x1, y1), (x2, y2), (255, 0, 0), -1)  # Change color to red
                            screen.show()
                            time.sleep(0.08)
                            cv2.rectangle(screen.screen, (x1, y1), (x2, y2), box_color, -1)  # Change back to original color
                            screen.show()
                            time.sleep(0.1)
                        #sound.play()

        recent_keys = []

    label = 'Stable Keys: ' + ''.join(all_stable_keys)
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    label_x = 10
    label_y = 30
    cv2.putText(screen.screen, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 50, 255), 2)

    # Draw boxes and labels
    for i in range(rows):
        for j in range(cols):
            # Calculate box coordinates
            x1 = j * box_width
            y1 = i * box_height
            x2 = (j + 1) * box_width
            y2 = (i + 1) * box_height

            # Draw box
            cv2.rectangle(screen.screen, (x1, y1), (x2, y2), box_color, box_thickness)

            # Draw label
            label = keyboard_layout[i][j]
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            label_x = x1 + (box_width - label_size[0]) // 2
            label_y = y1 + (box_height + label_size[1]) // 2
            cv2.putText(screen.screen, label, (label_x, label_y), cv2.FONT_HERSHEY_TRIPLEX, 1.3, (50, 155, 255), 4)



def calibrate(camera, screen, eye_tracker):
    N_REQ_COORDINATES = 10
    N_SKIP_COORDINATES = 0
    MAX_HISTORY_POINTS = 10

    screen.clean()

    base_model = MultiOutputRegressor(SVR(kernel='poly', C=150, degree=3, epsilon=0.1))
    stacking_model = LinearRegression()  # RandomForestRegressor()
    calibration = CalibrationSVR(base_model, stacking_model)
    calibration_points = calculate_points(screen)

    coordinates = []
    completed = False
    enough = 0
    skip = 0
    point = calibration_points.pop(0)
    screen.draw(point)
    screen.show()
    mouse_positions = []

    while point:
        the_list = calculate_points(screen)
        key = the_list.index(point)
        screen.draw(point)
        screen.change_background_color((0, 0, 0))
        screen.show()

        _, frame = camera.read()

        processed_frame = eye_tracker.process_frame(frame)
        right_pupil_coordinates = eye_tracker.gaze_calculator.right_pupil
        left_pupil_coordinates = eye_tracker.gaze_calculator.left_pupil
        nose_coordinates = eye_tracker.gaze_calculator.nose
        left_pupil_circumference = eye_tracker.gaze_calculator.left_pupil_circumfrence
        right_pupil_circumference = eye_tracker.gaze_calculator.right_pupil_circumfrence

        cv2.namedWindow("frame")
        dec_frame = processed_frame
        dec_frame = cv2.resize(dec_frame, (int(FRAME_WIDTH / 2), int(FRAME_HEIGHT / 2)))
        cv2.moveWindow("frame", 0, 0)
        cv2.imshow('frame', dec_frame)

        coordinates_tuple = (left_pupil_coordinates, right_pupil_coordinates, nose_coordinates)
        with open(csv_file_path, mode='a', newline='') as file_append:
            writer_append = csv.writer(file_append)
            writer_append.writerow([point, left_pupil_coordinates[0], left_pupil_coordinates[1],
                                    right_pupil_coordinates[0], right_pupil_coordinates[1],
                                    nose_coordinates[0], nose_coordinates[1]])

        if coordinates_tuple and skip < N_SKIP_COORDINATES:
            skip += 1
            continue

        if coordinates_tuple:
            coordinates.append((point, coordinates_tuple))
            enough += 1

        progress = len(coordinates) / N_REQ_COORDINATES
        screen.draw(point, progress=progress, point_index=key)
        screen.show()

        if enough >= N_REQ_COORDINATES and len(calibration_points) > 0:
            point = calibration_points.pop(0)
            skip = 0
            enough = 0
            screen.draw(point)
            screen.show()

        if enough >= N_REQ_COORDINATES and len(calibration_points) == 0:
            screen.clean()
            completed = True
            break

        k = cv2.waitKey(1) & 0xff

        if k == 1048603 or k == 27:
            screen.mode = "normal"
            screen.clean()
            screen.show()
            break

    if completed:
        
        calibration.update(coordinates)
        eye_tracker.calibration = calibration 

        screen.mode = "calibrated"
        screen.show()
        while True:
            _, frame = camera.read()

            processed_frame = eye_tracker.process_frame(frame)
            right_pupil_coordinates = eye_tracker.gaze_calculator.right_pupil
            left_pupil_coordinates = eye_tracker.gaze_calculator.left_pupil
            nose = eye_tracker.gaze_calculator.nose

            real_time_data = pd.DataFrame({
                'left_x': [left_pupil_coordinates[0]],
                'left_y': [left_pupil_coordinates[1]],
                'right_x': [right_pupil_coordinates[0]],
                'right_y': [right_pupil_coordinates[1]],
                'nose_x': [nose[0]],
                'nose_y': [nose[1]]
            })

            screen_coordinates_1 = calibration.predict(real_time_data)
            with open(csv_file_path2, mode='a', newline='') as file_append:
                writer_append = csv.writer(file_append)
                writer_append.writerow([screen_coordinates_1, left_pupil_coordinates[0], left_pupil_coordinates[1],
                                        right_pupil_coordinates[0], right_pupil_coordinates[1],
                                        nose[0], nose[1]])
            screen_prev_position = (0, 0)

            if mouse_positions:
                screen_prev_position = mouse_positions[-1]

            screen_coordinates = screen_coordinates_1[0]
            coordinates = ((screen_coordinates[0] + screen_prev_position[0]) / 2,
                           (screen_coordinates[1] + screen_prev_position[1]) / 2)



            mouse_positions.append(tuple(coordinates))
            mouse_positions = mouse_positions[-MAX_HISTORY_POINTS:]

            move_mouse(coordinates)
            visualize_mouse_movement(screen, mouse_positions, screen_coordinates)
            screen.show()

            #k = cv2.waitKey(1) & 0xff

            #if k == 1048603 or k == 27:
            #    break

            k1 = cv2.waitKey(1) & 0xFF
            if k1 == 27:
                break
            elif k1 == ord('c'):  # 'c' key to start calibration
                screen.mode = "calibration"

                calibrate(camera, screen, eye_tracker)


def calculate_points(screen):
     points = []

     # center
     p = (int(0.5 * screen.width), int(0.5 * screen.height))
     points.append(p)

     # top left
     p = (int(0.05 * screen.width), int(0.05 * screen.height))
     points.append(p)

     # top mid left
     p = (int(0.25 * screen.width), int(0.05 * screen.height))
     points.append(p)

     # top
     p = (int(0.5 * screen.width), int(0.05 * screen.height))
     points.append(p)

#     # top mid right
     p = (int(0.75 * screen.width), int(0.05 * screen.height))
     points.append(p)

     # top right
     p = (int(0.95 * screen.width), int(0.05 * screen.height))
     points.append(p)

#     # 2nd quad
     p = (int(0.25 * screen.width), int(0.25 * screen.height))
     points.append(p)

#     # mid st quad
     p = (int(0.50 * screen.width), int(0.25 * screen.height))
     points.append(p)

#     # 1st quad
     p = (int(0.75 * screen.width), int(0.25 * screen.height))
     points.append(p)

#     # left
     p = (int(0.05 * screen.width), int(0.5 * screen.height))
     points.append(p)

#     # left mid
     p = (int(0.25 * screen.width), int(0.5 * screen.height))
     points.append(p)

#     # right mid
     p = (int(0.75 * screen.width), int(0.5 * screen.height))
     points.append(p)

#     # right
     p = (int(0.95 * screen.width), int(0.5 * screen.height))
     points.append(p)

#     # 3rd quad
     p = (int(0.25 * screen.width), int(0.75 * screen.height))
     points.append(p)

#     # quad mid
     p = (int(0.50 * screen.width), int(0.75 * screen.height))
     points.append(p)

#     # 4th quad
     p = (int(0.75 * screen.width), int(0.75 * screen.height))
     points.append(p)

#     # bottom left
     p = (int(0.05 * screen.width), int(0.95 * screen.height))
     points.append(p)

#     # bottom left mid
     p = (int(0.25 * screen.width), int(0.95 * screen.height))
     points.append(p)

#     # bottom
     p = (int(0.5 * screen.width), int(0.95 * screen.height))
     points.append(p)

#     # bottom right mid
     p = (int(0.75 * screen.width), int(0.95 * screen.height))
     points.append(p)

#     # bottom right
     p = (int(0.95 * screen.width), int(0.95 * screen.height))
     points.append(p)

     return points

