
import pyautogui
import cv2
import tkinter as tk
from eye_tracker import EyeTracker
from calibration10 import calibrate
from calibration10 import CalibrationSVR
from screen import Screen


from instruction_screen import show_instruction_screen 
#imprort pygame 

calibration = CalibrationSVR()
RES_SCREEN = pyautogui.size()
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1200

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

    
def show_instruction_dialog() :
    dialog = tk.Toplevel()
    dialog.title("Instructions")
    dialog.geometry("500x300")

    instructions_label = tk.Label(dialog, text ="Follow the instructions and click OK when ready.")
    instructions_label.pack(pady=20)

    ok_button = tk.Button(dialog, text="OK", command=dialog.destroy)
    ok_button.pack(pady=10)


def main():
    # Show instruction screen before main implementation
    show_instruction_screen()
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Set up the eye tracker
    eye_tracker = EyeTracker()
    
    screen = Screen(SCREEN_WIDTH, SCREEN_HEIGHT)

    cv2.namedWindow("frame")

    screen.clean()
    calibration_completed = False

    while True:
        _, frame = camera.read()

        processed_frame = eye_tracker.process_frame(frame)
        img_h, img_w,img_c = frame.shape
        x_coordinate = int((RES_SCREEN[0] - img_w) // 2)
        y_coordinate = int((RES_SCREEN[1] - img_h) // 3)
        cv2.moveWindow("frame", x_coordinate, y_coordinate)# Process the current frame
        cv2.imshow('frame', processed_frame)


        # Handle user input
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Esc key to exit
            break
        elif k == ord('c'):  # 'c' key to start calibration
            #play_beep()
            screen.mode = "calibration"
            screen.draw_center()

            calibrate(camera, screen, eye_tracker)


            calibration_completed = True


if __name__ == '__main__':
    main()
