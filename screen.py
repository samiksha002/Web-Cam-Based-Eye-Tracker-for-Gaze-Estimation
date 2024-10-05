import numpy as np
import cv2
import pyautogui

RES_SCREEN = pyautogui.size()  # RES_SCREEN[0] -> width


# RES_SCREEN[1] -> heigth

class Screen:
    def __init__(self, width=1800, height=1169, background_color=(255, 255, 255)):
        self.width = width
        self.height = height
        self.pointer = (0, 0)
        self.mode = "normal"

        # Initialize the screen with the specified background color
        self.screen = np.full((height, width, 3), background_color, dtype=np.uint8)
        self.clean()

    def change_background_color(self, new_color):
        # Change the background color of the screen
        self.screen[:, :] = new_color
    

    def update(self, gaze):
        self.pointer = gaze

    def clean(self):
        self.screen[:, :] = [41, 44, 51]
        #self.print_instructions()

    def clean_1(self):
        self.screen[:, :] = [41, 44, 51]

    def draw(self, point, progress=0, point_index=0):
        x, y = point

        if progress - point_index == 1.0:
            cv2.circle(self.screen, (x, y), 22, (0, 0, 0), -1)
        else:
            cv2.circle(self.screen, (x, y), 22, (255, 255, 255), -1) #color 255,255,255

        if progress > 0:
            # Ellipse parameters
            radius = 15
            axes = (radius, radius)
            angle = 0
            start_angle = 0
            end_angle = 360 * (progress - point_index)
            cv2.ellipse(self.screen, (x, y), axes, angle, start_angle, end_angle, (0, 0, 0), 2)

    def draw_center(self):
        x, y = (int(0.5 * self.width), int(0.5 * self.height))
        cv2.circle(self.screen, (x, y), 22, (255, 255, 255), -1)

    def draw_pointer(self):
        x, y = self.pointer
        cv2.circle(self.screen, (x, y), 22, (0, 0, 0), -1)

    def print_instructions(self):
        x, y0, dy = int(0.03 * self.width), int(0.8 * self.height), 35

        if self.mode == "normal":
            instructions = "Press:\nESC to quit\nc to start calibration"
        if self.mode == "calibration":
            #            instructions = "Press:\nESC to terminate\nn to next calibration step"
            instructions = "Press:\nESC to terminate calibration"

        for i, line in enumerate(instructions.split('\n')):
            y = y0 + i * dy
            cv2.putText(img=self.screen, text=line, org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                        color=(0, 0, 0), thickness=2)

    def print_message(self, msg):

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 2
        th = 3

        for i, line in enumerate(msg.split('\n')):
            textsize = cv2.getTextSize(line, font, fs, th)[0]
            x = (self.width - textsize[0]) // 2
            y0, dy = (self.height + textsize[1]) // 2, textsize[1] + 30

            y = y0 + i * dy
            cv2.putText(img=self.screen, text=line, org=(x, y), fontFace=font, fontScale=fs, color=(0, 0, 0),
                        thickness=th)

    

    def show(self):
        cv2.namedWindow("screen", cv2.WINDOW_NORMAL)  # Create the window with the WINDOW_NORMAL flag

        # Calculate the x and y coordinates to center the window
        x_coordinate = int((RES_SCREEN[0] - self.width) // 2)
        y_coordinate = int((RES_SCREEN[1] - self.height) // 2)

        cv2.setWindowProperty("screen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # cv2.setWindowProperty("screen", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
        cv2.imshow("screen", self.screen)
    
    def draw_rectangles(self, border_color=(0, 0, 0), border_thickness=2):
        # Define the dimensions and properties of the grid
        rows = 4
        cols = 4
        rect_width = self.width // cols
        rect_height = self.height // rows

        # Define colors for the rectangles
        colors = [
            (255, 255, 255),  # Red
            (255, 255, 255),  # Green
            (255, 255, 255),  # Blue
            (255, 255, 255),  # Yellow
            (255, 255, 255),  # Magenta
            (255, 255, 255),  # Cyan
        ]

        # Ensure that the colors list has enough elements for the grid
        num_colors_needed = rows * cols
        colors_needed = colors * (num_colors_needed // len(colors)) + colors[:num_colors_needed % len(colors)]

        # Loop through rows and columns to draw rectangles with borders
        for row in range(rows):
            for col in range(cols):
                x1 = col * rect_width
                y1 = row * rect_height
                x2 = (col + 1) * rect_width
                y2 = (row + 1) * rect_height

                color = colors_needed[row * cols + col]  # Cycle through colors
                cv2.rectangle(self.screen, (x1, y1), (x2, y2), color, -1)  # -1 for filled rectangle
                cv2.rectangle(self.screen, (x1, y1), (x2, y2), border_color, border_thickness)  # Border

