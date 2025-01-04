import cv2
import numpy as np
import math
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import Tk, Button, filedialog, Label, Toplevel, Listbox, Scrollbar, simpledialog, messagebox
import logging
import os
import sys
import threading
import requests
from datetime import datetime
import joblib

# Setup logging
logging.basicConfig(
    filename="fall_detector.log",  # Log file
    level=logging.DEBUG,  # Log all levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s] %(message)s",  # Format for log messages
)
logging.info("Program started.")

# Global variable for video source


# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#Painting
video_source = None
drawing = False  # Whether the user is drawing
brush_radius = 10  # Radius of the paintbrush
draw_mode = True  # Draw mode flag
painting_complete = False  # Flag to indicate painting is done
drawn_contours = None #To save drawn contours

# Global variables to store the centroid's Y-coordinate
fall_centroid_y = None
up_centroid_y = None
is_first_frame = True  # Flag to identify the first frame
c=0
# Updated fall detection system
h_min=0
s_min=0
v_min=0
h_max=0
s_max=0
v_max=0
filtering_complete=False

static_counter = None
previous_frame = None
static_threshold=2
static_pixels=None
min_detection_confidence=0.7
Detect_Fail_Counter=0
y_DIFF_THRESHOLD = 0.01 
overlap=None

def reset_everything():
    global video_source, drawing, brush_radius, draw_mode, painting_complete, drawn_contours, fall_centroid_y, up_centroid_y, is_first_frame, c, h_min,s_min,v_min,h_max, s_max,v_max,filtering_complete ,static_counter,previous_frame,static_threshold,static_pixels,min_detection_confidence,Detect_Fail_Counter,y_DIFF_THRESHOLD,overlap
        
    #Painting
    video_source = None
    drawing = False  # Whether the user is drawing
    brush_radius = 10  # Radius of the paintbrush
    draw_mode = True  # Draw mode flag
    painting_complete = False  # Flag to indicate painting is done
    drawn_contours = None #To save drawn contours

    # Global variables to store the centroid's Y-coordinate
    fall_centroid_y = None
    up_centroid_y = None
    is_first_frame = True  # Flag to identify the first frame
    c=0

    #HSV
    h_min=0
    s_min=0
    v_min=0
    h_max=0
    s_max=0
    v_max=0
    filtering_complete=False

    #Static
    static_counter = None
    previous_frame = None
    static_threshold=2
    static_pixels=None
    overlap=None

    #Fall detection
    min_detection_confidence=0.7
    Detect_Fail_Counter=0
    y_DIFF_THRESHOLD = 0.01 



def toggle_draw(event, x, y, flags, param):
    """Mouse callback to handle drawing on the frame."""
    global drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(param, (x, y), brush_radius, (0, 0, 255), -1)  # Draw a red circle
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def nothing(x):
    pass


def ask_draw_mode():
    """Create a Tkinter popup with Yes/No buttons to ask if the user wants to draw."""
    global draw_mode
    root = Tk()
    root.title("Drawing Mode")

    def set_yes():
        """Set draw_mode to True and close the popup."""
        global draw_mode
        draw_mode = True
        root.destroy()

    def set_no():
        """Set draw_mode to False and close the popup."""
        global draw_mode
        draw_mode = False
        root.destroy()

def ask_filter_mode():
    """Create a Tkinter popup with Yes/No buttons to ask if the user wants to draw."""
    global filtering_complete
    root = Tk()
    root.title("Filtering Mode")

    def set_complete():
        """Set draw_mode to True and close the popup."""
        global filtering_complete
        filtering_complete= True
        cv2.destroyWindow("Thermal_Masking")
        root.destroy()     

    # Create label and button
    Label(root, text="Have you completed filtering?").pack(pady=10)
    Button(root, text="Yes", command=set_complete, width=10).pack(pady=20)
    # Center the window
    root.eval('tk::PlaceWindow . center')
    root.mainloop()

def paint_mode(frame):
    """Activate paint mode, allow drawing, and fill closed loops with green color."""
    global painting_complete, drawing, drawn_contours

    # Initialize the list to store contours
    drawn_contours = []

    # Create a blank image to store drawing and detect contours
    drawn_mask = np.zeros_like(frame, dtype=np.uint8)

    # Create a window for drawing
    cv2.namedWindow("Paint Mode")
    cv2.setMouseCallback("Paint Mode", toggle_draw, drawn_mask)

    while not painting_complete:
        # Display the drawing frame
        display_frame = cv2.addWeighted(frame, 0.5, drawn_mask, 0.5, 0)  # Blend original frame and drawing
        cv2.imshow("Paint Mode", display_frame)
        if cv2.getWindowProperty("Paint Mode", cv2.WND_PROP_VISIBLE) < 1:
            painting_complete = True
            break
        # Wait for the user to press 'q' to finish painting
        if cv2.waitKey(1) & 0xFF == ord("q"):
            painting_complete = True

        # Find contours in the drawn mask
        gray_mask = cv2.cvtColor(drawn_mask, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Save the contours
        drawn_contours = contours

        # Fill closed loops with green color
        for contour in contours:
            # Fill the contour area
            cv2.drawContours(frame, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)

    # Close the drawing window
    cv2.destroyWindow("Paint Mode")
    return drawn_contours

def resource_path(relative_path):
    try:
        base_path=sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path,relative_path)
# Load the AdaBoost model
model = joblib.load(resource_path("ada_boost_ensemble_model.pkl"))
scaler = joblib.load(resource_path("scaler.pkl"))

def calculate_distances(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def extract_features_for_prediction(landmarks):

    # Extract required landmarks
    left_shoulder = [landmarks[11].x, landmarks[11].y]
    right_shoulder = [landmarks[12].x, landmarks[12].y]
    left_hip = [landmarks[23].x, landmarks[23].y]
    right_hip = [landmarks[24].x, landmarks[24].y]
    left_knee = [landmarks[25].x, landmarks[25].y]
    right_knee = [landmarks[26].x, landmarks[26].y]
    left_ankle = [landmarks[27].x, landmarks[27].y]
    right_ankle = [landmarks[28].x, landmarks[28].y]

    # Calculate angles
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

    # Calculate distances
    shoulder_distance = calculate_distances(left_shoulder, right_shoulder)
    knee_distance = calculate_distances(left_knee, right_knee)
    left_ankle_to_hip = calculate_distances(left_ankle, left_hip)
    right_ankle_to_hip = calculate_distances(right_ankle, right_hip)

    # Return features as a list
    return [
        left_knee_angle, right_knee_angle, left_hip_angle, right_hip_angle,
        shoulder_distance, knee_distance, left_ankle_to_hip, right_ankle_to_hip
    ]

def prepare_features_for_model(landmarks,scaler=scaler):
    """
    Extract features from an image and format them for the model.
    :param image_path: Path to the image file.
    :param scaler: Pre-trained scaler used for training the model.
    :param pose: Mediapipe pose instance.
    :return: Preprocessed feature array for prediction or None if no landmarks are found.
    """
    features = extract_features_for_prediction(landmarks)
    
    # Convert features to a 2D array (1 sample, n_features)
    features_array = np.array(features).reshape(1, -1)

    # Scale the features using the scaler (ensures compatibility with the model)
    scaled_features = scaler.transform(features_array)
    
    return scaled_features


# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    try:
        a = np.array(a)  # First
        b = np.array(b)  # Mid
        c = np.array(c)  # End

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        logging.debug(f"Calculated angle: {angle}")
        return angle
    except Exception as e:
        logging.error(f"Error in calculate_angle: {e}")
        return None



def get_current_datetime():
    now = datetime.now()
    return now.strftime("%m/%d/%y,%H:%M:%S")

def send_message():
    url = "http://elderly-healthcare.infivr.com/api/events/create"
    payload = {
        "eventName": "Fall",
        "userName": "Satyam",
        "applicationName": "FALL_DETECTOR",
        "eventTime": get_current_datetime(),
        "severity": "high",
        "sensorId": "1"
    }
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)

# Function to list all available cameras
def list_cameras():
    try:
        index = 0
        available_cameras = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:  # If the camera cannot be opened, stop checking
                break
            available_cameras.append(f"Camera {index}")
            cap.release()
            index += 1
        logging.info(f"Available cameras: {available_cameras}")
        return available_cameras
    except Exception as e:
        logging.error(f"Error in list_cameras: {e}")
        return []


# Function to open a new window to select a camera
def select_camera_window():
    global video_source
    print(video_source)

    try:
        # Create a new window
        camera_window = Toplevel(root)
        camera_window.title("Select Camera")
        camera_window.geometry("300x300")

        # Label
        label = Label(camera_window, text="Select a Camera", font=("Helvetica", 12))
        label.pack(pady=10)

        # Listbox to show available cameras
        camera_listbox = Listbox(camera_window, width=30, height=10)
        camera_listbox.pack(pady=10)

        # Add a scrollbar
        scrollbar = Scrollbar(camera_window, orient="vertical", command=camera_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        camera_listbox.config(yscrollcommand=scrollbar.set)

        # List available cameras
        cameras = list_cameras()
        for cam in cameras:
            camera_listbox.insert("end", cam)

        # Function to set the selected camera and close the window
        def confirm_camera():
            selected_index = camera_listbox.curselection()
            if selected_index:
                video_source = int(selected_index[0])  # Get the camera index
                camera_window.destroy()  # Close the camera selection window
                root.destroy()  # Close the main window
                logging.info(f"Selected camera: {video_source}")
                fall_detection_system(video_source)

        # Button to confirm the selection
        confirm_button = Button(camera_window, text="Confirm", command=confirm_camera)
        confirm_button.pack(pady=10)

    except Exception as e:
        logging.error(f"Error in select_camera_window: {e}")

def check_fall(bbox):
    x_min, y_min, x_max, y_max = bbox
    width = abs(x_max - x_min)
    height = abs(y_max - y_min)
    print(height/width)
    return (height/width)<0.9

def check_up(bbox):
    x_min, y_min, x_max, y_max = bbox
    width = abs(x_max - x_min)
    height = abs(y_max - y_min)
    print(width/height)
    return (height/width)>1.5

def get_pixel_coordinates(landmark, width, height):
    """
    Convert normalized coordinates (from MediaPipe landmarks) to pixel values.
    """
    return int(landmark.x * width), int(landmark.y * height)

def calculate_midpoint(point1, point2):
    """
    Calculate the midpoint between two points.
    """
    return (
        (point1[0] + point2[0]) // 2,
        (point1[1] + point2[1]) // 2,
    )

def calculate_angle_with_vertical(midpoint1, midpoint2):
    """
    Calculate the angle between the line formed by two midpoints and the vertical axis (y-axis).
    """
    dx = midpoint2[0] - midpoint1[0]
    dy = midpoint2[1] - midpoint1[1]
    angle = np.degrees(np.arctan2(dx, dy))  # Angle in degrees
    return angle

def draw_line_and_calculate_angle(landmarks, frame):
    """
    Draw the line between the midpoints of shoulders and hips, and calculate the angle with the vertical axis.
    Parameters:
        landmarks: MediaPipe pose landmarks.
        frame: The current video frame (BGR).
    Returns:
        frame: The modified frame with the line and angle displayed.
    """
    # Extract frame dimensions
    h, w, _ = frame.shape

    # Get pixel coordinates of shoulders and hips
    left_shoulder_px = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], w, h)
    right_shoulder_px = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], w, h)
    left_hip_px = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_HIP], w, h)
    right_hip_px = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_HIP], w, h)

    # Calculate midpoints
    shoulder_midpoint = calculate_midpoint(left_shoulder_px, right_shoulder_px)
    hip_midpoint = calculate_midpoint(left_hip_px, right_hip_px)

    # Draw points and line
    cv2.circle(frame, shoulder_midpoint, 5, (0, 255, 0), -1)  # Green point
    cv2.circle(frame, hip_midpoint, 5, (0, 255, 0), -1)        # Green point
    cv2.line(frame, shoulder_midpoint, hip_midpoint, (0, 0, 255), 2)  # Red line

    # Calculate angle with vertical axis
    angle = calculate_angle_with_vertical(shoulder_midpoint, hip_midpoint)

    return frame, angle, draw_bounding_box(frame, landmarks, shoulder_midpoint)

def draw_bounding_box(frame, landmarks, chestpoint):
    """
    Draws a bounding box around the left hip, right hip, left knee, and right knee.

    Parameters:
        frame: The image frame (BGR) to draw on.
        landmarks: MediaPipe landmarks object containing pose landmarks.

    Returns:
        frame: The modified frame with the bounding box drawn.
    """
    # Extract relevant landmark indices
    relevant_indices = [23, 24, 25, 26]  # Left hip, right hip, left knee, right knee

    # Collect the (x, y) coordinates of the relevant landmarks
    points = []
    for index in relevant_indices:
        landmark = landmarks[index]
        if landmark.visibility > 0.5:  # Consider landmarks with good visibility
            x = int(landmark.x * frame.shape[1])  # Convert normalized x to pixel value
            y = int(landmark.y * frame.shape[0])  # Convert normalized y to pixel value
            points.append((x, y))

    if points:
        # Calculate the bounding box dimensions
        x_min = min(point[0] for point in points)
        y_min = min(point[1] for point in points)
        x_max = max(point[0] for point in points)
        y_max = max(point[1] for point in points)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
    else:
        return False
    return chestpoint[0]>x_min and chestpoint[1]<x_max and chestpoint[1]>y_min and chestpoint[1]<y_max
# Function to process the fall detection system
# Function to calculate the centroid of all joint points
# Threshold for Y-difference
 # Adjust this value based on the scaling of your coordinate system
# Function to calculate the centroid of a list of points
from PIL import Image, ImageEnhance

def calculate_centroid(points):
    try:
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)
        return centroid_x, centroid_y
    except Exception as e:
        logging.error(f"Error in calculate_centroid: {e}")
        return None, None

def process_frame(frame):
    # Convert the frame to a PIL image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Step 1: Enhance contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    contrast_image = enhancer.enhance(2.0)  # Increase contrast

    # Step 2: Convert to RGB and normalize
    rgb_image = contrast_image.convert("RGB")

    # Step 3: Convert to OpenCV format for additional preprocessing
    cv_image = np.array(rgb_image)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

    # Step 4: Denoise the image
    denoised_image = cv2.fastNlMeansDenoisingColored(cv_image, None, 10, 10, 7, 21)

    # Step 5: Highlight edges to make the pose more detectable
    gray_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, threshold1=50, threshold2=150)

    # Convert edges back to 3-channel image
    edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Blend edges with the original image
    blended = cv2.addWeighted(denoised_image, 0.7, edges_3channel, 0.3, 0)
    
    return blended

def fall_detection_system(video_source):
    logging.info("Fall Detection Started")
    global fall_centroid_y, up_centroid_y, is_first_frame, y_DIFF_THRESHOLD,c,draw_mode,filtering_complete,h_min, s_min, v_min, static_threshold,static_counter,previous_frame,static_pixels,min_detection_confidence,Detect_Fail_Counter,overlap
    try:
        if not painting_complete and draw_mode: draw_cap=cv2.VideoCapture(video_source)
        while draw_cap.isOpened() and not painting_complete and draw_mode:
            ret, frame = draw_cap.read()
            if not ret:
                logging.warning("Video capture ended or failed.")
                break

            # Capture the first frame
            captured_frame = frame.copy()  # Copy the frame to work on it

            # Reset the video to the 0th frame
            draw_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            draw_cap.release()  # Release draw_cap as we no longer need it

            # Perform drawing on the captured frame
            if draw_mode:
                global drawn_contours
                drawn_contours = paint_mode(captured_frame)  # Drawing occurs on the captured frame
                if drawn_contours:
                    print("Drawn contours processed.")
            else:
                break 
        # Create a window
        cv2.namedWindow("Trackbars")
        # Create trackbars for HSV min and max values
        cv2.createTrackbar("H Min", "Trackbars", 0, 179, nothing)
        cv2.createTrackbar("H Max", "Trackbars", 179, 179, nothing)
        cv2.createTrackbar("S Min", "Trackbars", 0, 255, nothing)
        cv2.createTrackbar("S Max", "Trackbars", 255, 255, nothing)
        cv2.createTrackbar("V Min", "Trackbars", 0, 255, nothing)
        cv2.createTrackbar("V Max", "Trackbars", 255, 255, nothing)    
        # cv2.namedWindow("Settings")
        # cv2.createTrackbar("Brightness Threshold MID", "Settings", 15, 240, nothing)
        while not filtering_complete:
                filter_cap=cv2.VideoCapture(video_source)
                ret, frame = filter_cap.read()
                # Capture the first frame
                captured_frame = frame.copy()  # Copy the frame to work on it
                # Reset the video to the 0th frame
                filter_cap.release()  # Release draw_cap as we no longer need it
                filter_cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
            # Recolor image to RGB
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # Get current positions of trackbars
                h_min = cv2.getTrackbarPos("H Min", "Trackbars")
                h_max = cv2.getTrackbarPos("H Max", "Trackbars")
                s_min = cv2.getTrackbarPos("S Min", "Trackbars")
                s_max = cv2.getTrackbarPos("S Max", "Trackbars")
                v_min = cv2.getTrackbarPos("V Min", "Trackbars")
                v_max = cv2.getTrackbarPos("V Max", "Trackbars")
                lower_bound = np.array([h_min, s_min, v_min])
                upper_bound = np.array([h_max, s_max, v_max])
                mask = cv2.inRange(hsv, lower_bound, upper_bound)
                thermal_image = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
                thermal_result = cv2.bitwise_and(thermal_image, thermal_image, mask=mask)
                cv2.imshow("Thermal_Masking",thermal_result)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # 'c' to call the popup
                    filtering_complete= True
                    cv2.destroyWindow("Thermal_Masking")
                    cv2.destroyWindow("Trackbars")

        cap = cv2.VideoCapture(video_source)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Curl counter variables
        counter = 0
        stage = None
        min_tracking_confidence=0.7
        # Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence) as pose:
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logging.warning("Video capture ended or failed.")
                    break
                
                # Create the mask based on trackbar positions
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_bound = np.array([h_min, s_min, v_min])
                upper_bound = np.array([h_max, s_max, v_max])
                mask = cv2.inRange(hsv, lower_bound, upper_bound)
                thermal_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                thermal_result = cv2.bitwise_and(thermal_image, thermal_image, mask=mask)
                # cv2.imshow("Thermal_Masking",thermal_result)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                gray_frame = cv2.cvtColor(thermal_result.copy(), cv2.COLOR_BGR2GRAY)
                gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
                # gray_frame = process_frame(gray_frame)

                results = pose.process(gray_frame)
                
                if not results.pose_landmarks and min_detection_confidence>=0.6 :
                    min_detection_confidence-=0.1
                    min_tracking_confidence-=0.1
                    results = pose.process(image)
                elif min_detection_confidence<0.7:
                    min_detection_confidence+=0.1
                    min_tracking_confidence+=0.1

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if not results.pose_landmarks:
                    cv2.rectangle(thermal_result, (0, 0), (225, 73), (245, 117, 16), -1)
                    cv2.putText(thermal_result, 'Fall Count', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(thermal_result, str(counter),
                                (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.putText(thermal_result, 'Stage', (120, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(thermal_result, stage,
                                (120, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Display
                    cv2.imshow('Fall Detection System', thermal_result)
                    continue

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    lower_body_landmarks = [
                        landmarks[23].visibility,  # Left hip
                        landmarks[24].visibility,  # Right hip
                        landmarks[25].visibility,  # Left knee
                        landmarks[26].visibility,  # Right knee
                        landmarks[27].visibility,  # Left ankle
                        landmarks[28].visibility,  # Right ankle
                        landmarks[29].visibility,  # Left heel
                        landmarks[30].visibility,  # Right heel
                        landmarks[31].visibility,  # Left foot index
                        landmarks[32].visibility,  # Right foot index
                    ]

                    upper_body_landmarks = [
                        landmarks[11].visibility,  # Left shoulder
                        landmarks[12].visibility,  # Right shoulder
                        landmarks[13].visibility,  # Left elbow
                        landmarks[14].visibility,  # Right elbow
                    ]

                    
                    # Threshold for visibility
                    visibility_threshold = 0.5
                    # Calculate bounding box
                    x_coords = [lm.x for lm in landmarks]
                    y_coords = [lm.y for lm in landmarks]

                    x_min = int(min(x_coords) * frame.shape[1])
                    x_max = int(max(x_coords) * frame.shape[1])
                    y_min = int(min(y_coords) * frame.shape[0])
                    y_max = int(max(y_coords) * frame.shape[0])
                    
                    color = (0, 255, 0)  # Green color for the bounding box (BGR format)
                    thickness = 2        # Thickness of the rectangle border
                    
                    cv2.rectangle(thermal_result, (x_min, y_min), (x_max, y_max), color, thickness)

                    # Get coordinates
                    keypoints = [
                        [landmark.x, landmark.y] for landmark in landmarks
                    ]  # List of [x, y] points

                    # Centroid calculation
                    centroid_x, centroid_y = calculate_centroid(keypoints)

                    # Get specific joint coordinates for angle calculation
        
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                    # Calculate angles
                    angle = calculate_angle(left_hip, left_knee, left_ankle)
                    angle_2 = calculate_angle(right_hip, right_knee, right_ankle)
                    logging.debug(f"Left angle: {angle}, Right angle: {angle_2}")

                    # print("y_DIFF_THRESHOLD",y_DIFF_THRESHOLD)
                    # First frame initialization
                    if is_first_frame:
                        if angle < 60 or angle_2 < 60:  # Fall state
                            stage = "fall"
                            fall_centroid_y = centroid_y  # Initialize fall centroid
                        else:  # Up state
                            stage = "up"
                            up_centroid_y = centroid_y  # Initialize up centroid
                        is_first_frame = False  # Disable first-frame logic
                        continue  # Skip threshold checks for the first frame
                    
                    if drawn_contours:  # Check if the user has drawn contours in the first frame
                        left_hip_point = (int(left_hip[0] * frame.shape[1]), int(left_hip[1] * frame.shape[0]))
                        right_hip_point = (int(right_hip[0] * frame.shape[1]), int(right_hip[1] * frame.shape[0]))

                        left_inside = False
                        right_inside = False

                        # Check each contour for inclusion
                        for contour in drawn_contours:
                            if cv2.pointPolygonTest(contour, left_hip_point, False) >= 0:  # Check if left hip is inside
                                left_inside = True
                            if cv2.pointPolygonTest(contour, right_hip_point, False) >= 0:  # Check if right hip is inside
                                right_inside = True

                        # Proceed only if both hips are inside at least one contour
                        if left_inside and right_inside:
                            logging.info("Both hips are inside the drawn contours. Continuing...")
                            continue
                    # Logic to check fall state
                    thermal_result,angle3,Fall_back=draw_line_and_calculate_angle(landmarks, thermal_result)
                    if (angle > 160 and angle_2 > 160) and check_up((x_min, y_min, x_max, y_max)):  # Check for "up" state
                        
                        c+=1
                        if c>=30:
                            y_DIFF_THRESHOLD=abs(fall_centroid_y - centroid_y)
                            c=0
                            stage = "up"

                        if fall_centroid_y is not None and abs(fall_centroid_y - centroid_y) > y_DIFF_THRESHOLD:
                            stage = "up"
                            up_centroid_y = centroid_y  # Store current "up" centroid
                            y_DIFF_THRESHOLD = abs(fall_centroid_y - centroid_y)
                            c=40
                            logging.info(f"Back to 'up' stage. Fall counter: {counter}")
                    
                    if ((angle < 60 or angle_2 < 60) and stage == "up")  and all(visibility < visibility_threshold for visibility in lower_body_landmarks) :  # Check for "fall" state
                        c-=1
                        if c<0:c=0
                        
                        if up_centroid_y is not None and abs(up_centroid_y - centroid_y) >= y_DIFF_THRESHOLD and abs(angle3)>45 :
                            send_message()
                            print("FALLING-",abs(up_centroid_y - centroid_y))
                            stage = "fall"
                            fall_centroid_y = centroid_y  # Store current "fall" centroid
                            counter += 1
                            c=0
                            y_DIFF_THRESHOLD = abs(up_centroid_y - centroid_y)
                            print("DUE TO FALLING- Angle",y_DIFF_THRESHOLD)
                            logging.info(f"Fall detected! Counter: {counter}")
                        elif y_DIFF_THRESHOLD-abs(up_centroid_y - centroid_y)<=0.04 and abs(angle3)>45 :
                            send_message()
                            print("FALLING-",abs(up_centroid_y - centroid_y))
                            stage = "fall"
                            fall_centroid_y = centroid_y  # Store current "fall" centroid
                            counter += 1
                            c=0
                            y_DIFF_THRESHOLD = abs(up_centroid_y - centroid_y)
                            print("DUE TO FALLING- Angle",y_DIFF_THRESHOLD)
                            logging.info(f"Fall detected! Counter: {counter}")
                    elif all(visibility > visibility_threshold for visibility in upper_body_landmarks) and all(visibility < visibility_threshold for visibility in lower_body_landmarks)  and y_max<frame.shape[1] and y_min>0 and x_min>0 and x_max<frame.shape[0] and stage == "up":
                            send_message()
                            stage = "fall"
                            fall_centroid_y = centroid_y 
                            counter += 1
                            c=0
                            y_DIFF_THRESHOLD = abs(up_centroid_y - centroid_y)
                            print("DUE TO FALLING- Upper Shoulders")
                            logging.info(f"Fall detected! Counter: {counter}")
                    elif check_fall((x_min, y_min, x_max, y_max)) and stage=="up" and abs(angle3)>45 :
                            send_message()
                            stage = "fall"
                            fall_centroid_y = centroid_y 
                            counter += 1
                            c=0
                            y_DIFF_THRESHOLD = abs(up_centroid_y - centroid_y)
                            print("DUE TO FALLING- Box dimension")
                            logging.info(f"Fall detected! Counter: {counter}")
                    elif stage=="up":
                            features = prepare_features_for_model(landmarks, scaler)
                            if (model.predict(features) and abs(angle3)>45) or (Fall_back and (angle < 60 and angle_2 < 60)):
                                send_message()
                                stage = "fall"
                                fall_centroid_y = centroid_y 
                                counter += 1
                                c=0
                                y_DIFF_THRESHOLD = abs(up_centroid_y - centroid_y)
                                print("DUE TO FALLING- MODEL Prediction")
                                logging.info(f"Fall detected! Counter: {counter}")


                except Exception as e:
                    logging.error(f"Error processing landmarks: {e}")
                # Render mediapipe landmarks
                mp_drawing.draw_landmarks(thermal_result, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
                
                # Render the fall counter
                cv2.rectangle(thermal_result, (0, 0), (225, 73), (245, 117, 16), -1)
                cv2.putText(thermal_result, 'Fall Count', (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(thermal_result, str(counter),
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(thermal_result, 'Stage', (120, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(thermal_result, stage,
                            (120, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display
                cv2.imshow('Fall Detection System', thermal_result)
                
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    logging.info("Exit key pressed. Closing application.")
                    break

            cap.release()
            cv2.destroyAllWindows()
            restart_program()
    except Exception as e:
        logging.error(f"Error in fall_detection_system: {e}")



# Function to select a video file
def select_video_file():
    reset_everything()
    global video_source
    try:
        video_source = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        if video_source:  # If a file is selected
            root.destroy()  # Close the GUI
            logging.info(f"Selected video file: {video_source}")
            fall_detection_system(resource_path(video_source))
    except Exception as e:
        logging.error(f"Error in select_video_file: {e}")

should_restart = False

def restart_program():
    # Display the restart dialog box
    response = messagebox.askyesno("Restart", "Do you want to restart?")
    if response:  # If the user clicks "Yes" 
        main()  # Restart the main function
    else:  # If the user clicks "No"
        root.quit()  # Exit the program

def main():
    global root
    root = Tk()
    root.title("Select Input Source")
    root.geometry("300x150")

    label = Label(root, text="Choose your input source:", font=("Helvetica", 12))
    label.pack(pady=10)

    button1 = Button(root, text="Select Camera", command=select_camera_window)
    button1.pack(pady=5)

    button2 = Button(root, text="Select Video File", command=select_video_file)
    button2.pack(pady=5)
    root.mainloop()

# Run the application
main()
