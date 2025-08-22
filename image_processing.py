#Image_processing.py
# image_processing.py
# This file will contain functions for capturing images, detecting ears,
# processing contours, and calculating measurements.
import tkinter as tk
from tkinter import ttk, messagebox, Label, filedialog, simpledialog
import cv2
import utils
import gui_pages  # For navigation and setup_background
import os
from PIL import Image, ImageTk
import threading
import time
import numpy as np
from ear_detector import EarDetector # Ensure this is imported
import ui_utils  # Import for styled UI elements
import joblib # To load the trained models
import json # For loading calibration data

# Global minimum thresholds for measurements - Fix #4
MIN_PIXEL_HEIGHT = 50
MIN_PIXEL_WIDTH = 25
DEFAULT_DISTANCE_MM = 150.0

def safe_button_config(button_ref, **kwargs):
    """Safely configure a button widget if it exists and is valid."""
    try:
        if button_ref and button_ref.winfo_exists():
            button_ref.config(**kwargs)
    except (tk.TclError, AttributeError):
        # Widget has been destroyed or is invalid
        pass

# Global variables
capture_globals = {
    'current_sensor_distance_sv': None,
    'status_message_sv': None,
    'detection_status_sv': None,
    'camera_feed_label': None,
    'thumbnail_labels': [],
    'thumbnail_images': [None] * 5,
    'capture_buttons': [], # This will now hold the "Recapture" buttons
    'main_capture_button': None, # New: For the single "Capture Image" button
    'photos_taken_count': 0,
    'patient_info_for_session': {},
    'start_hardware_button': None,
    'stop_hardware_button': None,
    'show_results_button': None,
    'video_loop_id': None,
    'camera_active': False,
    'camera_index': 0, # Default camera index - iVCam usually uses 0
    'cap': None, # OpenCV VideoCapture object
    'sensor_active': False,
    'sensor_thread': None,
    'stop_sensor_thread': False,
    'sensor_distance_mm': 0.0, # Holds the latest sensor distance
    'lock': threading.Lock(), # For thread-safe updates to sensor_distance_mm
    # 'ear_detector': None, # Removed: Will use utils.ear_detector_instance
    # 'yolo_model_loaded': False, # Removed: Will use utils.model or utils.ear_detector_instance status
    'display_width': 780, # For camera feed display
    'display_height': 480, # For camera feed display
    'root_window': None, # Reference to the main Tkinter root window
    'capture_window': None, # Reference to the actual capture window (main_frame in this case)
}

def safe_button_config(button, **kwargs):
    """Safely configure a button widget, checking if it exists first."""
    try:
        if button and button.winfo_exists():
            button.config(**kwargs)
    except tk.TclError:
        # Widget no longer exists, ignore
        pass

def safe_label_config(label, **kwargs):
    """Safely configure a label widget, checking if it exists first."""
    try:
        if label and label.winfo_exists():
            label.config(**kwargs)
    except tk.TclError:
        # Widget no longer exists, ignore
        pass

def load_calibration_models():
    """Load calibration models from disk"""
    calibration_file_base = "calibration_data_new" # Match what you saved in calibration
    try:
        utils.calibration_data['height_model'] = joblib.load(f"{calibration_file_base}_height_model.pkl")
        utils.calibration_data['width_model'] = joblib.load(f"{calibration_file_base}_width_model.pkl")
        # Also load the R2 scores and other data from JSON if needed
        with open(f"{calibration_file_base}.json", 'r') as f:
            cal_data = json.load(f)
            utils.calibration_data['r2_height'] = cal_data.get('r2_height')
            utils.calibration_data['r2_width'] = cal_data.get('r2_width')
            utils.calibration_data['focal_length_height'] = cal_data.get('focal_length_height')
            utils.calibration_data['focal_length_width'] = cal_data.get('focal_length_width')
        utils.is_calibrated = True
        print("Calibration models loaded successfully.")
    except Exception as e:
        print(f"Error loading calibration models: {e}")
        utils.is_calibrated = False

def calculate_real_dimensions(pixel_height, pixel_width, sensor_distance):
    """Calculate dimensions using focal length"""
    # Retrieve calibrated focal lengths
    focal_length_h = utils.calibration_data.get('focal_length_height')
    focal_length_w = utils.calibration_data.get('focal_length_width')

    if focal_length_h is None or focal_length_w is None or focal_length_h == 0 or focal_length_w == 0:
        messagebox.showerror("Calibration Error", "Focal length not calibrated. Please perform calibration first.")
        return None, None

    # Calculate actual dimensions using the focal length and current distance
    real_height_mm = (pixel_height * sensor_distance) / focal_length_h if focal_length_h > 0 else 0
    real_width_mm = (pixel_width * sensor_distance) / focal_length_w if focal_length_w > 0 else 0

    return real_height_mm, real_width_mm

def calculate_real_dimensions_using_regression(pixel_height, pixel_width, sensor_distance):
    """Calculate dimensions using regression model"""
    if not utils.is_calibrated or utils.calibration_data.get('height_model') is None:
        messagebox.showerror("Calibration Error", "Calibration models not loaded. Please calibrate the system.")
        return None, None

    # Predict pixels_per_mm for the current distance
    current_distance_array = np.array([[sensor_distance]])

    predicted_pph = utils.calibration_data['height_model'].predict(current_distance_array)[0]
    predicted_ppw = utils.calibration_data['width_model'].predict(current_distance_array)[0]

    if predicted_pph <= 0 or predicted_ppw <= 0:
        print(f"Warning: Predicted PPH/PPW is non-positive. PPH: {predicted_pph:.2f}, PPW: {predicted_ppw:.2f}")
        # Fallback or error handling
        return None, None

    real_height_mm = pixel_height / predicted_pph
    real_width_mm = pixel_width / predicted_ppw

    return real_height_mm, real_width_mm

def calculate_robust_average(measurements, method="trimmed_mean", trim_fraction=0.2):
    """Calculate robust average with outlier removal"""
    if not measurements:
        return "N/A"
    
    # Convert all measurements to float, handle potential non-numeric values
    numeric_measurements = []
    for m in measurements:
        try:
            numeric_measurements.append(float(m))
        except (ValueError, TypeError):
            continue # Skip invalid entries

    if not numeric_measurements:
        return "N/A"

    data = np.array(numeric_measurements)
    data = np.sort(data) # Sort for trimmed mean/median

    if method == "median":
        return np.median(data)
    elif method == "trimmed_mean":
        num_to_trim = int(len(data) * trim_fraction / 2) # Trim from both ends
        if len(data) <= 2 * num_to_trim: # If too few samples after trimming
            return np.mean(data) # Fallback to mean of all if trimming too much
        trimmed_data = data[num_to_trim : len(data) - num_to_trim]
        return np.mean(trimmed_data)
    else: # Default to simple mean if method is not recognized
        return np.mean(data)

def calculate_ear_dimensions_mm(pixel_height, pixel_width, sensor_distance_mm):
    """
    Calculates actual ear dimensions in millimeters using linear regression
    model coefficients from calibration data.
    """
    # Check if calibration data is available and in the new format
    if not utils.is_calibrated or not utils.calibration_data:
        return "N/A (Calibrate)", "N/A (Calibrate)"

    height_coefficients = utils.calibration_data.get('height_coefficients')
    height_intercept = utils.calibration_data.get('height_intercept')
    width_coefficients = utils.calibration_data.get('width_coefficients')
    width_intercept = utils.calibration_data.get('width_intercept')

    if (height_coefficients is None or height_intercept is None or
        width_coefficients is None or width_intercept is None):
        # Fallback for older calibration data or incomplete new format
        if 'pixel_to_mm_ratio' in utils.calibration_data:
            ratio = utils.calibration_data['pixel_to_mm_ratio']
            actual_height_mm = pixel_height * ratio
            actual_width_mm = pixel_width * ratio
            return round(actual_height_mm, 2), round(actual_width_mm, 2)
        else:
            return "N/A (Calibrate)", "N/A (Calibrate)"

    try:
        # Apply the linear regression formula:
        # predicted_value = coef[0] * pixel_value + coef[1] * sensor_distance + intercept
        actual_height_mm = (height_coefficients[0] * pixel_height +
                            height_coefficients[1] * sensor_distance_mm +
                            height_intercept)
        actual_width_mm = (width_coefficients[0] * pixel_width +
                           width_coefficients[1] * sensor_distance_mm +
                           width_intercept)

        return round(actual_height_mm, 2), round(actual_width_mm, 2)
    except Exception as e:
        print(f"Error during measurement calculation: {e}")
        return "N/A (Error)", "N/A (Error)"

def _init_tk_vars():
    """Initializes Tkinter StringVars for status messages."""
    global capture_globals
    capture_globals['current_sensor_distance_sv'] = tk.StringVar(value="Sensor Distance: N/A")
    capture_globals['status_message_sv'] = tk.StringVar(value="Status: Ready")
    capture_globals['detection_status_sv'] = tk.StringVar(value="Detection: Not Active")

def capture_images_entry_point(root, patient_info_dict):
    global capture_globals
    
    # Reset camera and sensor states when entering capture window
    if capture_globals['camera_active']:
        stop_hardware(root, on_close=False)
    
    capture_globals['root_window'] = root
    capture_globals['patient_info_for_session'] = patient_info_dict
    capture_globals['photos_taken_count'] = 0
    capture_globals['thumbnail_images'] = [None] * 5 # Reset thumbnails
    capture_globals['thumbnail_labels'] = [] # Reset thumbnail labels
    capture_globals['capture_buttons'] = [] # Reset recapture buttons

    # Clear previous window content
    gui_pages.clear_window(root)
    root.title("Capture Measurements")
    gui_pages.setup_background(root, window_key="capture_measurements_bg")

    # Create main container frame
    container = ttk.Frame(root, style="Dark.TFrame")
    container.pack(fill="both", expand=True)

    # Create canvas and scrollbar for scrollable window
    canvas = tk.Canvas(container, highlightthickness=0)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas, style="Dark.TFrame")

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    # Configure canvas window to fill the available width
    def configure_canvas_window(event):
        canvas_width = event.width
        canvas.itemconfig(canvas.find_all()[0], width=canvas_width)

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', configure_canvas_window)

    # Bind mouse wheel to canvas for scrolling
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Frame for the main content (now inside scrollable frame)
    main_frame = ttk.Frame(scrollable_frame, padding="20", style="Dark.TFrame")
    main_frame.pack(expand=True, fill="both")
    capture_globals['capture_window'] = main_frame # Set reference to the main frame

    # Title
    ui_utils.create_title_label(main_frame, "Capture Ear Measurements").pack(pady=(0, 20))

    # --- Live Feed and Side Controls Frame ---
    live_feed_and_controls_frame = ttk.Frame(main_frame, style="Light.TFrame", padding=10)
    live_feed_and_controls_frame.pack(pady=10, fill="both", expand=True)

    # Camera feed display frame (Left side)
    camera_display_frame = ttk.Frame(live_feed_and_controls_frame, style="Light.TFrame", padding=10)
    camera_display_frame.pack(side=tk.LEFT, expand=True, fill="both", padx=(0, 10))
    camera_feed_label = ui_utils.create_styled_label(camera_display_frame, "Camera Feed", bg="black", width=80, height=30)
    camera_feed_label.pack(expand=True, fill="both")
    capture_globals['camera_feed_label'] = camera_feed_label # Store reference

    # Side buttons frame (Right side)
    side_buttons_frame = ttk.Frame(live_feed_and_controls_frame, style="Dark.TFrame", padding="10")
    side_buttons_frame.pack(side=tk.RIGHT, fill="y", padx=(10, 0))
    
    # For backwards compatibility, also define button_frame
    button_frame = side_buttons_frame

    # "Capture Photo" button
    capture_globals['main_capture_button'] = ui_utils.create_primary_button(
        side_buttons_frame,
        "Capture Photo",
        lambda: capture_photo(capture_globals['photos_taken_count']),
        width=220, height=60
    )
    capture_globals['main_capture_button'].config(state=tk.DISABLED)
    capture_globals['main_capture_button'].pack(pady=10, fill="x")

    # "See Result" button
    capture_globals['show_results_button'] = ui_utils.create_success_button(
        side_buttons_frame,
        "See Result",
        lambda: show_results(root),
        width=220,
        height=60,
        state=tk.DISABLED # Disable initially
    )
    capture_globals['show_results_button'].pack(pady=10, fill="x")

    # "Back" button
    back_button = ui_utils.create_secondary_button(
        side_buttons_frame,
        "Back",
        lambda: gui_pages.create_patient_form(root),
        width=220,
        height=60
    )
    back_button.pack(pady=10, fill="x")

    # "Quit" button
    quit_button = ui_utils.create_danger_button(
        side_buttons_frame,
        "Quit",
        lambda: utils.on_app_close(root),
        width=220,
        height=60
    )
    quit_button.pack(pady=10, fill="x")

    # Other buttons are removed or commented out
    # These buttons have been replaced with the new buttons above

    # # "Show Results" button
    # capture_globals['show_results_button'] = ui_utils.create_success_button(
    #     side_buttons_frame,
    #     "Show Results",
    #     lambda: show_results(root),
    #     width=20,
    #     height=3,
    #     state=tk.DISABLED # Disable initially
    # )
    # capture_globals['show_results_button'].pack(pady=10, fill="x")

    # # "Back to Patient Info" button
    # back_button = ui_utils.create_secondary_button(
    #     side_buttons_frame,
    #     "Patient Details",
    #     lambda: gui_pages.create_patient_form(root), # Pass current patient info back
    #     width=20,
    #     height=3
    # )
    # back_button.pack(pady=10, fill="x")

    # # "Quit" button
    # quit_button = ui_utils.create_danger_button(
    #     side_buttons_frame,
    #     "Quit",
    #     lambda: utils.on_app_close(root),
    #     width=20,
    #     height=3
    # )
    # quit_button.pack(pady=10, fill="x")

    # Removed YOLO Model Load Section - model is loaded globally via home page
    # Removed Camera selection dropdown - iVCam handles it

    # Measurement info and status (Below Live Feed)
    info_frame = ttk.Frame(main_frame, padding="10", style="Light.TFrame")
    info_frame.pack(pady=10, fill="x")

    # Fix: Added text="" for textvariable labels
    ui_utils.create_styled_label(info_frame, text="", textvariable=capture_globals['current_sensor_distance_sv'], font=ui_utils.UI_FONTS["subheading"]).pack(side=tk.LEFT, padx=10)
    ui_utils.create_styled_label(info_frame, text="", textvariable=capture_globals['status_message_sv'], font=ui_utils.UI_FONTS["subheading"]).pack(side=tk.LEFT, padx=10)
    ui_utils.create_styled_label(info_frame, text="", textvariable=capture_globals['detection_status_sv'], font=ui_utils.UI_FONTS["subheading"]).pack(side=tk.LEFT, padx=10)

    # Thumbnail display area (Below Status Info)
    thumbnail_frame = ttk.Frame(main_frame, style="Dark.TFrame", padding=10)
    thumbnail_frame.pack(pady=10, fill="x")

    for i in range(5):
        # Create a frame for each thumbnail and its button
        thumb_slot_frame = ttk.Frame(thumbnail_frame, style="Dark.TFrame")
        thumb_slot_frame.pack(side=tk.LEFT, expand=True, padx=5)

        thumb_label = ui_utils.create_styled_label(thumb_slot_frame, f"Slot {i+1}", bg="gray", width=160, height=120)
        thumb_label.pack(pady=5)
        capture_globals['thumbnail_labels'].append(thumb_label)

        # "Recapture" button for each slot
        recapture_button = ui_utils.create_primary_button(
            thumb_slot_frame,
            f"Recapture {i+1}",
            lambda idx=i: capture_photo(idx),
            width=120,
            height=40,
            state=tk.DISABLED # Disable initially, enabled after capture
        )
        recapture_button.pack(pady=5, fill="x")
        capture_globals['capture_buttons'].append(recapture_button) # This now holds recapture buttons

    # Start hardware automatically when window loads
    # Use after to ensure all widgets are packed before starting camera
    capture_globals['root_window'].after(100, lambda: start_hardware(capture_globals['root_window']))

    # Ensure to stop hardware when window is closed
    capture_globals['root_window'].protocol("WM_DELETE_WINDOW", lambda: stop_hardware(capture_globals['root_window'], on_close=True))


def _start_camera_feed():
    """Starts the camera feed in a dedicated thread."""
    global capture_globals

    if capture_globals['camera_active'] and capture_globals['cap'] is not None and capture_globals['cap'].isOpened():
        print("Camera already active.")
        return

    # Use the default camera index (0) as iVCam handles it
    capture_globals['cap'] = cv2.VideoCapture(capture_globals['camera_index'])
    if not capture_globals['cap'].isOpened():
        messagebox.showerror("Camera Error", f"Could not open camera at index {capture_globals['camera_index']}. Please ensure iVCam is running or check camera connection.")
        capture_globals['status_message_sv'].set("Status: Camera Error")
        capture_globals['camera_active'] = False
        # Disable buttons if camera fails   
        safe_button_config(capture_globals['main_capture_button'], state=tk.DISABLED)
        for btn in capture_globals['capture_buttons']:
            if btn: btn.config(state=tk.DISABLED)
        return

    # Set resolution if needed, though iVCam handles it automatically often
    # capture_globals['cap'].set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # capture_globals['cap'].set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    capture_globals['camera_active'] = True
    capture_globals['status_message_sv'].set("Status: Camera Active")
    print(f"Camera started on index {capture_globals['camera_index']}")

    # Start the video update loop
    _update_camera_feed()

def _update_camera_feed():
    """Updates the camera feed display."""
    global capture_globals
    if not capture_globals['camera_active'] or capture_globals['cap'] is None or not capture_globals['cap'].isOpened():
        return

    ret, frame = capture_globals['cap'].read()
    if ret:
        diagnostic_frame = frame.copy() # Use a copy for drawing detections
        
        # Use YOLO segmentation model directly for live detection
        if utils.ear_detector_instance and utils.ear_detector_instance.model:
            try:
                # Use the YOLO model directly like in your working code
                # Apply detection with confidence and IoU thresholds
                results = utils.ear_detector_instance.model(frame, conf=0.5, iou=0.7, verbose=False)
                
                # Use the annotated frame from YOLO's plot() method
                diagnostic_frame = results[0].plot()
                
                # Check if we have any detections
                if len(results[0].boxes) > 0:
                    num_detections = len(results[0].boxes)
                    try:
                        if capture_globals['detection_status_sv']:
                            capture_globals['detection_status_sv'].set(f"Detection: Found {num_detections} ear(s)")
                    except tk.TclError:
                        pass
                    
                    safe_button_config(capture_globals['main_capture_button'], state=tk.NORMAL) # Enable capture button
                else:
                    try:
                        if capture_globals['detection_status_sv']:
                            capture_globals['detection_status_sv'].set("Detection: No ears found")
                    except tk.TclError:
                        pass
                    safe_button_config(capture_globals['main_capture_button'], state=tk.DISABLED) # Disable capture button
                        
            except Exception as e:
                try:
                    if capture_globals['detection_status_sv']:
                        capture_globals['detection_status_sv'].set(f"Detection Error: {e}")
                except tk.TclError:
                    pass
                print(f"Error during ear detection: {e}")
                safe_button_config(capture_globals['main_capture_button'], state=tk.DISABLED)
        else:
            try:
                if capture_globals['detection_status_sv']:
                    capture_globals['detection_status_sv'].set("Detection: Model Not Loaded")
            except tk.TclError:
                pass
            safe_button_config(capture_globals['main_capture_button'], state=tk.DISABLED)


        # Convert to PIL Image for display
        diagnostic_rgb = cv2.cvtColor(diagnostic_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(diagnostic_rgb)

        # Resize if needed for display
        img.thumbnail((capture_globals['display_width'], capture_globals['display_height']), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the label safely
        safe_label_config(capture_globals['camera_feed_label'], image=imgtk)
        try:
            if capture_globals['camera_feed_label'] and capture_globals['camera_feed_label'].winfo_exists():
                capture_globals['camera_feed_label'].image = imgtk  # Keep reference
        except (tk.TclError, AttributeError):
            pass  # Label destroyed or invalid

    # Schedule the next update safely
    try:
        if (capture_globals['capture_window'] and capture_globals['capture_window'].winfo_exists() and
            capture_globals['root_window'] and capture_globals['camera_active']):
            capture_globals['video_loop_id'] = capture_globals['root_window'].after(10, _update_camera_feed) # 10ms delay
    except tk.TclError:
        # Window has been destroyed, stop the update loop
        capture_globals['video_loop_id'] = None


def _stop_camera_feed():
    """Stops the camera feed."""
    global capture_globals
    if capture_globals['video_loop_id']:
        capture_globals['root_window'].after_cancel(capture_globals['video_loop_id'])
        capture_globals['video_loop_id'] = None
    if capture_globals['cap'] and capture_globals['cap'].isOpened():
        capture_globals['cap'].release()
        capture_globals['cap'] = None
    capture_globals['camera_active'] = False
    
    # Safely update camera feed label
    try:
        if capture_globals['camera_feed_label'] and capture_globals['camera_feed_label'].winfo_exists():
            capture_globals['camera_feed_label'].config(image=None, text="Camera Feed Stopped")
    except (tk.TclError, AttributeError):
        pass  # Label doesn't exist or has been destroyed
    
    # Safely update status variables
    try:
        if capture_globals['status_message_sv']:
            capture_globals['status_message_sv'].set("Status: Camera Stopped")
    except tk.TclError:
        pass
    
    try:
        if capture_globals['detection_status_sv']:
            capture_globals['detection_status_sv'].set("Detection: Inactive")
    except tk.TclError:
        pass
    
    print("Camera feed stopped.")

def start_hardware(root):
    """Starts the camera and ultrasonic sensor."""
    global capture_globals
    capture_globals['root_window'] = root # Ensure root_window is set

    # Start camera feed
    _start_camera_feed()

    # Start ultrasonic sensor (if not in simulation mode)
    if not utils.use_simulation_mode:
        if utils.ultrasonic_sensor_instance is None:
            messagebox.showinfo("Sensor Init", "Attempting to initialize ultrasonic sensor...")
            try:
                # Initialize the ultrasonic sensor
                sensor_success = utils.initialize_ultrasonic_sensor()
                if not sensor_success:
                    messagebox.showwarning("Sensor Warning", "Could not initialize ultrasonic sensor. Switching to simulation mode.")
                    utils.use_simulation_mode = True
            except Exception as e:
                messagebox.showerror("Sensor Error", f"Failed to initialize sensor: {e}")
                capture_globals['status_message_sv'].set("Status: Sensor Init Failed")
                utils.use_simulation_mode = True

        if utils.ultrasonic_sensor_instance and not capture_globals['sensor_active']:
            capture_globals['stop_sensor_thread'] = False
            capture_globals['sensor_thread'] = threading.Thread(target=read_sensor_continuously, daemon=True)
            capture_globals['sensor_thread'].start()
            capture_globals['sensor_active'] = True
            capture_globals['status_message_sv'].set("Status: Camera & Sensor Active")
            messagebox.showinfo("Hardware Status", "Camera and Sensor started successfully.")
        elif utils.use_simulation_mode and not capture_globals['sensor_active']:
            # Start simulated sensor readings
            capture_globals['stop_sensor_thread'] = False
            capture_globals['sensor_thread'] = threading.Thread(target=read_simulated_sensor_continuously, daemon=True)
            capture_globals['sensor_thread'].start()
            capture_globals['sensor_active'] = True
            capture_globals['status_message_sv'].set("Status: Camera Active (Simulation Mode)")
            messagebox.showinfo("Hardware Status", "Camera started (Simulation Mode).")
        else:
            messagebox.showwarning("Sensor Warning", "Sensor already active or not initialized properly.")
            capture_globals['status_message_sv'].set("Status: Camera Active (Sensor Issue)")
    else:
        capture_globals['status_message_sv'].set("Status: Camera Active (Simulation Mode)")
        messagebox.showinfo("Hardware Status", "Camera started (Simulation Mode).")

    # Disable start button and enable stop button
    if capture_globals['start_hardware_button']: # Check if button exists before configuring
        try:
            if capture_globals['start_hardware_button'].winfo_exists():
                capture_globals['start_hardware_button'].config(state=tk.DISABLED)
        except (tk.TclError, AttributeError):
            pass  # Button doesn't exist or has been destroyed
    if capture_globals['stop_hardware_button']: # Check if button exists before configuring
        try:
            if capture_globals['stop_hardware_button'].winfo_exists():
                capture_globals['stop_hardware_button'].config(state=tk.NORMAL)
        except (tk.TclError, AttributeError):
            pass  # Button doesn't exist or has been destroyed
    
    # Capture button state managed by _update_camera_feed based on detection
    # Recapture buttons are enabled after a photo is taken

def stop_hardware(root, on_close=False):
    """Stops the camera and ultrasonic sensor."""
    global capture_globals
    print("Attempting to stop hardware...")
    
    _stop_camera_feed()

    if capture_globals['sensor_active']:
        capture_globals['stop_sensor_thread'] = True
        if capture_globals['sensor_thread'] and capture_globals['sensor_thread'].is_alive():
            capture_globals['sensor_thread'].join(timeout=1) # Wait for thread to finish
            if capture_globals['sensor_thread'].is_alive():
                print("Warning: Sensor thread did not terminate cleanly.")
        if utils.ultrasonic_sensor_instance:
            utils.ultrasonic_sensor_instance.close()
            utils.ultrasonic_sensor_instance = None
        capture_globals['sensor_active'] = False
        print("Sensor stopped.")

    # Safely update status message
    try:
        if capture_globals['status_message_sv']:
            capture_globals['status_message_sv'].set("Status: Hardware Stopped")
    except tk.TclError:
        pass
    
    # Enable start button and disable stop button (safely)
    try:
        if capture_globals['start_hardware_button'] and capture_globals['start_hardware_button'].winfo_exists():
            capture_globals['start_hardware_button'].config(state=tk.NORMAL)
    except (tk.TclError, AttributeError):
        pass  # Button doesn't exist or has been destroyed
    
    try:
        if capture_globals['stop_hardware_button'] and capture_globals['stop_hardware_button'].winfo_exists():
            capture_globals['stop_hardware_button'].config(state=tk.DISABLED)
    except (tk.TclError, AttributeError):
        pass  # Button doesn't exist or has been destroyed
    
    # Disable capture and recapture buttons (safely)
    safe_button_config(capture_globals['main_capture_button'], state=tk.DISABLED)
    for btn in capture_globals['capture_buttons']:
        if btn:
            try:
                if btn.winfo_exists():
                    btn.config(state=tk.DISABLED)
            except (tk.TclError, AttributeError):
                pass  # Button has been destroyed
    
    try:
        if capture_globals['show_results_button'] and capture_globals['show_results_button'].winfo_exists():
            capture_globals['show_results_button'].config(state=tk.DISABLED)
    except (tk.TclError, AttributeError):
        pass  # Button doesn't exist or has been destroyed

    # Clean up patient info if exiting the session entirely
    if on_close:
        messagebox.showinfo("Exit", "Application is closing. Hardware stopped.")
        # Perform any final cleanup before exiting
        root.destroy()
    else:
        messagebox.showinfo("Hardware Status", "Camera and Sensor stopped successfully.")


def read_sensor_continuously():
    """Reads sensor data continuously in a separate thread."""
    global capture_globals
    if utils.ultrasonic_sensor_instance is None:
        print("Error: Ultrasonic sensor instance is not initialized.")
        return

    while not capture_globals['stop_sensor_thread']:
        try:
            distance = utils.ultrasonic_sensor_instance.get_distance_with_retry()
            if distance is not None and 0.1 <= distance <= 5000:  # Enhanced validation
                with capture_globals['lock']:
                    capture_globals['sensor_distance_mm'] = distance
                # Update Tkinter StringVar in the main thread safely
                try:
                    if capture_globals['root_window'] and capture_globals['current_sensor_distance_sv']:
                        capture_globals['root_window'].after(0, capture_globals['current_sensor_distance_sv'].set, f"Sensor Distance: {distance:.2f} mm")
                except tk.TclError:
                    pass  # Window or StringVar destroyed
                utils.current_sensor_distance = distance # Update global util variable
                print(f"[SENSOR] Distance reading: {distance:.2f}mm")
            else:
                print(f"[SENSOR] Invalid distance reading: {distance}")
                utils.current_sensor_distance = None  # Set to None for invalid readings
            time.sleep(0.1) # Read every 100ms
        except Exception as e:
            print(f"Error reading sensor: {e}")
            try:
                if capture_globals['root_window'] and capture_globals['current_sensor_distance_sv']:
                    capture_globals['root_window'].after(0, capture_globals['current_sensor_distance_sv'].set, "Sensor Error!")
            except tk.TclError:
                pass  # Window or StringVar destroyed
            utils.current_sensor_distance = None  # Set to None on error
            break # Exit loop on error
    print("Sensor reading thread terminated.")


def read_simulated_sensor_continuously():
    """Simulates sensor readings continuously for testing."""
    global capture_globals
    simulated_distance = DEFAULT_DISTANCE_MM
    
    while not capture_globals['stop_sensor_thread']:
        try:
            # Simulate a small fluctuation for realism
            simulated_distance = np.random.uniform(DEFAULT_DISTANCE_MM - 5, DEFAULT_DISTANCE_MM + 5)
            
            with capture_globals['lock']:
                capture_globals['sensor_distance_mm'] = simulated_distance
            
            # Update Tkinter StringVar in the main thread safely
            try:
                if capture_globals['root_window'] and capture_globals['current_sensor_distance_sv']:
                    capture_globals['root_window'].after(0, capture_globals['current_sensor_distance_sv'].set, f"Simulated Distance: {simulated_distance:.2f} mm")
            except tk.TclError:
                pass  # Window or StringVar destroyed
            utils.current_sensor_distance = simulated_distance
            print(f"[SENSOR SIMULATION] Distance reading: {simulated_distance:.1f}mm")
            
            time.sleep(0.1) # Simulate reading interval
        except Exception as e:
            print(f"Error in simulated sensor reading: {e}")
            try:
                if capture_globals['root_window'] and capture_globals['current_sensor_distance_sv']:
                    capture_globals['root_window'].after(0, capture_globals['current_sensor_distance_sv'].set, "Simulation Error!")
            except tk.TclError:
                pass  # Window or StringVar destroyed
            break
    
    print("Simulated sensor reading thread terminated.")


def capture_photo(slot_index):
    """Captures a photo, detects ears, and saves measurements."""
    global capture_globals
    print(f"Attempting to capture photo for slot {slot_index}...")

    if not capture_globals['cap'] or not capture_globals['cap'].isOpened():
        messagebox.showwarning("Capture Error", "Camera not active. Please start hardware.")
        return

    if not utils.is_calibrated:
        # Show warning only once per session
        if getattr(capture_globals['root_window'], '_cal_warning_shown', False):
            print("Calibration warning already shown for this session.")
        else:
            messagebox.showwarning("Calibration Required", "System is not calibrated. Measurements will be based on pixel ratio if available, or marked as 'N/A'. Please calibrate for accurate results.")
            setattr(capture_globals['root_window'], '_cal_warning_shown', True) # Set flag

    ret, frame = capture_globals['cap'].read()
    if not ret:
        messagebox.showerror("Capture Error", "Failed to capture image from camera.")
        return

    if utils.ear_detector_instance is None:
        messagebox.showwarning("Model Not Loaded", "YOLO ear detection model is not loaded. Cannot process image.")
        return

    # Acquire sensor distance
    with capture_globals['lock']:
        current_sensor_distance = capture_globals['sensor_distance_mm']
        if current_sensor_distance == 0.0 and not utils.use_simulation_mode:
            # If it's 0 and not simulation, sensor might not be working, use default
            current_sensor_distance = DEFAULT_DISTANCE_MM
            messagebox.showwarning("Sensor Data", f"Sensor distance not available or 0. Using default distance: {DEFAULT_DISTANCE_MM} mm for calculation.")

    try:
        # Use YOLO model directly for detection like in the live feed
        results = utils.ear_detector_instance.model(frame, conf=0.5, iou=0.7, verbose=False)
        
        ear_data = []
        if len(results[0].boxes) > 0:
            capture_globals['status_message_sv'].set(f"Status: Captured with {len(results[0].boxes)} ear(s) detected.")
            
            for i, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                pixel_width = x2 - x1
                pixel_height = y2 - y1

                # Validate pixel dimensions against minimum thresholds
                if pixel_height < MIN_PIXEL_HEIGHT or pixel_width < MIN_PIXEL_WIDTH:
                    messagebox.showwarning("Detection Too Small", f"Detected ear is too small ({pixel_height}x{pixel_width}px). Please move closer or adjust camera.")
                    continue

                # Calculate actual dimensions using the new dedicated function
                actual_height_mm, actual_width_mm = calculate_ear_dimensions_mm(
                    pixel_height, pixel_width, current_sensor_distance
                )

                # Add width consistency check
                # Check if measured width is within reasonable range for human ear (20-50mm typically)
                if isinstance(actual_width_mm, (int, float)) and (actual_width_mm < 20 or actual_width_mm > 50):
                    print("⚠️ Check width measurement definition")

                ear_data.append({
                    'box': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'class_id': class_id,
                    'pixel_width': pixel_width,
                    'pixel_height': pixel_height,
                    'sensor_distance_mm': current_sensor_distance,
                    'actual_height_mm': actual_height_mm,
                    'actual_width_mm': actual_width_mm
                })
                print(f"[CAPTURE] Ear {i+1}: {pixel_width}x{pixel_height}px, distance: {current_sensor_distance}mm, confidence: {confidence:.2f}")

                # For displaying on thumbnail, draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Ear: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            capture_globals['status_message_sv'].set("Status: Captured, no ears detected.")
            messagebox.showwarning("Detection Warning", "No ears detected in the captured image.")
            return

        if not ear_data:
            messagebox.showwarning("No Valid Detections", "No valid ear detections found (all were too small).")
            return

        # Convert to PIL Image for thumbnail
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        img.thumbnail((200, 150), Image.Resampling.LANCZOS) # Enlarged thumbnail size

        # Save the captured images to the specified directory
        save_directory = r"D:\Research\Research\Codes\Code_3\captured_images"
        os.makedirs(save_directory, exist_ok=True)
        
        # Save original image
        original_path = os.path.join(save_directory, f"original_img_{slot_index}.jpg")
        cv2.imwrite(original_path, capture_globals['cap'].read()[1] if capture_globals['cap'].read()[0] else frame)
        
        # Save annotated image (with detection boxes)
        annotated_path = os.path.join(save_directory, f"annotated_img_{slot_index}.jpg")
        cv2.imwrite(annotated_path, frame)
        
        # Save contour image (if needed - for now, same as annotated)
        contour_path = os.path.join(save_directory, f"contour_img_{slot_index}.jpg")
        cv2.imwrite(contour_path, frame)
        
        print(f"[SAVE] Images saved for slot {slot_index}: original, annotated, and contour")

        imgtk = ImageTk.PhotoImage(image=img)
        capture_globals['thumbnail_images'][slot_index] = {'image': imgtk, 'raw_frame': frame, 'detections': ear_data}
        capture_globals['thumbnail_labels'][slot_index].config(image=imgtk, text="") # Update label with image, clear text
        capture_globals['thumbnail_labels'][slot_index].image = imgtk # Keep reference

        # Enable recapture button for this slot
        if capture_globals['capture_buttons'] and slot_index < len(capture_globals['capture_buttons']) and capture_globals['capture_buttons'][slot_index]:
            capture_globals['capture_buttons'][slot_index].config(state=tk.NORMAL)
        
        # Increment photo count and update UI for next photo if using main capture button
        if slot_index == capture_globals['photos_taken_count']: # Only increment if it's a new capture, not a recapture
            capture_globals['photos_taken_count'] += 1
            if capture_globals['photos_taken_count'] >= 5:
                safe_button_config(capture_globals['main_capture_button'], state=tk.DISABLED, text="All Slots Full")
                messagebox.showinfo("All Slots Used", "All 5 measurement slots are now filled. You can use 'Recapture' to re-take a specific photo or 'See Result'.")
            else:
                safe_button_config(capture_globals['main_capture_button'], text=f"Capture Photo ({capture_globals['photos_taken_count']+1}/5)")
        
        # Enable Show Results button if at least one photo is taken
        if capture_globals['photos_taken_count'] > 0:
            if capture_globals['show_results_button']:
                capture_globals['show_results_button'].config(state=tk.NORMAL)

        # Show capture confirmation with sensor distance
        distance_info = f" (Distance: {current_sensor_distance:.1f}mm)" if current_sensor_distance > 0 else " (No distance data)"
        largest_ear = max(ear_data, key=lambda d: d['pixel_width'] * d['pixel_height'])
        messagebox.showinfo("Capture Success", f"Ear detected and measurement captured for slot {slot_index+1}!\n"
                            f"Pixel Dimensions: {largest_ear['pixel_height']}x{largest_ear['pixel_width']} px\n"
                            f"Estimated Actual: {largest_ear['actual_height_mm']}mm x {largest_ear['actual_width_mm']}mm{distance_info}")

    except Exception as e:
        messagebox.showerror("Processing Error", f"An error occurred during image processing: {e}")
        print(f"Image processing error: {e}")


def calculate_ear_dimensions_mm(pixel_height, pixel_width, sensor_distance_mm):
    """
    Calculate actual ear dimensions in mm using the linear regression model.
    
    Args:
        pixel_height: Height of the ear in pixels
        pixel_width: Width of the ear in pixels
        sensor_distance_mm: Distance from sensor to ear in mm
        
    Returns:
        tuple: (predicted_height_mm, predicted_width_mm)
    """
    if not utils.is_calibrated or not utils.calibration_data:
        # Handle case where system is not calibrated
        print("System not calibrated. Cannot calculate actual dimensions.")
        return "N/A (Calibrate)", "N/A (Calibrate)"

    calibration_data = utils.calibration_data

    # Check for new calibration format (with linear regression coefficients)
    if ('height_coefficients' in calibration_data and 
        'width_coefficients' in calibration_data and
        'height_intercept' in calibration_data and
        'width_intercept' in calibration_data):
        
        # Extract coefficients and intercepts for height
        height_coef = calibration_data['height_coefficients'] 
        height_intercept = calibration_data['height_intercept'] 

        # Extract coefficients and intercepts for width 
        width_coef = calibration_data['width_coefficients'] 
        width_intercept = calibration_data['width_intercept'] 

        # --- Perform the actual prediction using the linear regression model --- 
        # Predict height in mm 
        # Assumes height_coef[0] is for pixel_height and height_coef[1] is for sensor_distance 
        predicted_height_mm = (height_coef[0] * pixel_height) + \
                              (height_coef[1] * sensor_distance_mm) + \
                              height_intercept 

        # Predict width in mm 
        # Assumes width_coef[0] is for pixel_width and width_coef[1] is for sensor_distance 
        predicted_width_mm = (width_coef[0] * pixel_width) + \
                             (width_coef[1] * sensor_distance_mm) + \
                             width_intercept 

        return round(predicted_height_mm, 2), round(predicted_width_mm, 2)
        
    else:
        # Legacy format: check for old calibration data
        cal_pixel_height = calibration_data.get('pixel_height')
        cal_actual_height = calibration_data.get('actual_height_mm')
        cal_pixel_width = calibration_data.get('pixel_width')
        cal_actual_width = calibration_data.get('actual_width_mm')
        cal_distance = calibration_data.get('distance_mm')

        if cal_pixel_height and cal_actual_height and cal_pixel_width and cal_actual_width and cal_distance:
            mm_per_pixel_height_cal = cal_actual_height / cal_pixel_height
            mm_per_pixel_width_cal = cal_actual_width / cal_pixel_width

            distance_ratio = sensor_distance_mm / cal_distance

            mm_per_pixel_height_current = mm_per_pixel_height_cal * distance_ratio
            mm_per_pixel_width_current = mm_per_pixel_width_cal * distance_ratio

            actual_height_mm = pixel_height * mm_per_pixel_height_current
            actual_width_mm = pixel_width * mm_per_pixel_width_current

            return round(actual_height_mm, 2), round(actual_width_mm, 2)
        else:
            print("Legacy calibration data is incomplete. Please recalibrate.")
            return "N/A (Calibrate)", "N/A (Calibrate)"


def show_results(root):
    """
    Displays the measurement results, calculating averages based on the
    new focal length and regression model calculations with robust averaging.
    """
    global capture_globals
    if capture_globals['photos_taken_count'] == 0:
        messagebox.showwarning("No Photos", "Please capture at least one photo before viewing results.")
        return

    utils.measurement_results = [] # Clear previous results

    # Load calibration models if not already loaded
    if not utils.is_calibrated or not utils.calibration_data.get('height_model'):
        load_calibration_models()

    # Ensure current_sensor_distance is set for results calculation
    if utils.use_simulation_mode:
        if utils.current_sensor_distance is None or utils.current_sensor_distance == 0:
            sim_dist = simpledialog.askfloat("Simulation Distance", "Enter simulated sensor distance (mm):",
                                              initialvalue=DEFAULT_DISTANCE_MM, minvalue=1.0)
            if sim_dist is None:
                messagebox.showwarning("No Distance", "Measurement requires a sensor distance.")
                return
            utils.current_sensor_distance = sim_dist
    
    if utils.current_sensor_distance is None or utils.current_sensor_distance <= 0:
        messagebox.showerror("Missing Data", "Sensor distance is required for calculation. Please ensure sensor is active or enter a simulated value.")
        return

    # Temporarily store recalculated values for averaging
    recalculated_heights = []
    recalculated_widths = []

    for i in range(capture_globals['photos_taken_count']):
        thumb_data = capture_globals['thumbnail_images'][i]
        if thumb_data and thumb_data['raw_frame'] is not None and thumb_data['detections']:
            frame = thumb_data['raw_frame']
            detections_info = thumb_data['detections']

            for det_info in detections_info:
                detected_pixel_height = det_info['pixel_height']
                detected_pixel_width = det_info['pixel_width']
                # Use the sensor distance captured with this specific photo
                current_sensor_distance = det_info.get('sensor_distance_mm', utils.current_sensor_distance or 300.0)

                # Try regression model first, then fallback to focal length method
                if utils.calibration_data.get('height_model') is not None:
                    recalculated_h, recalculated_w = calculate_real_dimensions_using_regression(
                        detected_pixel_height, detected_pixel_width, current_sensor_distance
                    )
                else:
                    recalculated_h, recalculated_w = calculate_real_dimensions(
                        detected_pixel_height, detected_pixel_width, current_sensor_distance
                    )

                if recalculated_h is not None and recalculated_w is not None:
                    # Store these recalculated_h, recalculated_w for averaging later
                    recalculated_heights.append(recalculated_h)
                    recalculated_widths.append(recalculated_w)
                    
                    # Add image path for the annotated image
                    annotated_image_path = os.path.join(r"D:\Research\Research\Codes\Code_3\captured_images", f"annotated_img_{i}.jpg")
                    
                    utils.measurement_results.append({
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'image_slot': i + 1,
                        'ear_id_in_image': len(utils.measurement_results) + 1, # Simple ID for now
                        'pixel_height': detected_pixel_height,
                        'pixel_width': detected_pixel_width,
                        'actual_height_mm': round(recalculated_h, 2),
                        'actual_width_mm': round(recalculated_w, 2),
                        'sensor_distance_mm': current_sensor_distance,
                        'confidence': det_info['confidence'],
                        'image_path': annotated_image_path
                    })
                else:
                    print(f"Skipping measurement due to invalid calculated values for image slot {i+1}")

        else:
            print(f"No detections found for image slot {i+1} or image data is missing.")

    # Calculate robust average measurements if we have any results
    if recalculated_heights and recalculated_widths:
        # Use robust averaging:
        avg_height = calculate_robust_average(recalculated_heights, method="trimmed_mean", trim_fraction=0.2)
        avg_width = calculate_robust_average(recalculated_widths, method="trimmed_mean", trim_fraction=0.2)
        
        # Rounding
        if isinstance(avg_height, (int, float)): # Check if it's a number before rounding
            avg_height = round(avg_height, 2)
        if isinstance(avg_width, (int, float)):
            avg_width = round(avg_width, 2)
    else:
        avg_height = "N/A (No Valid Measurements)"
        avg_width = "N/A (No Valid Measurements)"
        
    # Get the closest/best image info (first one for now)
    closest_image_info = utils.measurement_results[0] if utils.measurement_results else None

    gui_pages.show_results_page(root, avg_height, avg_width, closest_image_info, capture_globals['patient_info_for_session'])

# These functions are now simplified/redundant given iVCam directly handles the camera
# We keep them to avoid errors if other parts of the code still call them,
# but they will have no effect on actual camera selection.
def update_camera_index(camera_index_var):
    global capture_globals
    print(f"Camera index update requested, but using default 0 for iVCam.")
    capture_globals['camera_index'] = 0


def refresh_camera_dropdown(camera_index_var, camera_combo):
    print("Camera dropdown refresh requested, but camera selection is bypassed for iVCam.")
    camera_options = ["0: iVCam (Default)"]
    camera_combo['values'] = camera_options
    camera_index_var.set(camera_options[0])
    capture_globals['camera_index'] = 0