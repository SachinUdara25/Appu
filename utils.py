# utils.py
import cv2
from ultralytics import YOLO # Ensure this is imported
import os
import tkinter as tk
from tkinter import messagebox
import numpy as np

from ear_detector import EarDetector

# --- Global Variables ---
camera_source = 0
custom_yolo_model_path = None
model = None
ear_detector_instance = None # Added
ultrasonic_sensor_instance = None
calibration_data = {}
is_calibrated = False

patient_info = {}
measurement_results = []

active_camera_object = None
root_tk = None
use_simulation_mode = False  # Use real sensor instead of simulation

# New global variables to hold current pixel dimensions and sensor distance
current_pixel_height = 0
current_pixel_width = 0
current_sensor_distance = None # Will hold the last read distance from sensor/simulation

def initialize_yolo_model():
    """
    Initialize the YOLO model from the specified path or try to find a suitable model.
    Returns True if successful, False otherwise.
    """
    global model, custom_yolo_model_path, ear_detector_instance

    # පෙර තිබූ model එක නිදහස් කරන්න (Clear up any existing model to free memory)
    if model is not None:
        try:
            # model යනු EarDetector instance එකයි. එහි ඇතුලේ ඇති YOLO model එකට ප්‍රවේශ විය යුතුයි.
            # 'model.model' යනු සැබෑ ultralytics.YOLO object එකයි.
            if hasattr(model, 'model') and hasattr(model.model, 'cpu'): 
                model.model.cpu() # මෙතනයි වෙනස්කම!
            model = None
            ear_detector_instance = None
            import gc
            gc.collect()  # Force garbage collection
        except Exception as e:
            print(f"Error releasing previous model: {e}")

    model_to_load = None
    if custom_yolo_model_path:
        model_to_load = custom_yolo_model_path
    else:
        # Default model path (ඔබට අවශ්‍ය නම් මෙය segmentation model එකක default path එකට වෙනස් කළ හැක)
        default_path = os.path.join(os.path.dirname(__file__), "dataset", "runs", "detect", "train2", "weights", "best.pt")
        if os.path.exists(default_path):
            model_to_load = default_path
        else:
            messagebox.showwarning("Model Not Found", "No custom model path specified and default detection model not found. Please select a model.")
            return False
    
    if not model_to_load:
        messagebox.showerror("Model Selection Error", "No YOLO model path provided or found.")
        return False

    try:
        # Initialize EarDetector instance and assign it to both utils.model and utils.ear_detector_instance
        model = EarDetector(model_path=model_to_load)
        ear_detector_instance = model  # Set the ear_detector_instance as well
        print(f"YOLO model (via EarDetector) loaded successfully from {model_to_load}")
        return True
    except Exception as e:
        print(f"Failed to initialize EarDetector: {e}")
        messagebox.showerror("Model Error", f"Failed to load the YOLO model. Please check the path and model type. Error: {e}")
        model = None # අසාර්ථක වුවහොත් model එක None බවට පත් කරන්න
        ear_detector_instance = None
        return False

def initialize_ultrasonic_sensor():
    """
    Initialize the ultrasonic sensor.
    Returns True if successful, False otherwise.
    """
    global ultrasonic_sensor_instance, use_simulation_mode
    
    try:
        from ultrasonic_sensor import UltrasonicSensor
        ultrasonic_sensor_instance = UltrasonicSensor()
        # Check if sensor was successfully initialized (ser is not None and port is open)
        if ultrasonic_sensor_instance.ser and ultrasonic_sensor_instance.ser.is_open:
            print("Ultrasonic sensor initialized successfully")
            use_simulation_mode = False
            return True
        else:
            print("Failed to connect to ultrasonic sensor")
            ultrasonic_sensor_instance = None
            use_simulation_mode = True
            return False
    except Exception as e:
        print(f"Error initializing ultrasonic sensor: {e}")
        ultrasonic_sensor_instance = None
        use_simulation_mode = True
        return False

def get_available_cameras(max_test=10):
    """Detect available video sources."""
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Found webcam at index {i}")
            available.append(i)
            cap.release()
    return available


def initialize_camera(source=None):
    """
    Initialize iVCam virtual camera.
    
    Args:
        source (int, optional): Camera index, defaults to 0 for iVCam.
        
    Returns:
        bool: True if camera was successfully initialized, False otherwise
    """
    global active_camera_object, camera_source
    
    # Always use index 0 for iVCam
    camera_source = 0
    
    # Release any existing camera
    if active_camera_object is not None:
        if hasattr(active_camera_object, 'isOpened') and active_camera_object.isOpened():
            print("Releasing previously active camera")
            active_camera_object.release()
        active_camera_object = None
    
    # If source is specified, use it
    if source is not None:
        camera_source = source
    
    # Try to open the specified camera
    try:
        print(f"Attempting to open camera at index {camera_source}")
        active_camera_object = cv2.VideoCapture(camera_source)
        
        if active_camera_object.isOpened():
            # Verify camera is working by reading a frame
            ret, frame = active_camera_object.read()
            
            if ret and frame is not None and frame.size > 0:
                # Get camera properties
                width = int(active_camera_object.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(active_camera_object.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = active_camera_object.get(cv2.CAP_PROP_FPS)
                
                print(f"Successfully opened camera at index {camera_source} ({width}x{height} @ {fps:.1f}fps)")
                
                # Set optimal camera properties if possible
                try:
                    # Try to set higher resolution if available
                    active_camera_object.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    active_camera_object.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    
                    # Try to set higher FPS if available
                    active_camera_object.set(cv2.CAP_PROP_FPS, 30)
                    
                    # Get updated properties
                    new_width = int(active_camera_object.get(cv2.CAP_PROP_FRAME_WIDTH))
                    new_height = int(active_camera_object.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    new_fps = active_camera_object.get(cv2.CAP_PROP_FPS)
                    
                    if new_width != width or new_height != height or new_fps != fps:
                        print(f"Camera properties updated: {new_width}x{new_height} @ {new_fps:.1f}fps")
                except Exception as e:
                    print(f"Note: Could not optimize camera properties: {e}")
                
                return True
            else:
                print(f"Camera at index {camera_source} opened but couldn't capture frames")
                active_camera_object.release()
                active_camera_object = None
        else:
            print(f"Failed to open camera at index {camera_source}")
    except Exception as e:
        print(f"Error opening camera at index {camera_source}: {e}")
        if active_camera_object is not None:
            active_camera_object.release()
            active_camera_object = None
    
    # If failed, try to find any available camera
    print("Attempting to find alternative cameras...")
    available_cameras, camera_info = get_available_cameras()
    
    if available_cameras:
        # First try to find iVCam
        ivcam_index = None
        for cam_info in camera_info:
            if "iVCam" in cam_info.get("name", ""):
                ivcam_index = cam_info["index"]
                break
                
        # Try iVCam first if found
        if ivcam_index is not None and ivcam_index != camera_source:
            try:
                print(f"Trying iVCam camera (index {ivcam_index})")
                active_camera_object = cv2.VideoCapture(ivcam_index)
                
                # Set properties for better compatibility with iVCam
                active_camera_object.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                active_camera_object.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                active_camera_object.set(cv2.CAP_PROP_FPS, 30)
                
                # Add a small delay to allow camera to initialize properly
                import time
                time.sleep(0.5)
                
                if active_camera_object.isOpened():
                    # Try multiple times to get a frame
                    max_retries = 3
                    for retry in range(max_retries):
                        ret, frame = active_camera_object.read()
                        if ret and frame is not None and frame.size > 0:
                            camera_source = ivcam_index
                            print(f"Fallback: Successfully opened iVCam camera (index {ivcam_index})")
                            return True
                        time.sleep(0.2)  # Short delay between retries
                    
                    # If we get here, we couldn't get a valid frame
                    active_camera_object.release()
                    active_camera_object = None
            except Exception as e:
                print(f"Error trying iVCam camera: {e}")
                if active_camera_object is not None:
                    active_camera_object.release()
                    active_camera_object = None
        
        # Try other available cameras
        for cam_info in camera_info:
            idx = cam_info["index"]
            if idx != camera_source and cam_info.get("working", True):  # Skip the one we already tried
                try:
                    print(f"Trying alternative camera: {cam_info['name']} (index {idx})")
                    active_camera_object = cv2.VideoCapture(idx)
                    if active_camera_object.isOpened():
                        ret, frame = active_camera_object.read()
                        if ret and frame is not None and frame.size > 0:
                            camera_source = idx
                            print(f"Fallback: Successfully opened camera at index {idx} ({cam_info['name']})")
                            return True
                        else:
                            active_camera_object.release()
                            active_camera_object = None
                except Exception as e:
                    print(f"Error trying fallback camera at index {idx}: {e}")
                    if active_camera_object is not None:
                        active_camera_object.release()
                        active_camera_object = None
    
    print("Failed to initialize any camera")
    
    # Provide troubleshooting information
    print("\nTroubleshooting tips:")
    print("1. Ensure your camera is properly connected")
    print("2. Check if another application is using the camera")
    print("3. Try restarting your computer")
    print("4. Check device manager to ensure camera drivers are installed correctly")
    
    return False


def _cleanup_before_exit():
    print("Performing cleanup before exit...")
    global active_camera_object, ultrasonic_sensor_instance

    if active_camera_object is not None:
        if hasattr(active_camera_object, 'isOpened') and active_camera_object.isOpened():
            print("Releasing active camera object.")
            active_camera_object.release()
        active_camera_object = None

    if ultrasonic_sensor_instance is not None:
        if hasattr(ultrasonic_sensor_instance, 'close'):
            print("Closing sensor instance.")
            ultrasonic_sensor_instance.close()
        ultrasonic_sensor_instance = None

    print("Cleanup complete.")

def on_app_close(root_window):
    """
    Handle application close event - moved from main.py to avoid circular imports
    """
    from tkinter import messagebox
    if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
        _cleanup_before_exit()
        root_window.destroy()


def normalize_image(image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve image quality."""
    if image is None:
        return None
        
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge channels
    limg = cv2.merge((cl, a, b))
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return enhanced


def refresh_camera_list():
    """
    Returns a fixed camera list for iVCam.
    iVCam typically shows up as a virtual camera at index 0 or 1.
    
    Returns:
        tuple: (list of available camera indices, list of camera info dictionaries)
    """
    # Return fixed camera info for iVCam
    available_cameras = [0]  # iVCam camera index
    camera_info = [{
        'index': 0,
        'name': 'iVCam',
        'resolution': (1920, 1080),  # iVCam default HD resolution
        'fps': 30.0
    }]
    
    # Release any existing camera that might be in use
    global active_camera_object
    if active_camera_object is not None:
        try:
            active_camera_object.release()
        except:
            pass
        active_camera_object = None
    
    # Force garbage collection to ensure resources are released
    import gc
    gc.collect()
    
    return available_cameras, camera_info
    time.sleep(0.5)
    
    # Rescan for available cameras
    available, camera_info = get_available_cameras(max_test=15)  # Scan more indices during refresh
    
    print(f"Camera refresh complete. Found {len(available)} working cameras.")
    
    return available, camera_info


def center_window(window):
    """
    Center a tkinter window on the screen.
    
    Args:
        window: The tkinter window to center
    """
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    x = (window.winfo_screenwidth() // 2) - (width // 2)
    y = (window.winfo_screenheight() // 2) - (height // 2)
    window.geometry('{}x{}+{}+{}'.format(width, height, x, y))

    
