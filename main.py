# main.py (or Code 2.0.py)

import tkinter as tk
from tkinter import messagebox
import os # For checking model path
import numpy as np
import cv2 # For cvtColor in dummy inference

# Import application modules
import utils
import sensor_function
import calibration
import image_processing # Though most of its functions are called via gui_pages
import gui_pages
# data_handling is mostly called from gui_pages as well

# --- Configuration ---
# Path to the YOLO model. This should be configurable or placed in a known relative path.
YOLO_MODEL_PATH = "best.pt" # Changed to relative path
# Consider adding a way to select this path via GUI if not found, or include it with the app.

# --- Initialization Functions ---
def initialize_yolo_model():
    """Loads the YOLOv8 model."""
    print("initialize_yolo_model: Attempting to load YOLO model...")
    try:
        from ultralytics import YOLO

        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path_to_check = YOLO_MODEL_PATH # Uses the global variable

        if not os.path.isabs(model_path_to_check):
            model_path_to_check = os.path.join(script_dir, model_path_to_check)

        model_path_to_check = os.path.normpath(model_path_to_check)
        print(f"initialize_yolo_model: Checking for model at: {model_path_to_check}")

        if not os.path.exists(model_path_to_check):
            print(f"initialize_yolo_model: Model file NOT FOUND at {model_path_to_check}")
            messagebox.showerror("YOLO Model Error",
                                 f"YOLO model file not found at '{model_path_to_check}'.\n"
                                 "Please ensure the model path is correct in main.py (e.g., 'best.pt' in the same directory as the script).") # Updated message hint
            utils.model = None
            return False

        print(f"initialize_yolo_model: Model file found. Loading with YOLO('{model_path_to_check}')...")
        utils.model = YOLO(model_path_to_check)
        print(f"YOLO model '{model_path_to_check}' loaded successfully from main.py.")

        try:
            print("Performing a dummy YOLO inference for initialization...")
            dummy_image_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
            dummy_image_rgb = cv2.cvtColor(dummy_image_bgr, cv2.COLOR_BGR2RGB)
            _ = utils.model(dummy_image_rgb, verbose=False)
            print("Dummy YOLO inference successful.")
        except Exception as e_dummy:
            print(f"Error during dummy YOLO inference: {e_dummy}")
            messagebox.showwarning("YOLO Init Warning",
                                   f"YOLO model loaded, but a test inference failed: {e_dummy}")

        return True

    except ImportError:
        print("initialize_yolo_model: Ultralytics YOLO library not found.")
        messagebox.showerror("Dependency Error",
                             "Ultralytics YOLO library not found. Please install it (e.g., pip install ultralytics).")
        utils.model = None
        return False
    except Exception as e:
        print(f"initialize_yolo_model: General error loading YOLO model: {e}")
        messagebox.showerror("YOLO Model Error", f"Error loading YOLO model: {e}")
        utils.model = None
        return False

def main():
    """Main function to setup and run the application."""
    root = tk.Tk()
    root.title("Ear Measurement System")

    try:
        root.geometry("1920x1080")
        print("Set geometry to 1920x1080")
    except tk.TclError as e:
        print(f"Could not set fullscreen or fixed geometry: {e}. Using default size.")

    root.resizable(False, False)
    root.update_idletasks()

    if not initialize_yolo_model():
        messagebox.showwarning("YOLO Warning", "YOLO model failed to load. Object detection will require manual selection.", parent=root)

    sensor_function.connect_arduino()
    calibration.load_calibration_model()
    gui_pages.create_home_page(root)

    print("Starting Tkinter mainloop...")
    root.mainloop()

    print("Application closed.")
    if utils.arduino and utils.arduino.is_open:
        try:
            utils.arduino.close()
            print("Arduino connection closed.")
        except Exception as e_arduino_close:
            print(f"Error closing Arduino connection: {e_arduino_close}")

    ipc_cap = image_processing.capture_globals.get("cap")
    if ipc_cap and ipc_cap.isOpened():
        try:
            ipc_cap.release()
            print("Released image_processing camera.")
        except Exception as e_ipc_release:
            print(f"Error releasing image_processing camera: {e_ipc_release}")

    calib_cap = calibration.calibration_globals.get("cap")
    if calib_cap and calib_cap.isOpened():
        try:
            calib_cap.release()
            print("Released calibration camera.")
        except Exception as e_calib_release:
            print(f"Error releasing calibration camera: {e_calib_release}")

if __name__ == "__main__":
    main()
