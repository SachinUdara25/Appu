    #main.py
# main.py
import tkinter as tk
from gui_pages import create_home_page
import utils
import calibration
import image_processing

# Using on_app_close from utils.py to avoid circular imports

# Add to your system - precise measurement points
MEASUREMENT_POINTS = {
    "height": "Top of helix to bottom of earlobe",
    "width": "Widest horizontal point including helix"
}

# Add width-specific validation
def validate_width_measurement(pixel_width, actual_width, distance):
    expected_ratio = actual_width / pixel_width
    
    if 0.15 < expected_ratio < 0.25:  # Expected range
        return "Valid"
    else:
        return "Check measurement points"

# Add width-specific calibration
# Collect additional width-focused samples
width_enhanced_samples = [
    # Focus on width precision
    {"pixel_width": 180, "actual_width": 36.1, "distance": 150},
    {"pixel_width": 185, "actual_width": 36.1, "distance": 155},
]

def main():
    root = tk.Tk()
    utils.root_tk = root
    root.title("Ear Measurement System")

    try:
        root.geometry("1920x1080")
    except tk.TclError as e:
        print(f"Note: Could not set initial geometry for root window: {e}")

    # Initialize Tkinter variables after root is created
    calibration._init_tk_vars()
    image_processing._init_tk_vars()
        
    # Load calibration models if available
    try:
        image_processing.load_calibration_models()
    except Exception as e:
        print(f"Could not load calibration models at startup: {e}")
    
    # Don't automatically load the YOLO model at startup
    # Let the user select the model path through the UI
    utils.custom_yolo_model_path = None
    utils.model = None
    print("Please select a YOLO model from the home page to proceed.")

    root.protocol("WM_DELETE_WINDOW", lambda: utils.on_app_close(root))
    create_home_page(root)
    root.mainloop()

if __name__ == '__main__':
    main()


