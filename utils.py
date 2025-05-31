# utils.py

# Shared global variables
arduino = None  # Serial connection object for Arduino
calibration_samples = []  # List to store calibration data points
scale_model = None  # Tuple (m, b) for the calibration model: scale = m * distance + b
results = []  # List to store measurement results for a patient: (length_mm, width_mm, temp_image_path)
captured_data = [] # List to store captured image frames and distances: (image_array, distance)
patient_info = {}  # Dictionary to store current patient's information
model = None  # YOLO model instance
droidcam_ip = None # String for DroidCam IP address
