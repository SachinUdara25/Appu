# calibration.py

import json
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox, Toplevel, Label, Button, Frame
from PIL import Image, ImageTk
import cv2
import os
import time
import utils
import sensor_function
import image_processing

# --- Calibration Model Load/Save ---
def load_calibration_model(filename="calibration_model.json"):
    try:
        with open(filename, 'r') as f: data = json.load(f)
        if 'm' in data and 'b' in data and isinstance(data['m'], (int, float)) and isinstance(data['b'], (int, float)):
            utils.scale_model = (float(data['m']), float(data['b']))
            print(f"Calibration model loaded: scale = {utils.scale_model[0]:.6f} * distance + {utils.scale_model[1]:.6f}")
            messagebox.showinfo("Load Success", "Calibration model loaded successfully.")
        else:
            utils.scale_model = None; print("Calibration model file corrupt."); messagebox.showwarning("Load Warning", "Corrupt calibration file.")
    except FileNotFoundError: utils.scale_model = None; print("No saved calibration model."); messagebox.showwarning("No Model", "No saved calibration model.")
    except json.JSONDecodeError: utils.scale_model = None; print(f"Error decoding JSON."); messagebox.showerror("Load Error", "Failed to decode calibration file.")
    except Exception as e: utils.scale_model = None; print(f"Error loading model: {e}"); messagebox.showerror("Load Error", f"Error: {e}")

def save_calibration_model(filename="calibration_model.json"):
    if utils.scale_model is None: messagebox.showwarning("No Model", "No calibration model to save."); return
    try:
        data = {"m": utils.scale_model[0], "b": utils.scale_model[1]}
        with open(filename, 'w') as f: json.dump(data, f, indent=4)
        messagebox.showinfo("Save Success", f"Calibration model saved to {filename}.")
    except Exception as e: messagebox.showerror("Save Error", f"Could not save calibration model: {e}")

calibration_globals = {
    "calib_win": None, "video_panel": None, "cap": None, "ip_camera_url": None,
    "num_samples": 0, "current_sample_index": 0, "calibration_data_points": [],
    "status_label": None, "distance_label": None, "root_ref": None,
    "current_frame_for_calib_capture": None,
    "consecutive_frame_read_failures": 0,
    "max_consecutive_failures": 5,
    "feed_retry_delay_ms": 5000,
    "original_feed_delay_ms": 100
}

def run_calibration_gui(root):
    from gui_pages import clear_window, create_home_page
    if calibration_globals.get("cap") and calibration_globals["cap"].isOpened():
        calibration_globals["cap"].release(); calibration_globals["cap"] = None
    calibration_globals["root_ref"] = root
    calibration_globals["calib_win"] = Toplevel(root)
    calibration_globals["calib_win"].title("System Calibration")
    calibration_globals["calib_win"].geometry("1920x1080"); calibration_globals["calib_win"].transient(root); calibration_globals["calib_win"].grab_set()
    main_frame = Frame(calibration_globals["calib_win"]); main_frame.pack(expand=True, fill="both", padx=20, pady=20)
    Label(main_frame, text="System Calibration", font=("Arial", 24, "bold")).pack(pady=20)
    instructions = ("Instructions:\n1. Enter # samples (min 3).\n2. Position ref. object at diff. distances.\n3. Auto-detect or manual select.\n4. Enter ACTUAL W/H of object (mm).\n5. Ensure good lighting.")
    Label(main_frame, text=instructions, font=("Arial", 12), justify=tk.LEFT).pack(pady=10)
    Button(main_frame, text="Start Calibration Process", command=_start_calibration_process, font=("Arial", 14), width=25, height=2).pack(pady=20)
    Button(main_frame, text="Cancel & Return Home", command=lambda: _cancel_calibration(create_home_page), font=("Arial", 14), width=25, height=2).pack(pady=10)
    calibration_globals["calib_win"].protocol("WM_DELETE_WINDOW", lambda: _cancel_calibration(create_home_page))

def _cancel_calibration(home_page_func):
    if calibration_globals.get("cap") and calibration_globals["cap"].isOpened():
        calibration_globals["cap"].release(); calibration_globals["cap"] = None
    if calibration_globals.get("calib_win"): calibration_globals["calib_win"].destroy(); calibration_globals["calib_win"] = None

def _start_calibration_process():
    num_s = simpledialog.askinteger("Number of Samples", "How many samples?", parent=calibration_globals["calib_win"], minvalue=3, maxvalue=10)
    if num_s is None: return
    calibration_globals["num_samples"] = num_s; calibration_globals["current_sample_index"] = 0; calibration_globals["calibration_data_points"].clear()
    calibration_globals["consecutive_frame_read_failures"] = 0
    for w in list(calibration_globals["calib_win"].winfo_children()): w.destroy()
    _setup_calibration_capture_ui()

def _setup_calibration_capture_ui():
    win = calibration_globals["calib_win"]
    Label(win, text="Calibration Capture", font=("Arial", 20, "bold")).pack(pady=10)
    calibration_globals["video_panel"] = Label(win); calibration_globals["video_panel"].pack(pady=10)
    info_frame = Frame(win); info_frame.pack(pady=5)
    calibration_globals["status_label"] = Label(info_frame, text=f"Sample: 1/{calibration_globals['num_samples']}", font=("Arial", 14)); calibration_globals["status_label"].pack(side=tk.LEFT, padx=10)
    calibration_globals["distance_label"] = Label(info_frame, text="Distance: --- mm", font=("Arial", 14)); calibration_globals["distance_label"].pack(side=tk.LEFT, padx=10)
    Button(win, text="Capture This Sample", command=_trigger_calibration_capture, font=("Arial", 14), width=20, height=2).pack(pady=10)
    if not utils.droidcam_ip:
        messagebox.showerror("IP Error", "DroidCam IP not set.", parent=win); _cancel_calibration(lambda r: image_processing.go_to_previous_page(lambda rr: gui_pages.create_home_page(rr))); return
    ip_url = f"http://{utils.droidcam_ip}:4747/video"
    calibration_globals["ip_camera_url"] = ip_url
    calibration_globals["cap"] = cv2.VideoCapture(ip_url)
    if not calibration_globals["cap"].isOpened():
        messagebox.showerror("Camera Error", f"Failed to open DroidCam: {ip_url}", parent=win)
        if calibration_globals["cap"]: calibration_globals["cap"].release(); calibration_globals["cap"] = None
        _cancel_calibration(lambda r: image_processing.go_to_previous_page(lambda rr: gui_pages.create_home_page(rr))); return
    _update_calibration_video_feed()

def _update_calibration_video_feed():
    cap = calibration_globals.get("cap")
    if cap and cap.isOpened():
        ret, frame_bgr = cap.read()
        if ret and frame_bgr is not None:
            calibration_globals["consecutive_frame_read_failures"] = 0
            frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
            dist = sensor_function.get_distance()
            dist_text = f"Distance: {dist:.1f} mm" if dist is not None and dist > 0 else "Distance: --- mm"
            if calibration_globals.get("distance_label"):
                 calibration_globals["distance_label"].config(text=dist_text)
            calibration_globals["current_frame_for_calib_capture"] = (frame_bgr.copy(), dist if dist is not None and dist > 0 else -1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb).resize((640, 480), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            if calibration_globals.get("video_panel"):
                calibration_globals["video_panel"].configure(image=img_tk); calibration_globals["video_panel"].image = img_tk
            if calibration_globals.get("calib_win"):
                calibration_globals["video_panel"].after(calibration_globals["original_feed_delay_ms"], _update_calibration_video_feed)
        else:
            calibration_globals["consecutive_frame_read_failures"] += 1
            print(f"Failed to read frame: calibration feed (Attempt: {calibration_globals['consecutive_frame_read_failures']}).")
            next_retry_delay = 200
            if calibration_globals["consecutive_frame_read_failures"] >= calibration_globals["max_consecutive_failures"]:
                if calibration_globals.get("distance_label"):
                    calibration_globals["distance_label"].config(text="Camera feed lost. Retrying...")
                print(f"Max frame read failures reached. Waiting for {calibration_globals['feed_retry_delay_ms']}ms.")
                next_retry_delay = calibration_globals['feed_retry_delay_ms']
            if calibration_globals.get("calib_win") and calibration_globals.get("video_panel"):
                 calibration_globals["video_panel"].after(next_retry_delay, _update_calibration_video_feed)
            else: print("Calibration window or video panel no longer exists. Stopping feed update attempts.")
    elif calibration_globals.get("calib_win"):
        print("Calibration camera is not open. Check connection.")

def _trigger_calibration_capture():
    print("_trigger_calibration_capture: Function started.") # New log
    image_np_bgr = None
    distance_mm = -1 # Default to invalid
    cap_object = calibration_globals.get("cap")
    ip_url = calibration_globals.get("ip_camera_url")

    if cap_object is None or not cap_object.isOpened():
        print("_trigger_calibration_capture: Camera not open or not initialized. Attempting to re-open...")
        if not ip_url:
            messagebox.showerror("Camera Error", "Camera IP URL not found for re-initialization.", parent=calibration_globals.get("calib_win"))
            return
        cap_object = cv2.VideoCapture(ip_url)
        calibration_globals["cap"] = cap_object
        if not cap_object.isOpened():
            messagebox.showerror("Camera Error", f"Failed to re-open camera at {ip_url}. Please check DroidCam connection.", parent=calibration_globals.get("calib_win"))
            return
        print(f"_trigger_calibration_capture: Camera re-opened successfully from {ip_url}")
        time.sleep(0.5)

    if cap_object and cap_object.isOpened():
        print("_trigger_calibration_capture: Attempting fresh camera read.")
        ret, fresh_frame_bgr = cap_object.read()
        if ret and fresh_frame_bgr is not None:
            fresh_frame_bgr = cv2.rotate(fresh_frame_bgr, cv2.ROTATE_90_CLOCKWISE)
            image_np_bgr = fresh_frame_bgr.copy()

            print("_trigger_calibration_capture: Attempting fresh distance read.") # New log
            current_dist = sensor_function.get_distance()
            dist_text_ui = "Distance: --- mm (calib capture)"
            if current_dist is not None and current_dist > 0:
                distance_mm = current_dist
                dist_text_ui = f"Distance: {distance_mm:.1f} mm"
            # else distance_mm remains -1
            if calibration_globals.get("distance_label"): calibration_globals["distance_label"].config(text=dist_text_ui)
            print(f"_trigger_calibration_capture: Fresh frame acquired and rotated, reported distance: {distance_mm:.1f}mm")
        else:
            print("_trigger_calibration_capture: Failed to get fresh frame, falling back to last known.")
            if hasattr(calibration_globals, "current_frame_for_calib_capture") and \
               calibration_globals.get("current_frame_for_calib_capture") is not None and \
               isinstance(calibration_globals["current_frame_for_calib_capture"], tuple) and \
               len(calibration_globals["current_frame_for_calib_capture"]) == 2 and \
               calibration_globals["current_frame_for_calib_capture"][0] is not None:
                image_np_bgr, distance_mm = calibration_globals["current_frame_for_calib_capture"] # This frame is already rotated
                print(f"_trigger_calibration_capture: Using cached frame. Cached distance: {distance_mm:.1f}mm")
            else:
                print("_trigger_calibration_capture: No cached frame available either.")
    else:
        print("_trigger_calibration_capture: Camera object not available or not open even after re-init attempt.")

    if image_np_bgr is None or image_np_bgr.size == 0:
        messagebox.showwarning("Capture Error", "No valid camera frame available for calibration (fresh or cached). Please ensure the camera is active.", parent=calibration_globals.get("calib_win"))
        return

    print(f"_trigger_calibration_capture: Current distance_mm before manual input check: {distance_mm:.1f}mm") # New log
    if distance_mm <= 0:
        print("_trigger_calibration_capture: Distance is invalid, prompting for manual input.") # New log
        if not messagebox.askyesno("Distance Error", "Sensor distance reading failed or invalid. Capture sample anyway with manual distance input?", parent=calibration_globals.get("calib_win")):
            print("_trigger_calibration_capture: User chose NOT to enter manual distance. Aborting sample.") # New log
            return
        manual_dist = simpledialog.askfloat("Manual Distance", "Enter current distance (mm):", parent=calibration_globals.get("calib_win"), minvalue=1.0)
        if manual_dist is None or manual_dist <= 0:
            messagebox.showwarning("Invalid Input", "Valid manual distance not provided.", parent=calibration_globals.get("calib_win"))
            print("_trigger_calibration_capture: User did not provide valid manual distance. Aborting sample.") # New log
            return
        distance_mm = manual_dist
        print(f"_trigger_calibration_capture: Using manually entered distance: {distance_mm:.1f}mm") # New log

    print(f"_trigger_calibration_capture: Proceeding to image processing. Image shape: {image_np_bgr.shape if image_np_bgr is not None else 'None'}, Distance for processing: {distance_mm:.1f}mm") # New log
    roi_coords = image_processing.detect_ear_yolo(image_np_bgr)
    if roi_coords is None: print("_trigger_calibration_capture: YOLO detection (detect_ear_yolo) returned None.")
    else: print(f"_trigger_calibration_capture: YOLO detection returned ROI: {roi_coords}")

    object_roi_bgr = None
    if roi_coords:
        x,y,w,h = roi_coords; object_roi_bgr = image_np_bgr[y:y+h, x:x+w]
    else:
        messagebox.showinfo("Manual Selection", "Auto-detection failed. Select object.", parent=calibration_globals.get("calib_win"))
        if calibration_globals.get("calib_win"): calibration_globals.get("calib_win").withdraw()
        manual_roi = cv2.selectROI("Select Object for Calibration", image_np_bgr, False, False)
        if calibration_globals.get("calib_win"): calibration_globals.get("calib_win").deiconify()
        cv2.destroyWindow("Select Object for Calibration")
        if not manual_roi or manual_roi == (0,0,0,0): messagebox.showwarning("Selection Skipped", "Manual selection skipped.", parent=calibration_globals.get("calib_win")); return
        x,y,w,h = [int(c) for c in manual_roi]
        if w == 0 or h == 0: messagebox.showwarning("Invalid ROI", "ROI zero W/H.", parent=calibration_globals.get("calib_win")); return
        object_roi_bgr = image_np_bgr[y:y+h, x:x+w]; roi_coords = (x,y,w,h)
        print(f"_trigger_calibration_capture: Manual ROI selected: {roi_coords}")
    if object_roi_bgr is None or object_roi_bgr.size == 0: messagebox.showerror("ROI Error", "No valid object region.", parent=calibration_globals.get("calib_win")); return

    object_roi_gray = cv2.cvtColor(object_roi_bgr, cv2.COLOR_BGR2GRAY)
    print("_trigger_calibration_capture: Calling get_ear_measurements_from_roi...")
    pixel_dims = image_processing.get_ear_measurements_from_roi(object_roi_gray)
    if not pixel_dims:
        print("_trigger_calibration_capture: get_ear_measurements_from_roi returned None. Using ROI box dimensions as fallback.")
        messagebox.showwarning("Measurement Error", "Could not get pixel dimensions from contours. Using ROI box dimensions.", parent=calibration_globals.get("calib_win"))
        pixel_width, pixel_height = roi_coords[2], roi_coords[3]
    else:
        pixel_width, pixel_height = pixel_dims
        print(f"_trigger_calibration_capture: get_ear_measurements_from_roi returned pixel_width={pixel_width}, pixel_height={pixel_height}")

    print("_trigger_calibration_capture: Prompting for actual dimensions.")
    actual_h = simpledialog.askfloat("Actual Height", "ACTUAL object HEIGHT (mm - horizontal in image):", parent=calibration_globals.get("calib_win"), minvalue=0.1)
    if actual_h is None: print("_trigger_calibration_capture: User cancelled actual height input."); return
    actual_w = simpledialog.askfloat("Actual Width", "ACTUAL object WIDTH (mm - vertical in image):", parent=calibration_globals.get("calib_win"), minvalue=0.1)
    if actual_w is None: print("_trigger_calibration_capture: User cancelled actual width input."); return

    print(f"_trigger_calibration_capture: Storing calibration data. Dist: {distance_mm}, PixelW(H): {pixel_width}, ActualH: {actual_h}, PixelH(W): {pixel_height}, ActualW: {actual_w}")
    if pixel_width > 0: calibration_globals["calibration_data_points"].append((distance_mm, pixel_width, actual_h))
    if pixel_height > 0: calibration_globals["calibration_data_points"].append((distance_mm, pixel_height, actual_w))

    calibration_globals["current_sample_index"] += 1
    if calibration_globals["current_sample_index"] < calibration_globals["num_samples"]:
        calibration_globals["status_label"].config(text=f"Sample: {calibration_globals['current_sample_index']+1}/{calibration_globals['num_samples']}")
    else:
        print("_trigger_calibration_capture: All samples collected. Finishing calibration.") # New log
        if calibration_globals.get("cap") and calibration_globals["cap"].isOpened():
            calibration_globals["cap"].release(); calibration_globals["cap"] = None
        _finish_calibration_calculation()

def _finish_calibration_calculation():
    from gui_pages import create_home_page
    if not calibration_globals["calibration_data_points"] or len(calibration_globals["calibration_data_points"]) < 1:
        messagebox.showerror("Calibration Error", "Not enough data for calibration.", parent=calibration_globals.get("calib_win")); _cancel_calibration(create_home_page); return
    distances = np.array([dp[0] for dp in calibration_globals["calibration_data_points"]]).reshape(-1, 1)
    scales_y = np.array([dp[2] / dp[1] for dp in calibration_globals["calibration_data_points"] if dp[1] > 0])
    if len(scales_y) < 2 :
        messagebox.showerror("Calibration Error", "Not enough valid data (min 2 with non-zero pixel dims).", parent=calibration_globals.get("calib_win")); _cancel_calibration(create_home_page); return
    A = np.hstack([distances[:len(scales_y)], np.ones((len(scales_y), 1))])
    try:
        m,b = np.linalg.lstsq(A, scales_y, rcond=None)[0]
        utils.scale_model = (float(m), float(b)); save_calibration_model()
        msg = (f"Calibration OK!\nScale(mm/px) = {m:.6f}*dist(mm) + {b:.6f}\nSaved.")
        messagebox.showinfo("Calibration Complete", msg, parent=calibration_globals.get("calib_win"))
    except np.linalg.LinAlgError as e: messagebox.showerror("Math Error", f"LinAlgError: {e}", parent=calibration_globals.get("calib_win")); utils.scale_model = None
    except Exception as e: messagebox.showerror("Error", f"Calc error: {e}", parent=calibration_globals.get("calib_win")); utils.scale_model = None
    _cancel_calibration(create_home_page)

if __name__ == '__main__':
    root = tk.Tk(); root.title("Calibration Test"); root.geometry("300x200")
    class MockUtils:
        def __init__(self):
            self.scale_model = None
            self.droidcam_ip = "192.168.1.101"
            self.model = None
    utils=MockUtils()
    class MockSensor:
        def get_distance(self):
            return 100.0 + np.random.rand() * 10
    sensor_function=MockSensor()
    class MockImageProcessing:
        def detect_ear_yolo(self,img):
            h,w,_=img.shape
            return(w//4,h//4,w//2,h//2)if np.random.rand()>0.1 else None
        def get_ear_measurements_from_roi(self,roi_gray):
            return(roi_gray.shape[1]-20,roi_gray.shape[0]-20)if roi_gray.size>0 else None
    image_processing=MockImageProcessing()
    Button(root,text="Run Calibration",command=lambda:run_calibration_gui(root)).pack(pady=20)
    Button(root,text="Load Model",command=load_calibration_model).pack(pady=5)
    root.mainloop()
