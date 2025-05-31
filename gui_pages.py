# gui_pages.py

import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, filedialog, Frame, Label, Button, Entry
from PIL import Image, ImageTk
import cv2 # Only if needed for direct camera interaction here, prefer image_processing.py
import os # For path joining, checking file existence
import utils # For accessing shared data like droidcam_ip, patient_info, results
import sensor_function # For connect_arduino (though typically called at app start)
import calibration # For run_calibration_gui, load_calibration_model
import image_processing # For capture_images_entry_point
import data_handling # For save_measurements_and_generate_report

# --- Global variable for background PhotoImage to prevent garbage collection ---
# This should be managed carefully, perhaps one per Toplevel or a shared one if image is same
background_photo_references = {}

# --- Helper Functions ---
def clear_window(window):
    """Destroys all widgets in a given window or frame."""
    for widget in window.winfo_children():
        widget.destroy()

def setup_background(window_or_frame, window_key="default_bg"):
    """Sets up a background image for the given window or frame."""
    # Placeholder path - this should be configurable or relative
    bg_image_path = r"D:\Campus\Lessons\4th Year\Research\Codes\0.2\Calm_lakes_sunset-Nature_HD_Wallpapers_2560x1600.jpg"
    # Fallback if the above path is an issue during development:
    # Check if a local 'background.jpg' or 'background.png' exists
    local_bg_jpg = "background.jpg" # Expected to be in the same directory as the script
    local_bg_png = "background.png" # Expected to be in the same directory as the script

    # Determine script's directory for local fallbacks
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_bg_jpg_path = os.path.join(script_dir, local_bg_jpg)
    local_bg_png_path = os.path.join(script_dir, local_bg_png)

    if os.path.exists(local_bg_jpg_path):
        bg_image_path = local_bg_jpg_path
    elif os.path.exists(local_bg_png_path):
        bg_image_path = local_bg_png_path
    elif not os.path.exists(bg_image_path): # If original path also doesn't exist
        print(f"Background image not found at primary path, local JPG, or local PNG. Using fallback color. Checked: {bg_image_path}, {local_bg_jpg_path}, {local_bg_png_path}")
        window_or_frame.configure(bg="lightgrey") # Fallback background color
        return

    try:
        # Ensure window/frame dimensions are updated before getting size for image
        window_or_frame.update_idletasks()
        win_width = window_or_frame.winfo_width()
        win_height = window_or_frame.winfo_height()

        # If window is not yet sized (e.g. during initial setup), use default full screen
        if win_width < 100 or win_height < 100: # Arbitrary small threshold
            win_width = 1920
            win_height = 1080
            # If it's the main root window, attempt to set its geometry
            # Check if it's the root window (Tk instance)
            if isinstance(window_or_frame, tk.Tk) and not isinstance(window_or_frame, tk.Toplevel):
                 try:
                    window_or_frame.geometry(f"{win_width}x{win_height}")
                 except tk.TclError as e:
                    # This can happen if the window is already managed by a wm that prevents this
                    print(f"Could not set geometry for main window: {e}")


        bg_image = Image.open(bg_image_path).resize((win_width, win_height), Image.Resampling.LANCZOS)
        bg_photo = ImageTk.PhotoImage(bg_image)

        # Determine how to apply the background
        # Using a Label that is placed to cover the entire area and then raising other widgets above it
        # is generally more robust for complex UIs than a Canvas packed first.

        bg_label = Label(window_or_frame, image=bg_photo)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        bg_label.image = bg_photo # Keep a reference
        background_photo_references[window_key + "_bglbl"] = bg_label # Store label ref

        # Ensure other content is stacked on top of this background label
        # This might involve calling .lift() on content frames after they are created,
        # or ensuring the bg_label is created first in its parent.

    except FileNotFoundError:
        print(f"Background image file not found: {bg_image_path}. Using fallback color.")
        window_or_frame.configure(bg="lightblue") # Fallback
    except Exception as e:
        print(f"Error setting background image: {e}")
        window_or_frame.configure(bg="lightyellow") # Fallback for other errors

def go_home(root):
    """Clears the current root window and recreates the home page."""
    # Release camera if active from image_processing or calibration
    if image_processing.capture_globals.get("cap") and image_processing.capture_globals["cap"].isOpened():
        image_processing.capture_globals["cap"].release()
        image_processing.capture_globals["cap"] = None
        print("Released image_processing camera.")
    if calibration.calibration_globals.get("cap") and calibration.calibration_globals["cap"].isOpened():
        calibration.calibration_globals["cap"].release()
        calibration.calibration_globals["cap"] = None
        print("Released calibration camera.")

    clear_window(root)
    create_home_page(root)

# --- Page Creation Functions ---

def create_home_page(root):
    """Creates the initial home page for IP entry and main navigation."""
    clear_window(root)
    root.title("Ear Measurement System - Home")

    # Attempt to set fullscreen or fixed large size for the root window itself
    try:
        # root.attributes('-fullscreen', True) # This can be problematic across OSes
        root.geometry("1920x1080")
    except tk.TclError as e:
        print(f"Note: Could not set fullscreen/geometry for root on home page: {e}")

    # Container frame will hold the background and the content frame
    # This container_frame is what setup_background will apply the image to.
    container_frame = Frame(root)
    container_frame.pack(expand=True, fill="both")
    setup_background(container_frame, "home_page_bg_container")

    # Content frame is placed on top of the container_frame (which has the background)
    # Make its background transparent by not setting one, or matching parent if needed
    content_frame = Frame(container_frame)
    content_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)


    Label(content_frame, text="Ear Measurement System", font=("Arial", 32, "bold"), bg=content_frame.cget('bg')).pack(pady=(0,30))

    ip_frame = Frame(content_frame, bg=content_frame.cget('bg'))
    ip_frame.pack(pady=20)
    Label(ip_frame, text="Enter DroidCam IP Address:", font=("Arial", 16), bg=ip_frame.cget('bg')).pack(side=tk.LEFT, padx=5)
    ip_entry_var = tk.StringVar(value=utils.droidcam_ip or "")
    ip_entry = Entry(ip_frame, textvariable=ip_entry_var, width=20, font=("Arial", 16))
    ip_entry.pack(side=tk.LEFT, padx=5)

    def submit_ip_and_show_buttons():
        ip = ip_entry_var.get()
        if not ip:
            messagebox.showwarning("IP Required", "Please enter the DroidCam IP address.")
            return
        utils.droidcam_ip = ip
        print(f"DroidCam IP set to: {utils.droidcam_ip}")
        messagebox.showinfo("IP Set", f"DroidCam IP address set to: {utils.droidcam_ip}", parent=root) # Explicit parent for Toplevel

        ip_frame.pack_forget()
        ip_submit_button.pack_forget()
        _show_main_buttons_in_frame(content_frame, root)

    ip_submit_button = Button(content_frame, text="Submit IP", command=submit_ip_and_show_buttons, font=("Arial", 14), width=15, height=2)
    ip_submit_button.pack(pady=10)

    if utils.droidcam_ip:
        ip_frame.pack_forget()
        ip_submit_button.pack_forget()
        _show_main_buttons_in_frame(content_frame, root)


def _show_main_buttons_in_frame(frame_to_populate, root_window):
    """Helper to populate a frame with the main navigation buttons."""
    btn_font = ("Arial", 16)
    btn_width = 25
    btn_height = 3
    btn_pady = 15

    # Ensure buttons also have transparent background if frame_to_populate is transparent
    button_bg = frame_to_populate.cget('bg')

    Button(frame_to_populate, text="Calibrate System", command=lambda: calibration.run_calibration_gui(root_window),
           font=btn_font, width=btn_width, height=btn_height, bg=button_bg).pack(pady=btn_pady)

    Button(frame_to_populate, text="Load Calibrated Model", command=calibration.load_calibration_model,
           font=btn_font, width=btn_width, height=btn_height, bg=button_bg).pack(pady=btn_pady)

    Button(frame_to_populate, text="New Patient Measurement", command=lambda: create_patient_form(root_window),
           font=btn_font, width=btn_width, height=btn_height, bg=button_bg).pack(pady=btn_pady)

    Button(frame_to_populate, text="Quit Application", command=root_window.quit,
            font=btn_font, width=btn_width, height=btn_height, bg="salmon").pack(pady=btn_pady)


def create_patient_form(root):
    """Creates the form for entering new patient details."""
    clear_window(root)
    root.title("Ear Measurement System - New Patient")
    try:
        root.geometry("1920x1080")
    except tk.TclError as e:
        print(f"Note: Could not set geometry for patient form: {e}")

    container_frame = Frame(root)
    container_frame.pack(expand=True, fill="both")
    setup_background(container_frame, "patient_form_bg_container")

    content_frame = Frame(container_frame)
    content_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    content_bg = content_frame.cget('bg') # For child widgets

    Label(content_frame, text="Patient Details", font=("Arial", 24, "bold"), bg=content_bg).grid(row=0, column=0, columnspan=2, pady=20)

    form_fields = [
        ("Name:", tk.Entry, "Name"),
        ("Address Line 1:", tk.Entry, "Address1"),
        ("Address Line 2:", tk.Entry, "Address2"),
        ("Address Line 3:", tk.Entry, "Address3"),
        ("Phone (10 digits):", tk.Entry, "Phone"),
        ("Gender:", ttk.Combobox, "Gender"),
        ("Birthday (YYYY-MM-DD):", tk.Entry, "DOB")
    ]

    patient_form_entries = {}

    for i, (label_text, widget_type, key) in enumerate(form_fields):
        Label(content_frame, text=label_text, font=("Arial", 14), bg=content_bg).grid(row=i+1, column=0, sticky="w", padx=10, pady=5)
        if widget_type == tk.Entry:
            entry = widget_type(content_frame, width=40, font=("Arial", 14))
        elif widget_type == ttk.Combobox:
            entry = widget_type(content_frame, values=["Male", "Female", "Other"], state="readonly", width=38, font=("Arial", 14))
        entry.grid(row=i+1, column=1, sticky="ew", padx=10, pady=5)
        patient_form_entries[key] = entry

    button_frame = Frame(content_frame, bg=content_bg)
    button_frame.grid(row=len(form_fields)+1, column=0, columnspan=2, pady=30)

    def submit_patient_info():
        phone = patient_form_entries["Phone"].get()
        if not (phone.isdigit() and len(phone) == 10):
            messagebox.showerror("Validation Error", "Phone number must be 10 digits.", parent=root)
            return

        dob = patient_form_entries["DOB"].get()
        # Basic YYYY-MM-DD format check (can be improved with regex or dateutil.parser)
        if dob and not (len(dob) == 10 and dob[4] == '-' and dob[7] == '-'):
             messagebox.showerror("Validation Error", "Date of Birth should be in YYYY-MM-DD format.", parent=root)
             return

        utils.patient_info.clear()
        for key, widget in patient_form_entries.items():
            utils.patient_info[key] = widget.get()

        print("Patient Info:", utils.patient_info)
        image_processing.capture_images_entry_point(root, utils.patient_info)

    def clear_patient_form_entries():
        for widget in patient_form_entries.values():
            if isinstance(widget, tk.Entry):
                widget.delete(0, tk.END)
            elif isinstance(widget, ttk.Combobox):
                widget.set('')

    Button(button_frame, text="Take Measurement", command=submit_patient_info, font=("Arial", 14, "bold"), width=20, height=2).pack(side=tk.LEFT, padx=10)
    Button(button_frame, text="Clear Form", command=clear_patient_form_entries, font=("Arial", 14), width=15, height=2).pack(side=tk.LEFT, padx=10)
    Button(button_frame, text="Homepage", command=lambda: go_home(root), font=("Arial", 14), width=15, height=2).pack(side=tk.LEFT, padx=10)


def show_results_page(root, avg_ear_height, avg_ear_width, closest_image_info, patient_info_dict):
    clear_window(root)
    root.title("Ear Measurement System - Results")
    try:
        root.geometry("1920x1080")
    except tk.TclError as e:
        print(f"Note: Could not set geometry for results page: {e}")

    container_frame = Frame(root)
    container_frame.pack(expand=True, fill="both")
    setup_background(container_frame, "results_page_bg_container")

    content_frame = Frame(container_frame)
    content_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    content_bg = content_frame.cget('bg')

    Label(content_frame, text="Measurement Results", font=("Arial", 24, "bold"), bg=content_bg).pack(pady=20)
    Label(content_frame, text=f"Average Ear Height (Length): {avg_ear_height:.2f} mm", font=("Arial", 16), bg=content_bg).pack(pady=5)
    Label(content_frame, text=f"Average Ear Width: {avg_ear_width:.2f} mm", font=("Arial", 16), bg=content_bg).pack(pady=10)
    Label(content_frame, text="Representative Image:", font=("Arial", 16, "italic"), bg=content_bg).pack(pady=(10,5))

    image_panel_results = Label(content_frame, bg=content_bg)
    image_panel_results.pack(pady=10)

    if closest_image_info and closest_image_info.get('image_path') and os.path.exists(closest_image_info['image_path']):
        try:
            img_pil = Image.open(closest_image_info['image_path'])
            display_w, display_h = 480, 360
            img_w_orig, img_h_orig = img_pil.width, img_pil.height
            aspect_ratio_orig = img_h_orig / float(img_w_orig) if img_w_orig > 0 else 1.0

            current_w, current_h = display_w, int(display_w * aspect_ratio_orig)
            if current_h > display_h: # If height constraint is violated
                current_h = display_h
                current_w = int(display_h / aspect_ratio_orig) if aspect_ratio_orig > 0 else display_w

            img_pil_resized = img_pil.resize((current_w, current_h), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_pil_resized)

            image_panel_results.config(image=img_tk)
            image_panel_results.image = img_tk

            Label(content_frame,
                  text=f"Details for this image: H: {closest_image_info['ear_height_mm']:.2f}mm, W: {closest_image_info['ear_width_mm']:.2f}mm @ {closest_image_info.get('distance', 'N/A'):.1f}mm",
                  font=("Arial", 12), bg=content_bg).pack()
        except Exception as e:
            image_panel_results.config(text=f"Error displaying image: {e}", font=("Arial", 12), fg="red", bg=content_bg)
            print(f"Error loading/displaying result image: {e}")
    else:
        image_panel_results.config(text="No representative image available or path is invalid.", font=("Arial", 12), bg=content_bg)

    button_frame_results = Frame(content_frame, bg=content_bg)
    button_frame_results.pack(pady=30)
    btn_font_results = ("Arial", 14)
    btn_width_results = 18
    btn_height_results = 2

    Button(button_frame_results, text="Save Data & Report", command=data_handling.save_measurements_and_generate_report,
           font=btn_font_results, width=btn_width_results, height=btn_height_results).pack(side=tk.LEFT, padx=10)
    Button(button_frame_results, text="New Patient", command=lambda: create_patient_form(root),
           font=btn_font_results, width=btn_width_results, height=btn_height_results).pack(side=tk.LEFT, padx=10)

    def go_back_to_capture_current_patient():
        image_processing.capture_images_entry_point(root, patient_info_dict) # patient_info_dict is from this scope

    Button(button_frame_results, text="Recapture", command=go_back_to_capture_current_patient,
           font=btn_font_results, width=btn_width_results, height=btn_height_results).pack(side=tk.LEFT, padx=10)

    Button(button_frame_results, text="Quit", command=root.quit,
           font=btn_font_results, width=btn_width_results, height=btn_height_results, bg="salmon").pack(side=tk.LEFT, padx=10)


if __name__ == '__main__':
    root = tk.Tk()
    # root.geometry("1920x1080")
    # Fullscreen or large geometry for root is better handled inside create_home_page or specific pages
    # to ensure it's applied after clear_window if root is reused.

    class MockUtils:
        def __init__(self):
            self.droidcam_ip = None
            self.patient_info = {}
            self.results = []
            self.scale_model = (0.0015, 0.05)
            self.model = "YOLO Model (Mocked)"
    utils = MockUtils()

    class MockSensor:
        def connect_arduino(self): print("Mock connect_arduino called")
        def get_distance(self): return 150.0
    sensor_function = MockSensor()

    class MockCalibration:
        def run_calibration_gui(self, r):
            print(f"Mock run_calibration_gui called with root: {r}")
            messagebox.showinfo("Mock Calibration", "Calibration GUI (mocked).", parent=r)
        def load_calibration_model(self):
            print("Mock load_calibration_model called")
            messagebox.showinfo("Mock Load Calibration", "Loading calibration model (mocked).")
    calibration = MockCalibration()

    dummy_image_path_for_test = "dummy_closest_ear.png" # Define at module level for access in mock

    class MockImageProcessing:
        capture_globals = {"cap": None} # Mock this attribute
        def capture_images_entry_point(self, r, pi):
            print(f"Mock capture_images_entry_point called with root: {r}, patient_info: {pi}")

            # Simulate some results for show_results_page
            utils.results = [
                {'ear_height_mm': 50, 'ear_width_mm': 30, 'image_path': None, 'distance':100, 'original_image':None},
                {'ear_height_mm': 52, 'ear_width_mm': 31, 'image_path': None, 'distance':110, 'original_image':None}
            ]

            # Ensure dummy image for testing results page
            try:
                Image.new('RGB', (300,200), 'blue').save(dummy_image_path_for_test)
                print(f"Created dummy image: {dummy_image_path_for_test}")
            except Exception as e:
                print(f"Could not create dummy image for test: {e}")

            # Use the path if image was created, else None
            closest_img_path = dummy_image_path_for_test if os.path.exists(dummy_image_path_for_test) else None
            closest_dummy = {'image_path': closest_img_path, 'ear_height_mm': 51, 'ear_width_mm': 30.5, 'distance': 105}

            show_results_page(r, 51.0, 30.5, closest_dummy, pi)

    image_processing = MockImageProcessing()

    class MockDataHandling:
        def save_measurements_and_generate_report(self):
            print("Mock save_measurements_and_generate_report called")
            messagebox.showinfo("Mock Save", "Data saving and report generation (mocked).")
    data_handling = MockDataHandling()

    create_home_page(root) # Start the application flow

    root.mainloop()

    if os.path.exists(dummy_image_path_for_test):
        try:
            os.remove(dummy_image_path_for_test)
            print(f"Cleaned up dummy image: {dummy_image_path_for_test}")
        except Exception as e:
            print(f"Error cleaning up dummy image: {e}")
