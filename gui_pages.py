#gui_pages.py
# gui_pages.py

import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, filedialog, Frame, Label, Button, Entry
from PIL import Image, ImageTk
import cv2
import os
import utils
# import sensor_function # This import will likely cause an error, might need to be removed or adapted
import calibration
import image_processing
import data_handling
# gui_pages.py
from utils import initialize_yolo_model
try:
    from tkcalendar import DateEntry
    TKCALENDAR_AVAILABLE = True
except ImportError:
    print("Warning: tkcalendar not available. Date picker will use basic text entry.")
    TKCALENDAR_AVAILABLE = False
import ui_utils  # Import our new UI utilities
# utils.initialize_yolo_model will be used directly
# Using on_app_close from utils.py to avoid circular imports

# --- Global variable for background PhotoImage to prevent garbage collection ---
background_photo_references = {}

# --- Helper Functions ---
def clear_window(window):
    """Safely clear all widgets from a window, handling destroyed widgets gracefully."""
    try:
        children = window.winfo_children()
        for widget in children:
            try:
                if widget.winfo_exists():
                    widget.destroy()
            except tk.TclError:
                # Widget is already destroyed or invalid, skip it
                pass
    except tk.TclError:
        # Parent window might be destroyed, nothing to clear
        pass

def setup_background(window_or_frame, window_key="default_bg"):
    # Use relative path to background image
    bg_image_path = "background.jpg"

    # Use our enhanced background setup function
    bg_label = ui_utils.setup_enhanced_background(window_or_frame, bg_image_path, window_key)
    
    if bg_label and bg_label.image:
        # Keep reference to prevent garbage collection
        background_photo_references[window_key] = bg_label.image
        
        # Make sure the background label is at the bottom layer
        bg_label.lower()

def go_home(root):
    # Release camera resources if they are open
    if image_processing.capture_globals.get("cap") and image_processing.capture_globals["cap"].isOpened():
        image_processing.capture_globals["cap"].release()
        image_processing.capture_globals["cap"] = None
        print("Released image_processing camera.")
    if calibration.calibration_globals.get("cap") and calibration.calibration_globals["cap"].isOpened():
        calibration.calibration_globals["cap"].release()
        calibration.calibration_globals["cap"] = None
        print("Released calibration camera.")

    if utils.ultrasonic_sensor_instance and hasattr(utils.ultrasonic_sensor_instance,
                                                    'ser') and utils.ultrasonic_sensor_instance.ser is not None:
        if utils.ultrasonic_sensor_instance.ser.is_open:
            print("go_home: Closing ultrasonic sensor.")
            utils.ultrasonic_sensor_instance.close()
        # Clear the instance in utils, as it might be re-initialized later by calibration or capture pages
        # This assumes that pages needing the sensor will always re-initialize it.
        utils.ultrasonic_sensor_instance = None

    clear_window(root)
    create_home_page(root)

# --- New Helper Functions for Home Page Logic ---
def _check_and_enable_main_buttons(buttons_ref_dict, model_load_button, root_ref):
    model_path_is_set = bool(utils.custom_yolo_model_path)
    model_object_is_loaded = bool(utils.model)

    actions_ready = model_path_is_set and model_object_is_loaded
    action_button_state = tk.NORMAL if actions_ready else tk.DISABLED

    if buttons_ref_dict.get('calibrate'):
        buttons_ref_dict['calibrate'].config(state=action_button_state)
    if buttons_ref_dict.get('new_patient'):
        buttons_ref_dict['new_patient'].config(state=action_button_state)

    # Always enable the model load button to allow users to change models
    if model_load_button:
        model_load_button.config(state=tk.NORMAL)

def _load_yolo_model_path_and_init(root_ref, buttons_ref_dict, model_load_button):
    # Create a dialog to select between default model or custom model
    model_select_window = tk.Toplevel(root_ref)
    model_select_window.title("Select YOLO Model")
    model_select_window.geometry("600x450")  # Increased size for better visibility
    model_select_window.transient(root_ref)
    model_select_window.grab_set()
    model_select_window.lift()  # Ensure window is on top
    
    # Center the window
    model_select_window.update_idletasks()
    width = model_select_window.winfo_width()
    height = model_select_window.winfo_height()
    x = (model_select_window.winfo_screenwidth() // 2) - (width // 2)
    y = (model_select_window.winfo_screenheight() // 2) - (height // 2)
    model_select_window.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    
    # Add a frame with padding
    frame = ttk.Frame(model_select_window, padding="20")
    frame.pack(expand=True, fill="both")
    
    # Add a label
    ttk.Label(frame, text="Select YOLO Model", font=("Arial", 16, "bold")).pack(pady=10)
    
    # Add radio buttons for model selection
    model_choice = tk.StringVar(value="segmentation")  # Set default to segmentation model
    
    # Create a style with larger font for better visibility
    style = ttk.Style()
    style.configure("TRadiobutton", font=("Arial", 12))
    
    # Add a header label
    ttk.Label(
        frame,
        text="Please select a YOLO model for ear detection:",
        font=("Arial", 14, "bold")
    ).pack(anchor="w", pady=(0, 15))
    
    # Add info label
    info_label = ttk.Label(
        frame,
        text="Note: The segmentation model provides better detection accuracy and visual feedback.",
        font=("Arial", 10, "italic"),
        foreground="blue"
    )
    info_label.pack(anchor="w", pady=(0, 10))
    
    
    ttk.Radiobutton(
        frame,
        text="Use Default Ear Detection Model\n(dataset/runs/detect/train2/weights/best.pt)",
        value="default",
        variable=model_choice,
        style="TRadiobutton"
    ).pack(anchor="w", pady=10)
    
    ttk.Radiobutton(
        frame,
        text="Use Ear Segmentation Model (Recommended)\n(dataset3/runs/segment/train7/weights/best.pt)",
        value="segmentation",
        variable=model_choice,
        style="TRadiobutton"
    ).pack(anchor="w", pady=10)
    
    ttk.Radiobutton(
        frame,
        text="Browse for Custom Model...",
        value="custom",
        variable=model_choice,
        style="TRadiobutton"
    ).pack(anchor="w", pady=10)
    
    # Function to handle model selection
    def on_select_model():
        choice = model_choice.get()
        
        # Check if model files exist before closing the dialog
        
        if choice == "default":
            # Use the ear detection model
            file_path = os.path.join("dataset", "runs", "detect", "train2", "weights", "best.pt")
            if not os.path.exists(file_path):
                messagebox.showerror("Model Not Found",
                                    f"The default ear detection model was not found at:\n{file_path}\n\n"
                                    "Please make sure the model file exists or select another option.",
                                    parent=model_select_window)
                return
        elif choice == "segmentation":
            # Use the ear segmentation model
            file_path = os.path.join("dataset3", "runs", "segment", "train7", "weights", "best.pt")
            if not os.path.exists(file_path):
                messagebox.showerror("Model Not Found",
                                    f"The ear segmentation model was not found at:\n{file_path}\n\n"
                                    "Please make sure the model file exists or select another option.",
                                    parent=model_select_window)
                return
        elif choice == "custom":
            # Let the user browse for a custom model
            file_path = filedialog.askopenfilename(
                title="Select YOLO Model File",
                filetypes=[("PyTorch Model Files", "*.pt"), ("All files", "*.*")],
                parent=model_select_window
            )
            if not file_path:
                messagebox.showwarning("Model Load Cancelled", "No YOLO model file was selected.", parent=model_select_window)
                return
            
        # Close the dialog now that we have a valid model path
        model_select_window.destroy()
        
        # Release any existing model before loading a new one
        if utils.model is not None:
            try:
                # Attempt to release resources if possible
                if hasattr(utils.model, 'cpu'):
                    utils.model.cpu()
                utils.model = None
                import gc
                gc.collect()  # Force garbage collection
            except Exception as e:
                print(f"Error releasing previous model: {e}")
        
        # Set the model path and initialize
        utils.custom_yolo_model_path = file_path
        messagebox.showinfo("Model Path Set", f"YOLO model path set to: {file_path}\nAttempting to initialize...",
                            parent=root_ref)
        
        # Initialize the YOLO model - this will set both utils.model and utils.ear_detector_instance
        if utils.initialize_yolo_model():
            messagebox.showinfo("Model Initialized",
                               f"YOLO model and Ear Detector initialized successfully!\n\n"
                               f"Model type: {type(utils.model)}\n"
                               f"Classes: {len(utils.model.names)}\n"
                               f"Class names: {', '.join(utils.model.names.values())}",
                               parent=root_ref)
        else:
            # initialize_yolo_model shows its own error messagebox
            utils.custom_yolo_model_path = None  # Reset path if model init fails
            utils.model = None  # Ensure model object is also None
            utils.ear_detector_instance = None
        
        _check_and_enable_main_buttons(buttons_ref_dict, model_load_button, root_ref)
    
    # Add buttons
    button_frame = ttk.Frame(frame)
    button_frame.pack(pady=20)
    
    # Create a style for buttons
    style.configure("TButton", font=("Arial", 12, "bold"))
    
    ttk.Button(
        button_frame,
        text="Select Model",
        command=on_select_model,
        style="TButton"
    ).pack(side="left", padx=20)
    
    ttk.Button(
        button_frame,
        text="Cancel",
        command=model_select_window.destroy,
        style="TButton"
    ).pack(side="left", padx=20)

# Camera selection functions removed as they are not needed for iVCam

def _attempt_start_new_patient(root_window):
    if not utils.model:
        messagebox.showerror("Model Not Ready",
                             "YOLO model is not initialized. Please load and initialize a model from the home page first.",
                             parent=root_window)
        return
    if not utils.is_calibrated:
        messagebox.showwarning("Not Calibrated",
                               "System is not calibrated. Measurements may not be accurate. Consider calibrating first.",
                               parent=root_window)
        # Optionally, allow proceeding or force calibration:
        # if not messagebox.askyesno("Proceed?", "System not calibrated. Proceed with potentially inaccurate measurements?"):
        #     return
    # Add a new warning for calibration status
    if not utils.is_calibrated:
        messagebox.showwarning("Calibration Recommended",
                               "The system has not been calibrated. Measurement accuracy may be affected. "
                               "It is recommended to calibrate the system first via the Home page.",
                               parent=root_window)
        # Decide if you want to strictly prevent or just warn.
        # For now, it's a warning and allows proceeding.
    create_patient_form(root_window)

# --- Modified create_home_page ---
def create_home_page(root):
    clear_window(root)
    root.title("Ear Measurement System - Home")
    try:
        root.geometry("1920x1080")
    except tk.TclError as e:
        print(f"Note: Could not set fullscreen/geometry for root on home page: {e}")

    container_frame = Frame(root)
    container_frame.pack(expand=True, fill="both")
    setup_background(container_frame, "home_page_bg_container")

    # Ensure the frame is properly updated
    root.update_idletasks()

    # Create a semi-transparent content frame
    content_frame = ui_utils.create_content_frame(container_frame, padding=30, bg_color="#f0f0f0")
    content_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    buttons_dict = {}  # To store references to main action buttons

    # Use modern font for title
    ui_utils.create_styled_label(
        content_frame,
        "Ear Measurement System",
        font=ui_utils.UI_FONTS["title"],
        fg=ui_utils.UI_COLORS["primary"]
    ).pack(pady=(0, 30))

    # Model Load UI - Make it more prominent with modern styling
    model_load_frame = Frame(content_frame, bg="#f0f0f0", relief=tk.FLAT, bd=0)
    model_load_frame.pack(pady=20, padx=20, fill=tk.X)
    
    # Add a header for the model selection section
    ui_utils.create_styled_label(
        model_load_frame,
        "YOLO Model Selection",
        font=ui_utils.UI_FONTS["heading"],
        fg=ui_utils.UI_COLORS["info"]
    ).pack(pady=(10, 5))
    
    # Add descriptive text with more emphasis
    ui_utils.create_styled_label(
        model_load_frame,
        "⚠️ You must select a YOLO model before using the system ⚠️\nThis model is used for ear detection and measurement.",
        font=ui_utils.UI_FONTS["small_bold"],
        fg=ui_utils.UI_COLORS["error"]
    ).pack(pady=(0, 10))
    
    # Create a more prominent button
    model_status_text = "Current Model: " + (os.path.basename(utils.custom_yolo_model_path) if utils.custom_yolo_model_path else "None (Please select a model)")
    ui_utils.create_styled_label(
        model_load_frame,
        model_status_text,
        font=ui_utils.UI_FONTS["small"],
        fg=ui_utils.UI_COLORS["text"]
    ).pack(pady=(0, 10))
    
    # Create a rounded rectangle button for model selection
    model_load_button = ui_utils.create_secondary_button(
        model_load_frame,
        "➡️ Select YOLO Model ⬅️",
        lambda: _load_yolo_model_path_and_init(root, buttons_dict, model_load_button),
        width=300,
        height=60
    )
    model_load_button.pack(pady=10)

    # Camera Information UI with modern styling
    ip_frame = Frame(content_frame, bg="#f0f0f0", relief=tk.FLAT, bd=0)
    ip_frame.pack(pady=20, padx=20, fill=tk.X)
    
    # Add a header for the camera information section
    ui_utils.create_styled_label(
        ip_frame,
        "Camera Information",
        font=ui_utils.UI_FONTS["heading"],
        fg=ui_utils.UI_COLORS["info"]
    ).pack(pady=(10, 5))
    
    # Add message about using iVCam
    ui_utils.create_styled_label(
        ip_frame,
        "Using iVCam (iPhone) via USB.",
        font=ui_utils.UI_FONTS["body"]
    ).pack(side=tk.LEFT, padx=5)

    _show_main_buttons_in_frame(content_frame, root, buttons_dict)
    _check_and_enable_main_buttons(buttons_dict, model_load_button, root)

def _show_main_buttons_in_frame(frame_to_populate, root_window, buttons_dict_param):
    btn_width = 300
    btn_height = 60
    btn_pady = 15

    # Create rounded rectangle buttons with hover effects
    buttons_dict_param['calibrate'] = ui_utils.create_primary_button(
        frame_to_populate,
        "Calibrate System",
        lambda: calibration.create_calibration_page(root_window),
        width=btn_width,
        height=btn_height
    )
    buttons_dict_param['calibrate'].pack(pady=btn_pady)
    buttons_dict_param['calibrate'].config(state=tk.DISABLED)


    
    buttons_dict_param['new_patient'] = ui_utils.create_primary_button(
        frame_to_populate,
        "Start Measurement",
        lambda: _attempt_start_new_patient(root_window),
        width=btn_width,
        height=btn_height
    )
    buttons_dict_param['new_patient'].pack(pady=btn_pady)
    buttons_dict_param['new_patient'].config(state=tk.DISABLED)

    buttons_dict_param['quit'] = ui_utils.create_danger_button(
        frame_to_populate,
        "Quit Application",
        lambda: utils.on_app_close(root_window),
        width=btn_width,
        height=btn_height
    )
    buttons_dict_param['quit'].pack(pady=btn_pady)

def create_patient_form(root):
    # Reset hardware state when starting new patient session
    import image_processing
    try:
        # Stop any active hardware from previous session
        image_processing.stop_hardware(root, on_close=False)
        # Reset capture globals to clean state
        image_processing.capture_globals['sensor_active'] = False
        image_processing.capture_globals['camera_active'] = False
        image_processing.capture_globals['photos_taken_count'] = 0
        print("Hardware state reset for new patient session")
    except Exception as e:
        print(f"Note: Error resetting hardware state: {e}")
    
    # Clear window safely before proceeding
    clear_window(root)
    root.title("Ear Measurement System - New Patient")
    try:
        root.geometry("1920x1080")
    except tk.TclError as e:
        print(f"Note: Could not set geometry for patient form: {e}")

    if not utils.model:  # Safety check
        messagebox.showerror("Model Not Ready", "YOLO model is not available. Cannot proceed.", parent=root)
        go_home(root)
        return

    container_frame = Frame(root)
    container_frame.pack(expand=True, fill="both")
    setup_background(container_frame, "patient_form_bg_container")
    
    # Create a semi-transparent content frame with modern styling
    content_frame = ui_utils.create_content_frame(container_frame, padding=30, bg_color="#f0f0f0")
    content_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    content_bg = content_frame.cget('bg')
    
    # Add a styled title
    ui_utils.create_styled_label(
        content_frame,
        "Patient Details",
        font=ui_utils.UI_FONTS["subtitle"],
        fg=ui_utils.UI_COLORS["primary"]
    ).grid(row=0, column=0, columnspan=2, pady=20)
    
    form_fields = [
        ("Name:", tk.Entry, "Name"),
        ("Address:", tk.Entry, "Address"),
        ("Phone (10 digits):", tk.Entry, "Phone"),
        ("Gender:", ttk.Combobox, "Gender"),
        ("Birthday:", DateEntry, "DOB")
    ]
    
    patient_form_entries = {}
    for i, (label_text, widget_type, key) in enumerate(form_fields):
        # Create styled labels for form fields
        ui_utils.create_styled_label(
            content_frame,
            label_text,
            font=ui_utils.UI_FONTS["body"],
            fg=ui_utils.UI_COLORS["text"]
        ).grid(row=i + 1, column=0, sticky="w", padx=10, pady=10)
        
        if widget_type == tk.Entry:
            # Create styled entry fields
            entry = ui_utils.create_styled_entry(content_frame, width=40, font=ui_utils.UI_FONTS["body"])
        elif widget_type == ttk.Combobox:  # ttk.Combobox for gender
            # Create styled combobox
            entry = ui_utils.create_styled_combobox(
                content_frame,
                values=["Male", "Female", "Other"],
                state="readonly",
                width=38,
                font=ui_utils.UI_FONTS["body"]
            )
        elif widget_type == DateEntry:  # DateEntry for birthday
            # Style the DateEntry
            entry = widget_type(
                content_frame,
                width=38,
                font=ui_utils.UI_FONTS["body"],
                date_pattern='yyyy-mm-dd',
                background=ui_utils.UI_COLORS["primary"],
                foreground='white',
                borderwidth=2,
                year=2000
            )
            
        entry.grid(row=i + 1, column=1, sticky="ew", padx=10, pady=10)
        patient_form_entries[key] = entry

    button_frame = Frame(content_frame, bg="#f0f0f0")
    button_frame.grid(row=len(form_fields) + 1, column=0, columnspan=2, pady=30)

    def submit_patient_info_wrapper():
        phone = patient_form_entries["Phone"].get()
        if not (phone.isdigit() and len(phone) == 10):
            messagebox.showerror("Validation Error", "Phone number must be 10 digits.", parent=root)
            return

        utils.patient_info.clear()
        for key, widget in patient_form_entries.items():
            if key == "DOB" and isinstance(widget, DateEntry):
                # Get the date in YYYY-MM-DD format from DateEntry
                utils.patient_info[key] = widget.get_date().strftime('%Y-%m-%d')
            else:
                utils.patient_info[key] = widget.get()

        # Proceed to image capture
        image_processing.capture_images_entry_point(root, utils.patient_info)

    def clear_patient_form_entries_wrapper():
        for key, widget in patient_form_entries.items():
            if isinstance(widget, tk.Entry):
                widget.delete(0, tk.END)
            elif isinstance(widget, ttk.Combobox):
                widget.set('')  # Clear selection
            elif isinstance(widget, DateEntry):
                # Reset to default date
                widget.set_date(widget._date)

    # Create styled buttons with rounded corners
    ui_utils.create_primary_button(
        button_frame,
        "Take Measurement",
        submit_patient_info_wrapper,
        width=240,
        height=50
    ).pack(side=tk.LEFT, padx=10)
    
    ui_utils.create_secondary_button(
        button_frame,
        "Clear Form",
        clear_patient_form_entries_wrapper,
        width=180,
        height=50
    ).pack(side=tk.LEFT, padx=10)
    
    ui_utils.create_secondary_button(
        button_frame,
        "Homepage",
        lambda: go_home(root),
        width=180,
        height=50
    ).pack(side=tk.LEFT, padx=10)

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
    
    # Create a semi-transparent content frame with modern styling
    content_frame = ui_utils.create_content_frame(container_frame, padding=30, bg_color="#f0f0f0")
    content_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    content_bg = content_frame.cget('bg')

    # Add a styled title
    ui_utils.create_styled_label(
        content_frame,
        "Measurement Results",
        font=ui_utils.UI_FONTS["subtitle"],
        fg=ui_utils.UI_COLORS["primary"]
    ).pack(pady=20)
    
    # Add measurement results with modern styling
    # Format values properly based on type
    height_display = f"{avg_ear_height:.2f} mm" if isinstance(avg_ear_height, (int, float)) else str(avg_ear_height)
    width_display = f"{avg_ear_width:.2f} mm" if isinstance(avg_ear_width, (int, float)) else str(avg_ear_width)
    
    ui_utils.create_styled_label(
        content_frame,
        f"Average Ear Height (Length): {height_display}",
        font=ui_utils.UI_FONTS["body_bold"],
        fg=ui_utils.UI_COLORS["text"]
    ).pack(pady=5)
    
    ui_utils.create_styled_label(
        content_frame,
        f"Average Ear Width: {width_display}",
        font=ui_utils.UI_FONTS["body_bold"],
        fg=ui_utils.UI_COLORS["text"]
    ).pack(pady=10)

    # Add representative image section
    ui_utils.create_styled_label(
        content_frame,
        "Representative Image:",
        font=ui_utils.UI_FONTS["body"],
        fg=ui_utils.UI_COLORS["text"]
    ).pack(pady=(10, 5))
    
    # Create a frame with subtle shadow for the image
    image_frame = Frame(content_frame, bg="#f0f0f0", bd=0, relief=tk.FLAT)
    image_frame.pack(pady=10, padx=10)
    
    image_panel_results = Label(image_frame, bg="#f0f0f0")
    image_panel_results.pack(pady=10, padx=10)

    if closest_image_info and closest_image_info.get('image_path') and os.path.exists(closest_image_info['image_path']):
        try:
            img_pil = Image.open(closest_image_info['image_path'])
            display_w, display_h = 480, 360  # Define desired display box size
            img_w_orig, img_h_orig = img_pil.width, img_pil.height

            aspect_ratio_orig = img_h_orig / float(img_w_orig) if img_w_orig > 0 else 1.0

            current_w, current_h = display_w, int(display_w * aspect_ratio_orig)
            if current_h > display_h:
                current_h = display_h
                current_w = int(display_h / aspect_ratio_orig) if aspect_ratio_orig > 0 else display_w

            img_pil_resized = img_pil.resize((current_w, current_h), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_pil_resized)
            image_panel_results.config(image=img_tk)
            image_panel_results.image = img_tk  # Keep reference

            # Display specific measurements for this representative image
            rep_h = closest_image_info.get('actual_height_mm', 0.0)
            rep_w = closest_image_info.get('actual_width_mm', 0.0)
            rep_d = closest_image_info.get('sensor_distance_mm', 'N/A')
            
            # Format height and width - handle both numeric and string values
            if isinstance(rep_h, (int, float)):
                rep_h_str = f"{rep_h:.2f}mm"
            else:
                rep_h_str = str(rep_h)
                
            if isinstance(rep_w, (int, float)):
                rep_w_str = f"{rep_w:.2f}mm"
            else:
                rep_w_str = str(rep_w)
                
            if isinstance(rep_d, float): 
                rep_d_str = f"{rep_d:.1f}mm"
            else:
                rep_d_str = str(rep_d)

            ui_utils.create_styled_label(
                content_frame,
                f"Details: H: {rep_h_str}, W: {rep_w_str} @ {rep_d_str}",
                font=ui_utils.UI_FONTS["small_bold"],
                fg=ui_utils.UI_COLORS["info"]
            ).pack()
        except Exception as e:
            image_panel_results.config(
                text=f"Error displaying image: {e}",
                font=ui_utils.UI_FONTS["small"],
                fg=ui_utils.UI_COLORS["error"],
                bg=content_bg
            )
    else:
        image_panel_results.config(
            text="No representative image available or path is invalid.",
            font=ui_utils.UI_FONTS["small"],
            fg=ui_utils.UI_COLORS["warning"],
            bg=content_bg
        )

    # Add a label explaining the save functionality
    ui_utils.create_styled_label(
        content_frame,
        "Click 'Save to Excel & PDF' to export measurement data and generate a detailed report",
        font=ui_utils.UI_FONTS["small"],
        fg=ui_utils.UI_COLORS["text"]
    ).pack(pady=(20, 10))

    button_frame_results = Frame(content_frame, bg="#f0f0f0")
    button_frame_results.pack(pady=20)

    # Save Data button now calls a wrapper that includes PDF saving dialog
    def save_all_data_wrapper():
        # This function in data_handling now manages both Excel and PDF.
        # It will internally ask for a base filename.
        success = data_handling.save_measurements_and_generate_report()
        if success:
            # Highlight that both Excel and PDF were saved
            messagebox.showinfo("Save Complete",
                               "Measurement data has been successfully saved to both Excel and PDF formats.")

    # Create a more prominent save button with rounded corners
    ui_utils.create_primary_button(
        button_frame_results,
        "Save to Excel & PDF",
        save_all_data_wrapper,
        width=240,
        height=50
    ).pack(side=tk.LEFT, padx=15)
    
    # Create styled buttons for other actions
    ui_utils.create_secondary_button(
        button_frame_results,
        "New Patient",
        lambda: create_patient_form(root),
        width=180,
        height=50
    ).pack(side=tk.LEFT, padx=15)

    def go_back_to_capture_current_patient_wrapper():
        # This needs to re-open the image capture screen for the *current* patient
        # utils.patient_info should still hold the current patient's data.
        image_processing.capture_images_entry_point(root, patient_info_dict)  # Pass the current patient_info

    ui_utils.create_secondary_button(
        button_frame_results,
        "Recapture",
        go_back_to_capture_current_patient_wrapper,
        width=180,
        height=50
    ).pack(side=tk.LEFT, padx=15)

    ui_utils.create_danger_button(
        button_frame_results,
        "Quit",
        lambda: utils.on_app_close(root),
        width=180,
        height=50
    ).pack(side=tk.LEFT, padx=15)

# Remove the problematic import of sensor_function
# Ensure initialize_yolo_model is correctly handled (e.g., moved to utils.py and called from there)
# If sensor_function was meant to be ultrasonic_sensor, update imports accordingly if needed.
# The current gui_pages.py doesn't use sensor_function directly.
# The `utils.initialize_yolo_model()` call needs that function to exist in `utils.py`.

# End of gui_pages.py


