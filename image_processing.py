# image_processing.py

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, simpledialog, Toplevel, Label, Button
import os
import tempfile
import time
import utils
import traceback
import sensor_function

# Adjustable YOLO confidence threshold
YOLO_CONFIDENCE_THRESHOLD = 0.5

def detect_ear_yolo(image_np):
    if image_np is None or image_np.size == 0:
        print("detect_ear_yolo: Received empty image.")
        return None
    print(f"detect_ear_yolo: Detecting ear in image with shape: {image_np.shape}")
    # cv2.imwrite("debug_yolo_input.jpg", image_np)
    if utils.model is None:
        messagebox.showerror("YOLO Model Error", "YOLO Model not loaded.")
        return None
    try:
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results_yolo = utils.model(image_rgb, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)
        if not results_yolo or not results_yolo[0].boxes:
            model_name_for_log = utils.model.ckpt_path if hasattr(utils.model, 'ckpt_path') and utils.model.ckpt_path else 'N/A'
            if model_name_for_log == 'N/A' and hasattr(utils.model, 'model') and hasattr(utils.model.model, 'yaml_file'):
                 model_name_for_log = utils.model.model.yaml_file
            print(f"detect_ear_yolo: No detections. Confidence: {YOLO_CONFIDENCE_THRESHOLD}, Model: {model_name_for_log}")
            return None
        boxes_xywh = results_yolo[0].boxes.xywh.cpu().numpy()
        confidences = results_yolo[0].boxes.conf.cpu().numpy()
        if len(boxes_xywh) > 0:
            best_box_index = 0
            x_center, y_center, w, h = boxes_xywh[best_box_index]
            x = int(x_center - w / 2); y = int(y_center - h / 2)
            w_int = int(w); h_int = int(h)
            if w_int <= 0 or h_int <= 0:
                print(f"detect_ear_yolo: Bad box dim ({w_int}x{h_int}). Skip."); return None
            return (x, y, w_int, h_int)
        else:
            model_name_for_log = utils.model.ckpt_path if hasattr(utils.model, 'ckpt_path') and utils.model.ckpt_path else 'N/A'
            if model_name_for_log == 'N/A' and hasattr(utils.model, 'model') and hasattr(utils.model.model, 'yaml_file'):
                 model_name_for_log = utils.model.model.yaml_file
            print(f"detect_ear_yolo: No ear (0 boxes). Confidence: {YOLO_CONFIDENCE_THRESHOLD}, Model: {model_name_for_log}")
            return None
    except Exception as e:
        messagebox.showerror("Detection Error", f"YOLO detection failure: {e}")
        print(f"detect_ear_yolo: YOLO error: {e}"); print(traceback.format_exc())
        return None

def get_ear_measurements_from_roi(image_roi_gray):
    if image_roi_gray is None or image_roi_gray.size == 0: return None
    blurred_roi = cv2.GaussianBlur(image_roi_gray, (5, 5), 0)
    edges = cv2.Canny(blurred_roi, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: print("No contours in ROI."); return None
    largest_contour = max(contours, key=cv2.contourArea)
    _, _, w_contour, h_contour = cv2.boundingRect(largest_contour)
    return int(w_contour), int(h_contour)

def process_single_image_for_measurement(image_bgr, distance_mm):
    if image_bgr is None or image_bgr.size == 0:
        messagebox.showerror("Image Error", "Cannot process empty image."); return None
    roi_yolo = detect_ear_yolo(image_bgr)
    annotated_image = image_bgr.copy()
    final_roi_coords = None
    ear_region_bgr = None
    if roi_yolo:
        x, y, w, h = roi_yolo
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(annotated_image, "YOLO Ear", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        final_roi_coords = roi_yolo
        ear_region_bgr = image_bgr[y : y + h, x : x + w]
    else:
        messagebox.showinfo("Manual Selection", "YOLO ear detection failed. Select region manually.")
        try:
            manual_roi = cv2.selectROI("Select Ear Region Manually", image_bgr, False, False)
            cv2.destroyWindow("Select Ear Region Manually")
            if manual_roi == (0,0,0,0) or not manual_roi: messagebox.showwarning("Skipped", "ROI selection skipped."); return None
            x,y,w,h = [int(c) for c in manual_roi]
            if w==0 or h==0: messagebox.showwarning("Invalid ROI", "ROI zero W/H."); return None
            cv2.rectangle(annotated_image, (x,y), (x+w,y+h), (0,255,255),2); cv2.putText(annotated_image, "Manual Ear", (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
            final_roi_coords=(x,y,w,h); ear_region_bgr=image_bgr[y:y+h,x:x+w]
        except Exception as e: messagebox.showerror("ROI Error", f"Manual ROI error: {e}"); return None
    if final_roi_coords is None or ear_region_bgr is None or ear_region_bgr.size == 0:
        messagebox.showerror("ROI Error", "No valid ear region."); return None
    ear_region_gray = cv2.cvtColor(ear_region_bgr, cv2.COLOR_BGR2GRAY)
    pixel_measurements = get_ear_measurements_from_roi(ear_region_gray)
    pixel_width, pixel_height = (final_roi_coords[2], final_roi_coords[3]) if not pixel_measurements else pixel_measurements
    if not pixel_measurements: messagebox.showwarning("Measurement Error", "No contours for pixel dims.")

    scale_current_px_to_mm = 0.5
    if utils.scale_model is None: messagebox.showwarning("Calibration Missing", "Using default scale 0.5.")
    else:
        m,b=utils.scale_model
        if distance_mm <= 0: messagebox.showwarning("Distance Error", f"Invalid dist ({distance_mm}mm). Using default scale.")
        else: scale_val = m*distance_mm+b; scale_current_px_to_mm = scale_val if scale_val > 0 else 0.5
        if scale_current_px_to_mm == 0.5 and distance_mm > 0 : messagebox.showwarning("Calibration Error", "Bad scale from calib. Using default.")

    ear_height_mm = round(pixel_width * scale_current_px_to_mm, 2)
    ear_width_mm = round(pixel_height * scale_current_px_to_mm, 2)
    roi_x, roi_y, _, _ = final_roi_coords
    txt = f"H: {ear_height_mm}mm, W: {ear_width_mm}mm @ {distance_mm:.1f}mm"
    cv2.putText(annotated_image, txt, (roi_x, roi_y+final_roi_coords[3]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
    try:
        fname=f"ear_proc_{len(utils.results)}_{int(time.time())}.jpg"; path=os.path.join(tempfile.gettempdir(),fname)
        Image.fromarray(cv2.cvtColor(annotated_image,cv2.COLOR_BGR2RGB)).save(path)
    except Exception as e: messagebox.showerror("File Save Error",f"Could not save temp image: {e}"); return None
    return ear_height_mm, ear_width_mm, path

capture_globals = {
    "cap": None, "ip_camera_url": None, "capture_window": None, "image_panel": None,
    "captured_images_with_data": [], "current_root": None, "patient_info_dict_ref": None,
    "max_captures": 5, "status_label": None, "distance_label": None,
    "current_frame_for_capture": None,
    "consecutive_frame_read_failures": 0,      # Added
    "max_consecutive_failures": 5,             # Added
    "feed_retry_delay_ms": 5000,               # Added
    "original_feed_delay_ms": 100              # Added (normal update rate)
}

def setup_capture_gui(root, patient_info_dict_param):
    from gui_pages import clear_window, create_patient_form
    clear_window(root); capture_globals.update({
        "current_root":root, "patient_info_dict_ref":patient_info_dict_param,
        "captured_images_with_data":[], "capture_window":root,
        "consecutive_frame_read_failures": 0 # Reset on new setup
        })
    main_frame=tk.Frame(root); main_frame.pack(pady=20,padx=20,fill="both",expand=True)
    capture_globals["image_panel"]=Label(main_frame); capture_globals["image_panel"].pack(pady=10)
    info_f=tk.Frame(main_frame); info_f.pack(pady=5)
    capture_globals["status_label"]=Label(info_f,text=f"Captured: 0/{capture_globals['max_captures']}",font=("Arial",14)); capture_globals["status_label"].pack(side=tk.LEFT,padx=10)
    capture_globals["distance_label"]=Label(info_f,text="Distance: --- mm",font=("Arial",14)); capture_globals["distance_label"].pack(side=tk.LEFT,padx=10)
    btn_f=tk.Frame(main_frame); btn_f.pack(pady=10)
    Button(btn_f,text="Capture",command=trigger_capture_image,width=15,height=2).pack(side=tk.LEFT,padx=5)
    Button(btn_f,text="Result",command=lambda:process_and_show_results(False),width=15,height=2).pack(side=tk.LEFT,padx=5)
    Button(btn_f,text="Prev Page",command=lambda:go_to_previous_page(create_patient_form),width=15,height=2).pack(side=tk.LEFT,padx=5)
    cap_img_outer_f=tk.Frame(main_frame); cap_img_outer_f.pack(pady=10,fill=tk.X)
    tk.Label(cap_img_outer_f,text="Captured Images:",font=("Arial",12)).pack()
    cv_scroll=tk.Canvas(cap_img_outer_f,height=150); scrollbar=tk.Scrollbar(cap_img_outer_f,orient="horizontal",command=cv_scroll.xview)
    capture_globals["captured_images_display_frame"]=tk.Frame(cv_scroll); cv_scroll.configure(xscrollcommand=scrollbar.set)
    cv_scroll.pack(side=tk.TOP,fill=tk.X,expand=True); scrollbar.pack(side=tk.BOTTOM,fill=tk.X)
    cv_scroll.create_window((0,0),window=capture_globals["captured_images_display_frame"],anchor="nw")
    capture_globals["captured_images_display_frame"].bind("<Configure>",lambda e:cv_scroll.configure(scrollregion=cv_scroll.bbox("all")))
    if not utils.droidcam_ip: messagebox.showerror("IP Error","DroidCam IP not set."); go_to_previous_page(create_patient_form); return
    ip_url=f"http://{utils.droidcam_ip}:4747/video"; capture_globals["ip_camera_url"]=ip_url
    capture_globals["cap"]=cv2.VideoCapture(ip_url)
    if not capture_globals["cap"].isOpened():
        messagebox.showerror("Cam Error",f"Failed to open DroidCam: {ip_url}.")
        if capture_globals["cap"]:capture_globals["cap"].release();capture_globals["cap"]=None
        go_to_previous_page(create_patient_form); return
    _update_video_feed()

def _update_video_feed():
    cap = capture_globals.get("cap")
    if cap and cap.isOpened():
        ret, frame_bgr = cap.read()
        if ret and frame_bgr is not None:
            capture_globals["consecutive_frame_read_failures"] = 0 # Reset counter
            frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)

            dist = sensor_function.get_distance()
            dist_text = f"Distance: {dist:.1f} mm" if dist is not None and dist > 0 else "Distance: --- mm"
            if capture_globals.get("distance_label"):
                capture_globals["distance_label"].config(text=dist_text)
            capture_globals["current_frame_for_capture"] = (frame_bgr.copy(), dist if dist is not None and dist > 0 else -1)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            target_w = 640
            w_orig, h_orig = img_pil.width, img_pil.height
            aspect_ratio = h_orig/w_orig if w_orig > 0 else 1.0
            target_h = int(target_w * aspect_ratio)
            if target_h == 0: target_h = int(target_w * (480/640.0)) # Fallback for bad aspect ratio

            img_pil_resized = img_pil.resize((target_w, target_h if target_h > 0 else 480), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(image=img_pil_resized)

            if capture_globals.get("image_panel"):
                capture_globals["image_panel"].configure(image=img_tk); capture_globals["image_panel"].image = img_tk

            if capture_globals.get("capture_window"): # Check if window still active
                 capture_globals["image_panel"].after(capture_globals["original_feed_delay_ms"], _update_video_feed)
        else:
            capture_globals["consecutive_frame_read_failures"] += 1
            print(f"Failed to read frame: patient image capture (Attempt: {capture_globals['consecutive_frame_read_failures']}).")

            next_retry_delay = 200 # Default shorter retry
            if capture_globals["consecutive_frame_read_failures"] >= capture_globals["max_consecutive_failures"]:
                if capture_globals.get("distance_label"):
                    capture_globals["distance_label"].config(text="Camera feed lost. Retrying...")
                # if capture_globals.get("image_panel"): capture_globals["image_panel"].image = None
                print(f"Max frame read failures. Waiting {capture_globals['feed_retry_delay_ms']}ms.")
                next_retry_delay = capture_globals['feed_retry_delay_ms']

            if capture_globals.get("capture_window") and capture_globals.get("image_panel"):
                 capture_globals["image_panel"].after(next_retry_delay, _update_video_feed)
            else:
                 print("Capture window or image panel no longer exists. Stopping feed update attempts.")

    elif capture_globals.get("capture_window"):
        print("Patient camera is not open. Check connection.")

def trigger_capture_image():
    if len(capture_globals["captured_images_with_data"]) >= capture_globals["max_captures"]:
        messagebox.showinfo("Limit Reached", f"Max {capture_globals['max_captures']} images."); return
    img_np_bgr, dist = None, -1
    cap, ip_url = capture_globals.get("cap"), capture_globals.get("ip_camera_url")
    if cap is None or not cap.isOpened():
        print("trigger_capture_image: Cam re-opening...");
        if not ip_url: messagebox.showerror("Cam Error", "Cam IP URL missing."); return
        cap=cv2.VideoCapture(ip_url); capture_globals["cap"]=cap
        if not cap.isOpened(): messagebox.showerror("Cam Error", f"Failed to re-open cam: {ip_url}."); return
        print(f"Cam re-opened: {ip_url}"); time.sleep(0.5)
    if cap and cap.isOpened():
        print("trigger_capture_image: Fresh read."); ret, fresh_f = cap.read()
        if ret and fresh_f is not None:
            fresh_f = cv2.rotate(fresh_f, cv2.ROTATE_90_CLOCKWISE)
            img_np_bgr = fresh_f.copy()
            cur_d = sensor_function.get_distance()
            dist_txt = "Dist: --- mm (capture)"
            if cur_d is not None and cur_d > 0: dist=cur_d; dist_txt=f"Dist: {dist:.1f} mm"
            else: dist = -1
            if capture_globals.get("distance_label"): capture_globals["distance_label"].config(text=dist_txt)
            print(f"Fresh frame, dist: {dist}")
        else:
            print("Fresh read failed, using cached.");
            if capture_globals.get("current_frame_for_capture") is not None: img_np_bgr,dist=capture_globals["current_frame_for_capture"]
    else:
         print("Cam not open after attempt, using cached.");
         if capture_globals.get("current_frame_for_capture") is not None: img_np_bgr,dist=capture_globals["current_frame_for_capture"]
    if img_np_bgr is None or img_np_bgr.size==0: messagebox.showwarning("Cap Error","No valid frame. Ensure cam active."); return
    if dist == -1 and not messagebox.askyesno("Dist Error","Sensor fail. Capture anyway?"): return
    capture_globals["captured_images_with_data"].append((img_np_bgr.copy(),dist))
    _update_captured_images_display()
    status_txt=f"Captured: {len(capture_globals['captured_images_with_data'])}/{capture_globals['max_captures']}"
    capture_globals["status_label"].config(text=status_txt)
    if len(capture_globals["captured_images_with_data"])==capture_globals["max_captures"]: messagebox.showinfo("Done","All images captured. Click 'Result'.")

def _update_captured_images_display():
    for w in capture_globals["captured_images_display_frame"].winfo_children(): w.destroy()
    for i,(img_data,dist) in enumerate(capture_globals["captured_images_with_data"]):
        img_rgb=cv2.cvtColor(img_data,cv2.COLOR_BGR2RGB)
        pil_img=Image.fromarray(img_rgb).resize((120,90),Image.Resampling.LANCZOS); tk_img=ImageTk.PhotoImage(pil_img)
        f=tk.Frame(capture_globals["captured_images_display_frame"],bd=1,relief="solid"); f.pack(side=tk.LEFT,padx=5,pady=5)
        lbl=Label(f,image=tk_img); lbl.image=tk_img; lbl.pack()
        dist_text = f"{dist:.1f}mm" if dist > 0 else "Dist. N/A"
        Label(f,text=f"Img {i+1} ({dist_text})").pack()
        Button(f,text="Recapture",command=lambda idx=i:_recapture_specific_image(idx)).pack()

def _recapture_specific_image(index_to_recapture):
    if not (0<=index_to_recapture<len(capture_globals["captured_images_with_data"])): messagebox.showerror("Recap Error","Invalid index."); return
    img_np_bgr,dist = None,-1
    cap,ip_url = capture_globals.get("cap"),capture_globals.get("ip_camera_url")
    if cap is None or not cap.isOpened():
        print("_recapture: Cam re-opening...");
        if not ip_url: messagebox.showerror("Cam Error","Cam IP URL missing."); return
        cap=cv2.VideoCapture(ip_url); capture_globals["cap"]=cap
        if not cap.isOpened(): messagebox.showerror("Cam Error",f"Failed to re-open: {ip_url}."); return
        print(f"Cam re-opened: {ip_url}"); time.sleep(0.5)
    if cap and cap.isOpened():
        print("_recapture: Fresh read."); ret,fresh_f = cap.read()
        if ret and fresh_f is not None:
            fresh_f = cv2.rotate(fresh_f, cv2.ROTATE_90_CLOCKWISE)
            img_np_bgr = fresh_f.copy()
            cur_d = sensor_function.get_distance()
            dist_txt = "Dist: --- mm (recapture)"
            if cur_d is not None and cur_d > 0: dist=cur_d; dist_txt=f"Dist: {dist:.1f} mm"
            else: dist = -1
            if capture_globals.get("distance_label"): capture_globals["distance_label"].config(text=dist_txt)
            print(f"Fresh frame, dist: {dist}")
        else:
            print("Fresh read failed, using cached.");
            if capture_globals.get("current_frame_for_capture") is not None: img_np_bgr,dist=capture_globals["current_frame_for_capture"]
    else:
        print("Cam not open after attempt, using cached.");
        if capture_globals.get("current_frame_for_capture") is not None: img_np_bgr,dist=capture_globals["current_frame_for_capture"]
    if img_np_bgr is None or img_np_bgr.size==0: messagebox.showwarning("Recap Error","No valid frame. Ensure cam active."); return
    if dist == -1 and not messagebox.askyesno("Dist Error","Sensor fail. Recapture anyway?"): return
    capture_globals["captured_images_with_data"][index_to_recapture]=(img_np_bgr.copy(),dist)
    _update_captured_images_display(); messagebox.showinfo("Recaptured",f"Image {index_to_recapture+1} recaptured.")

def process_and_show_results(is_retry=False):
    from gui_pages import show_results_page
    if not capture_globals["captured_images_with_data"]: messagebox.showwarning("No Images","No images captured."); return
    if len(capture_globals["captured_images_with_data"])<capture_globals["max_captures"] and not is_retry:
        if not messagebox.askyesno("Confirm","Not all images captured. Proceed?"): return
    if capture_globals.get("cap"): capture_globals["cap"].release();capture_globals["cap"]=None
    utils.results.clear(); all_h,all_w=[],[]
    for i,(img_bgr,dist_mm) in enumerate(capture_globals["captured_images_with_data"]):
        if dist_mm<=0 and utils.scale_model: messagebox.showwarning("Proc Warn",f"Img {i+1} invalid dist ({dist_mm}mm).")
        res_t = process_single_image_for_measurement(img_bgr,dist_mm)
        if res_t:
            h,w,p = res_t
            utils.results.append({'ear_height_mm':h,'ear_width_mm':w,'image_path':p,'original_image':img_bgr,'distance':dist_mm})
            all_h.append(h);all_w.append(w)
        else:
            messagebox.showwarning("Proc Fail",f"Could not process image {i+1}.")
            utils.results.append({'ear_height_mm':0,'ear_width_mm':0,'image_path':None,'original_image':img_bgr,'distance':dist_mm,'error':'Processing failed'})
    if not all_h or not all_w:
        messagebox.showerror("Proc Error","No images successfully processed."); setup_capture_gui(capture_globals["current_root"],capture_globals["patient_info_dict_ref"]); return
    avg_h=round(np.mean(all_h),2) if all_h else 0; avg_w=round(np.mean(all_w),2) if all_w else 0
    closest_img,min_d_avg=None,float('inf')
    if avg_h>0 and avg_w>0:
        for r_dict in utils.results:
            if r_dict.get('error'):continue
            diff=abs(r_dict['ear_height_mm']-avg_h)+abs(r_dict['ear_width_mm']-avg_w)
            if diff<min_d_avg:min_d_avg=diff;closest_img=r_dict
    if not closest_img and utils.results:
        for r_dict in utils.results:
            if not r_dict.get('error'):closest_img=r_dict;break
    show_results_page(capture_globals["current_root"],avg_h,avg_w,closest_img,capture_globals["patient_info_dict_ref"])

def go_to_previous_page(patient_form_creation_func):
    if capture_globals.get("cap"):capture_globals["cap"].release();capture_globals["cap"]=None
    patient_form_creation_func(capture_globals["current_root"])

def capture_images_entry_point(root, patient_info_dict):
    if utils.model is None: messagebox.showerror("Model Error","YOLO model not loaded."); return
    setup_capture_gui(root,patient_info_dict)
