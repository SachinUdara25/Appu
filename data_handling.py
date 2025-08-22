#Data_handling.py
# data_handling.py
import pandas as pd
from tkinter import filedialog, messagebox
import utils  # To access utils.patient_info, utils.results
import os
import tempfile
from PIL import Image
import tkinter as tk

# Ensure ReportLab is available, provide guidance if not.
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    print("Warning: ReportLab not available. PDF generation will be disabled.")
    REPORTLAB_AVAILABLE = False

def save_measurements_and_generate_report():
    """
    Saves measurement results to Excel and generates a PDF report.
    Returns True if both are successful, False otherwise.
    """
    if not utils.patient_info or not utils.measurement_results:
        messagebox.showwarning("No Data", "No patient info or measurements to save.")
        return False

    # Prepare file paths
    base_dir = filedialog.askdirectory(title="Select Folder to Save Files")
    if not base_dir:
        return False
    
    # Create a sanitized patient name for filenames
    patient_name = utils.patient_info.get('Name', 'Unknown_Patient')
    # Replace invalid filename characters
    sanitized_name = "".join([c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in patient_name])
    
    # Add timestamp to ensure unique filenames
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use a consistent Excel file name for all patients (ensure no spaces in filename)
    excel_file_name = "patient_measurements.xlsx"
    
    # Create a subdirectory for data files if it doesn't exist
    data_dir = os.path.join(base_dir, "ear_measurement_data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Use normalized paths with consistent slashes
    excel_file_path = os.path.normpath(os.path.join(data_dir, excel_file_name))
    pdf_file_path = os.path.normpath(os.path.join(data_dir, f"{sanitized_name}_report_{timestamp}.pdf"))
    
    # Show the paths that will be used
    messagebox.showinfo("Save Locations",
                       f"Files will be saved as:\n\nExcel: {os.path.basename(excel_file_path)}\n\nPDF: {os.path.basename(pdf_file_path)}")

    # Convert results to DataFrame
    valid_results = [res for res in utils.measurement_results if res.get('error') is None]
    if not valid_results:
        messagebox.showwarning("No Valid Data", "No valid measurement results to save.")
        return False

    # Calculate average height and width
    valid_heights = [res['actual_height_mm'] for res in valid_results if isinstance(res.get('actual_height_mm'), (int, float)) and res['actual_height_mm'] > 0]
    valid_widths = [res['actual_width_mm'] for res in valid_results if isinstance(res.get('actual_width_mm'), (int, float)) and res['actual_width_mm'] > 0]
    avg_ear_height = round(pd.Series(valid_heights).mean(), 2) if valid_heights else 0.0
    avg_ear_width = round(pd.Series(valid_widths).mean(), 2) if valid_widths else 0.0

    # Prepare data for Excel
    excel_data_dict = {
        # Patient Information
        "Patient Name": utils.patient_info.get("Name", "N/A"),
        "Date of Birth": utils.patient_info.get("DOB", "N/A"),
        "Gender": utils.patient_info.get("Gender", "N/A"),
        "Phone": utils.patient_info.get("Phone", "N/A"),
        "Address": utils.patient_info.get("Address", "N/A"),
        
        # Measurement Results
        "Average Ear Height (mm)": avg_ear_height,
        "Average Ear Width (mm)": avg_ear_width,
        
        # Measurement Date/Time
        "Measurement Date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "Measurement Time": pd.Timestamp.now().strftime("%H:%M:%S"),
    }

    # Add individual measurements for all images
    for i in range(len(utils.measurement_results)):
        result = utils.measurement_results[i]
        # Only add valid measurements (no errors)
        if result.get('error') is None:
            excel_data_dict[f"Ear {i+1} Height (mm)"] = result.get('actual_height_mm', 0.0)
            excel_data_dict[f"Ear {i+1} Width (mm)"] = result.get('actual_width_mm', 0.0)
            excel_data_dict[f"Ear {i+1} Distance (mm)"] = result.get('sensor_distance_mm', 0.0)
        else:
            excel_data_dict[f"Ear {i+1} Height (mm)"] = 0.0
            excel_data_dict[f"Ear {i+1} Width (mm)"] = 0.0
            excel_data_dict[f"Ear {i+1} Distance (mm)"] = 0.0

    # Check if an existing Excel file exists to append to
    excel_exists = os.path.exists(excel_file_path)
    
    try:
        # If file exists, read it and append new data
        if excel_exists:
            try:
                existing_df = pd.read_excel(excel_file_path)
                new_df = pd.DataFrame([excel_data_dict])
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.to_excel(excel_file_path, index=False)
            except PermissionError as pe:
                # If file is open in Excel or otherwise locked, save with a different name
                alt_excel_path = os.path.normpath(os.path.join(data_dir, f"patient_measurements_{timestamp}.xlsx"))
                messagebox.showwarning("File Access Issue",
                                      f"The file {excel_file_name} is currently open or locked. Saving to {os.path.basename(alt_excel_path)} instead.")
                df = pd.DataFrame([excel_data_dict])
                df.to_excel(alt_excel_path, index=False)
                excel_file_path = alt_excel_path  # Update the path for later reference
        else:
            # Create new Excel file
            df = pd.DataFrame([excel_data_dict])
            df.to_excel(excel_file_path, index=False)
        
        excel_saved_successfully = True
    except Exception as e:
        messagebox.showerror("Excel Save Failed", f"Could not save Excel file: {e}")
        excel_saved_successfully = False

    # Generate PDF
    pdf_generated_successfully = True  # Default to True for when PDF generation is skipped
    if REPORTLAB_AVAILABLE:
        try:
            doc = SimpleDocTemplate(pdf_file_path)
            styles = getSampleStyleSheet()
            story = []

            # Title
            story.append(Paragraph("<b>Ear Measurement Report</b>", styles['Title']))
            story.append(Spacer(1, 0.25 * inch))

            # Patient Information
            story.append(Paragraph("<b>Patient Information:</b>", styles['h2']))
            story.append(Paragraph(f"<b>Name:</b> {utils.patient_info.get('Name', 'N/A')}", styles['Normal']))
            story.append(Paragraph(f"<b>Date of Birth:</b> {utils.patient_info.get('DOB', 'N/A')}", styles['Normal']))
            story.append(Paragraph(f"<b>Gender:</b> {utils.patient_info.get('Gender', 'N/A')}", styles['Normal']))
            story.append(Paragraph(f"<b>Phone:</b> {utils.patient_info.get('Phone', 'N/A')}", styles['Normal']))
            story.append(Paragraph(f"<b>Address:</b> {utils.patient_info.get('Address', 'N/A')}", styles['Normal']))

            story.append(Spacer(1, 0.25 * inch))

            # Measurements
            story.append(Paragraph("<b>Measurement Results:</b>", styles['h2']))
            story.append(Paragraph(f"Ear Height: {avg_ear_height} mm", styles['Normal']))
            story.append(Paragraph(f"Ear Width: {avg_ear_width} mm", styles['Normal']))
            
            # Add a spacer before the images section
            story.append(Spacer(1, 0.25 * inch))

            # Add images and measurements
            for i, res in enumerate(utils.measurement_results):
                # First add the measurements
                story.append(Paragraph(f"<b>Image {i+1}:</b>", styles['h3']))
                
                if 'error' in res:
                    # Skip the error message for images
                    continue
                else:
                    # First add the measurements
                    height_val = res.get('actual_height_mm', 'N/A')
                    width_val = res.get('actual_width_mm', 'N/A')
                    distance_val = res.get('sensor_distance_mm', 'N/A')
                    
                    print(f"=== PDF GENERATION DEBUG ===")
                    print(f"Writing to PDF for Image {i+1}:")
                    print(f"  Height value: {height_val} (type: {type(height_val)})")
                    print(f"  Width value: {width_val} (type: {type(width_val)})")
                    print(f"  Distance value: {distance_val} (type: {type(distance_val)})")
                    print(f"===========================")
                    
                    # Format values properly for display
                    height_display = f"{height_val:.1f} mm" if isinstance(height_val, (int, float)) else str(height_val)
                    width_display = f"{width_val:.1f} mm" if isinstance(width_val, (int, float)) else str(width_val)
                    distance_display = f"{distance_val:.2f} mm" if isinstance(distance_val, (int, float)) else str(distance_val)
                    
                    story.append(Paragraph(f"<b>Ear Height:</b> {height_display}", styles['Normal']))
                    story.append(Paragraph(f"<b>Ear Width:</b> {width_display}", styles['Normal']))
                    story.append(Paragraph(f"<b>Distance from Camera:</b> {distance_display}", styles['Normal']))
                    story.append(Paragraph(f"<b>Pixel Dimensions:</b> {res.get('pixel_height', 'N/A')}px x {res.get('pixel_width', 'N/A')}px", styles['Normal']))
                    story.append(Spacer(1, 0.1 * inch))
                    
                    # Directly include the images from captured_images directory
                    # These are the 5 images we take in the "Take Measurement" window
                    img_paths = [
                        os.path.join("captured_images", f"annotated_img_{i}.jpg")
                    ]
                    
                    images_added = 0
                    
                    # Add each image to the PDF
                    for img_path in img_paths:
                        if os.path.exists(img_path):
                            try:
                                # Add the image to the PDF
                                img = Image.open(img_path)
                                # Resize if needed
                                max_width = 5 * inch
                                max_height = 4 * inch
                                img_width, img_height = img.size
                                
                                # Calculate aspect ratio
                                aspect = img_width / float(img_height)
                                
                                # Determine new dimensions
                                if img_width > max_width:
                                    img_width = max_width
                                    img_height = img_width / aspect
                                
                                if img_height > max_height:
                                    img_height = max_height
                                    img_width = img_height * aspect
                                
                                # Add image to PDF with explicit path
                                img_for_pdf = RLImage(os.path.abspath(img_path), width=img_width, height=img_height)
                                story.append(img_for_pdf)
                                story.append(Spacer(1, 0.1 * inch))
                                images_added += 1
                                print(f"Added image to PDF: {os.path.abspath(img_path)}")
                            except Exception as img_error:
                                story.append(Paragraph(f"Error loading image: {img_error}", styles['Normal']))
                    
                    if images_added == 0:
                        story.append(Paragraph("No images available for this measurement", styles['Normal']))
                    
                    # Add pixel dimensions if available
                    if 'pixel_height' in res and 'pixel_width' in res:
                        story.append(Paragraph(f"<b>Pixel Dimensions:</b> {res['pixel_height']}px × {res['pixel_width']}px", styles['Normal']))
                
                story.append(Spacer(1, 0.25 * inch))

            # Build PDF
            doc.build(story)
            pdf_generated_successfully = True
        except Exception as e:
            messagebox.showerror("PDF Generation Failed", f"Could not generate PDF: {e}")
            pdf_generated_successfully = False
    else:
        print("PDF generation skipped: ReportLab not available")
        pdf_generated_successfully = False

    # Function to open files with default applications
    def open_file(file_path):
        import subprocess
        import platform
        
        try:
            if platform.system() == 'Windows':
                os.startfile(file_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.call(('open', file_path))
            else:  # Linux
                subprocess.call(('xdg-open', file_path))
            return True
        except Exception as e:
            print(f"Error opening file {file_path}: {e}")
            return False
    
    # Final feedback with option to open files
    if excel_saved_successfully and pdf_generated_successfully:
        result = messagebox.askyesno(
            "Success",
            f"Data saved successfully!\n\nExcel: {excel_file_path}\nPDF: {pdf_file_path}\n\nWould you like to open these files now?",
            icon=messagebox.INFO
        )
        if result:
            # Try to open both files
            pdf_opened = open_file(pdf_file_path)
            excel_opened = open_file(excel_file_path)
            
            if not pdf_opened or not excel_opened:
                messagebox.showinfo("Note", "Some files could not be opened automatically. Please navigate to the save location to view them.")
    elif excel_saved_successfully:
        messagebox.showinfo("Partial Success", f"Data saved to Excel: {excel_file_path}\nPDF generation failed or was skipped.")
    elif pdf_generated_successfully:
        messagebox.showinfo("Partial Success", f"PDF Report generated: {pdf_file_path}\nExcel save failed.")
    else:
        messagebox.showerror("Save Failed", "Both Excel and PDF saving failed. Please check console for errors.")

    return excel_saved_successfully or pdf_generated_successfully

if __name__ == '__main__':
    # This block is for testing data_handling.py independently.
    # It requires a Tkinter root window for filedialogs.
    pass


