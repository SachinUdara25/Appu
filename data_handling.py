# data_handling.py

import pandas as pd
from tkinter import filedialog, messagebox
import utils # To access utils.patient_info, utils.results
import os # For path manipulation
import tempfile # For dummy image creation in main
from PIL import Image # For dummy image creation in main
import tkinter as tk # For main test block

# Ensure ReportLab is available, provide guidance if not.
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.lib.pagesizes import letter
except ImportError:
    messagebox.showerror(
        "Dependency Missing",
        "ReportLab library is not installed. PDF reporting will not be available. "
        "Please install it by running: pip install reportlab"
    )
    # Define dummy classes or functions if ReportLab is critical and you want to avoid NameErrors
    # For now, functions using it will fail gracefully if this import fails.
    class SimpleDocTemplate: pass # Dummy
    class Paragraph: pass # Dummy
    # etc. or just let it raise NameError later.

def save_measurements_and_generate_report():
    """
    Saves patient measurements to an Excel file and generates a PDF report.
    Assumes utils.patient_info and utils.results are populated.
    utils.results is expected to be a list of dictionaries:
    [{'ear_height_mm': h, 'ear_width_mm': w, 'image_path': path,
      'original_image': img_bgr, 'distance': dist, 'error': 'optional error msg'}, ...]
    """
    if not utils.patient_info:
        messagebox.showerror("Data Error", "Patient information is missing.")
        return
    if not utils.results:
        messagebox.showerror("Data Error", "No measurement results found to save.")
        return

    # Calculate average length (height) and width from valid results
    valid_heights = [res['ear_height_mm'] for res in utils.results if 'error' not in res and res['ear_height_mm'] > 0]
    valid_widths = [res['ear_width_mm'] for res in utils.results if 'error' not in res and res['ear_width_mm'] > 0]

    avg_ear_height = round(pd.Series(valid_heights).mean(), 2) if valid_heights else 0
    avg_ear_width = round(pd.Series(valid_widths).mean(), 2) if valid_widths else 0

    # --- Prepare data for Excel ---
    excel_data = {
        "Patient Name": [utils.patient_info.get("Name", "N/A")],
        "Date of Birth": [utils.patient_info.get("DOB", "N/A")],
        "Gender": [utils.patient_info.get("Gender", "N/A")],
        "Phone": [utils.patient_info.get("Phone", "N/A")],
        "Address Line 1": [utils.patient_info.get("Address1", "N/A")],
        "Address Line 2": [utils.patient_info.get("Address2", "N/A")],
        "Address Line 3": [utils.patient_info.get("Address3", "N/A")],
        "Average Ear Height (mm)": [avg_ear_height],
        "Average Ear Width (mm)": [avg_ear_width]
    }

    for i, res_dict in enumerate(utils.results):
        excel_data[f"Image {i+1} Ear Height (mm)"] = [res_dict['ear_height_mm'] if 'error' not in res_dict else 'Error']
        excel_data[f"Image {i+1} Ear Width (mm)"] = [res_dict['ear_width_mm'] if 'error' not in res_dict else 'Error']
        excel_data[f"Image {i+1} Distance (mm)"] = [res_dict['distance'] if res_dict['distance'] > 0 else 'N/A']
        excel_data[f"Image {i+1} Processed Path"] = [res_dict['image_path'] if 'error' not in res_dict else 'N/A']
        if 'error' in res_dict:
             excel_data[f"Image {i+1} Error"] = [res_dict['error']]


    df = pd.DataFrame(excel_data)

    # --- Ask for save location (basename) ---
    # User provides a single filename, we'll append .xlsx and _report.pdf
    base_file_path = filedialog.asksaveasfilename(
        title="Save Report As",
        defaultextension=".xlsx", # Default for the Excel file
        filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
    )

    if not base_file_path:
        messagebox.showinfo("Cancelled", "Save operation cancelled.")
        return

    excel_file_path = base_file_path
    if not excel_file_path.lower().endswith(".xlsx"):
        excel_file_path += ".xlsx"

    pdf_file_path = os.path.splitext(base_file_path)[0] + "_report.pdf"


    # --- Save Excel File ---
    excel_saved_successfully = False
    try:
        df.to_excel(excel_file_path, index=False)
        print(f"Data saved to Excel: {excel_file_path}")
        excel_saved_successfully = True
    except Exception as e:
        messagebox.showerror("Excel Save Error", f"Could not save Excel file: {e}")
        # Optionally, ask if user wants to proceed with PDF anyway
        if not messagebox.askyesno("Continue?", "Excel save failed. Continue with PDF report generation?"):
            return # Stop if Excel fails and user doesn't want to continue

    # --- Generate PDF Report ---
    if 'SimpleDocTemplate' not in globals() or not hasattr(globals()['SimpleDocTemplate'], '__call__'): # Check if ReportLab failed to import or is dummy
        messagebox.showerror("PDF Error", "ReportLab library not found or not functional. Cannot generate PDF report.")
        if excel_saved_successfully: # Check if excel was saved
             messagebox.showinfo("Excel Saved", f"Data successfully saved to:\n{excel_file_path}")
        return

    pdf_generated_successfully = False
    try:
        doc = SimpleDocTemplate(pdf_file_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("Ear Measurement Report", styles['h1']))
        story.append(Spacer(1, 0.25 * inch))

        # Patient Information
        story.append(Paragraph("<b>Patient Information:</b>", styles['h2']))
        story.append(Paragraph(f"<b>Name:</b> {utils.patient_info.get('Name', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"<b>Date of Birth:</b> {utils.patient_info.get('DOB', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"<b>Gender:</b> {utils.patient_info.get('Gender', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"<b>Phone:</b> {utils.patient_info.get('Phone', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"<b>Address:</b> {utils.patient_info.get('Address1', '')}", styles['Normal']))
        if utils.patient_info.get('Address2'):
            story.append(Paragraph(f"{utils.patient_info.get('Address2', '')}", styles['Normal']))
        if utils.patient_info.get('Address3'):
            story.append(Paragraph(f"{utils.patient_info.get('Address3', '')}", styles['Normal']))
        story.append(Spacer(1, 0.25 * inch))

        # Summary Measurements
        story.append(Paragraph("<b>Average Measurements:</b>", styles['h2']))
        story.append(Paragraph(f"Average Ear Height (Length): {avg_ear_height} mm", styles['Normal']))
        story.append(Paragraph(f"Average Ear Width: {avg_ear_width} mm", styles['Normal']))
        story.append(Spacer(1, 0.25 * inch))

        # Individual Image Results
        story.append(Paragraph("<b>Captured Image Details:</b>", styles['h2']))
        story.append(Spacer(1, 0.1 * inch))

        max_img_width_pdf = 6 * inch # Max width for images in PDF
        max_img_height_pdf = 4.5 * inch # Max height for images in PDF

        for i, res_dict in enumerate(utils.results):
            story.append(Paragraph(f"<b>Image {i + 1}:</b>", styles['h3']))
            if 'error' in res_dict:
                story.append(Paragraph(f"<i>Error processing this image: {res_dict['error']}</i>", styles['Italic']))
                # Optionally, include the original (unprocessed) image if available and useful
            elif res_dict['image_path'] and os.path.exists(res_dict['image_path']):
                try:
                    # Display the annotated image
                    img = RLImage(res_dict['image_path'])

                    # Scale image to fit if too large
                    img_w, img_h = img.imageWidth, img.imageHeight
                    aspect = img_h / float(img_w) if img_w > 0 else 1 # Avoid division by zero

                    current_draw_width = img_w
                    current_draw_height = img_h

                    if img_w > max_img_width_pdf:
                        current_draw_width = max_img_width_pdf
                        current_draw_height = max_img_width_pdf * aspect

                    # Check height again if width scaling made it too tall, or if original height was too tall
                    if current_draw_height > max_img_height_pdf:
                        current_draw_height = max_img_height_pdf
                        current_draw_width = max_img_height_pdf / aspect if aspect > 0 else max_img_width_pdf # Avoid division by zero

                    img.drawWidth = current_draw_width
                    img.drawHeight = current_draw_height

                    story.append(img)
                    story.append(Spacer(1, 0.1 * inch))
                    story.append(Paragraph(f"Measured Ear Height: {res_dict['ear_height_mm']} mm", styles['Normal']))
                    story.append(Paragraph(f"Measured Ear Width: {res_dict['ear_width_mm']} mm", styles['Normal']))
                    story.append(Paragraph(f"Sensor Distance: {res_dict['distance'] if res_dict['distance'] > 0 else 'N/A'} mm", styles['Normal']))
                except Exception as img_e:
                    story.append(Paragraph(f"<i>Could not load/display image {i+1}: {img_e}</i>", styles['Italic']))
            else:
                story.append(Paragraph("<i>Annotated image not available.</i>", styles['Italic']))
            story.append(Spacer(1, 0.2 * inch))
            if (i + 1) % 2 == 0 and (i + 1) < len(utils.results) : # Add a page break after every 2 images, for example
                 story.append(PageBreak())


        doc.build(story)
        print(f"PDF report generated: {pdf_file_path}")
        pdf_generated_successfully = True

        if excel_saved_successfully and pdf_generated_successfully:
            messagebox.showinfo("Success", f"Data saved to Excel: {excel_file_path}\nPDF Report generated: {pdf_file_path}")
        elif excel_saved_successfully:
            messagebox.showinfo("Partial Success", f"Data saved to Excel: {excel_file_path}\nPDF generation failed.")
        elif pdf_generated_successfully: # Unlikely scenario if excel save was first, but for completeness
             messagebox.showinfo("Partial Success", f"PDF Report generated: {pdf_file_path}\nExcel save failed.")
        # If both failed, errors would have been shown already.

    except ImportError: # Catches if ReportLab was not available from the start
        messagebox.showerror("PDF Error", "ReportLab library is not installed. PDF report not generated.")
        if excel_saved_successfully:
             messagebox.showinfo("Excel Saved", f"Data successfully saved to:\n{excel_file_path}")
    except Exception as e:
        messagebox.showerror("PDF Generation Error", f"Could not generate PDF report: {e}")
        if excel_saved_successfully:
             messagebox.showinfo("Excel Saved", f"Data successfully saved to:\n{excel_file_path}")


if __name__ == '__main__':
    # Example Usage (requires a running Tkinter app for filedialog)

    # Mock utils for testing
    class MockUtils:
        def __init__(self):
            self.patient_info = {
                "Name": "Test Patient", "DOB": "2000-01-01", "Gender": "Other",
                "Phone": "1234567890", "Address1": "123 Test St", "Address2": "Apt B", "Address3": "Testville, TS 12345"
            }
            # Create some dummy image files for testing PDF generation
            self.temp_dir = tempfile.mkdtemp()
            self.dummy_image_paths = []
            for i in range(5):
                try:
                    # Create images with different aspect ratios
                    width = 300 + i*50
                    height = 200 + i*30
                    img = Image.new('RGB', (width, height), color = (73, 109, 137))
                    # Add some text to image to differentiate
                    # from PIL import ImageDraw # Not available in sandbox by default
                    # draw = ImageDraw.Draw(img)
                    # draw.text((10,10), f"Image {i}", fill=(255,255,0))
                    path = os.path.join(self.temp_dir, f"dummy_ear_{i}.png")
                    img.save(path)
                    self.dummy_image_paths.append(path)
                except Exception as e:
                    print(f"Error creating dummy image {i}: {e}")


            self.results = [
                {'ear_height_mm': 55.1, 'ear_width_mm': 30.5, 'image_path': self.dummy_image_paths[0] if len(self.dummy_image_paths)>0 else None, 'distance': 100.0, 'original_image': None},
                {'ear_height_mm': 56.2, 'ear_width_mm': 31.0, 'image_path': self.dummy_image_paths[1] if len(self.dummy_image_paths)>1 else None, 'distance': 105.0, 'original_image': None},
                {'ear_height_mm': 0, 'ear_width_mm': 0, 'image_path': None, 'distance': 110.0, 'original_image': None, 'error': 'Simulated processing error'},
                {'ear_height_mm': 54.8, 'ear_width_mm': 30.1, 'image_path': self.dummy_image_paths[3] if len(self.dummy_image_paths)>3 else None, 'distance': 98.0, 'original_image': None},
                {'ear_height_mm': 55.5, 'ear_width_mm': 30.7, 'image_path': self.dummy_image_paths[4] if len(self.dummy_image_paths)>4 else None, 'distance': 102.0, 'original_image': None},
            ]
            # Add a result with a missing image path to test that case
            if len(self.dummy_image_paths) > 2: # ensure index 2 is valid
                 self.results.insert(2, {'ear_height_mm': 50.0, 'ear_width_mm': 25.0, 'image_path': "non_existent_image.png", 'distance': 100.0, 'original_image': None})


    # --- To run this test block: ---
    # 1. Ensure pandas and reportlab are installed.
    # 2. Uncomment the Tkinter root window setup.
    # 3. Run this file directly. It will prompt for a save location.

    # print("Setting up Tkinter root for test...")
    # root_test = tk.Tk()
    # root_test.withdraw() # Hide the dummy root window, filedialog will still work

    # global utils # Make the mock utils global for the function to see
    # utils = MockUtils()
    # print("Mock utils and results created.")
    # # print(f"Dummy image paths: {utils.dummy_image_paths}")
    # # for p in utils.dummy_image_paths: print(f"Exists {p}: {os.path.exists(p)}")

    # # Call the function
    # print("Calling save_measurements_and_generate_report()...")
    # save_measurements_and_generate_report()

    # # Clean up dummy images and dir
    # print("Cleaning up temporary files...")
    # try:
    #     for path in utils.dummy_image_paths:
    #         if os.path.exists(path):
    #             os.remove(path)
    #     if os.path.exists(utils.temp_dir):
    #         os.rmdir(utils.temp_dir)
    #     print("Cleaned up temporary files.")
    # except Exception as e:
    #     print(f"Error during cleanup: {e}")

    # print("Destroying Tkinter root...")
    # root_test.destroy() # Properly close the hidden root window
    # print("Test finished.")
    pass # Keep the if __name__ block valid even if testing part is commented
