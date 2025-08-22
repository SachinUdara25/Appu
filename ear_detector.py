#ear_detector.py
from ultralytics import YOLO
import cv2
import numpy as np
import os

class EarDetector:
    def __init__(self, model_path=None):
        """Initialize the ear detector with a YOLO model"""
        import torch
        if model_path is None:
            model_path = os.path.join("dataset", "runs", "detect", "train2", "weights", "best.pt")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            self.model = YOLO(model_path)
            # Add this line to make the 'names' attribute accessible directly via EarDetector instance
            self.names = self.model.names  # <--- මෙම පේළිය එක් කරන්න
            
            if self.model.task not in ['detect', 'segment']:
                raise ValueError(f"Unsupported YOLO model task: {self.model.task}. Expected 'detect' or 'segment'.")
            
            if self.model.task == 'segment':
                print(f"Info: Loaded a YOLO segmentation model. Ensure downstream code handles 'boxes' attribute correctly.")
            elif self.model.task == 'detect':
                print(f"Info: Loaded a YOLO object detection model.")
            else:
                print(f"Info: Loaded a YOLO model with task: {self.model.task}. Check compatibility.")

            self.model.to(self.device)
            print(f"Ear detector initialized with model: {model_path} on device: {self.device}")
        except Exception as e:
            print(f"Error initializing EarDetector with model {model_path}: {e}")
            print("Please ensure the model path is correct and the model file is valid.")
            print("If you are loading a segmentation model, ensure your ultralytics version is compatible.")
            raise

    def detect_ear(self, frame):
        """
        Detect ears in the given frame
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            tuple: (annotated_frame, ear_detected, bounding_boxes, segmentation_masks)
                   segmentation_masks will be None if not a segmentation model or no masks found
        """
        # Make a copy of the frame for annotation
        annotated_frame = frame.copy()
        
        # Log which device is being used for detection
        # print(f"Performing detection on device: {self.device}") # Too chatty, uncomment for debug
        
        ear_detected = False
        bounding_boxes = []
        segmentation_masks = None # Initialize for segmentation masks
        all_detections = []  # Initialize all_detections list

        try:
            # Predict method returns a list of Results objects
            # For a single image, results will contain one Results object
            # Setting verbose=False to reduce console output during live feed
            results = self.model.predict(frame, conf=0.5, iou=0.7, verbose=False) # Adjust conf and iou as needed

            if results and len(results) > 0:
                result = results[0] # Assuming one result object per frame

                if result.boxes: # Check if bounding boxes are present (for detect and segment models)
                    for i, box in enumerate(result.boxes):
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0]) # Class ID

                        all_detections.append({
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'confidence': confidence,
                            'class_id': class_id
                        })


                        bounding_boxes.append({
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'confidence': confidence,
                            'class_id': class_id,
                            'height': y2 - y1,
                            'width': x2 - x1
                        })

                        # Use color based on confidence (red for low, green for high)
                        color = (0, int(255 * confidence), int(255 * (1 - confidence)))
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add label with confidence and dimensions
                        height = y2 - y1
                        width = x2 - x1
                        label = f"#{i} Conf: {confidence:.2f} ({height}x{width}px)" # Format confidence better
                        
                        # Background for text
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + text_size[0], y1), color, -1)
                        
                        # Text
                        cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Check for segmentation masks if the model provides them (i.e., it's a segmentation model)
                if result.masks:
                    # masks.data contains the binary masks (NxHxW)
                    # masks.xy contains the polygon coordinates (list of numpy arrays)
                    # We might want to return the binary masks or process them further
                    # For simplicity, let's return a list of numpy arrays for the masks
                    segmentation_masks = result.masks.data.cpu().numpy() # Convert to numpy array on CPU

                    # Optionally, draw segmentation masks on the annotated_frame
                    # This part can be resource-intensive or depend on desired visualization
                    # For a simple overlay:
                    mask_overlay = np.zeros_like(annotated_frame, dtype=np.uint8)
                    for i, mask in enumerate(segmentation_masks):
                        # Resize mask to original frame dimensions if necessary (predict usually scales it)
                        mask_resized = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]))
                        
                        # Create a colored overlay for the mask
                        color_mask = np.zeros_like(annotated_frame, dtype=np.uint8)
                        color_mask[mask_resized > 0] = [0, 255, 0] # Green color for masks
                        
                        # Blend the mask with the annotated frame
                        annotated_frame = cv2.addWeighted(annotated_frame, 1, color_mask, 0.3, 0) # 30% opacity

            # Add model info (e.g., model name, version, device)
            # This part remains mostly the same, ensuring it doesn't break if `model` is not yet defined
            if hasattr(self, 'model') and self.model:
                model_name = getattr(self.model, 'model_name', 'YOLO Model')
                model_version = getattr(self.model, 'version', 'N/A')
                device_info = self.device
                
                info_text = f"Model: {model_name} | Device: {device_info}"
                cv2.putText(annotated_frame, info_text, (10, annotated_frame.shape[0] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        except Exception as e:
            print(f"Error during ear detection: {e}")
            # Optionally, re-raise the error or return an empty result
            return annotated_frame, False, [], None # Return empty results on error
            
        return annotated_frame, ear_detected, bounding_boxes, segmentation_masks # Return masks as well

    
    def diagnostic_detection(self, frame):
        """
        Run detection with diagnostic visualization to help troubleshoot model issues.
        Shows raw model output with all detections regardless of confidence.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            tuple: (diagnostic_frame, all_detections)
        """
        # Make a copy of the frame for annotation
        diagnostic_frame = frame.copy()
        
        # Add frame dimensions
        height, width = frame.shape[:2]
        cv2.putText(diagnostic_frame, f"Frame: {width}x{height}px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Log which device is being used for diagnostic detection
        print(f"[EarDetector] Running diagnostic detection on device: {self.device}")
        # Run YOLO detection with very low confidence to see all potential detections
        results = self.model(frame, device=self.device, conf=0.1)
        
        all_detections = []
        
        # Process all results, even low confidence ones
        for result in results:
            boxes = result.boxes
            
            # Add detection count
            cv2.putText(diagnostic_frame, f"Detections: {len(boxes)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Process each detection
            for i, box in enumerate(boxes):
                confidence = box.conf.item()
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls.item())
                
                # Store detection info
                all_detections.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'confidence': confidence,
                    'class_id': class_id
                })
                
                # Use color based on confidence (red for low, green for high)
                color = (0, int(255 * confidence), int(255 * (1 - confidence)))
                
                # Draw bounding box
                cv2.rectangle(diagnostic_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add label with confidence and dimensions
                height = y2 - y1
                width = x2 - x1
                label = f"#{i} Conf: {confidence:.3f} ({height}x{width}px)"
                
                # Background for text
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(diagnostic_frame, (x1, y1 - 20), (x1 + text_size[0], y1), color, -1)
                
                # Text
                cv2.putText(diagnostic_frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add model info
        model_info = f"Model: {os.path.basename(self.model.ckpt_path)}"
        cv2.putText(diagnostic_frame, model_info, (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return diagnostic_frame, all_detections

