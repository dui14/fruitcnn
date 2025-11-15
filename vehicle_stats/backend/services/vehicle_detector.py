import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os
from typing import Tuple, Dict

class VehicleDetector:
    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model = YOLO(model_name)
        
        # COCO dataset class IDs for vehicles
        self.vehicle_classes = {
            2: 'car',
            3: 'motorbike',
            5: 'bus',
            7: 'truck'
        }
        
        # Map to our categories
        self.category_mapping = {
            'car': 'cars',
            'motorbike': 'motorbikes',
            'bus': 'trucks',
            'truck': 'trucks'
        }
    
    def detect_vehicles_image(self, image_path: str, output_path: str) -> Dict[str, int]:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        results = self.model(img, conf=0.3)
        
        vehicle_counts = {
            'motorbikes': 0,
            'cars': 0,
            'trucks': 0
        }
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.vehicle_classes:
                    vehicle_type = self.vehicle_classes[cls_id]
                    category = self.category_mapping[vehicle_type]
                    vehicle_counts[category] += 1
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Different colors for different vehicle types
                    colors = {
                        'motorbikes': (0, 255, 0),    # Green
                        'cars': (255, 0, 0),           # Blue
                        'trucks': (0, 0, 255)          # Red
                    }
                    color = colors[category]
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    label = f"{vehicle_type}: {conf:.2f}"
                    cv2.putText(img, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add summary text
        y_offset = 30
        for vehicle_type, count in vehicle_counts.items():
            text = f"{vehicle_type.capitalize()}: {count}"
            cv2.putText(img, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_offset += 30
        
        cv2.imwrite(output_path, img)
        return vehicle_counts
    
    def detect_vehicles_video(self, video_path: str, output_path: str) -> Dict[str, int]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Track maximum counts across all frames
        max_vehicle_counts = {
            'motorbikes': 0,
            'cars': 0,
            'trucks': 0
        }
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every frame (or skip frames for faster processing)
            if frame_count % 1 == 0:  # Process every frame
                results = self.model(frame, conf=0.3, verbose=False)
                
                frame_vehicle_counts = {
                    'motorbikes': 0,
                    'cars': 0,
                    'trucks': 0
                }
                
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        if cls_id in self.vehicle_classes:
                            vehicle_type = self.vehicle_classes[cls_id]
                            category = self.category_mapping[vehicle_type]
                            frame_vehicle_counts[category] += 1
                            
                            # Draw bounding box
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            
                            colors = {
                                'motorbikes': (0, 255, 0),
                                'cars': (255, 0, 0),
                                'trucks': (0, 0, 255)
                            }
                            color = colors[category]
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            label = f"{vehicle_type}: {conf:.2f}"
                            cv2.putText(frame, label, (x1, y1 - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Update maximum counts
                for key in max_vehicle_counts:
                    max_vehicle_counts[key] = max(max_vehicle_counts[key], 
                                                  frame_vehicle_counts[key])
                
                # Add frame counts
                y_offset = 30
                for vehicle_type, count in frame_vehicle_counts.items():
                    text = f"{vehicle_type.capitalize()}: {count}"
                    cv2.putText(frame, text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    y_offset += 30
            
            out.write(frame)
        
        cap.release()
        out.release()
        
        return max_vehicle_counts
    
    def process_file(self, input_path: str, output_path: str, file_type: str) -> Dict[str, int]:
        if file_type in ['jpg', 'jpeg', 'png', 'bmp']:
            return self.detect_vehicles_image(input_path, output_path)
        elif file_type in ['mp4', 'avi', 'mov', 'mkv']:
            return self.detect_vehicles_video(input_path, output_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
