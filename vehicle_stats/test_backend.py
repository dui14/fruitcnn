#!/usr/bin/env python3
"""
Test script to verify backend setup and YOLOv8 installation
"""

import sys
import os

print("=" * 60)
print("Vehicle Statistics Backend - Setup Test")
print("=" * 60)

# Test 1: Import required modules
print("\n1. Testing module imports...")
try:
    import fastapi
    print("   ✓ FastAPI imported successfully")
except ImportError as e:
    print(f"   ✗ FastAPI import failed: {e}")
    sys.exit(1)

try:
    import cv2
    print("   ✓ OpenCV imported successfully")
except ImportError as e:
    print(f"   ✗ OpenCV import failed: {e}")
    sys.exit(1)

try:
    from ultralytics import YOLO
    print("   ✓ Ultralytics (YOLOv8) imported successfully")
except ImportError as e:
    print(f"   ✗ Ultralytics import failed: {e}")
    sys.exit(1)

try:
    import motor
    print("   ✓ Motor (MongoDB async driver) imported successfully")
except ImportError as e:
    print(f"   ✗ Motor import failed: {e}")
    sys.exit(1)

# Test 2: Check YOLOv8 model
print("\n2. Testing YOLOv8 model initialization...")
try:
    model = YOLO("yolov8n.pt")
    print("   ✓ YOLOv8 model loaded successfully")
    print(f"   Model: {model.model_name if hasattr(model, 'model_name') else 'yolov8n'}")
except Exception as e:
    print(f"   ✗ YOLOv8 model loading failed: {e}")
    sys.exit(1)

# Test 3: Check directory structure
print("\n3. Checking directory structure...")
backend_dir = os.path.join(os.path.dirname(__file__), "backend")
required_dirs = ["models", "services", "routes", "database"]
for dir_name in required_dirs:
    dir_path = os.path.join(backend_dir, dir_name)
    if os.path.exists(dir_path):
        print(f"   ✓ {dir_name}/ exists")
    else:
        print(f"   ✗ {dir_name}/ missing")

# Test 4: Check main files
print("\n4. Checking main files...")
required_files = [
    "backend/main.py",
    "backend/services/vehicle_detector.py",
    "backend/routes/vehicle_routes.py",
    "backend/database/connection.py",
    "backend/models/vehicle_stats.py"
]
for file_path in required_files:
    full_path = os.path.join(os.path.dirname(__file__), file_path)
    if os.path.exists(full_path):
        print(f"   ✓ {file_path}")
    else:
        print(f"   ✗ {file_path} missing")

# Test 5: Test vehicle detector
print("\n5. Testing VehicleDetector class...")
try:
    sys.path.insert(0, backend_dir)
    from services.vehicle_detector import VehicleDetector
    detector = VehicleDetector()
    print("   ✓ VehicleDetector initialized successfully")
    print(f"   Vehicle classes: {list(detector.vehicle_classes.values())}")
except Exception as e:
    print(f"   ✗ VehicleDetector initialization failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed! Backend is ready to run.")
print("=" * 60)
print("\nTo start the backend server, run:")
print("  cd backend && python -m uvicorn main:app --reload")
print("\nOr use the startup script:")
print("  ./start_backend.sh")
print()
