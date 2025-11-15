from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from typing import Dict
import os
import shutil
from pathlib import Path
from datetime import datetime
from bson import ObjectId

from backend.services.vehicle_detector import VehicleDetector
from backend.models.vehicle_stats import VehicleStats, VehicleCount
from backend.database.connection import get_database

router = APIRouter(prefix="/api/vehicles", tags=["vehicles"])

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

detector = VehicleDetector()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_extension = file.filename.split('.')[-1].lower()
        supported_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'mp4', 'avi', 'mov', 'mkv']
        
        if file_extension not in supported_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Supported: {', '.join(supported_extensions)}"
            )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "success": True,
            "filename": filename,
            "file_type": file_extension,
            "message": "File uploaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect/{filename}")
async def detect_vehicles(filename: str):
    try:
        input_path = os.path.join(UPLOAD_DIR, filename)
        
        if not os.path.exists(input_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        file_extension = filename.split('.')[-1].lower()
        output_filename = f"detected_{filename}"
        
        if file_extension in ['mp4', 'avi', 'mov', 'mkv']:
            output_filename = output_filename.rsplit('.', 1)[0] + '.mp4'
        
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        vehicle_counts = detector.process_file(input_path, output_path, file_extension)
        
        return {
            "success": True,
            "filename": filename,
            "output_filename": output_filename,
            "vehicle_counts": vehicle_counts,
            "message": "Detection completed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/save")
async def save_statistics(
    filename: str,
    output_filename: str,
    vehicle_counts: Dict[str, int],
    db = Depends(get_database)
):
    try:
        file_extension = filename.split('.')[-1].lower()
        
        stats = {
            "filename": filename,
            "file_type": file_extension,
            "vehicle_counts": vehicle_counts,
            "processed_at": datetime.utcnow(),
            "output_path": output_filename
        }
        
        result = await db.vehicle_statistics.insert_one(stats)
        
        return {
            "success": True,
            "id": str(result.inserted_id),
            "message": "Statistics saved to database"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{filename}")
async def download_result(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

@router.get("/statistics")
async def get_all_statistics(db = Depends(get_database)):
    try:
        cursor = db.vehicle_statistics.find().sort("processed_at", -1).limit(50)
        statistics = []
        
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            statistics.append(doc)
        
        return {
            "success": True,
            "data": statistics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics/{stat_id}")
async def get_statistic_by_id(stat_id: str, db = Depends(get_database)):
    try:
        doc = await db.vehicle_statistics.find_one({"_id": ObjectId(stat_id)})
        
        if not doc:
            raise HTTPException(status_code=404, detail="Statistic not found")
        
        doc["_id"] = str(doc["_id"])
        
        return {
            "success": True,
            "data": doc
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cleanup")
async def cleanup_files():
    try:
        for folder in [UPLOAD_DIR, OUTPUT_DIR]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        
        return {
            "success": True,
            "message": "All temporary files cleaned up"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
