from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime

class VehicleCount(BaseModel):
    motorbikes: int = 0
    cars: int = 0
    trucks: int = 0

class VehicleStats(BaseModel):
    filename: str
    file_type: str
    vehicle_counts: VehicleCount
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    output_path: Optional[str] = None
    metadata: Optional[Dict] = None

class VehicleStatsInDB(VehicleStats):
    id: str = Field(alias="_id")
    
    class Config:
        populate_by_name = True
