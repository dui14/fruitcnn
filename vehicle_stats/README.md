# Vehicle Statistics Program using YOLOv8

A full-stack application for detecting and counting vehicles (motorbikes, cars, trucks) in images and videos using YOLOv8 deep learning model.

## Features

- 🚗 Detect and count vehicles: motorbikes, cars, and trucks
- 📸 Support for images (JPG, PNG, BMP) and videos (MP4, AVI, MOV, MKV)
- 🎯 Real-time detection with bounding boxes
- 💾 Save statistics to MongoDB database
- 📊 Visual results with annotated images/videos
- 🌐 Simple and user-friendly web interface

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **YOLOv8**: State-of-the-art object detection model
- **MongoDB**: NoSQL database for storing statistics
- **OpenCV**: Computer vision library
- **Ultralytics**: YOLOv8 implementation

### Frontend
- **React**: JavaScript library for building UI
- **Axios**: HTTP client for API requests

## Project Structure

```
vehicle_stats/
├── backend/
│   ├── main.py                 # FastAPI application entry point
│   ├── database/
│   │   └── connection.py       # MongoDB connection
│   ├── models/
│   │   └── vehicle_stats.py    # Data models
│   ├── services/
│   │   └── vehicle_detector.py # YOLOv8 detection service
│   ├── routes/
│   │   └── vehicle_routes.py   # API endpoints
│   ├── uploads/                # Temporary upload directory
│   └── outputs/                # Processed results directory
├── frontend/
│   ├── src/
│   │   ├── App.js              # Main React component
│   │   ├── App.css             # Styling
│   │   └── index.js            # React entry point
│   └── public/
│       └── index.html          # HTML template
└── requirements.txt            # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8+
- Node.js 16+
- MongoDB (running locally or remote)

### Backend Setup

1. Navigate to the project directory:
```bash
cd vehicle_stats
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the `backend` directory:
```bash
cd backend
cp .env.example .env
```

4. Edit `.env` with your MongoDB connection:
```
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=vehicle_stats_db
UPLOAD_DIR=uploads
OUTPUT_DIR=outputs
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install Node.js dependencies:
```bash
npm install
```

## Running the Application

### Start MongoDB

Make sure MongoDB is running on your system:
```bash
# On Linux/Mac
sudo systemctl start mongod

# Or using Docker
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

### Start Backend Server

```bash
cd backend
python main.py
```

The API will be available at `http://localhost:8000`

### Start Frontend Development Server

```bash
cd frontend
npm start
```

The React app will open at `http://localhost:3000`

## Usage

1. **Browse File**: Click the "Browse File" button to select an image or video file
2. **Run Detection**: Click "Run Detection" to upload and process the file with YOLOv8
3. **View Results**: See the detected vehicle counts and annotated output
4. **Save to Database**: Click "Save to Database" to store the statistics in MongoDB

## API Endpoints

### Upload File
```
POST /api/vehicles/upload
Content-Type: multipart/form-data
Body: file (image or video)
```

### Run Detection
```
POST /api/vehicles/detect/{filename}
```

### Save Statistics
```
POST /api/vehicles/save
Params: filename, output_filename, vehicle_counts
```

### Download Result
```
GET /api/vehicles/download/{filename}
```

### Get All Statistics
```
GET /api/vehicles/statistics
```

### Get Statistic by ID
```
GET /api/vehicles/statistics/{stat_id}
```

## Vehicle Detection

The application uses YOLOv8 pre-trained on the COCO dataset to detect:

- **Motorbikes** (class ID: 3) - Green bounding boxes
- **Cars** (class ID: 2) - Blue bounding boxes
- **Trucks** (class IDs: 5, 7 - buses and trucks) - Red bounding boxes

## Database Schema

```javascript
{
  "_id": ObjectId,
  "filename": String,
  "file_type": String,
  "vehicle_counts": {
    "motorbikes": Number,
    "cars": Number,
    "trucks": Number
  },
  "processed_at": DateTime,
  "output_path": String
}
```

## Troubleshooting

### MongoDB Connection Error
- Ensure MongoDB is running
- Check the `MONGODB_URL` in your `.env` file
- Verify network connectivity

### YOLOv8 Model Download
- On first run, YOLOv8 will download the model weights (~6MB)
- Ensure you have internet connectivity
- The model will be cached for future use

### CORS Issues
- The backend is configured to allow all origins
- For production, update CORS settings in `main.py`

## Performance Tips

- For faster video processing, adjust frame skip in `vehicle_detector.py`
- Use smaller YOLOv8 models (yolov8n.pt) for speed
- Use larger models (yolov8x.pt) for better accuracy

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
