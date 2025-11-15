import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadedFilename, setUploadedFilename] = useState('');
  const [fileType, setFileType] = useState('');
  const [detectionResults, setDetectionResults] = useState(null);
  const [outputFilename, setOutputFilename] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [savedId, setSavedId] = useState('');

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setMessage(`Selected: ${file.name}`);
      setDetectionResults(null);
      setOutputFilename('');
      setSavedId('');
    }
  };

  const handleUploadAndRun = async () => {
    if (!selectedFile) {
      setMessage('Please select a file first');
      return;
    }

    setLoading(true);
    setMessage('Uploading file...');

    try {
      // Step 1: Upload file
      const formData = new FormData();
      formData.append('file', selectedFile);

      const uploadResponse = await axios.post(
        `${API_BASE_URL}/api/vehicles/upload`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      if (uploadResponse.data.success) {
        const filename = uploadResponse.data.filename;
        const fileType = uploadResponse.data.file_type;
        setUploadedFilename(filename);
        setFileType(fileType);
        setMessage('File uploaded. Running detection...');

        // Step 2: Run detection
        const detectResponse = await axios.post(
          `${API_BASE_URL}/api/vehicles/detect/${filename}`
        );

        if (detectResponse.data.success) {
          setDetectionResults(detectResponse.data.vehicle_counts);
          setOutputFilename(detectResponse.data.output_filename);
          setMessage('Detection completed successfully!');
        }
      }
    } catch (error) {
      setMessage(`Error: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    if (!detectionResults || !uploadedFilename) {
      setMessage('No detection results to save');
      return;
    }

    setLoading(true);
    setMessage('Saving to database...');

    try {
      const saveResponse = await axios.post(
        `${API_BASE_URL}/api/vehicles/save`,
        null,
        {
          params: {
            filename: uploadedFilename,
            output_filename: outputFilename,
            vehicle_counts: JSON.stringify(detectionResults),
          },
        }
      );

      if (saveResponse.data.success) {
        setSavedId(saveResponse.data.id);
        setMessage(`Saved successfully! ID: ${saveResponse.data.id}`);
      }
    } catch (error) {
      setMessage(`Error saving: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (outputFilename) {
      window.open(`${API_BASE_URL}/api/vehicles/download/${outputFilename}`, '_blank');
    }
  };

  return (
    <div className="App">
      <div className="container">
        <h1>Vehicle Statistics Program</h1>
        <p className="subtitle">Detect and count vehicles using YOLOv8</p>

        <div className="card">
          <div className="file-input-section">
            <label htmlFor="file-input" className="file-label">
              Browse File
            </label>
            <input
              id="file-input"
              type="file"
              accept="image/*,video/*"
              onChange={handleFileSelect}
              className="file-input"
            />
            {selectedFile && (
              <p className="file-name">{selectedFile.name}</p>
            )}
          </div>

          <div className="button-group">
            <button
              onClick={handleUploadAndRun}
              disabled={!selectedFile || loading}
              className="btn btn-primary"
            >
              {loading ? 'Processing...' : 'Run Detection'}
            </button>

            <button
              onClick={handleSave}
              disabled={!detectionResults || loading || savedId}
              className="btn btn-success"
            >
              {savedId ? 'Saved ✓' : 'Save to Database'}
            </button>
          </div>

          {message && (
            <div className={`message ${message.includes('Error') ? 'error' : 'success'}`}>
              {message}
            </div>
          )}
        </div>

        {detectionResults && (
          <div className="results-card">
            <h2>Detection Results</h2>
            <div className="stats-grid">
              <div className="stat-item motorbike">
                <div className="stat-icon">🏍️</div>
                <div className="stat-label">Motorbikes</div>
                <div className="stat-value">{detectionResults.motorbikes}</div>
              </div>
              <div className="stat-item car">
                <div className="stat-icon">🚗</div>
                <div className="stat-label">Cars</div>
                <div className="stat-value">{detectionResults.cars}</div>
              </div>
              <div className="stat-item truck">
                <div className="stat-icon">🚚</div>
                <div className="stat-label">Trucks</div>
                <div className="stat-value">{detectionResults.trucks}</div>
              </div>
            </div>

            {outputFilename && (
              <div className="output-section">
                <h3>Processed Output</h3>
                {fileType && ['jpg', 'jpeg', 'png', 'bmp'].includes(fileType) ? (
                  <img
                    src={`${API_BASE_URL}/outputs/${outputFilename}`}
                    alt="Detection result"
                    className="result-image"
                  />
                ) : (
                  <video
                    src={`${API_BASE_URL}/outputs/${outputFilename}`}
                    controls
                    className="result-video"
                  />
                )}
                <button onClick={handleDownload} className="btn btn-download">
                  Download Result
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
