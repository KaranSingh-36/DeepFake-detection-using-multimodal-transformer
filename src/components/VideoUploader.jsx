import { useRef, useState, useEffect } from "react";
import { downloadAnalysisReport } from "../utils/downloadAnalysisReport";

function VideoUploader({ onAnalysisComplete }) {
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [analysisTimestamp, setAnalysisTimestamp] = useState(null);

  const handleFile = (file) => {
    if (!file) return;

    const allowedTypes = ["video/mp4", "video/quicktime", "video/x-msvideo"];
    if (!allowedTypes.includes(file.type)) {
      alert("Please select an MP4, MOV, or AVI video.");
      return;
    }

    const maxSize = 500 * 1024 * 1024;
    if (file.size > maxSize) {
      alert("Video size must be less than 500 MB.");
      return;
    }

    setVideoUrl((previousUrl) => {
      if (previousUrl) URL.revokeObjectURL(previousUrl);
      return URL.createObjectURL(file);
    });

    setSelectedFile(file);
    setAnalysisResult(null);
    setAnalysisTimestamp(null);
    setIsAnalyzing(false);
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    handleFile(file);
    event.target.value = "";
  };

  const handleChooseVideo = () => {
    fileInputRef.current?.click();
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (event) => {
    event.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    handleFile(event.dataTransfer.files[0]);
  };

  const handleAnalyze = () => {
    if (!selectedFile) {
      alert("Please select a video first.");
      return;
    }

    setIsAnalyzing(true);
    setAnalysisResult(null);

    // Temporary frontend demo. Replace with the teammate-owned API request later.
    window.setTimeout(() => {
      const result = {
        prediction: "DEEPFAKE",
        confidence: 96.8,
        processingTime: 12.4,
      };
      const analyzedAt = new Date().toLocaleString();

      setIsAnalyzing(false);
      setAnalysisResult(result);
      setAnalysisTimestamp(analyzedAt);

      onAnalysisComplete?.({
        id: `${Date.now()}-${selectedFile.name}`,
        fileName: selectedFile.name,
        prediction: result.prediction,
        confidence: result.confidence,
        processingTime: result.processingTime,
        analyzedAt,
        isDemo: true,
      });
    }, 2500);
  };

  const handleDownloadReport = () => {
    if (!analysisResult || !selectedFile) {
      alert("Please analyze a video before downloading a report.");
      return;
    }

    downloadAnalysisReport({
      id: `${Date.now()}-${selectedFile.name}`,
      fileName: selectedFile.name,
      prediction: analysisResult.prediction,
      confidence: analysisResult.confidence,
      processingTime: analysisResult.processingTime,
      analyzedAt: analysisTimestamp || new Date().toLocaleString(),
      isDemo: true,
    });
  };

  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  const isDeepfake = analysisResult?.prediction?.toUpperCase() === "DEEPFAKE";

  return (
    <>
      <input
        ref={fileInputRef}
        type="file"
        accept="video/mp4,video/quicktime,video/x-msvideo"
        onChange={handleFileChange}
        hidden
      />

      <div className="analysis-layout">
        {/* LEFT SIDE - VIDEO UPLOAD */}
        <div className="upload-panel">
          {!selectedFile ? (
            <div
              className={`upload-area ${isDragging ? "dragging" : ""}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <div className="upload-icon">↑</div>
              <h4>{isDragging ? "Drop your video here" : "Drop your video here"}</h4>
              <p>or click to browse from your computer</p>
              <span className="file-info">MP4, MOV, AVI • Maximum 500 MB</span>
              <button className="secondary-button" onClick={handleChooseVideo}>
                Choose Video
              </button>
            </div>
          ) : (
            <>
              <div className="video-preview">
                <video src={videoUrl} controls preload="metadata" />
              </div>
              <div className="selected-file-info">
                <strong>{selectedFile.name}</strong>
                <span>{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</span>
              </div>
              <div className="video-actions">
                <button
                  className="secondary-button"
                  onClick={handleChooseVideo}
                  disabled={isAnalyzing}
                >
                  Change Video
                </button>
                <button
                  className="primary-analyze-button"
                  onClick={handleAnalyze}
                  disabled={isAnalyzing}
                >
                  {isAnalyzing ? "Analyzing..." : "Analyze Video →"}
                </button>
              </div>
            </>
          )}
        </div>

        {/* RIGHT SIDE - ANALYSIS RESULT */}
        <div className="result-panel">
          {!analysisResult && !isAnalyzing && (
            <div className="empty-result">
              <div className="result-placeholder-icon">◉</div>
              <h3>Analysis Result</h3>
              <p>
                Upload a video and click <strong>Analyze Video</strong> to see the
                detection result.
              </p>
            </div>
          )}

          {isAnalyzing && (
            <div className="empty-result">
              <div className="loading-spinner"></div>
              <h3>Analyzing Video...</h3>
              <p>Processing visual and audio information</p>
              <div className="analysis-progress"><div></div></div>
            </div>
          )}

          {analysisResult && (
            <>
              <div className="result-title-row">
                <div>
                  <span className="section-label">ANALYSIS RESULT</span>
                  <h3>Detection Result</h3>
                </div>
                <button
                  className="report-button"
                  onClick={handleDownloadReport}
                  title="Download the report for this result"
                >
                  ↓ Report
                </button>
              </div>

              <div className={`result-alert ${isDeepfake ? "deepfake-alert" : "real-alert"}`}>
                <div className="alert-icon">{isDeepfake ? "!" : "✓"}</div>
                <div>
                  <strong>{analysisResult.prediction}</strong>
                  <p>
                    {isDeepfake
                      ? "This video is likely AI-generated or manipulated."
                      : "This video appears to be authentic based on the model analysis."}
                  </p>
                  <p className="demo-label">
                    DEMO RESULT — not a real model prediction
                  </p>
                </div>
              </div>

              <div className="result-information">
                <div className="confidence-circle">
                  <div>
                    <strong>{analysisResult.confidence}%</strong>
                    <span>Demo Confidence</span>
                  </div>
                </div>

                <div className="result-table">
                  <div><span>Prediction</span><strong className={isDeepfake ? "danger-text" : "success-text"}>{analysisResult.prediction}</strong></div>
                  <div><span>Confidence</span><strong>{analysisResult.confidence}%</strong></div>
                  <div><span>Processing Time</span><strong>{analysisResult.processingTime}s</strong></div>
                  <div><span>Model Used</span><strong>Multimodal Transformer (demo)</strong></div>
                  <div><span>Architecture</span><strong>ViT + Wav2Vec2 + Cross-Attention</strong></div>
                  <div><span>Dataset</span><strong>LAV-DF</strong></div>
                </div>
              </div>

              <div className="preprocessing-box">
                <h4>⚙ Preprocessing Information</h4>
                <div className="preprocessing-grid">
                  <div className="processing-card visual-processing">
                    <strong>▣ Visual Processing</strong>
                    <ul>
                      <li>8 frames (equal spacing)</li>
                      <li>Resize: 224 × 224</li>
                      <li>ImageNet normalization</li>
                    </ul>
                  </div>
                  <div className="processing-card audio-processing">
                    <strong>♫ Audio Processing</strong>
                    <ul>
                      <li>Mono audio</li>
                      <li>16 kHz sampling rate</li>
                      <li>4 seconds (64,000 samples)</li>
                      <li>Padding / truncation</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="another-video-container">
                <button className="secondary-button" onClick={handleChooseVideo}>
                  Analyze Another Video
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}

export default VideoUploader;
