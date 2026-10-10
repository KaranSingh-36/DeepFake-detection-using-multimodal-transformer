
import { useEffect, useState } from "react";
import VideoUploader from "./components/VideoUploader";
import { downloadAnalysisReport } from "./utils/downloadAnalysisReport";







function App() {
  const [activeSection, setActiveSection] = useState("home");

  const [analysisHistory, setAnalysisHistory] = useState(() => {
    try {
      const savedHistory = localStorage.getItem(
        "fusiontrace-analysis-history"
      );

      if (!savedHistory) return [];

      const parsedHistory = JSON.parse(savedHistory);

      return Array.isArray(parsedHistory) ? parsedHistory : [];
    } catch (error) {
      console.error("Could not load analysis history:", error);
      return [];
    }
  });

  const [selectedAnalysis, setSelectedAnalysis] = useState(null);

  useEffect(() => {
    try {
      localStorage.setItem(
        "fusiontrace-analysis-history",
        JSON.stringify(analysisHistory)
      );
    } catch (error) {
      console.error("Could not save analysis history:", error);
    }
  }, [analysisHistory]);

  function handleAnalysisComplete(result) {
    setAnalysisHistory((previous) => [result, ...previous]);
  }
  function handleClearHistory() {
  const confirmed = window.confirm(
    "Are you sure you want to clear all saved analysis history? This cannot be undone."
  );

  if (!confirmed) return;

  setAnalysisHistory([]);
  setSelectedAnalysis(null);
  localStorage.removeItem("fusiontrace-analysis-history");
}

  // Keep the rest of your existing App.jsx code here.

  const navItems = [
    { id: "home", icon: "⌂", label: "Home" },
    { id: "video-analysis", icon: "▶", label: "Video Analysis" },
    { id: "analysis-history", icon: "◷", label: "Analysis History" },
    { id: "model-information", icon: "▣", label: "Model Information" },
    { id: "dataset", icon: "▤", label: "Dataset" },
    { id: "about-project", icon: "ⓘ", label: "About Project" },
    { id: "contact", icon: "✉", label: "Contact" },
  ];

  const sectionTitles = {
    home: "Home",
    "video-analysis": "Video Analysis",
    "analysis-history": "Analysis History",
    "model-information": "Model Information",
    dataset: "Dataset",
    "about-project": "About Project",
    contact: "Contact",
  };

  function navigateTo(id) {
    setActiveSection(id);

    document.getElementById(id)?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }

  return (
    <div className="app">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="logo">
          <div className="logo-icon">🛡</div>
          <div>
            <h2>FusionTrace AI</h2>
            <span>Multimodal Detection</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          {navItems.map((item) => (
            <a
              key={item.id}
              href={`#${item.id}`}
              className={`nav-item ${
                activeSection === item.id ? "active" : ""
              }`}
              onClick={(event) => {
                event.preventDefault();
                navigateTo(item.id);
              }}
            >
              <span>{item.icon}</span>
              {item.label}
            </a>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="security-box">
            <div className="status-dot"></div>
            <div>
              <strong>System Secure</strong>
              <p>Detection service active</p>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        {/* Top Navbar */}
        <header className="topbar" id="home">
          <div>
            <p className="breadcrumb">
              Dashboard / {sectionTitles[activeSection]}
            </p>
            <h1>Deepfake Detection Dashboard</h1>
          </div>

          <div className="model-status">
            <span className="online-dot"></span>
            <div>
              <strong>Model Status: Online</strong>
              <small>Multimodal Transformer • 25K</small>
            </div>
          </div>
        </header>

        {/* Hero Section */}
        <section className="hero">
          <div className="hero-content">
            <span className="hero-label">
              AI-POWERED MEDIA VERIFICATION
            </span>

            <h2>
              Detect Deepfakes,
              <br />
              <span>Strengthen Digital Trust.</span>
            </h2>

            <p>
              Analyze video content using our multimodal transformer
              architecture combining visual and audio information.
            </p>

            <button
              className="primary-button"
              onClick={() => navigateTo("video-analysis")}
            >
              Analyze a Video →
            </button>
          </div>

          <div className="hero-visual">
            <div className="scan-circle">
              <div className="shield-large">🛡</div>
              <div className="scan-line"></div>
            </div>
          </div>
        </section>

        {/* Stats */}
        <section className="stats-grid">
          <div className="stat-card">
            <div className="stat-icon blue">◉</div>
            <div>
              <span>Model Accuracy</span>
              <strong>96.93%</strong>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-icon purple">◈</div>
            <div>
              <span>ROC-AUC</span>
              <strong>0.9988</strong>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-icon green">✓</div>
            <div>
              <span>Training Samples</span>
              <strong>25,000+</strong>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-icon orange">⚡</div>
            <div>
              <span>Detection Type</span>
              <strong>Audio + Video</strong>
            </div>
          </div>
        </section>

        {/* Video Analysis */}
        <section
          id="video-analysis"
          className="video-analysis-section"
        >
          <div className="video-analysis-header">
            <div>
              <span className="section-label">VIDEO ANALYSIS</span>
              <h3>Upload Video</h3>
              <p>
                Upload a video and let the multimodal transformer
                analyze visual and audio information for deepfake
                detection.
              </p>
            </div>
          </div>

          <VideoUploader onAnalysisComplete={handleAnalysisComplete} />
        </section>

        {/* Analysis History */}
        
{/* Analysis History */}
<section id="analysis-history" className="recent-section">
  
<div className="section-heading">
  <div>
    <span className="section-label">SESSION ACTIVITY</span>
    <h3>Analysis History</h3>
    <span className="section-note">
      {analysisHistory.length} saved result(s)
    </span>
  </div>

  <button
    className="text-button"
    onClick={handleClearHistory}
    disabled={analysisHistory.length === 0}
    title="Clear all saved analysis history"
  >
    Clear History
  </button>
</div>


  <div className="table-card">
    <div className="table-header">
      <span>VIDEO</span>
      <span>RESULT</span>
      <span>CONFIDENCE</span>
      <span>TIME</span>
      <span>ACTION</span>
    </div>

    {analysisHistory.length === 0 ? (
      <div className="history-empty">
        <h4>No analyses in this session yet</h4>
        <p>
          Upload a video and run an analysis to see it listed here.
        </p>
        <button
          className="primary-button"
          onClick={() => navigateTo("video-analysis")}
        >
          Analyze a Video →
        </button>
      </div>
    ) : (
      analysisHistory.map((item) => (
        <div className="table-row" key={item.id}>
          <div className="history-video-name">
            <strong>{item.fileName}</strong>
            <small className="demo-label">
              DEMO RESULT · Not a real model prediction
            </small>
          </div>

          <span
            className={`result ${
              item.prediction === "REAL" ? "real" : "fake"
            }`}
          >
            ● {item.prediction}
          </span>

          <span>{item.confidence}%</span>
          <span>{item.processingTime}s</span>

          <button onClick={() => setSelectedAnalysis(item)}>
            View
          </button>
        </div>
      ))
    )}
  </div>

  <p className="section-note">
    History is stored in React state for this page session.
    Demo results are simulated and must not be treated as verified
    deepfake detections.
  </p>

  {selectedAnalysis && (
    <div className="table-card history-details">
      <div className="section-heading">
        <h3>Analysis Details</h3>
        <button onClick={() => setSelectedAnalysis(null)}>
          Close
        </button>
      </div>

      <p>
        <strong>Video:</strong> {selectedAnalysis.fileName}
      </p>
      <p>
        <strong>Prediction:</strong> {selectedAnalysis.prediction}
      </p>
      <p>
        <strong>Confidence:</strong> {selectedAnalysis.confidence}%
      </p>
      <p>
        <strong>Processing time:</strong>{" "}
        {selectedAnalysis.processingTime}s
      </p>
      <p>
        <strong>Analyzed at:</strong> {selectedAnalysis.analyzedAt}
      </p>
      <button
  className="primary-button"
  onClick={() => downloadAnalysisReport(selectedAnalysis)}
>
  Download Report ↓
</button>
      <p className="demo-label">
        DEMO RESULT — generated by the frontend, not the trained model.
      </p>
    </div>
  )}
</section>


        {/* Model Information */}
        <section id="model-information" className="recent-section">
          <div className="section-heading">
            <div>
              <span className="section-label">MODEL ARCHITECTURE</span>
              <h3>Model Information</h3>
            </div>
          </div>

          <div className="table-card info-card">
            <h4>Multimodal Transformer</h4>
            <p>
              FusionTrace AI is designed to combine visual and audio
              features to classify media as REAL or DEEPFAKE.
            </p>
            <ul>
              <li>Visual feature extraction: Vision Transformer (ViT)</li>
              <li>Audio feature extraction: Wav2Vec2</li>
              <li>Feature fusion: Bidirectional cross-attention</li>
              <li>Output: Binary media classification</li>
            </ul>
            <p className="section-note">
              Displayed architecture describes the project design;
              live inference depends on the connected model service.
            </p>
          </div>
        </section>

        {/* Dataset */}
        <section id="dataset" className="recent-section">
          <div className="section-heading">
            <div>
              <span className="section-label">DATA & EVALUATION</span>
              <h3>Dataset</h3>
            </div>
          </div>

          <div className="table-card info-card">
            <h4>LAV-DF</h4>
            <p>
              The project uses the LAV-DF dataset for multimodal
              audio-video deepfake detection experiments.
            </p>
            <p>
              Dataset samples are processed into video frames and
              audio inputs before being passed to the model.
            </p>
            <p className="section-note">
              Dataset counts and evaluation metrics should be updated
              to reflect the final verified experiment.
            </p>
          </div>
        </section>

        {/* About Project */}
        <section id="about-project" className="recent-section">
          <div className="section-heading">
            <div>
              <span className="section-label">OUR PROJECT</span>
              <h3>About FusionTrace AI</h3>
            </div>
          </div>

          <div className="table-card info-card">
            <p>
              FusionTrace AI is an academic cybersecurity project
              focused on detecting manipulated media through
              multimodal transformer architecture.
            </p>
            <p>
              By combining visual and audio information, the system
              aims to support media authenticity analysis.
            </p>
          </div>
        </section>

        {/* Contact */}
        <section id="contact" className="recent-section">
          <div className="section-heading">
            <div>
              <span className="section-label">GET IN TOUCH</span>
              <h3>Contact & Project Support</h3>
            </div>
          </div>

          <div className="table-card info-card">
            <p>
              For project questions, implementation details, or
              academic collaboration, contact the project team.
            </p>
            <p>
              <strong>Project:</strong> Deepfake Detection Using
              Multimodal Transformer Architecture for Cybersecurity
              Applications
            </p>
            <p className="section-note">
              Karan Singh (2501940063@krmu.edu.in), Tushar Singh (2501940055@krmu.edu.in)
            </p>
          </div>
        </section>

        {/* Footer */}
        <footer>
          FusionTrace AI • Multimodal Deepfake Detection System
          <span>Built for Cybersecurity Research</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
