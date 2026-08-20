// Handles navigation between routes
import { useNavigate } from "react-router-dom";
// Drag & drop upload zone component
import UploadZone from "../components/UploadZone";
// Upload page styles
import "../styles/upload.css";
import API_BASE_URL from "../config/api";  // new import for static web app

export default function UploadPage() {
  // React Router navigation hook
  const navigate = useNavigate();

  // Converts ArrayBuffer to Base64 string for image preview
  function arrayBufferToBase64(buffer) {
    let binary = "";
    const bytes = new Uint8Array(buffer);

    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }

    return window.btoa(binary);
  }

  // Handles file upload, backend request, and navigation to preview
  async function handleSelect(file) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE_URL}/upload/hand-landmarks-debug`, {    // old local "http://127.0.0.1:8000/upload/hand-landmarks-debug"
      method: "POST",
      body: formData,
    });

    // Reads measurement data from backend response headers
    const approxPalmWidth = response.headers.get("X-Approx-Palm-Width-MM");
    const approxHandLength = response.headers.get("X-Approx-Hand-Length-MM");
    const handFound = response.headers.get("X-Hand-Found");
    const a4Found = response.headers.get("X-A4-Found");

    // Converts backend debug image to Base64 for preview
    const rawBuffer = await response.arrayBuffer();
    const debugImageBase64 = arrayBufferToBase64(rawBuffer);

    // Builds analysis object passed to PreviewPage
    const analysis = {
      palmWidth: approxPalmWidth ? Number(approxPalmWidth) : null,
      palmLength: approxHandLength ? Number(approxHandLength) : null,
      handType: handFound === "True" ? "Detected" : "Not detected",
      confidence: handFound === "True" ? 100 : 0,

      euSize: approxPalmWidth ? Math.round(Number(approxPalmWidth) / 10) : "N/A",
      usSize: approxPalmWidth ? Math.round(Number(approxPalmWidth) / 12) : "N/A",
      fit: "Regular",

      a4Found: a4Found === "True",
      handFound: handFound === "True",

      debugImageBase64,
    };

    // Navigates to preview page with file and analysis data
    navigate("/preview", { state: { file, analysis } });
  }

  // Renders upload page layout and content
  return (
    <div className="upload-container">
      {/* Left side: instructions and upload controls */}
      <div className="upload-left">
        <h1 className="page-title">Upload your hand photo</h1>

        <p className="page-subtitle">
          Place your hand on an A4 sheet and take a photo with all four corners clearly visible. Make sure your hand does not cover any corner, and the lighting is even across the entire paper.
        </p>

        {/* Drag & drop upload zone */}
        <UploadZone onSelect={handleSelect} />

        {/* Manual upload button using global primary style */}
        <button
          className="btn btn-primary"
          onClick={() => document.getElementById("manual-upload").click()}
        >
          Upload manually
        </button>

        {/* Hidden file input for manual upload */}
        <input
          id="manual-upload"
          type="file"
          accept="image/*"
          style={{ display: "none" }}
          onChange={(e) => {
            const file = e.target.files[0];
            if (file) handleSelect(file);
          }}
        />

        {/* Upload info text */}
        <p className="upload-info">
          Supported formats: JPG, JPEG, PNG<br />
          Max size: 10 MB<br />
          Images are not stored
        </p>
      </div>

      {/* Right side: info card and example image */}
      <div className="upload-right">
        <div className="info-card">
          <h2>Hand size measurement tool</h2>
          <p className="info-subtitle">Accurate hand measurement from a single photo</p>

          <p className="info-description">
            Upload a hand photo placed on an A4 sheet. The algorithm detects the sheet, calibrates scale, and measures your hand automatically.
          </p>

          <img
            src="/example-hand.jpg"
            alt="Example hand on A4"
            className="example-image"
          />
        </div>
      </div>
    </div>
  );
}
