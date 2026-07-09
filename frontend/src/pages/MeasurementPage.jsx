import { useLocation, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import "../styles/measurement.css";

export default function MeasurementPage() {
  const navigate = useNavigate();
  const location = useLocation();



  // Extracting data passed through React Router navigation state.
  // These values are provided by ResultPage via navigate("/measurement", { state: {...} }).
  const file = location.state?.file;          // Uploaded image file (Blob)
  const analysis = location.state?.analysis;  // ⭐ Real backend analysis data (JSON)
  const isPreview = location.state?.preview;  // Preview mode flag (boolean)
 
  // Useful for verifying that navigation state is correctly forwarded.
  console.log("📏 MeasurementPage received analysis:", analysis);
  const previewMessage = location.state?.message; // Preview message text (string)


  const [imageURL, setImageURL] = useState(null);

  // Create a temporary URL for the uploaded image so it can be displayed.
  // URL.createObjectURL() converts the File object into a browser‑readable URL.
  useEffect(() => {
    if (!file) return;

    const url = URL.createObjectURL(file);
    setImageURL(url);

    // Cleanup: revoke URL when component unmounts.
    // Prevents memory leaks caused by unused object URLs.
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // Fallback: if user opened this page directly (without uploading an image).
  // This prevents the page from breaking when accessed without navigation state.
  if (!file || !analysis) {
    return (
      <div className="measurement-container">
        <div className="empty-message">
          <h1>No measurement data</h1>
          <p>You opened this page directly. Please upload a photo first.</p>
          <button className="buy-btn" onClick={() => navigate("/")}>
            Go to upload
          </button>
        </div>
      </div>
    );
  }

  // ⭐ The backend response is used directly as measurement results.
  // No transformation is needed — the backend already returns final values.
  const results = analysis;

  return (
    <div className="measurement-container">

      {/* Preview mode banner */}
      {/* Displayed only when the user is in preview mode (before final confirmation). */}
      {isPreview && (
        <div className="preview-banner">
          <strong>Preview mode:</strong> {previewMessage}
        </div>
      )}

      {/* Left side: image + title */}
      {/* Shows the uploaded hand image and basic page description. */}
      <div className="measurement-left">
        <h1 className="measurement-title">Your hand measurement</h1>
        <p className="measurement-subtitle">
          Based on your uploaded photo, here are your hand dimensions.
        </p>

        {imageURL && (
          <img src={imageURL} alt="Hand" className="measurement-image" />
        )}
      </div>

      {/* Right side: measurement results */}
      {/* All values come directly from backend analysis JSON. */}
      <div className="measurement-right">
        <div className="result-card">
          <h2>Hand details</h2>

          <div className="result-row">
            <span>Hand type:</span>
            <strong>{results.handType}</strong>
          </div>

          <div className="result-row">
            <span>Palm width:</span>
            <strong>{results.palmWidth} mm</strong>
          </div>

          <div className="result-row">
            <span>Palm length:</span>
            <strong>{results.palmLength} mm</strong>
          </div>

          <div className="result-row">
            <span>Confidence:</span>
            <strong>{results.confidence}%</strong>
          </div>
        </div>

        <div className="result-card">
          <h2>Recommended glove size</h2>

          <div className="result-row">
            <span>EU size:</span>
            <strong>{results.euSize}</strong>
          </div>

          <div className="result-row">
            <span>US size:</span>
            <strong>{results.usSize}</strong>
          </div>

          <div className="result-row">
            <span>Fit:</span>
            <strong>{results.fit}</strong>
          </div>
        </div>

        {/* Placeholder button for future e‑commerce integration */}
        <button className="buy-btn">
          Go to buy gloves
        </button>
      </div>
    </div>
  );
}
