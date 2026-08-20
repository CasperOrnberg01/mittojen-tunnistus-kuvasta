import { useLocation, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import "../styles/measurement.css";

/* Measurement page: shows processed hand image and measurement results */
export default function MeasurementPage() {
  const navigate = useNavigate();
  const location = useLocation();

  /* Data passed from previous page */
  const file = location.state?.file;
  const analysis = location.state?.analysis;
  const isPreview = location.state?.preview;
  const previewMessage = location.state?.message;

  /* Log analysis when it changes */
  useEffect(() => {
    console.log("MeasurementPage received analysis:", analysis);
  }, [analysis]);

  const [imageURL, setImageURL] = useState(null);

  /* Create temporary URL for uploaded image */
  useEffect(() => {
    if (!file) return;

    const url = URL.createObjectURL(file);
    setImageURL(url);

    return () => URL.revokeObjectURL(url);
  }, [file]);

  /* Fallback screen if no data */
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

  const results = analysis;

  return (
    <div className="measurement-container">

      {/* Preview mode banner */}
      {isPreview && (
        <div className="preview-banner">
          <strong>Preview mode:</strong> {previewMessage}
        </div>
      )}

      {/* Left side: processed image */}
      <div className="measurement-left">
        <h1 className="measurement-title">Your hand measurement</h1>

        <p className="measurement-subtitle">
          Based on your uploaded photo, here are your hand dimensions.
        </p>

        {results.debugImageBase64 ? (
          <img
            src={`data:image/jpeg;base64,${results.debugImageBase64}`}
            alt="Processed hand"
            className="measurement-image"
          />
        ) : (
          imageURL && (
            <img src={imageURL} alt="Hand" className="measurement-image" />
          )
        )}
      </div>

      {/* Right side: measurement results */}
      <div className="measurement-right">

        {/* Hand details */}
        <div className="result-card">
          <h2>Hand details</h2>

          <div className="result-row">
            <span>Hand type:</span>
            <strong>{results.handType}</strong>
          </div>

          <div className="result-row">
            <span>Palm width:</span>
            <strong>
              {results.palmWidth} mm ({(results.palmWidth / 10).toFixed(1)} cm)
            </strong>
          </div>

          <div className="result-row">
            <span>Hand length:</span>
            <strong>
              {results.palmLength} mm ({(results.palmLength / 10).toFixed(1)} cm)
            </strong>
          </div>

          <div className="result-row">
            <span>Confidence:</span>
            <strong>{results.confidence}%</strong>
          </div>
        </div>

        {/* Glove size */}
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

        {/* Buttons */}
        <button className="buy-btn">
          Go to buy gloves
        </button>

        {/* Yellow button */}
        <button
          className="back-btn"
          onClick={() => navigate("/")}
        >
          Choose another photo
        </button>
      </div>
    </div>
  );
}
