import { useLocation, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import "../styles/measurement.css";

export default function MeasurementPage() {
  const navigate = useNavigate();
  const location = useLocation();

  const file = location.state?.file;
  const analysis = location.state?.analysis;
  const isPreview = location.state?.preview;
  const previewMessage = location.state?.message;

  useEffect(() => {
    console.log("📏 MeasurementPage received analysis:", analysis);
  }, [analysis]);


  const [imageURL, setImageURL] = useState(null);

  useEffect(() => {
    if (!file) return;

    const url = URL.createObjectURL(file);
    setImageURL(url);

    return () => URL.revokeObjectURL(url);
  }, [file]);

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

      {isPreview && (
        <div className="preview-banner">
          <strong>Preview mode:</strong> {previewMessage}
        </div>
      )}

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
            <span>Hand length:</span>
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

        <button className="buy-btn">
          Go to buy gloves
        </button>

        {/* button for buying gloves */}
        <button
          className="buy-btn"
          onClick={() => navigate("/")}
        >
          Choose another photo
        </button>
      </div>
    </div>
  );
}
