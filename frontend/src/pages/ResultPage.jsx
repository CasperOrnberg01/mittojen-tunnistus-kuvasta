import { useLocation, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import "../styles/result.css";

export default function ResultPage() {
  const navigate = useNavigate();
  const location = useLocation();

  // Extracting data passed from PreviewPage
  const file = location.state?.file;            // Uploaded image file (Blob)
  const analysis = location.state?.analysis;    // ⭐ Backend analysis data

  const [imageURL, setImageURL] = useState(null);

  // Create a temporary URL for the uploaded image so it can be displayed
  useEffect(() => {
    // If user opened this page directly without uploading → redirect
    if (!file) {
      navigate("/");
      return;
    }

    // Convert File object into a browser‑readable temporary URL
    const url = URL.createObjectURL(file);
    setImageURL(url);

    // Cleanup: revoke URL when component unmounts to avoid memory leaks
    return () => URL.revokeObjectURL(url);
  }, [file, navigate]);

  // Continue to MeasurementPage with backend analysis
  function handleCalculate() {
    console.log("🔍 ResultPage received analysis:", analysis);

    navigate("/measurement", {
      state: {
        file,
        analysis,   // ⭐ Passing backend data forward
        preview: false
      }
    });
  }

  return (
    <div className="result-container">
      <div className="result-left">
        <h1 className="result-title">Image analysis</h1>
        <p className="result-subtitle">
          Your photo has been processed. Review the detection results below.
        </p>

        {/* Display uploaded image */}
        {imageURL && (
          <img src={imageURL} alt="Uploaded" className="result-image" />
        )}
      </div>

      <div className="result-right">
        <h2 className="result-section-title">Detection status</h2>

        <div className="status-list">

          {/* A4 detection status */}
          {/* If analysis exists → backend successfully detected A4 */}
          <div className={`status-item ${analysis ? "success" : "fail"}`}>
            <span className="status-icon">✓</span>
            A4 detected
          </div>

          {/* Hand detection status */}
          {/* Backend confirms hand landmarks were found */}
          <div className={`status-item ${analysis ? "success" : "fail"}`}>
            <span className="status-icon">✓</span>
            Hand detected
          </div>

          {/* Always ready to calculate if previous steps succeeded */}
          <div className="status-item success">
            <span className="status-icon">✓</span>
            Ready to calculate
          </div>
        </div>

        {/* Continue to measurement page */}
        <button className="calculate-btn" onClick={handleCalculate}>
          Calculate Now
        </button>
      </div>
    </div>
  );
}
