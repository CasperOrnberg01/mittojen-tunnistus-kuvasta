import { useLocation, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import "../styles/preview.css";

/**
 * PreviewPage
 * -----------
 * Displays:
 *  - preview of the uploaded image
 *  - basic file information
 * Allows user to continue to the ResultPage.
 */

export default function PreviewPage() {
  const navigate = useNavigate();
  const location = useLocation();

  // Extracting data passed from UploadPage
  const file = location.state?.file;            // Uploaded image file (Blob)
  const analysis = location.state?.analysis;    // Backend analysis data

  const [imageURL, setImageURL] = useState(null);

  /**
   * Log analysis only once when it changes.
   * Prevents console spam caused by React re-renders.
   */
  useEffect(() => {
    if (analysis) {
      console.log("PreviewPage received analysis:", analysis);
    }
  }, [analysis]);

  /**
   * Create a temporary URL for the uploaded image so it can be displayed.
   * If user opens this page directly → redirect to upload page.
   */
  useEffect(() => {
    if (!file) {
      navigate("/");
      return;
    }

    const url = URL.createObjectURL(file);
    setImageURL(url);

    // Cleanup: revoke URL when component unmounts to avoid memory leaks
    return () => URL.revokeObjectURL(url);
  }, [file, navigate]);

  /**
   * Replace image → return to upload page
   */
  function handleReplace() {
    navigate("/");
  }

  /**
   * Continue → pass file + backend analysis to ResultPage
   */
  function handleContinue() {
    navigate("/result", {
      state: {
        file,
        analysis,
      },
    });
  }

  return (
    <div className="preview-container">
      <div className="preview-card">
        <h1 className="preview-title">Preview your photo</h1>
        <p className="preview-subtitle">
          Make sure your hand is clearly visible and placed on an A4 sheet.
        </p>

        {/* Display uploaded image preview */}
        {imageURL && (
          <img src={imageURL} alt="Uploaded preview" className="preview-image" />
        )}

        {/* File information section */}
        <div className="preview-info">
          <p><strong>File name:</strong> {file?.name}</p>
          <p><strong>Size:</strong> {(file?.size / 1024 / 1024).toFixed(2)} MB</p>
          <p><strong>Type:</strong> {file?.type}</p>
        </div>

        {/* Buttons: replace or continue */}
        <div className="preview-buttons">
          <button className="replace-btn" onClick={handleReplace}>
            Replace image
          </button>

          <button className="continue-btn" onClick={handleContinue}>
            Continue
          </button>
        </div>
      </div>
    </div>
  );
}
