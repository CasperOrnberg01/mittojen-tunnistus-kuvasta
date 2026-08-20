import { useLocation, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import "../styles/preview.css";

/* Preview page: shows uploaded image and basic file info */
export default function PreviewPage() {
  const navigate = useNavigate();
  const location = useLocation();

  /* Data passed from UploadPage */
  const file = location.state?.file;
  const analysis = location.state?.analysis;

  const [imageURL, setImageURL] = useState(null);

  /* Log analysis when it changes */
  useEffect(() => {
    if (analysis) {
      console.log("PreviewPage received analysis:", analysis);
    }
  }, [analysis]);

  /* Create temporary URL for uploaded image */
  useEffect(() => {
    if (!file) {
      navigate("/");
      return;
    }

    const url = URL.createObjectURL(file);
    setImageURL(url);

    return () => URL.revokeObjectURL(url);
  }, [file, navigate]);

  /* Replace image → go back to upload page */
  function handleReplace() {
    navigate("/");
  }

  /* Continue → go to ResultPage with file + analysis */
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

        {/* Uploaded image preview */}
        {imageURL && (
          <img src={imageURL} alt="Uploaded preview" className="preview-image" />
        )}

        {/* File information */}
        <div className="preview-info">
          <p><strong>File name:</strong> {file?.name}</p>
          <p><strong>Size:</strong> {(file?.size / 1024 / 1024).toFixed(2)} MB</p>
          <p><strong>Type:</strong> {file?.type}</p>
        </div>

        {/* Action buttons */}
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
