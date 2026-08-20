// React hooks for refs and local state
import { useRef, useState } from "react";

export default function UploadZone({ onSelect }) {
  // Ref to hidden file input element
  const fileInputRef = useRef(null);
  // Drag state for visual feedback
  const [isDragging, setIsDragging] = useState(false);
  // Error message for invalid files
  const [error, setError] = useState("");

  // Validates file type and size before upload
  function validateFile(file) {
    if (!file.type.startsWith("image/")) {
      setError("Only image files are allowed.");
      return false;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError("File is too large. Max size is 10MB.");
      return false;
    }
    setError("");
    return true;
  }

  // Handles file selection from input
  function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file && validateFile(file)) {
      onSelect(file);
    }
  }

  // Handles file drop from drag & drop
  function handleDrop(e) {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && validateFile(file)) {
      onSelect(file);
    }
  }

  // Renders drag & drop zone and hidden file input
  return (
    <div className="upload-zone-wrapper">
      <div
        className={`upload-zone ${isDragging ? "dragging" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current.click()}
      >
        <div className="upload-icon">📤</div>
        <p className="upload-title">Drag & drop your image</p>
        <p className="upload-subtitle">or click to select</p>

        {/* Hidden file input triggered by click */}
        <input
          type="file"
          accept="image/*"
          ref={fileInputRef}
          style={{ display: "none" }}
          onChange={handleFileSelect}
        />
      </div>

      {/* Error message for invalid file selection */}
      {error && <p className="upload-error">{error}</p>}
    </div>
  );
}
