import { useRef, useState } from "react";

export default function UploadZone({ onSelect }) {
  const fileInputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState("");

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

  function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file && validateFile(file)) {
      onSelect(file);
    }
  }

  function handleDrop(e) {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && validateFile(file)) {
      onSelect(file);
    }
  }

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

        <input
          type="file"
          accept="image/*"
          ref={fileInputRef}
          style={{ display: "none" }}
          onChange={handleFileSelect}
        />
      </div>

      {error && <p className="upload-error">{error}</p>}
    </div>
  );
}
