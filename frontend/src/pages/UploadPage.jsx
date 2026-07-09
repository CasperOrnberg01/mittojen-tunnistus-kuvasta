import { useNavigate } from "react-router-dom";
import UploadZone from "../components/UploadZone";
import "../styles/upload.css";

export default function UploadPage() {
  const navigate = useNavigate();

  // Main function — handles file selection and sends the file to the backend.
  async function handleSelect(file) {
    console.log("📤 Sending file to backend:", file);

    const formData = new FormData();
    formData.append("file", file);

    // ⭐ IMPORTANT: use the correct endpoint that performs hand measurement
    const response = await fetch("http://127.0.0.1:8000/upload/hand-landmarks-debug", {
      method: "POST",
      body: formData,
    });

    // ⭐ The endpoint returns JPEG, not JSON.
    // Measurements are inside HTTP headers.
    const approxPalmWidth = response.headers.get("X-Approx-Palm-Width-MM");
    const approxHandLength = response.headers.get("X-Approx-Hand-Length-MM");
    const handFound = response.headers.get("X-Hand-Found");
    const a4Found = response.headers.get("X-A4-Found");

    console.log("📥 Backend headers:", {
      approxPalmWidth,
      approxHandLength,
      handFound,
      a4Found
    });

    // ⭐ Build analysis object manually
    const analysis = {
      palmWidth: approxPalmWidth ? Number(approxPalmWidth) : null,
      palmLength: approxHandLength ? Number(approxHandLength) : null,
      handType: handFound === "True" ? "Detected" : "Not detected",
      confidence: handFound === "True" ? 100 : 0,

      // Temporary glove size logic (you can replace later)
      euSize: approxPalmWidth ? Math.round(Number(approxPalmWidth) / 10) : "N/A",
      usSize: approxPalmWidth ? Math.round(Number(approxPalmWidth) / 12) : "N/A",
      fit: "Regular",

      // Debug image (base64)
      debugImageBase64: await response.arrayBuffer()
    };

    console.log("📥 Final analysis object:", analysis);

    // Navigate to PreviewPage and pass file + analysis
    navigate("/preview", {
      state: {
        file,
        analysis
      }
    });
  }

  return (
    <div className="upload-container">
      <div className="upload-left">
        <h1 className="page-title">Upload your hand photo</h1>
        <p className="page-subtitle">
          Take a photo of your hand on an A4 sheet and upload it for precise measurement.
        </p>

        <UploadZone onSelect={handleSelect} />

        <button
          className="manual-upload-btn"
          onClick={() => document.getElementById("manual-upload").click()}
        >
          Upload manually
        </button>

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

        <p className="upload-info">
          Supported formats: JPG, JPEG, PNG<br />
          Max size: 10 MB<br />
          Images are not stored
        </p>
      </div>

      <div className="upload-right">
        <div className="info-card">
          <h2>Hand size measurement tool</h2>
          <p className="info-subtitle">Accurate hand measurement from a single photo</p>
          <p className="info-description">
            Upload a hand photo placed on an A4 sheet for precise sizing.
            Our algorithm detects the sheet, calibrates scale, and measures your hand automatically.
          </p>

          <img
            src="/example-hand.png"
            alt="Example hand on A4"
            className="example-image"
          />
        </div>
      </div>
    </div>
  );
}
