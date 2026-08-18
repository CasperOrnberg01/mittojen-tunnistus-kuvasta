import { useNavigate } from "react-router-dom";
import UploadZone from "../components/UploadZone";
import "../styles/upload.css";
import API_BASE_URL from "../config/api";  // new import for static web app

export default function UploadPage() {
  const navigate = useNavigate();

  // Convert ArrayBuffer → Base64
  function arrayBufferToBase64(buffer) {
    let binary = "";
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;

    for (let i = 0; i < len; i++) {
      binary += String.fromCharCode(bytes[i]);
    }

    return window.btoa(binary);
  }

  async function handleSelect(file) {
    console.log(" Sending file to backend:", file);

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE_URL}/upload/hand-landmarks-debug`, {    // old local "http://127.0.0.1:8000/upload/hand-landmarks-debug"
      method: "POST",
      body: formData,
    });

    // Read headers
    const approxPalmWidth = response.headers.get("X-Approx-Palm-Width-MM");
    const approxHandLength = response.headers.get("X-Approx-Hand-Length-MM");
    const handFound = response.headers.get("X-Hand-Found");
    const a4Found = response.headers.get("X-A4-Found");

    console.log(" Backend headers:", {
      approxPalmWidth,
      approxHandLength,
      handFound,
      a4Found
    });

    // Convert backend image to Base64
    const rawBuffer = await response.arrayBuffer();
    const debugImageBase64 = arrayBufferToBase64(rawBuffer);

    // Build analysis object
    const analysis = {
      palmWidth: approxPalmWidth ? Number(approxPalmWidth) : null,
      palmLength: approxHandLength ? Number(approxHandLength) : null,
      handType: handFound === "True" ? "Detected" : "Not detected",
      confidence: handFound === "True" ? 100 : 0,

      euSize: approxPalmWidth ? Math.round(Number(approxPalmWidth) / 10) : "N/A",
      usSize: approxPalmWidth ? Math.round(Number(approxPalmWidth) / 12) : "N/A",
      fit: "Regular",

      //  Flags for ResultPage
      a4Found: a4Found === "True",
      handFound: handFound === "True",

      //  Backend image
      debugImageBase64
    };

    console.log(" Final analysis object:", analysis);

    navigate("/preview", {
      state: { file, analysis }
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
