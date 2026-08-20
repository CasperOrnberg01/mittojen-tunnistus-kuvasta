import { useState } from "react";
import CanvasOverlay from "./CanvasOverlay";
import "../styles/calibration.css";
import API_BASE_URL from "../config/api"; // new import for static web app

/* Manual calibration modal: allows user to select A4 corners manually */
export default function ManualCalibrationModal({
  file,
  a4DebugBase64,
  onClose,
  autoCorners,
  onSave,
}) {
  const [corners, setCorners] = useState(null);

  /* Show loading state if debug image is missing */
  if (!a4DebugBase64) {
    return (
      <div className="calibration-modal">
        <div className="calibration-content">
          <h2>Manual A4 Calibration</h2>
          <p>Loading A4 debug image...</p>
        </div>
      </div>
    );
  }

  /* Send selected corners to backend and build new analysis object */
  async function handleSave() {
    if (!corners) return;

    const formData = new FormData();
    formData.append("file", file);

    formData.append("top_left_x", corners.topLeft.x);
    formData.append("top_left_y", corners.topLeft.y);

    formData.append("top_right_x", corners.topRight.x);
    formData.append("top_right_y", corners.topRight.y);

    formData.append("bottom_right_x", corners.bottomRight.x);
    formData.append("bottom_right_y", corners.bottomRight.y);

    formData.append("bottom_left_x", corners.bottomLeft.x);
    formData.append("bottom_left_y", corners.bottomLeft.y);

    const response = await fetch(
      `${API_BASE_URL}/upload/hand-landmarks-manual-debug`,  // old local "http://127.0.0.1:8000/upload/hand-landmarks-manual-debug"
      {
        method: "POST",
        body: formData,
      }
    );

    const headers = response.headers;

    const approxPalmWidth = headers.get("X-Approx-Palm-Width-MM");
    const approxHandLength = headers.get("X-Approx-Hand-Length-MM");
    const handFound = headers.get("X-Hand-Found");

    const buffer = await response.arrayBuffer();
    const base64 = btoa(
      new Uint8Array(buffer).reduce(
        (data, byte) => data + String.fromCharCode(byte),
        ""
      )
    );

    const newAnalysis = {
      palmWidth: approxPalmWidth ? Number(approxPalmWidth) : null,
      palmLength: approxHandLength ? Number(approxHandLength) : null,
      handType: handFound === "True" ? "Detected" : "Not detected",
      confidence: handFound === "True" ? 100 : 0,
      euSize: approxPalmWidth ? Math.round(Number(approxPalmWidth) / 10) : "N/A",
      usSize: approxPalmWidth ? Math.round(Number(approxPalmWidth) / 12) : "N/A",
      fit: "Regular",
      a4Found: true,
      handFound: handFound === "True",
      debugImageBase64: base64,
    };

    onSave(newAnalysis);
  }

  return (
    <div className="calibration-modal">
      <div className="calibration-content">
        <h2>Manual A4 Calibration</h2>

        {/* Canvas overlay for selecting corners */}
        <CanvasOverlay
          imageBase64={a4DebugBase64}
          autoCorners={autoCorners}
          onCornersSelected={(c) => setCorners(c)}
        />

        {/* Modal buttons */}
        <div className="calibration-buttons">
          <button className="modal-btn-cancel" onClick={onClose}>
            Cancel
          </button>

          <button className="modal-btn-save" onClick={handleSave}>
            Save calibration
          </button>
        </div>
      </div>
    </div>
  );
}
