import { useLocation, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import ManualCalibrationModal from "../components/ManualCalibrationModal";
import "../styles/result.css";
import API_BASE_URL from "../config/api"; // new import for static web app

/* Result page: shows uploaded image, debug image, detection status, and calibration modal */
export default function ResultPage() {
  const navigate = useNavigate();
  const location = useLocation();

  /* Data passed from previous page */
  const file = location.state?.file;
  const analysis = location.state?.analysis;

  /* Local state */
  const [imageURL, setImageURL] = useState(null);
  const [a4DebugBase64, setA4DebugBase64] = useState(null);
  const [autoCorners, setAutoCorners] = useState(null);
  const [showCalibrationModal, setShowCalibrationModal] = useState(false);

  /* Log analysis when it changes */
  useEffect(() => {
    if (analysis) {
      console.log("FULL ANALYSIS:", analysis);
    }
  }, [analysis]);

  /* Converts ArrayBuffer to Base64 string */
  function arrayBufferToBase64(buffer) {
    let binary = "";
    const bytes = new Uint8Array(buffer);

    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }

    return window.btoa(binary);
  }

  /* Create temporary URL for uploaded image */
  useEffect(() => {
    if (!file) return;

    const url = URL.createObjectURL(file);
    setImageURL(url);

    return () => URL.revokeObjectURL(url);
  }, [file]);

  /* Fetch A4 debug image + corner headers */
  useEffect(() => {
    async function loadA4Debug() {
      if (!file) return;

      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE_URL}/upload/debug`, {  // old local: "http://127.0.0.1:8000/upload/debug"
        method: "POST",
        body: formData,
      });

      /* Log all headers */
      const headerObj = {};
      response.headers.forEach((value, key) => {
        headerObj[key] = value;
      });
      console.log("Backend headers:", headerObj);

      /* Read auto-detected A4 corners */
      const tl = response.headers.get("x-a4-top-left");
      const tr = response.headers.get("x-a4-top-right");
      const br = response.headers.get("x-a4-bottom-right");
      const bl = response.headers.get("x-a4-bottom-left");

      console.log("RAW A4 HEADERS:", { tl, tr, br, bl });

      if (tl && tr && br && bl) {
        const parsed = {
          top_left: tl.split(",").map(Number),
          top_right: tr.split(",").map(Number),
          bottom_right: br.split(",").map(Number),
          bottom_left: bl.split(",").map(Number),
        };
        setAutoCorners(parsed);
      } else {
        console.warn("A4 corners not found in headers");
      }

      /* Read JPEG debug image */
      const buffer = await response.arrayBuffer();
      const base64 = arrayBufferToBase64(buffer);
      setA4DebugBase64(base64);
    }

    loadA4Debug();
  }, [file]);

  /* Log auto corners when they change */
  useEffect(() => {
    if (autoCorners) {
      console.log("PARSED AUTO CORNERS:", autoCorners);
    }
  }, [autoCorners]);

  /* Log debug image when loaded */
  useEffect(() => {
    if (a4DebugBase64) {
      console.log("A4 debug loaded, base64 length:", a4DebugBase64.length);
    }
  }, [a4DebugBase64]);

  /* Fallback screen if no data */
  if (!file || !analysis) {
    return (
      <div className="result-container">
        <div className="empty-message">
          <h1>No data</h1>
          <p>Please upload a photo first.</p>
          <button className="back-btn" onClick={() => navigate("/")}>
            Go to upload
          </button>
        </div>
      </div>
    );
  }

  /* Detection flags */
  const a4Ok = analysis?.a4Found === true;
  const handOk = analysis?.handFound === true;
  const measurementOk =
    handOk &&
    analysis?.palmLength != null &&
    analysis?.palmWidth != null &&
    analysis?.confidence >= 50;

  return (
    <div className="result-container">

      {/* Top section: text + status + actions */}
      <div className="result-top">
        <h1 className="result-title">Image analysis</h1>
        <p className="result-subtitle">Your photo has been processed.</p>

        <h2 className="result-section-title">Detection status</h2>

        <div className="status-list">
          <div className={`status-item ${a4Ok ? "success" : "fail"}`}>
            <span className="status-icon">{a4Ok ? "✓" : "✖"}</span>
            A4 detected
          </div>

          <div className={`status-item ${handOk ? "success" : "fail"}`}>
            <span className="status-icon">{handOk ? "✓" : "✖"}</span>
            Hand detected
          </div>

          <div className={`status-item ${measurementOk ? "success" : "fail"}`}>
            <span className="status-icon">{measurementOk ? "✓" : "✖"}</span>
            Measurements ready
          </div>
        </div>

        <div className="result-actions">
          <button
            className="calculate-btn"
            onClick={() =>
              navigate("/measurement", {
                state: { file, analysis, preview: false },
              })
            }
          >
            Calculate Now
          </button>

          <button
            className="manual-btn"
            disabled={!a4DebugBase64}
            onClick={() => setShowCalibrationModal(true)}
          >
            Calibrate A4 manually
          </button>

          <button className="back-btn" onClick={() => navigate(-1)}>
            ← Back
          </button>
        </div>
      </div>

      {/* Bottom section: two images on the same level */}
      <div className="result-images-row">
        {imageURL && (
          <img src={imageURL} alt="Uploaded" className="result-image" />
        )}

        {a4DebugBase64 ? (
          <img
            src={`data:image/jpeg;base64,${a4DebugBase64}`}
            alt="A4 debug"
            className="result-image"
          />
        ) : (
          <p>Loading A4 debug image...</p>
        )}
      </div>

      {/* Manual calibration modal */}
      {showCalibrationModal && a4DebugBase64 && (
        <ManualCalibrationModal
          file={file}
          a4DebugBase64={a4DebugBase64}
          autoCorners={autoCorners}
          onClose={() => setShowCalibrationModal(false)}
          onSave={(newAnalysis) =>
            navigate("/measurement", {
              state: { file, analysis: newAnalysis, preview: false },
            })
          }
        />
      )}
    </div>
  );
}
