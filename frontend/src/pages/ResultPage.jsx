import { useLocation, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import ManualCalibrationModal from "../components/ManualCalibrationModal";
import "../styles/result.css";

/**
 * ResultPage
 * -----------
 * Displays:
 *  - original uploaded image
 *  - A4 debug overlay image
 *  - detection status (A4, hand, measurements)
 *  - auto‑detected A4 corners
 *  - manual calibration modal
 *
 * This page receives:
 *  - file (original uploaded image)
 *  - analysis (hand measurement results)
 *
 * It also fetches:
 *  - A4 debug image from backend
 *  - A4 auto‑detected corners from response headers
 */

export default function ResultPage() {
  const navigate = useNavigate();
  const location = useLocation();

  // Data passed from previous page
  const file = location.state?.file;
  const analysis = location.state?.analysis;

  // Local state
  const [imageURL, setImageURL] = useState(null);
  const [a4DebugBase64, setA4DebugBase64] = useState(null);
  const [autoCorners, setAutoCorners] = useState(null);
  const [showCalibrationModal, setShowCalibrationModal] = useState(false);

  /**
   * Log analysis only when it changes.
   * Prevents console spam caused by React re-renders.
   */
  useEffect(() => {
    if (analysis) {
      console.log("FULL ANALYSIS:", analysis);
    }
  }, [analysis]);

  /**
   * Convert ArrayBuffer → Base64 string
   * Used for converting backend JPEG debug image.
   */
  function arrayBufferToBase64(buffer) {
    let binary = "";
    const bytes = new Uint8Array(buffer);

    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }

    return window.btoa(binary);
  }

  /**
   * Create a temporary URL for the uploaded image
   * and clean it up when component unmounts.
   */
  useEffect(() => {
    if (!file) return;

    const url = URL.createObjectURL(file);
    setImageURL(url);

    return () => URL.revokeObjectURL(url);
  }, [file]);

  /**
   * Fetch A4 debug image + A4 corner headers from backend.
   * This runs only once when the file changes.
   */
  useEffect(() => {
    async function loadA4Debug() {
      if (!file) return;

      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://127.0.0.1:8000/upload/debug", {
        method: "POST",
        body: formData,
      });

      /**
       * Log all headers once (clean, readable)
       */
      const headerObj = {};
      response.headers.forEach((value, key) => {
        headerObj[key] = value;
      });
      console.log(" Backend headers:", headerObj);

      /**
       * Read auto-detected A4 corners from headers
       */
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
        console.warn("⚠ A4 corners not found in headers");
      }

      /**
       * Read JPEG debug image body
       */
      const buffer = await response.arrayBuffer();
      const base64 = arrayBufferToBase64(buffer);

      setA4DebugBase64(base64);
    }

    loadA4Debug();
  }, [file]);

  /**
   * Log auto corners only when they change
   */
  useEffect(() => {
    if (autoCorners) {
      console.log("PARSED AUTO CORNERS:", autoCorners);
    }
  }, [autoCorners]);

  /**
   * Log debug image only when it loads
   */
  useEffect(() => {
    if (a4DebugBase64) {
      console.log("A4 debug loaded, base64 length:", a4DebugBase64.length);
    }
  }, [a4DebugBase64]);

  /**
   * If no data was passed, show fallback screen
   */
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

  // Detection flags
  const a4Ok = analysis?.a4Found === true;
  const handOk = analysis?.handFound === true;
  const measurementOk =
    handOk &&
    analysis?.palmLength != null &&
    analysis?.palmWidth != null &&
    analysis?.confidence >= 50;

  return (
    <div className="result-container">

      {/* LEFT SIDE — images */}
      <div className="result-left">
        <h1 className="result-title">Image analysis</h1>
        <p className="result-subtitle">Your photo has been processed.</p>

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

        <button className="back-btn" onClick={() => navigate(-1)}>
          ← Back
        </button>
      </div>

      {/* RIGHT SIDE — status + actions */}
      <div className="result-right">
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
        </div>
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
