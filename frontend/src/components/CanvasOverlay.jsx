import { useEffect, useRef, useState } from "react";
import "../styles/calibration.css";

export default function CanvasOverlay({ imageBase64, autoCorners, onCornersSelected }) {

  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  // Initialize corner points (auto-detected or default manual positions)
  const [points, setPoints] = useState(() => {
    if (autoCorners) {
      return {
        topLeft:     { x: autoCorners.top_left[0],     y: autoCorners.top_left[1] },
        topRight:    { x: autoCorners.top_right[0],    y: autoCorners.top_right[1] },
        bottomRight: { x: autoCorners.bottom_right[0], y: autoCorners.bottom_right[1] },
        bottomLeft:  { x: autoCorners.bottom_left[0],  y: autoCorners.bottom_left[1] },
      };
    }

    return {
      topLeft: { x: 100, y: 100 },
      topRight: { x: 300, y: 100 },
      bottomRight: { x: 300, y: 400 },
      bottomLeft: { x: 100, y: 400 },
    };
  });

  const [dragging, setDragging] = useState(null);

  // Load image and draw canvas whenever image or points change
  useEffect(() => {
    if (!imageBase64) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const img = new Image();
    img.src = `data:image/jpeg;base64,${imageBase64}`;

    img.onload = () => {
      imgRef.current = img;

      // Set canvas to real image size
      canvas.width = img.width;
      canvas.height = img.height;

      draw(ctx, img);
    };
  }, [imageBase64, points]);

  // Draw image, polygon and corner points
  function draw(ctx, img) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.drawImage(img, 0, 0);

    ctx.fillStyle = "red";      // manual corner points color
    ctx.strokeStyle = "red";    // manual polygon color
    ctx.lineWidth = 10;         // polygon thickness

    const p = points;

    // Draw polygon connecting the four corners
    ctx.beginPath();
    ctx.moveTo(p.topLeft.x, p.topLeft.y);
    ctx.lineTo(p.topRight.x, p.topRight.y);
    ctx.lineTo(p.bottomRight.x, p.bottomRight.y);
    ctx.lineTo(p.bottomLeft.x, p.bottomLeft.y);
    ctx.closePath();
    ctx.stroke();

    // Draw draggable corner points
    Object.values(p).forEach((pt) => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 25, 0, Math.PI * 2); // point radius
      ctx.fill();
    });

    // Send updated corner positions to parent
    onCornersSelected(points);
  }

  // Convert mouse coordinates to canvas coordinates (accounting for CSS scaling)
  function getMousePos(e) {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();

    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  }

  // Detect if user clicked on a corner point
  function handleMouseDown(e) {
    const { x, y } = getMousePos(e);

    for (const key of Object.keys(points)) {
      const pt = points[key];
      const dx = pt.x - x;
      const dy = pt.y - y;

      // Check if click is inside point radius
      if (dx * dx + dy * dy < 15 * 15) {
        setDragging(key);
        return;
      }
    }
  }

  // Update point position while dragging
  function handleMouseMove(e) {
    if (!dragging) return;

    const { x, y } = getMousePos(e);

    setPoints((prev) => ({
      ...prev,
      [dragging]: { x, y },
    }));
  }

  // Stop dragging on mouse release
  function handleMouseUp() {
    setDragging(null);
  }

  return (
    <canvas
      ref={canvasRef}
      className="calibration-canvas"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    />
  );
}
