import { useEffect, useRef, useState } from "react";
import "../styles/calibration.css";

export default function CanvasOverlay({ imageBase64, autoCorners, onCornersSelected }) {

  const canvasRef = useRef(null);
  const imgRef = useRef(null);

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

  useEffect(() => {
    if (!imageBase64) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const img = new Image();
    img.src = `data:image/jpeg;base64,${imageBase64}`;

    img.onload = () => {
      imgRef.current = img;

      // Set REAL canvas size
      canvas.width = img.width;
      canvas.height = img.height;

      draw(ctx, img);
    };
  }, [imageBase64, points]);

  function draw(ctx, img) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.drawImage(img, 0, 0);

    ctx.fillStyle = "red";
    ctx.strokeStyle = "red";
    ctx.lineWidth = 3;

    const p = points;

    ctx.beginPath();
    ctx.moveTo(p.topLeft.x, p.topLeft.y);
    ctx.lineTo(p.topRight.x, p.topRight.y);
    ctx.lineTo(p.bottomRight.x, p.bottomRight.y);
    ctx.lineTo(p.bottomLeft.x, p.bottomLeft.y);
    ctx.closePath();
    ctx.stroke();

    Object.values(p).forEach((pt) => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 8, 0, Math.PI * 2);
      ctx.fill();
    });

    onCornersSelected(points);
  }

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

  function handleMouseDown(e) {
    const { x, y } = getMousePos(e);

    for (const key of Object.keys(points)) {
      const pt = points[key];
      const dx = pt.x - x;
      const dy = pt.y - y;
      if (dx * dx + dy * dy < 15 * 15) {
        setDragging(key);
        return;
      }
    }
  }

  function handleMouseMove(e) {
    if (!dragging) return;

    const { x, y } = getMousePos(e);

    setPoints((prev) => ({
      ...prev,
      [dragging]: { x, y },
    }));
  }

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
