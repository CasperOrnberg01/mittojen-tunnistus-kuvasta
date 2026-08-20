# Mittojen tunnistus kuvasta

Image measurement application for estimating hand measurements from a single image.
User places their hand on a A4 paper sheet, without covering any on the A4's corners (top left, top right, bottom right, bottom left)
A4 sheet is used as known size reference object, allowing application to detect the hand and estimate hand length and palm width.


## User flow

```mermaid
flowchart LR
    A[Upload photo] --> B[Preview]
    B --> C[Image analysis]
    C --> D{Detection OK?}
    D -->|Yes| E[Calculate now]
    D -->|No| F[Calibrate A4 manually]
    F --> E
```

## Application Flow

```mermaid
flowchart TD
    A[User uploads hand photo] --> B[React frontend]
    B --> C[FastAPI backend]
    C --> D[A4 sheet detection]
    D --> E[Perspective correction]
    E --> F[Pixel to mm calibration]
    F --> G[MediaPipe hand detection]
    G --> H[Hand segmentation]
    H --> I[Hand measurements]
    I --> J[Results shown in frontend]
```



## Technologies

| Area | Technologies |
|---|---|
| Frontend | React, Vite |
| Backend | Python, FastAPI |
| Image Processing | OpenCV, NumPy |
| Hand Detection | MediaPipe hand landmarks |
| Hand Segmentation (mainly for more accurate palm width) | OpenCV, GrabCut, contours, LAB/HSV colors |
| Deployment | Docker, Microsoft Azure |


## Project structure

```text
mittojen-tunnistus-kuvasta/
├── backend/
│   ├── app/
│   │   ├── models/
│   │   ├── routes/
│   │   ├── services/
│   │   └── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── docs/
├── frontend/
│   ├── public/
│   └── src/
│       ├── components/
│       ├── config/
│       ├── pages/
│       ├── styles/
│       ├── App.jsx
│       ├── main.jsx
│       └── router.jsx
└── README.md
```
