// Base URL used for all backend API requests.
// In production, Vite reads VITE_API_BASE_URL from .env.production.
// During local development, it falls back to the local FastAPI server.
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export default API_BASE_URL;