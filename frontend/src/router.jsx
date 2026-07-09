import { BrowserRouter, Routes, Route } from "react-router-dom";
import UploadPage from "./pages/UploadPage";
import PreviewPage from "./pages/PreviewPage";
import ResultPage from "./pages/ResultPage";
import MeasurementPage from "./pages/MeasurementPage";

export default function AppRouter() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<UploadPage />} />
        <Route path="/preview" element={<PreviewPage />} />
        <Route path="/result" element={<ResultPage />} />
        <Route path="/measurement" element={<MeasurementPage />} />
      </Routes>
    </BrowserRouter>
  );
}
