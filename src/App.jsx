import Header from "./components/Header";
import { Routes, Route, Navigate } from "react-router-dom";
import HomePage from "./pages/HomePage";
import MapPage from "./pages/MapPage";
import ProgPage from "./pages/ProgPage";
import ProgDetail from "./pages/ProgDetail";
function App() {
  return (
    <div className="app">
      <Header />

      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/chat" element={<MapPage />} />
        <Route path="/programs" element={<ProgPage />} />
        <Route path="/program/:id" element={<ProgDetail />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}

export default App;
