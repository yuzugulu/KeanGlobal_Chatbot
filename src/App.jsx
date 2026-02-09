import { useState } from "react";
import Header from "./components/Header";
import ChatPanel from "./components/ChatPanel";
import MapPanel from "./components/MapPanel";

function App() {
  const [showMap, setShowMap] = useState(false);
  const [routeRequest, setRouteRequest] = useState(null);

  return (
    <div className="app">
      <Header />

      <div className={showMap ? "main-layout two-col" : "main-layout one-col"}>
        <ChatPanel setShowMap={setShowMap} setRouteRequest={setRouteRequest} />
        {showMap && <MapPanel setShowMap={setShowMap} routeRequest={routeRequest} />}

      </div>
    </div>
  );
}

export default App;
