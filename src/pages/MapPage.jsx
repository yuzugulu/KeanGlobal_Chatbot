import { useState } from "react";
import ChatPanel from "../components/ChatPanel";
import MapPanel from "../components/MapPanel";

export default function MapPage() {
  const [showMap, setShowMap] = useState(false);
  const [routeRequest, setRouteRequest] = useState(null);

  return (
    <div className={showMap ? "main-layout two-col" : "main-layout one-col"}>
      <ChatPanel setShowMap={setShowMap} setRouteRequest={setRouteRequest} />
      {showMap && (
        <MapPanel setShowMap={setShowMap} routeRequest={routeRequest} />
      )}
    </div>
  );
}
