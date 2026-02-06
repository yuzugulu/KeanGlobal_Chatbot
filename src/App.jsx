import { useState } from "react";
import Header from "./components/Header";
import ChatPanel from "./components/ChatPanel";
import MapPanel from "./components/MapPanel";

function App() {
  const [showMap, setShowMap] = useState(false);

  return (
    <div className="app">
      <Header />

      <div className={showMap ? "main-layout two-col" : "main-layout one-col"}>
        <ChatPanel setShowMap={setShowMap} />
       {showMap && <MapPanel setShowMap={setShowMap} />}

      </div>
    </div>
  );
}

export default App;
