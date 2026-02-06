function MapPanel({ setShowMap }) {
  return (
    <div className="panel map-panel">
      <div className="map-header">
        <h3 className="panel-title">Campus Map</h3>

        <button
          className="btn-secondary"
          onClick={() => setShowMap(false)}
        >
          Close Map
        </button>
      </div>

      <div className="map-placeholder">
        Map will appear here
      </div>
    </div>
  );
}

export default MapPanel;
