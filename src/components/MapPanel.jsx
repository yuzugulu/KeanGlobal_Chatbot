import { useEffect, useMemo, useState } from "react";
import L from "leaflet";
import { CircleMarker, MapContainer, Marker, Polyline, Popup, TileLayer, useMap } from "react-leaflet";
import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

const KEAN_MAIN_CAMPUS = [40.6798, -74.2341];
const CAMPUS_BOUNDS = {
  north: 40.6825,
  south: 40.6772,
  east: -74.2312,
  west: -74.2378
};

const BUILDINGS = [
  { id: "kean_hall", name: "Kean Hall", position: [40.6798, -74.2341] },
  { id: "glassman_hall", name: "Glassman Hall (GLAB)", position: [40.6802, -74.2353] },
  { id: "library", name: "Nancy Thompson Library", position: [40.6791, -74.2328] },
  { id: "stem", name: "STEM Building", position: [40.6804, -74.2332] },
  { id: "downs_hall", name: "Downs Hall", position: [40.6811, -74.2347] },
  { id: "harwood", name: "Harwood Arena", position: [40.6788, -74.2356] },
  { id: "uc", name: "University Center", position: [40.6789, -74.2338] }
];

const CAMPUS_PATHS = [
  ["kean_hall", "library"],
  ["kean_hall", "glassman_hall"],
  ["kean_hall", "uc"],
  ["kean_hall", "harwood"],
  ["glassman_hall", "downs_hall"],
  ["library", "stem"],
  ["library", "uc"],
  ["stem", "downs_hall"],
  ["uc", "harwood"],
  ["uc", "library"],
  ["downs_hall", "kean_hall"]
];

const campusMarkerIcon = L.icon({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
});

function haversineDistanceMeters([lat1, lon1], [lat2, lon2]) {
  const toRadians = value => (value * Math.PI) / 180;
  const earthRadius = 6371000;
  const dLat = toRadians(lat2 - lat1);
  const dLon = toRadians(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRadians(lat1)) * Math.cos(toRadians(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * earthRadius * Math.asin(Math.sqrt(a));
}

function isInsideCampus([lat, lon]) {
  return (
    lat <= CAMPUS_BOUNDS.north &&
    lat >= CAMPUS_BOUNDS.south &&
    lon <= CAMPUS_BOUNDS.east &&
    lon >= CAMPUS_BOUNDS.west
  );
}

function getBuildingById(id) {
  return BUILDINGS.find(building => building.id === id);
}

function getNearestBuilding(position) {
  let nearest = BUILDINGS[0];
  let minDistance = Number.POSITIVE_INFINITY;

  BUILDINGS.forEach(building => {
    const distance = haversineDistanceMeters(position, building.position);
    if (distance < minDistance) {
      minDistance = distance;
      nearest = building;
    }
  });

  return nearest;
}

function buildGraph() {
  const graph = {};

  BUILDINGS.forEach(building => {
    graph[building.id] = [];
  });

  CAMPUS_PATHS.forEach(([fromId, toId]) => {
    const from = getBuildingById(fromId);
    const to = getBuildingById(toId);
    const weight = haversineDistanceMeters(from.position, to.position);
    graph[fromId].push({ id: toId, weight });
    graph[toId].push({ id: fromId, weight });
  });

  return graph;
}

function dijkstra(startId, endId, graph) {
  if (startId === endId) return [startId];

  const distances = {};
  const previous = {};
  const unvisited = new Set(Object.keys(graph));

  Object.keys(graph).forEach(id => {
    distances[id] = Number.POSITIVE_INFINITY;
    previous[id] = null;
  });

  distances[startId] = 0;

  while (unvisited.size > 0) {
    let current = null;
    let bestDistance = Number.POSITIVE_INFINITY;

    unvisited.forEach(id => {
      if (distances[id] < bestDistance) {
        bestDistance = distances[id];
        current = id;
      }
    });

    if (!current || bestDistance === Number.POSITIVE_INFINITY) break;
    if (current === endId) break;

    unvisited.delete(current);

    graph[current].forEach(({ id: neighborId, weight }) => {
      if (!unvisited.has(neighborId)) return;
      const candidate = distances[current] + weight;
      if (candidate < distances[neighborId]) {
        distances[neighborId] = candidate;
        previous[neighborId] = current;
      }
    });
  }

  const route = [];
  let cursor = endId;

  while (cursor) {
    route.unshift(cursor);
    cursor = previous[cursor];
  }

  if (route[0] !== startId) return [];
  return route;
}

function RouteViewport({ routeCoordinates }) {
  const map = useMap();

  useEffect(() => {
    if (routeCoordinates.length < 2) return;
    const bounds = L.latLngBounds(routeCoordinates);
    map.fitBounds(bounds, { padding: [40, 40], maxZoom: 18 });
  }, [map, routeCoordinates]);

  return null;
}

function MapPanel({ setShowMap, routeRequest }) {
  const [startId, setStartId] = useState("kean_hall");
  const [endId, setEndId] = useState("library");
  const [userPosition, setUserPosition] = useState(null);
  const [locationStatus, setLocationStatus] = useState("");

  const graph = useMemo(() => buildGraph(), []);

  const routeBuildingIds = useMemo(() => dijkstra(startId, endId, graph), [endId, graph, startId]);

  const routeCoordinates = useMemo(
    () =>
      routeBuildingIds
        .map(id => getBuildingById(id))
        .filter(Boolean)
        .map(building => building.position),
    [routeBuildingIds]
  );

  const routeDistanceMeters = useMemo(() => {
    if (routeCoordinates.length < 2) return 0;
    let total = 0;
    for (let i = 1; i < routeCoordinates.length; i += 1) {
      total += haversineDistanceMeters(routeCoordinates[i - 1], routeCoordinates[i]);
    }
    return total;
  }, [routeCoordinates]);

  function setMyLocationAsStart() {
    if (!navigator.geolocation) {
      setLocationStatus("Geolocation not supported in this browser.");
      return;
    }

    navigator.geolocation.getCurrentPosition(
      position => {
        const coords = [position.coords.latitude, position.coords.longitude];
        setUserPosition(coords);

        if (!isInsideCampus(coords)) {
          setLocationStatus("You are outside Kean campus bounds. Select buildings manually.");
          return;
        }

        const nearest = getNearestBuilding(coords);
        setStartId(nearest.id);
        setLocationStatus(`Using your location. Nearest start point: ${nearest.name}.`);
      },
      () => {
        setLocationStatus("Could not read your location. Check browser location permission.");
      },
      { enableHighAccuracy: true, timeout: 10000 }
    );
  }

  function swapRoute() {
    setStartId(endId);
    setEndId(startId);
  }

  const startBuilding = getBuildingById(startId);
  const endBuilding = getBuildingById(endId);

  useEffect(() => {
    if (!routeRequest) return;

    if (routeRequest.destinationId && getBuildingById(routeRequest.destinationId)) {
      setEndId(routeRequest.destinationId);
    }

    if (routeRequest.useCurrentLocation) {
      setMyLocationAsStart();
    }
  }, [routeRequest]);

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

      <div className="route-controls">
        <label className="route-field">
          Start
          <select value={startId} onChange={event => setStartId(event.target.value)}>
            {BUILDINGS.map(building => (
              <option key={building.id} value={building.id}>
                {building.name}
              </option>
            ))}
          </select>
        </label>

        <button type="button" className="btn-secondary route-swap" onClick={swapRoute}>
          Swap
        </button>

        <label className="route-field">
          Destination
          <select value={endId} onChange={event => setEndId(event.target.value)}>
            {BUILDINGS.map(building => (
              <option key={building.id} value={building.id}>
                {building.name}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="route-actions">
        <button type="button" className="btn-primary" onClick={setMyLocationAsStart}>
          Use My Location
        </button>
        {locationStatus && <span className="route-note">{locationStatus}</span>}
      </div>

      <div className="route-summary">
        <strong>{startBuilding?.name}</strong> to <strong>{endBuilding?.name}</strong>
        {" - "}
        {routeCoordinates.length > 1
          ? `${Math.round(routeDistanceMeters)} m estimated path`
          : "No route found in campus graph"}
      </div>

      <div className="leaflet-wrapper">
        <MapContainer center={KEAN_MAIN_CAMPUS} zoom={16} className="leaflet-map">
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {BUILDINGS.map(building => (
            <Marker key={building.id} position={building.position} icon={campusMarkerIcon}>
              <Popup>{building.name}</Popup>
            </Marker>
          ))}
          <RouteViewport routeCoordinates={routeCoordinates} />
          {routeCoordinates.length > 1 && (
            <>
              <Polyline positions={routeCoordinates} pathOptions={{ color: "#fdb813", weight: 10, opacity: 0.55 }} />
              <Polyline positions={routeCoordinates} pathOptions={{ color: "#003667", weight: 6, opacity: 0.95 }} />
              <CircleMarker center={routeCoordinates[0]} radius={8} pathOptions={{ color: "#16a34a", fillOpacity: 1 }}>
                <Popup>Route Start</Popup>
              </CircleMarker>
              <CircleMarker
                center={routeCoordinates[routeCoordinates.length - 1]}
                radius={8}
                pathOptions={{ color: "#dc2626", fillOpacity: 1 }}
              >
                <Popup>Route Destination</Popup>
              </CircleMarker>
            </>
          )}
          {userPosition && (
            <CircleMarker center={userPosition} radius={7} pathOptions={{ color: "#fdb813", fillOpacity: 0.9 }}>
              <Popup>Your current location</Popup>
            </CircleMarker>
          )}
        </MapContainer>
      </div>
    </div>
  );
}

export default MapPanel;
