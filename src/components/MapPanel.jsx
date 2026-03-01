import { useCallback, useEffect, useMemo, useState } from "react";
import L from "leaflet";
import { CircleMarker, MapContainer, Marker, Polygon, Polyline, Popup, TileLayer, useMap } from "react-leaflet";
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

const LOCATION_CSV_URL = new URL("../data/campus_locations_main_east_full.csv", import.meta.url).href;
const PARKING_CSV_URL = new URL("../data/kean_parking_lots.csv", import.meta.url).href;
const PATHS_CSV_URL = new URL("../data/campus_edges_walkpaths_balanced_skeleton Final.csv", import.meta.url).href;
const PARKING_TYPE_COLORS = {
  student: "#2563eb",
  faculty_staff: "#f97316",
  overnight: "#16a34a"
};

const LOCATION_TYPE_LABELS = {
  building: "Building",
  entrance: "Entrance",
  parking: "Parking",
  lawn: "Open Space",
  field: "Athletics"
};

const campusMarkerIcon = L.icon({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
});

function normalizeId(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/\uFEFF/g, "")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function toBoolean(value) {
  const normalized = String(value || "").trim().toLowerCase();
  return normalized === "yes" || normalized === "true" || normalized === "1";
}

function parseCsvRow(line) {
  const cells = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    const next = line[i + 1];

    if (char === '"' && inQuotes && next === '"') {
      current += '"';
      i += 1;
      continue;
    }

    if (char === '"') {
      inQuotes = !inQuotes;
      continue;
    }

    if (char === "," && !inQuotes) {
      cells.push(current.trim());
      current = "";
      continue;
    }

    current += char;
  }

  cells.push(current.trim());
  return cells;
}

function parseCsv(text) {
  return text
    .replace(/\r/g, "")
    .split("\n")
    .filter(line => line.trim().length > 0)
    .map(parseCsvRow);
}

function parseLatLngPair(value) {
  const parts = String(value || "")
    .split(",")
    .map(piece => Number(piece.trim()));
  if (parts.length !== 2 || Number.isNaN(parts[0]) || Number.isNaN(parts[1])) return null;
  return [parts[0], parts[1]];
}

function parseLatLngList(value) {
  return String(value || "")
    .split(";")
    .map(parseLatLngPair)
    .filter(Boolean);
}

function parseLocationsCsv(text) {
  const rows = parseCsv(text);
  return rows
    .slice(1)
    .map(columns => {
      const position = parseLatLngPair(columns[2]);
      const id = String(columns[0] || "").replace(/\uFEFF/g, "").trim();
      if (!id || !position) return null;
      return {
        id,
        name: columns[1] || id,
        position,
        campus: columns[3] || "Main",
        routable: toBoolean(columns[4]),
        parent: (columns[5] || "").trim() || null,
        accessibility: toBoolean(columns[6]),
        type: (columns[7] || "").trim().toLowerCase() || "location"
      };
    })
    .filter(Boolean);
}

function parseParkingCsv(text) {
  const rows = parseCsv(text);
  return rows
    .slice(1)
    .map(columns => {
      const id = (columns[0] || "").trim();
      const polygon = parseLatLngList(columns[4]);
      if (!id || polygon.length < 3) return null;
      return {
        id,
        name: columns[1] || id,
        parkingType: (columns[2] || "").trim().toLowerCase(),
        polygon
      };
    })
    .filter(Boolean);
}

function parsePathsCsv(text) {
  const rows = parseCsv(text);
  return rows
    .slice(1)
    .map(columns => {
      const edgeId = (columns[0] || "").replace(/\uFEFF/g, "").trim();
      const fromId = (columns[1] || "").trim();
      const toId = (columns[2] || "").trim();
      if (!edgeId || !fromId || !toId) return null;
      return {
        edgeId,
        fromId,
        toId,
        mode: (columns[3] || "").trim().toLowerCase(),
        pathCoordinates: parseLatLngList(columns[5])
      };
    })
    .filter(Boolean);
}

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

function getLocationById(id, locationsById) {
  return locationsById.get(id);
}

function getNearestLocation(position, locations) {
  let nearest = locations[0];
  let minDistance = Number.POSITIVE_INFINITY;

  locations.forEach(location => {
    const distance = haversineDistanceMeters(position, location.position);
    if (distance < minDistance) {
      minDistance = distance;
      nearest = location;
    }
  });

  return nearest;
}

function buildGraph(routableLocations, locationsById, edges, resolveRoutableId) {
  const graph = {};
  const edgeGeometryByDirection = {};

  routableLocations.forEach(location => {
    graph[location.id] = [];
  });

  edges.forEach(edge => {
    if (edge.mode && edge.mode !== "walk") return;
    const fromId = resolveRoutableId(edge.fromId);
    const toId = resolveRoutableId(edge.toId);
    if (!fromId || !toId || fromId === toId) return;

    const from = getLocationById(fromId, locationsById);
    const to = getLocationById(toId, locationsById);
    if (!from || !to) return;

    let coordinates = edge.pathCoordinates.length > 1 ? edge.pathCoordinates : [from.position, to.position];
    const firstDist = haversineDistanceMeters(coordinates[0], from.position);
    const lastDist = haversineDistanceMeters(coordinates[coordinates.length - 1], from.position);
    if (lastDist < firstDist) {
      coordinates = [...coordinates].reverse();
    }

    let weight = 0;
    for (let i = 1; i < coordinates.length; i += 1) {
      weight += haversineDistanceMeters(coordinates[i - 1], coordinates[i]);
    }

    graph[fromId].push({ id: toId, weight });
    graph[toId].push({ id: fromId, weight });
    edgeGeometryByDirection[`${fromId}=>${toId}`] = coordinates;
    edgeGeometryByDirection[`${toId}=>${fromId}`] = [...coordinates].reverse();
  });

  return { graph, edgeGeometryByDirection };
}

function dijkstra(startId, endId, graph) {
  if (!startId || !endId || !graph[startId] || !graph[endId]) return [];
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

function routeIdsToCoordinates(routeIds, edgeGeometryByDirection, locationsById) {
  if (routeIds.length === 0) return [];
  if (routeIds.length === 1) {
    const only = locationsById.get(routeIds[0]);
    return only ? [only.position] : [];
  }

  const coordinates = [];
  for (let i = 1; i < routeIds.length; i += 1) {
    const fromId = routeIds[i - 1];
    const toId = routeIds[i];
    const segment =
      edgeGeometryByDirection[`${fromId}=>${toId}`] ||
      [locationsById.get(fromId)?.position, locationsById.get(toId)?.position].filter(Boolean);
    if (segment.length === 0) continue;
    if (coordinates.length === 0) {
      coordinates.push(...segment);
    } else {
      coordinates.push(...segment.slice(1));
    }
  }
  return coordinates;
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

function HighlightViewport({ destination, enabled }) {
  const map = useMap();

  useEffect(() => {
    if (!enabled || !destination) return;
    map.setView(destination, 19, { animate: true });
  }, [destination, enabled, map]);

  return null;
}

function MapPanel({ setShowMap, routeRequest }) {
  const [startId, setStartId] = useState("");
  const [endId, setEndId] = useState("");
  const [userPosition, setUserPosition] = useState(null);
  const [locationStatus, setLocationStatus] = useState("");
  const [campusFilter, setCampusFilter] = useState("All Campuses");
  const [directoryQuery, setDirectoryQuery] = useState("");
  const [locationMode, setLocationMode] = useState("directions");
  const [locations, setLocations] = useState([]);
  const [pathEdges, setPathEdges] = useState([]);
  const [parkingLots, setParkingLots] = useState([]);
  const [dataError, setDataError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadCsvData() {
      try {
        const [locationsResponse, parkingResponse, pathsResponse] = await Promise.all([
          fetch(LOCATION_CSV_URL),
          fetch(PARKING_CSV_URL),
          fetch(PATHS_CSV_URL)
        ]);

        if (!locationsResponse.ok || !parkingResponse.ok || !pathsResponse.ok) {
          throw new Error("CSV files could not be loaded.");
        }

        const [locationsText, parkingText, pathsText] = await Promise.all([
          locationsResponse.text(),
          parkingResponse.text(),
          pathsResponse.text()
        ]);

        if (cancelled) return;
        setLocations(parseLocationsCsv(locationsText));
        setParkingLots(parseParkingCsv(parkingText));
        setPathEdges(parsePathsCsv(pathsText));
        setDataError("");
      } catch (error) {
        if (cancelled) return;
        setDataError(error instanceof Error ? error.message : "Map data failed to load.");
      }
    }

    loadCsvData();
    return () => {
      cancelled = true;
    };
  }, []);

  const locationsById = useMemo(() => {
    const map = new Map();
    locations.forEach(location => {
      map.set(location.id, location);
    });
    return map;
  }, [locations]);

  const childrenByParent = useMemo(() => {
    const map = new Map();
    locations.forEach(location => {
      if (!location.parent) return;
      if (!map.has(location.parent)) map.set(location.parent, []);
      map.get(location.parent).push(location.id);
    });
    return map;
  }, [locations]);

  const routableLocationIds = useMemo(() => {
    const set = new Set();
    locations.forEach(location => {
      if (location.routable) set.add(location.id);
    });
    return set;
  }, [locations]);

  const aliasCandidatesByNormalizedId = useMemo(() => {
    const map = new Map();
    const addAlias = (alias, id) => {
      const normalized = normalizeId(alias);
      if (!normalized) return;
      if (!map.has(normalized)) map.set(normalized, []);
      if (!map.get(normalized).includes(id)) map.get(normalized).push(id);
    };

    locations.forEach(location => {
      const id = location.id;
      const normalizedId = normalizeId(id);
      addAlias(id, id);
      addAlias(normalizedId, id);

      const base = normalizedId
        .replace(/_entrance_(front|rear|side)$/g, "")
        .replace(/_(front|rear|side)$/g, "")
        .replace(/_main$/g, "");
      if (base) {
        addAlias(base, id);
        addAlias(`${base}_main`, id);
      }
    });

    return map;
  }, [locations]);

  const resolveRoutableId = useMemo(() => {
    return rawId => {
      if (!rawId) return null;
      const requested = String(rawId).trim();
      if (!requested) return null;

      const candidates = [];
      const pushCandidate = candidate => {
        if (!candidate) return;
        if (!candidates.includes(candidate)) candidates.push(candidate);
      };

      pushCandidate(requested);

      const normalizedCandidates = aliasCandidatesByNormalizedId.get(normalizeId(requested)) || [];
      normalizedCandidates.forEach(pushCandidate);

      candidates.forEach(candidate => {
        const children = childrenByParent.get(candidate) || [];
        children.forEach(pushCandidate);
      });

      for (const candidate of candidates) {
        if (routableLocationIds.has(candidate)) return candidate;
      }
      return null;
    };
  }, [aliasCandidatesByNormalizedId, childrenByParent, routableLocationIds]);

  const routableLocations = useMemo(
    () => locations.filter(location => location.routable).sort((a, b) => a.name.localeCompare(b.name)),
    [locations]
  );

  const buildingMarkers = useMemo(
    () => locations.filter(location => location.type === "building"),
    [locations]
  );

  const entranceMarkers = useMemo(
    () => locations.filter(location => location.type === "entrance"),
    [locations]
  );

  const directoryPlaces = useMemo(() => {
    return locations
      .map(location => {
        const resolvedDestinationId = resolveRoutableId(location.id);
        const parentName = location.parent ? locationsById.get(location.parent)?.name : null;
        return {
          id: location.id,
          name: location.name,
          campus: location.campus || "Main",
          category: LOCATION_TYPE_LABELS[location.type] || "Location",
          description: parentName
            ? `${LOCATION_TYPE_LABELS[location.type] || "Location"} for ${parentName}`
            : `${LOCATION_TYPE_LABELS[location.type] || "Location"} on campus`,
          destinationId: resolvedDestinationId || null
        };
      })
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [locations, locationsById, resolveRoutableId]);

  useEffect(() => {
    if (routableLocations.length === 0) return;

    const defaultStart =
      resolveRoutableId("kean_hall_main") ||
      resolveRoutableId("kean_hall_entrance_front") ||
      routableLocations[0].id;
    const defaultEnd =
      resolveRoutableId("library_main") ||
      resolveRoutableId("library") ||
      routableLocations[Math.min(1, routableLocations.length - 1)].id;

    setStartId(prev => resolveRoutableId(prev) || defaultStart);
    setEndId(prev => resolveRoutableId(prev) || defaultEnd);
  }, [resolveRoutableId, routableLocations]);

  const { graph, edgeGeometryByDirection } = useMemo(
    () => buildGraph(routableLocations, locationsById, pathEdges, resolveRoutableId),
    [locationsById, pathEdges, resolveRoutableId, routableLocations]
  );

  const routeBuildingIds = useMemo(() => dijkstra(startId, endId, graph), [endId, graph, startId]);

  const routeCoordinates = useMemo(
    () => routeIdsToCoordinates(routeBuildingIds, edgeGeometryByDirection, locationsById),
    [edgeGeometryByDirection, locationsById, routeBuildingIds]
  );

  const routeDistanceMeters = useMemo(() => {
    if (routeCoordinates.length < 2) return 0;
    let total = 0;
    for (let i = 1; i < routeCoordinates.length; i += 1) {
      total += haversineDistanceMeters(routeCoordinates[i - 1], routeCoordinates[i]);
    }
    return total;
  }, [routeCoordinates]);

  const displayRouteCoordinates = useMemo(
    () => (locationMode === "highlight" ? [] : routeCoordinates),
    [locationMode, routeCoordinates]
  );

  const campusOptions = useMemo(() => ["All Campuses", ...new Set(directoryPlaces.map(place => place.campus))], [directoryPlaces]);
  const filteredDirectoryPlaces = useMemo(() => {
    const query = directoryQuery.trim().toLowerCase();
    return directoryPlaces.filter(place => {
      const matchesCampus = campusFilter === "All Campuses" || place.campus === campusFilter;
      if (!matchesCampus) return false;
      if (!query) return true;
      return (
        place.name.toLowerCase().includes(query) ||
        place.category.toLowerCase().includes(query) ||
        place.description.toLowerCase().includes(query)
      );
    });
  }, [campusFilter, directoryPlaces, directoryQuery]);

  const setMyLocationAsStart = useCallback(() => {
    if (routableLocations.length === 0) {
      setLocationStatus("Map data is still loading.");
      return;
    }

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

        const nearest = getNearestLocation(coords, routableLocations);
        setStartId(nearest.id);
        setLocationStatus(`Using your location. Nearest start point: ${nearest.name}.`);
      },
      () => {
        setLocationStatus("Could not read your location. Check browser location permission.");
      },
      { enableHighAccuracy: true, timeout: 10000 }
    );
  }, [routableLocations]);

  function swapRoute() {
    setLocationMode("directions");
    setStartId(endId);
    setEndId(startId);
  }
  function routeToPlace(place) {
    const resolvedDestinationId = resolveRoutableId(place.destinationId);
    if (!resolvedDestinationId) {
      setLocationStatus("Route data not set for this location yet. Select a mapped destination above.");
      return;
    }

    setLocationMode("directions");
    setEndId(resolvedDestinationId);
    setShowMap(true);
    setMyLocationAsStart();
  }

  const startBuilding = getLocationById(startId, locationsById);
  const endBuilding = getLocationById(endId, locationsById);

  useEffect(() => {
    if (!routeRequest) return;

    const mappedDestination = resolveRoutableId(routeRequest.destinationId);
    if (mappedDestination) {
      setEndId(mappedDestination);
    }

    const nextMode = routeRequest.locationMode === "directions" ? "directions" : "highlight";
    setLocationMode(nextMode);

    if (routeRequest.useCurrentLocation || nextMode === "directions") {
      setLocationMode("directions");
      setMyLocationAsStart();
    } else if (mappedDestination) {
      const destination = getLocationById(mappedDestination, locationsById);
      if (destination) {
        setStartId(mappedDestination);
        setLocationStatus(`Showing ${destination.name} on the map.`);
      }
    }
  }, [locationsById, resolveRoutableId, routeRequest, setMyLocationAsStart]);

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
            {routableLocations.map(location => (
              <option key={location.id} value={location.id}>
                {location.name}
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
            {routableLocations.map(location => (
              <option key={location.id} value={location.id}>
                {location.name}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="route-actions">
        <button type="button" className="btn-primary" onClick={setMyLocationAsStart}>
          Use My Location
        </button>
        {dataError && <span className="route-note">{dataError}</span>}
        {locationStatus && <span className="route-note">{locationStatus}</span>}
      </div>

      <div className="route-summary">
        <strong>{startBuilding?.name}</strong> to <strong>{endBuilding?.name}</strong>
        {" - "}
        {locationMode === "highlight"
          ? "Showing destination only"
          : routeCoordinates.length > 1
          ? `${Math.round(routeDistanceMeters)} m estimated path`
          : "No route found in campus graph"}
      </div>
      <div className="route-note">
        Parking overlay colors: Student (blue), Faculty/Staff (orange), Overnight (green).
      </div>

      <div className="directory-panel">
        <div className="directory-head">
          <strong>Kean Campus Directory</strong>
          <span>{filteredDirectoryPlaces.length} locations</span>
        </div>
        <div className="directory-controls">
          <select value={campusFilter} onChange={event => setCampusFilter(event.target.value)}>
            {campusOptions.map(option => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
          <input
            value={directoryQuery}
            onChange={event => setDirectoryQuery(event.target.value)}
            placeholder="Search building, service, or campus..."
          />
        </div>
        <div className="directory-list">
          {filteredDirectoryPlaces.map(place => (
            <div key={place.id} className="directory-item">
              <div className="directory-item-main">
                <div className="directory-item-name">{place.name}</div>
                <div className="directory-item-meta">{place.campus} • {place.category}</div>
                <div className="directory-item-desc">{place.description}</div>
              </div>
              <button
                type="button"
                className="btn-secondary directory-route-btn"
                onClick={() => routeToPlace(place)}
                disabled={!place.destinationId}
              >
                {place.destinationId ? "Route" : "Info Only"}
              </button>
            </div>
          ))}
        </div>
      </div>

      <div className="leaflet-wrapper">
        <MapContainer center={KEAN_MAIN_CAMPUS} zoom={18} className="leaflet-map">
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {parkingLots.map(lot => (
            <Polygon
              key={lot.id}
              positions={lot.polygon}
              pathOptions={{
                color: PARKING_TYPE_COLORS[lot.parkingType] || "#64748b",
                fillColor: PARKING_TYPE_COLORS[lot.parkingType] || "#64748b",
                fillOpacity: 0.22,
                weight: 2
              }}
            >
              <Popup>
                <strong>{lot.name}</strong>
                <br />
                {lot.parkingType || "parking"}
              </Popup>
            </Polygon>
          ))}
          {buildingMarkers.map(building => (
            <Marker key={building.id} position={building.position} icon={campusMarkerIcon}>
              <Popup>{building.name}</Popup>
            </Marker>
          ))}
          {entranceMarkers.map(entrance => (
            <CircleMarker
              key={entrance.id}
              center={entrance.position}
              radius={4}
              pathOptions={{ color: "#0f172a", fillColor: "#f8fafc", fillOpacity: 1, weight: 1 }}
            >
              <Popup>{entrance.name}</Popup>
            </CircleMarker>
          ))}
          <RouteViewport routeCoordinates={displayRouteCoordinates} />
          <HighlightViewport destination={endBuilding?.position} enabled={locationMode === "highlight"} />
          {displayRouteCoordinates.length > 1 && (
            <>
              <Polyline positions={displayRouteCoordinates} pathOptions={{ color: "#fdb813", weight: 10, opacity: 0.55 }} />
              <Polyline positions={displayRouteCoordinates} pathOptions={{ color: "#003667", weight: 6, opacity: 0.95 }} />
              <CircleMarker center={displayRouteCoordinates[0]} radius={8} pathOptions={{ color: "#16a34a", fillOpacity: 1 }}>
                <Popup>Route Start</Popup>
              </CircleMarker>
              <CircleMarker
                center={displayRouteCoordinates[displayRouteCoordinates.length - 1]}
                radius={8}
                pathOptions={{ color: "#dc2626", fillOpacity: 1 }}
              >
                <Popup>Route Destination</Popup>
              </CircleMarker>
            </>
          )}
          {locationMode === "highlight" && endBuilding?.position && (
            <CircleMarker center={endBuilding.position} radius={12} pathOptions={{ color: "#dc2626", fillOpacity: 0.25, weight: 3 }}>
              <Popup>{endBuilding.name}</Popup>
            </CircleMarker>
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
