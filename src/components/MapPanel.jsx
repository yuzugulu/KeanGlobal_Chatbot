import { Fragment, useCallback, useEffect, useMemo, useState } from "react";
import L from "leaflet";
import { CircleMarker, MapContainer, Marker, Polygon, Polyline, Popup, TileLayer, useMap } from "react-leaflet";
import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";
import buildingProfiles from "../data/building_profiles.json";
import campusLocationsData from "../data/kean_locations.json";
import missingPathsData from "../data/kean_paths_missing.json";
import routingGraphData from "../data/kean_routing_graph.json";
import parkingLotsGeoJsonRaw from "../data/kean_parking_lots.geojson?raw";
import pathsGeoJsonRaw from "../data/kean_paths.geojson?raw";

const buildingImageModules = import.meta.glob("../assets/buildings/*.{png,jpg,jpeg,webp,gif}", {
  eager: true,
  import: "default"
});

const KEAN_MAIN_CAMPUS = [40.6798, -74.2341];
const CAMPUS_BOUNDS = {
  north: 40.6825,
  south: 40.6772,
  east: -74.2312,
  west: -74.2378
};
const EXPANDED_CAMPUS_BOUNDS = {
  north: 40.6855,
  south: 40.674,
  east: -74.221,
  west: -74.2415
};

const PARKING_TYPE_COLORS = {
  student: "#2563eb",
  faculty_staff: "#f97316",
  overnight: "#16a34a"
};
const ROUTE_LINE_COLORS = ["#2563eb", "#f97316", "#9333ea"];

const LOCATION_TYPE_LABELS = {
  building: "Building",
  entrance: "Entrance",
  parking: "Parking",
  lawn: "Open Space",
  field: "Athletics"
};

const ROUTE_NODE_ALIASES = {
  "downs_hall_entrance_front": "downs_hall_main",
  "kean_hall_entrance_front": "kean_hall_main",
  "kean_hall_entrance_side": "kean_hall_main",
  "miron_center_main": "miron_main",
  "naab_entrance_front": "naab_main",
  "north_ave_academic_main": "naab_main",
  "vaughn_eames_front": "vaughn_eames_main",
  "wilkins_theatre_front": "wilkins_theatre_main"
};

const IMAGE_TOKEN_STOP_WORDS = new Set([
  "building",
  "hall",
  "center",
  "campus",
  "main",
  "new",
  "university",
  "residence",
  "student",
  "academic",
  "for",
  "and",
  "the",
  "of",
  "at"
]);

const BUILDING_IMAGE_ALIASES = {
  administration: ["administration building"],
  barnes_nobles_glab: ["kean barnes and noble", "barnes and noble at glab", "barnes noble glab", "bookstore at glab"],
  bartlett_hall: ["bartlett hall"],
  basketball_courts: ["kean basketball courts", "basketball courts", "basketball court"],
  burch_hall: ["burch hall"],
  cas: ["cas building", "center for academic success"],
  cougar_hall: ["cougar hall"],
  d_angola_gym: ["d angola gym", "dangola gym", "d'angola gymnasium", "d angola gymnasium"],
  d_angola_pool: ["d angola pool", "dangola pool", "d'angola pool", "pool"],
  enlow_hall_lawn: ["the lawn at enlow hall", "enlow hall lawn", "the lawn"],
  dougall_hall: ["dougall hall"],
  downs_hall: ["downs hall"],
  east_campus_building: ["east campus", "east campus building"],
  freshman_residence_hall: ["freshman hall", "freshman residence hall"],
  glab: ["glab", "green lane academic building"],
  harwood_arena: ["harwood arena"],
  hennings_hall: ["george hennings hall", "hennings hall"],
  hennings_research: ["george hennings research", "hennings research building"],
  hutchinson_hall: ["hutchinson hall"],
  kean_hall: ["kean hall"],
  library: ["nancy thompson library", "library"],
  miron_center: ["miron student center", "miron center"],
  msc_turf_field: ["miron turf field", "msc turf field", "miron student center turf field", "turf field"],
  naab: ["naab", "north avenue academic building"],
  nathan_weiss_building: ["nathan weiss east campus", "nathan weiss building"],
  rogers_hall: ["rogers hall"],
  sozio_hall: ["sozio hall"],
  stem: ["stem building", "stem"],
  townsend_hall: ["townsend hall"],
  union_train_station: ["union train station", "union township train station", "train station"],
  upperclassman_residence_hall: ["upperclassmen hall", "upperclassman residence hall"],
  vaughn_eames: ["vaughn eames hall", "vaughn-eames hall"],
  whiteman_hall: ["whiteman hall"],
  wilkins_theatre: ["wilkins theater", "wilkins theatre"]
};

const parkingLotsGeoJson = JSON.parse(parkingLotsGeoJsonRaw);
const pathsGeoJson = JSON.parse(pathsGeoJsonRaw);
const buildingImageRecords = buildBuildingImageRecords(buildingImageModules);
const HIDDEN_BUILDING_MARKER_IDS = new Set(["george_hennings_research"]);

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

function stripImageExtension(value) {
  return String(value || "").replace(/\.[^.]+$/, "");
}

function tokenizeMeaningfulName(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/&/g, " and ")
    .replace(/['’]/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .split(/\s+/)
    .filter(token => token && !IMAGE_TOKEN_STOP_WORDS.has(token));
}

function buildBuildingImageRecords(modules) {
  return Object.entries(modules).map(([path, src]) => {
    const filename = path.split("/").pop() || "";
    const normalizedName = normalizeId(stripImageExtension(filename));
    const tokens = new Set(tokenizeMeaningfulName(filename));
    return { path, src, filename, normalizedName, tokens };
  });
}

function scoreImageCandidate(candidate, image) {
  if (!candidate) return 0;
  const normalized = normalizeId(candidate);
  const tokens = tokenizeMeaningfulName(candidate);
  let score = 0;

  if (normalized && image.normalizedName === normalized) score += 200;
  if (normalized && image.normalizedName.includes(normalized)) score += 120;
  if (normalized && normalized.includes(image.normalizedName)) score += 80;

  const overlap = tokens.filter(token => image.tokens.has(token)).length;
  if (overlap) {
    score += overlap * 40;
    score += Math.round((overlap / Math.max(1, tokens.length)) * 30);
  }

  return score;
}

function toBoolean(value) {
  const normalized = String(value || "").trim().toLowerCase();
  return normalized === "yes" || normalized === "true" || normalized === "1";
}

function prettifyRouteNodeName(id) {
  return String(id || "")
    .replace(/\bmain\d*\b/gi, "Main")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, letter => letter.toUpperCase());
}

function isWithinBounds([lat, lon], bounds) {
  return lat <= bounds.north && lat >= bounds.south && lon <= bounds.east && lon >= bounds.west;
}

function normalizeCoordinatePair(value) {
  if (!Array.isArray(value) || value.length < 2) return null;

  const first = Number(value[0]);
  const second = Number(value[1]);
  if (Number.isNaN(first) || Number.isNaN(second)) return null;

  const asLatLng = [first, second];
  const asLngLat = [second, first];

  const latLngInCampus = isWithinBounds(asLatLng, EXPANDED_CAMPUS_BOUNDS);
  const lngLatInCampus = isWithinBounds(asLngLat, EXPANDED_CAMPUS_BOUNDS);

  if (latLngInCampus && !lngLatInCampus) return asLatLng;
  if (lngLatInCampus && !latLngInCampus) return asLngLat;

  if (Math.abs(first) <= 90 && Math.abs(second) <= 180) return asLatLng;
  if (Math.abs(second) <= 90 && Math.abs(first) <= 180) return asLngLat;
  return null;
}

function parseLatLngPair(value) {
  if (Array.isArray(value)) {
    return normalizeCoordinatePair(value);
  }

  const parts = String(value || "")
    .split(",")
    .map(piece => Number(piece.trim()));
  if (parts.length !== 2 || Number.isNaN(parts[0]) || Number.isNaN(parts[1])) return null;
  return [parts[0], parts[1]];
}

function getLocationPosition(location) {
  if (Number.isFinite(location?.latitude) && Number.isFinite(location?.longitude)) {
    return [Number(location.latitude), Number(location.longitude)];
  }

  const latLng = parseLatLngPair(location?.latlng);
  if (latLng) return latLng;

  return normalizeCoordinatePair(location?.coordinates);
}

function parseLocationRecords(data) {
  return data
    .map(location => {
      const position = getLocationPosition(location);
      const id = String(location?.id || "").replace(/\uFEFF/g, "").trim();
      if (!id || !position) return null;
      return {
        id,
        name: location?.name || id,
        position,
        campus: location?.campus || "Main",
        routable: toBoolean(location?.routable),
        parent: (location?.parent || "").trim() || null,
        accessibility: toBoolean(location?.accessibility),
        type: (location?.type || "location").trim().toLowerCase()
      };
    })
    .filter(Boolean);
}

function parseRoutingNodes(graphData, existingLocations) {
  const existingIds = new Set(existingLocations.map(location => location.id));
  return (graphData?.nodes || [])
    .map(node => {
      const id = String(node?.id || "").trim();
      const position = normalizeCoordinatePair(node?.coordinates);
      if (!id || !position || existingIds.has(id)) return null;
      return {
        id,
        name: prettifyRouteNodeName(id),
        position,
        campus: isWithinBounds(position, CAMPUS_BOUNDS) ? "Main" : "Other",
        routable: true,
        parent: null,
        accessibility: true,
        type: "route_node"
      };
    })
    .filter(Boolean);
}

function parseParkingGeoJson(data) {
  return (data?.features || [])
    .map(feature => {
      const ring = feature?.geometry?.coordinates?.[0];
      const polygon = Array.isArray(ring) ? ring.map(normalizeCoordinatePair).filter(Boolean) : [];
      if (polygon.length < 3) return null;

      const properties = feature?.properties || {};
      const name = properties.name || properties.parking_id || "Parking";
      return {
        id: String(properties.parking_id || normalizeId(name)),
        name,
        parkingType: String(properties.parking_type || "").trim().toLowerCase() || inferParkingTypeFromName(name),
        polygon
      };
    })
    .filter(Boolean);
}

function parsePathFeatures(data) {
  return (data?.features || [])
    .map(feature => {
      const properties = feature?.properties || {};
      const pathCoordinates = Array.isArray(feature?.geometry?.coordinates)
        ? feature.geometry.coordinates.map(normalizeCoordinatePair).filter(Boolean)
        : [];

      const edgeId = String(properties.edge_id || "").replace(/\uFEFF/g, "").trim();
      const fromId = String(properties.from_id || "").trim();
      const toId = String(properties.to_id || "").trim();
      if (!edgeId || !fromId || !toId) return null;

      return {
        edgeId,
        fromId,
        toId,
        mode: String(properties.mode || "").trim().toLowerCase(),
        pathCoordinates,
        notes: String(properties.notes || "").trim(),
        source: String(properties.source || "").trim().toLowerCase()
      };
    })
    .filter(Boolean);
}

function parseMissingPathRecords(data) {
  return (data || [])
    .map(edge => {
      const edgeId = String(edge?.edge_id || "").replace(/\uFEFF/g, "").trim();
      const fromId = String(edge?.from_id || "").trim();
      const toId = String(edge?.to_id || "").trim();
      if (!edgeId || !fromId || !toId) return null;

      return {
        edgeId,
        fromId,
        toId,
        mode: String(edge?.mode || "").trim().toLowerCase(),
        pathCoordinates: [],
        notes: String(edge?.notes || "").trim(),
        source: "auto_proximity"
      };
    })
    .filter(Boolean);
}

function formatParkingTypeLabel(parkingType) {
  const normalized = String(parkingType || "").trim().toLowerCase();
  if (normalized === "faculty_staff") return "Faculty/Staff Parking";
  if (normalized === "overnight") return "Overnight Parking";
  if (normalized === "visitor") return "Visitor Parking";
  if (normalized === "student") return "Student Parking";
  return "Parking";
}

function inferParkingTypeFromName(name) {
  const text = String(name || "").toLowerCase();
  if (text.includes("overnight")) return "overnight";
  if (text.includes("faculty") || text.includes("staff")) return "faculty_staff";
  if (text.includes("visitor")) return "visitor";
  return "student";
}

function getBuildingImage(building) {
  const candidates = [
    building?.id,
    building?.name,
    ...(BUILDING_IMAGE_ALIASES[normalizeId(building?.id)] || [])
  ].filter(Boolean);

  let bestImage = null;
  let bestScore = 0;

  for (const image of buildingImageRecords) {
    let imageScore = 0;
    for (const candidate of candidates) {
      imageScore = Math.max(imageScore, scoreImageCandidate(candidate, image));
    }
    if (imageScore > bestScore) {
      bestScore = imageScore;
      bestImage = image;
    }
  }

  return bestScore >= 60 ? bestImage?.src || null : null;
}

function createSquareAroundPoint([lat, lon], delta = 0.00012) {
  return [
    [lat + delta, lon - delta],
    [lat + delta, lon + delta],
    [lat - delta, lon + delta],
    [lat - delta, lon - delta]
  ];
}

function getBuildingProfile(building) {
  const profile = buildingProfiles[building.id];
  if (profile) return profile;

  const isResidence = /hall|residence/i.test(building.name);
  if (isResidence) {
    return {
      usage: "Residential facility for campus housing.",
      departments: ["Housing and Residence Life"],
      notes: "Contact Housing and Residence Life for occupancy and assignment details."
    };
  }

  return {
    usage: "Campus building used for academic, student, or operational functions.",
    departments: ["Details available from campus directory"],
    notes: "Detailed profile for this building is being added."
  };
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
  return isWithinBounds([lat, lon], CAMPUS_BOUNDS);
}

function getLocationById(id, locationsById) {
  return locationsById.get(id);
}

function getLocationDisplayName(location, locationsById) {
  if (!location) return "";
  if (location.type !== "entrance") return location.name;

  const parent = location.parent ? locationsById.get(location.parent) : null;
  if (!parent) return location.name;

  const parentName = parent.name;
  const rawName = String(location.name || "");
  const withoutParent = rawName
    .replace(parentName, "")
    .replace(/\s+/g, " ")
    .trim();

  if (!withoutParent) return parentName;

  const normalizedSuffix = withoutParent
    .replace(/^entrance\s*/i, "")
    .replace(/\s+/g, " ")
    .trim();

  if (!normalizedSuffix) return `${parentName} Main Entrance`;

  const prettySuffix = normalizedSuffix
    .replace(/\bfront(\d+)\b/i, "Front Entrance $1")
    .replace(/\brear(\d+)\b/i, "Rear Entrance $1")
    .replace(/\bside(\d+)\b/i, "Side Entrance $1")
    .replace(/\bfront\b/i, "Front Entrance")
    .replace(/\brear\b/i, "Rear Entrance")
    .replace(/\bside\b/i, "Side Entrance")
    .replace(/\bmain\b/i, "Main Entrance")
    .trim();

  return `${parentName} - ${prettySuffix}`;
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

function resolveEdgeNodeId(rawId, locationsById, routableLocationIds, childrenByParent, resolveRoutableId) {
  const requested = String(rawId || "").trim();
  if (!requested) return null;

  if (routableLocationIds.has(requested)) return requested;

  const aliasedRequested = ROUTE_NODE_ALIASES[requested];
  if (aliasedRequested && routableLocationIds.has(aliasedRequested)) {
    return aliasedRequested;
  }

  const location = locationsById.get(requested);
  if (location) {
    const candidates = [];

    const pushRoutable = candidateId => {
      if (!candidateId || !routableLocationIds.has(candidateId)) return;
      const candidate = locationsById.get(candidateId);
      if (!candidate || !candidate.position || !location.position) return;
      const distance = haversineDistanceMeters(location.position, candidate.position);
      candidates.push({ id: candidateId, distance });
    };

    const siblingParent = location.parent || null;
    if (siblingParent && childrenByParent.has(siblingParent)) {
      childrenByParent.get(siblingParent).forEach(pushRoutable);
    }
    if (childrenByParent.has(requested)) {
      childrenByParent.get(requested).forEach(pushRoutable);
    }

    if (candidates.length > 0) {
      candidates.sort((a, b) => a.distance - b.distance);
      return candidates[0].id;
    }
  }

  return resolveRoutableId(requested);
}

function buildGraph(routableLocations, locationsById, edges, resolveRoutableId, routableLocationIds, childrenByParent) {
  const graph = {};
  const edgeGeometryByKey = {};

  routableLocations.forEach(location => {
    graph[location.id] = [];
  });

  edges.forEach(edge => {
    if (edge.mode && edge.mode !== "walk") return;
    const fromId = resolveEdgeNodeId(edge.fromId, locationsById, routableLocationIds, childrenByParent, resolveRoutableId);
    const toId = resolveEdgeNodeId(edge.toId, locationsById, routableLocationIds, childrenByParent, resolveRoutableId);
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
    const hasTracedGeometry = edge.pathCoordinates.length > 1;
    if (!hasTracedGeometry) {
      const sourcePenalty = edge.source === "auto_proximity" ? 5.0 : 3.5;
      weight *= sourcePenalty;
    }

    const forwardEdgeKey = `${edge.edgeId}:${fromId}=>${toId}`;
    const reverseEdgeKey = `${edge.edgeId}:${toId}=>${fromId}`;

    graph[fromId].push({ id: toId, weight, edgeKey: forwardEdgeKey });
    graph[toId].push({ id: fromId, weight, edgeKey: reverseEdgeKey });
    edgeGeometryByKey[forwardEdgeKey] = coordinates;
    edgeGeometryByKey[reverseEdgeKey] = [...coordinates].reverse();
  });

  return { graph, edgeGeometryByKey };
}

function dijkstra(startId, endId, graph) {
  if (!startId || !endId || !graph[startId] || !graph[endId]) return { routeIds: [], edgeKeys: [] };
  if (startId === endId) return { routeIds: [startId], edgeKeys: [] };

  const distances = {};
  const previous = {};
  const unvisited = new Set(Object.keys(graph));

  Object.keys(graph).forEach(id => {
    distances[id] = Number.POSITIVE_INFINITY;
    previous[id] = { nodeId: null, edgeKey: null };
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

    graph[current].forEach(({ id: neighborId, weight, edgeKey }) => {
      if (!unvisited.has(neighborId)) return;
      const candidate = distances[current] + weight;
      if (candidate < distances[neighborId]) {
        distances[neighborId] = candidate;
        previous[neighborId] = { nodeId: current, edgeKey };
      }
    });
  }

  const routeIds = [];
  const edgeKeys = [];
  let cursor = endId;

  while (cursor) {
    routeIds.unshift(cursor);
    const previousStep = previous[cursor];
    if (previousStep?.edgeKey) {
      edgeKeys.unshift(previousStep.edgeKey);
    }
    cursor = previousStep?.nodeId;
  }

  if (routeIds[0] !== startId) return { routeIds: [], edgeKeys: [] };
  return { routeIds, edgeKeys };
}

function createPenalizedGraph(graph, edgeKeys, penaltyMultiplier = 1.9) {
  if (edgeKeys.length === 0) return graph;

  const penalizedEdges = new Set(edgeKeys);

  const nextGraph = {};
  Object.entries(graph).forEach(([nodeId, edges]) => {
    nextGraph[nodeId] = edges.map(edge => ({
      ...edge,
      weight: penalizedEdges.has(edge.edgeKey) ? edge.weight * penaltyMultiplier : edge.weight
    }));
  });
  return nextGraph;
}

function buildRouteVariants(startId, endId, graph, edgeGeometryByKey, locationsById, maxRoutes = 3) {
  const routes = [];
  const seen = new Set();
  let workingGraph = graph;

  for (let index = 0; index < maxRoutes; index += 1) {
    const route = dijkstra(startId, endId, workingGraph);
    if (route.routeIds.length < 2) break;

    const signature = route.edgeKeys.join("|");
    if (seen.has(signature)) break;
    seen.add(signature);

    const coordinates = routeIdsToCoordinates(route.routeIds, route.edgeKeys, edgeGeometryByKey, locationsById);
    if (coordinates.length < 2) break;

    routes.push({ routeIds: route.routeIds, edgeKeys: route.edgeKeys, coordinates });
    workingGraph = createPenalizedGraph(workingGraph, route.edgeKeys);
  }

  return routes;
}

function routeIdsToCoordinates(routeIds, edgeKeys, edgeGeometryByKey, locationsById) {
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
      edgeGeometryByKey[edgeKeys[i - 1]] ||
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
    const timer = window.setTimeout(() => {
      map.invalidateSize({ pan: false });
      const bounds = L.latLngBounds(routeCoordinates);
      const handleMoveEnd = () => {
        window.setTimeout(() => map.invalidateSize({ pan: false }), 60);
      };
      map.once("moveend", handleMoveEnd);
      map.fitBounds(bounds, { padding: [40, 40], maxZoom: 18, animate: false });
    }, 180);
    return () => window.clearTimeout(timer);
  }, [map, routeCoordinates]);

  return null;
}

function HighlightViewport({ destination, enabled }) {
  const map = useMap();

  useEffect(() => {
    if (!enabled || !destination) return;
    const timer = window.setTimeout(() => {
      map.invalidateSize({ pan: false });
      const handleMoveEnd = () => {
        window.setTimeout(() => map.invalidateSize({ pan: false }), 60);
      };
      map.once("moveend", handleMoveEnd);
      map.setView(destination, 20, { animate: false });
    }, 180);
    return () => window.clearTimeout(timer);
  }, [destination, enabled, map]);

  return null;
}

function MapSizeInvalidator() {
  const map = useMap();

  useEffect(() => {
    const timer = window.setTimeout(() => {
      map.invalidateSize({ pan: false });
    }, 180);
    return () => window.clearTimeout(timer);
  }, [map]);

  return null;
}

function MapResizeObserver() {
  const map = useMap();

  useEffect(() => {
    const container = map.getContainer();
    if (!container || typeof ResizeObserver === "undefined") return undefined;

    let frameId = 0;
    const observer = new ResizeObserver(() => {
      if (frameId) window.cancelAnimationFrame(frameId);
      frameId = window.requestAnimationFrame(() => {
        map.invalidateSize({ pan: false });
      });
    });

    observer.observe(container);

    return () => {
      observer.disconnect();
      if (frameId) window.cancelAnimationFrame(frameId);
    };
  }, [map]);

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
  const [showBuildingList, setShowBuildingList] = useState(true);
  const [showDirectoryPanel, setShowDirectoryPanel] = useState(false);
  const [locations, setLocations] = useState([]);
  const [pathEdges, setPathEdges] = useState([]);
  const [parkingLots, setParkingLots] = useState([]);
  const [dataError, setDataError] = useState("");
  const [highlightTargetId, setHighlightTargetId] = useState(null);

  useEffect(() => {
    try {
      const parsedLocations = parseLocationRecords(campusLocationsData);
      const routingNodes = parseRoutingNodes(routingGraphData, parsedLocations);
      setLocations([...parsedLocations, ...routingNodes]);
      setParkingLots(parseParkingGeoJson(parkingLotsGeoJson));
      setPathEdges([...parsePathFeatures(pathsGeoJson), ...parseMissingPathRecords(missingPathsData)]);
      setDataError("");
    } catch (error) {
      setDataError(error instanceof Error ? error.message : "Map data failed to load.");
    }
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
      addAlias(location.name, id);

      const base = normalizedId
        .replace(/_entrance_(front|rear|side)\d*$/g, "")
        .replace(/_(front|rear|side)\d*$/g, "")
        .replace(/_main\d*$/g, "")
        .replace(/_entrance$/g, "");
      if (base) {
        addAlias(base, id);
        addAlias(`${base}_main`, id);
      }
    });

    Object.entries(ROUTE_NODE_ALIASES).forEach(([sourceId, targetId]) => {
      addAlias(sourceId, targetId);
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
      pushCandidate(ROUTE_NODE_ALIASES[requested]);

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

  const graphLocations = useMemo(() => locations.filter(location => location.routable), [locations]);

  const routableLocations = useMemo(
    () =>
      locations
        .filter(location => location.routable && location.type !== "route_node")
        .sort((a, b) => a.name.localeCompare(b.name)),
    [locations]
  );

  const buildingMarkers = useMemo(
    () =>
      locations.filter(
        location =>
          location.type === "building" &&
          !HIDDEN_BUILDING_MARKER_IDS.has(location.id)
      ),
    [locations]
  );

  const entranceMarkers = useMemo(
    () => locations.filter(location => location.type === "entrance"),
    [locations]
  );

  const supplementalParkingLots = useMemo(() => {
    if (parkingLots.length > 0) {
      return [];
    }

    const existingNames = new Set(parkingLots.map(lot => normalizeId(lot.name)));
    const existingIds = new Set(parkingLots.map(lot => normalizeId(lot.id)));

    return locations
      .filter(location => location.type === "parking")
      .filter(location => !existingNames.has(normalizeId(location.name)) && !existingIds.has(normalizeId(location.id)))
      .map(location => ({
        id: `supp_${location.id}`,
        name: location.name,
        parkingType: inferParkingTypeFromName(location.name),
        polygon: createSquareAroundPoint(location.position),
        approximate: true
      }));
  }, [locations, parkingLots]);

  const allParkingLots = useMemo(() => [...parkingLots, ...supplementalParkingLots], [parkingLots, supplementalParkingLots]);

  const directoryPlaces = useMemo(() => {
    return locations
      .filter(location => location.type !== "route_node")
      .map(location => {
        const resolvedDestinationId = resolveRoutableId(location.id);
        const parentName = location.parent ? locationsById.get(location.parent)?.name : null;
        const displayName = getLocationDisplayName(location, locationsById);
        return {
          id: location.id,
          name: displayName,
          rawName: location.name,
          campus: location.campus || "Main",
          category: LOCATION_TYPE_LABELS[location.type] || "Location",
          description: parentName
            ? `${LOCATION_TYPE_LABELS[location.type] || "Location"} for ${parentName}`
            : `${LOCATION_TYPE_LABELS[location.type] || "Location"} on campus`,
          locationType: location.type,
          destinationId: resolvedDestinationId || null
        };
      })
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [locations, locationsById, resolveRoutableId]);

  const routeOptions = useMemo(() => {
    const byDestination = new Map();

    const getPriority = place => {
      if (place.locationType === "building") return 3;
      if (place.locationType === "parking") return 2;
      return 1;
    };

    directoryPlaces
      .filter(place => place.destinationId)
      .forEach(place => {
        const existing = byDestination.get(place.destinationId);
        if (!existing || getPriority(place) > getPriority(existing)) {
          byDestination.set(place.destinationId, place);
        }
      });

    return [...byDestination.values()].sort((a, b) => a.name.localeCompare(b.name));
  }, [directoryPlaces]);

  const routeOptionById = useMemo(() => {
    const map = new Map();
    routeOptions.forEach(option => {
      map.set(option.destinationId, option);
    });
    return map;
  }, [routeOptions]);

  useEffect(() => {
    if (routableLocations.length === 0) return;

    const defaultStart =
      resolveRoutableId("kean_hall_entrance_front") ||
      resolveRoutableId("kean_hall_main") ||
      routableLocations[0].id;
    const defaultEnd =
      resolveRoutableId("library_main") ||
      resolveRoutableId("library") ||
      routableLocations[Math.min(1, routableLocations.length - 1)].id;

    setStartId(prev => resolveRoutableId(prev) || defaultStart);
    setEndId(prev => resolveRoutableId(prev) || defaultEnd);
  }, [resolveRoutableId, routableLocations]);

  const { graph, edgeGeometryByKey } = useMemo(
    () => buildGraph(graphLocations, locationsById, pathEdges, resolveRoutableId, routableLocationIds, childrenByParent),
    [childrenByParent, graphLocations, locationsById, pathEdges, resolveRoutableId, routableLocationIds]
  );

  const routeVariants = useMemo(
    () => buildRouteVariants(startId, endId, graph, edgeGeometryByKey, locationsById),
    [edgeGeometryByKey, endId, graph, locationsById, startId]
  );

  const routeBuildingIds = routeVariants[0]?.routeIds || [];
  const routeCoordinates = routeVariants[0]?.coordinates || [];

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
  const displayRouteVariants = useMemo(
    () => (locationMode === "highlight" ? [] : routeVariants),
    [locationMode, routeVariants]
  );
  const routeOverlayKey = useMemo(
    () => `${startId}:${endId}:${displayRouteVariants.map(variant => variant.edgeKeys.join("|")).join("~")}`,
    [displayRouteVariants, endId, startId]
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
        place.rawName.toLowerCase().includes(query) ||
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

        const nearest = getNearestLocation(coords, graphLocations);
        setStartId(nearest.id);
        setLocationStatus(`Using your location. Nearest start point: ${nearest.name}.`);
      },
      () => {
        setLocationStatus("Could not read your location. Check browser location permission.");
      },
      { enableHighAccuracy: true, timeout: 10000 }
    );
  }, [graphLocations, routableLocations.length]);

  function swapRoute() {
    setLocationMode("directions");
    setStartId(endId);
    setEndId(startId);
  }

  function showDirections() {
    setLocationMode("directions");
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
  const startLabel = routeOptionById.get(startId)?.name || startBuilding?.name;
  const endLabel = routeOptionById.get(endId)?.name || endBuilding?.name;
  const highlightTarget = getLocationById(highlightTargetId, locationsById) || endBuilding;

  useEffect(() => {
    if (!routeRequest) return;

    const requestedDestination = routeRequest.destinationId ? getLocationById(routeRequest.destinationId, locationsById) : null;
    const mappedDestination = resolveRoutableId(routeRequest.destinationId);
    if (mappedDestination) {
      setEndId(mappedDestination);
    }
    setHighlightTargetId(routeRequest.destinationId || mappedDestination || null);

    const nextMode = routeRequest.locationMode === "directions" ? "directions" : "highlight";
    setLocationMode(nextMode);

    if (routeRequest.useCurrentLocation || nextMode === "directions") {
      setLocationMode("directions");
      setMyLocationAsStart();
    } else if (mappedDestination) {
      const destination = requestedDestination || getLocationById(mappedDestination, locationsById);
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
        <div className="map-header-actions">
          <button
            type="button"
            className="btn-secondary"
            onClick={() => setShowBuildingList(prev => !prev)}
          >
            {showBuildingList ? "Hide Directions" : "Show Directions"}
          </button>
          <button
            className="btn-secondary"
            onClick={() => setShowMap(false)}
          >
            Close Map
          </button>
        </div>
      </div>

      {showBuildingList && (
        <div className="route-controls">
          <label className="route-field">
            Start
            <select value={startId} onChange={event => setStartId(event.target.value)}>
              {routeOptions.map(option => (
                <option key={option.destinationId} value={option.destinationId}>
                  {option.name}
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
              {routeOptions.map(option => (
                <option key={option.destinationId} value={option.destinationId}>
                  {option.name}
                </option>
              ))}
            </select>
          </label>
        </div>
      )}

      <div className="route-actions">
        <button type="button" className="btn-primary" onClick={setMyLocationAsStart}>
          Use My Location
        </button>
        <button type="button" className="btn-secondary" onClick={showDirections}>
          Show Route
        </button>
        {dataError && <span className="route-note">{dataError}</span>}
        {locationStatus && <span className="route-note">{locationStatus}</span>}
      </div>

      <div className="route-summary">
        <strong>{startLabel}</strong> to <strong>{endLabel}</strong>
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
          <strong>{showDirectoryPanel ? "Kean Campus Directory" : "Directory Hidden"}</strong>
          <div className="directory-head-actions">
            <span>{filteredDirectoryPlaces.length} locations</span>
            <button
              type="button"
              className="btn-secondary directory-toggle-btn"
              onClick={() => setShowDirectoryPanel(prev => !prev)}
            >
              {showDirectoryPanel ? "Hide Directory" : "Show Directory"}
            </button>
          </div>
        </div>
        {showDirectoryPanel && (
          <>
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
          </>
        )}
      </div>

      <div className="leaflet-wrapper">
        <MapContainer center={KEAN_MAIN_CAMPUS} zoom={18} className="leaflet-map">
          <MapSizeInvalidator />
          <MapResizeObserver />
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {allParkingLots.map(lot => (
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
                {formatParkingTypeLabel(lot.parkingType)}
                {lot.approximate ? " (approximate)" : ""}
              </Popup>
            </Polygon>
          ))}
          {buildingMarkers.map(building => {
            const profile = getBuildingProfile(building);
            const imageSrc = getBuildingImage(building);
            return (
              <Marker key={building.id} position={building.position} icon={campusMarkerIcon}>
                <Popup>
                  <div className="building-popup">
                    {imageSrc ? (
                      <img
                        src={imageSrc}
                        alt={building.name}
                        className="building-popup-image"
                        loading="lazy"
                      />
                    ) : null}
                    <h4 className="building-popup-title">{building.name}</h4>
                    <div className="building-popup-campus">{building.campus || "Main"} Campus</div>
                    <div className="building-popup-section">
                      <div className="building-popup-label">Use</div>
                      <div className="building-popup-text">{profile.usage}</div>
                    </div>
                    <div className="building-popup-section">
                      <div className="building-popup-label">Departments</div>
                      <ul className="building-popup-list">
                        {profile.departments.map(department => (
                          <li key={department}>{department}</li>
                        ))}
                      </ul>
                    </div>
                    <div className="building-popup-section">
                      <div className="building-popup-label">Relevant Info</div>
                      <div className="building-popup-text">{profile.notes}</div>
                    </div>
                  </div>
                </Popup>
              </Marker>
            );
          })}
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
          <HighlightViewport destination={highlightTarget?.position} enabled={locationMode === "highlight"} />
          {displayRouteVariants.length > 0 && (
            <Fragment key={routeOverlayKey}>
              {displayRouteVariants.map((variant, index) => (
                <Polyline
                  key={`route-${variant.routeIds.join("-")}`}
                  positions={variant.coordinates}
                  pathOptions={{
                    color: ROUTE_LINE_COLORS[index] || ROUTE_LINE_COLORS[ROUTE_LINE_COLORS.length - 1],
                    weight: index === 0 ? 7 : 5,
                    opacity: index === 0 ? 0.95 : 0.75
                  }}
                />
              ))}
              <CircleMarker center={displayRouteVariants[0].coordinates[0]} radius={8} pathOptions={{ color: "#16a34a", fillOpacity: 1 }}>
                <Popup>Route Start</Popup>
              </CircleMarker>
              <CircleMarker
                center={displayRouteVariants[0].coordinates[displayRouteVariants[0].coordinates.length - 1]}
                radius={8}
                pathOptions={{ color: "#dc2626", fillOpacity: 1 }}
              >
                <Popup>Route Destination</Popup>
              </CircleMarker>
            </Fragment>
          )}
          {locationMode === "highlight" && highlightTarget?.position && (
            <CircleMarker center={highlightTarget.position} radius={12} pathOptions={{ color: "#dc2626", fillOpacity: 0.25, weight: 3 }}>
              <Popup>{highlightTarget.name}</Popup>
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
