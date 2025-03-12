// data.js


// -------------------- Color mappings --------------------
const colorMap = {
  "river": "blue",
  "road": "grey",
  "ditch": "lightblue",
  "farmland": "yellow",
  "scrub": "lightgreen",
  "building": "red",
  "shed": "brown",
  "water": "darkblue",
  "house": "orange",
  "forest": "darkgreen",
  "garage": "darkgrey",
  "driveway": "lightgrey",
  "apartments": "purple",
  "stream": "teal",
  "meadow": "lime",
  "path": "saddlebrown",
  "cycleway": "darkslateblue",
  "greenhouse": "palegreen",
  "service": "silver",
  "hotel": "gold",
  "grassland": "forestgreen"
};

// -------------------- GeoJSON data --------------------

// Fetch GeoJSON data from the /data/features.geojson file in the server
// And parse the data into a JSON object
async function fetchGeoJSONData() {
  try {
    const response = await fetch("/data/features.geojson");
    if (!response.ok) {
      throw new Error(`Failed to load GeoJSON: ${response.statusText}`);
    }
    const data = await response.json();
    console.log("GeoJSON data loaded successfully");
    return data;
  } catch (error) {
    console.error("Error fetching GeoJSON data:", error);
    return null; // Handle the error gracefully
  }
}

// Reorder GeoJSON data, first with Polygons and then LineStrings
function reorderGeoJSON(geoJSON) {
  if (!geoJSON || !geoJSON.features) {
    console.error("Invalid GeoJSON data");
    return geoJSON;
  }

  const polygons = geoJSON.features.filter(feature => feature.geometry.type === 'Polygon');
  const lineStrings = geoJSON.features.filter(feature => feature.geometry.type === 'LineString');
  
  return { ...geoJSON, features: polygons.concat(lineStrings) };
}

// Fetch and process the GeoJSON data
var geoJSON = null;
async function loadAndProcessGeoJSON() {
  geoJSON = await fetchGeoJSONData();
  if (geoJSON) {
    geoJSON = reorderGeoJSON(geoJSON);
    console.log("GeoJSON data reordered successfully");
  }
  return geoJSON;
}
