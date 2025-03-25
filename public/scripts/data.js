// data.js


// -------------------- Variables --------------------

var ENV = null;
var CONFIG = null;
var VIS = null;

var geoJSON = null;
let images = {};
let satImages = {};
let zoom = null;
let extraTiles = null;
let center = null;
let orig_bbox = null;
let bbox = null;
let topLeft = null;
let bottomRight = null;

// -------------------- Read config --------------------

async function fetchConfig() {
  try {
    const simResponse = await fetch('/data/config/simulation.json');
    if (!simResponse.ok) {
      throw new Error(`Failed to load sim config: ${simResponse.statusText}`);
    }

    const simData = await simResponse.json();
    ENV = simData.env;
    CONFIG = simData.config;
    console.log("ENV & CONFIG loaded successfully");

    const visResponse = await fetch('/data/config/visualization.json');
    if (!visResponse.ok) {
      throw new Error(`Failed to load vis config: ${visResponse.statusText}`);
    }
    const visData = await visResponse.json();
    VIS = visData;
    console.log("VIS loaded successfully");
    set_osm_variables();
    console.log("OSM variables set successfully");
    loadTiles();
    console.log("Tiles loaded successfully");
  } catch (error) {
    console.error("Error fetching config:", error);
    return null;
  }
}



// ----------------- Bounding box and tiles -----------------
function set_osm_variables() {
  center = ENV['center'];
  zoom = ENV['zoom'] || 16;
  extraTiles = ENV['extra_tiles'] || 1;
  orig_bbox = VIS['bbox'];
  bbox = VIS['tiles_adapted_bbox'];

  topLeft = VIS["top_left_tile"]
  bottomRight = VIS["bottom_right_tile"]
  tilesX = bottomRight[0] - topLeft[0] + 1;
  tilesY = bottomRight[1] - topLeft[1] + 1;
  // console.log(`TilesX: ${tilesX}, TilesY: ${tilesY}`);
}

function getTileURL(x, y, zoom) {
  return `https://tile.openstreetmap.org/${zoom}/${x}/${y}.png`;
}

function getSatelliteTileURL(x, y, zoom) {
  return `https://api.maptiler.com/tiles/satellite-v2/${zoom}/${x}/${y}.jpg?key=sMDJDFByHsGaNYXwE0g4`;
}

function loadTiles() {
  images = {}
  
  for (let x = topLeft[0] - extraTiles; x <= bottomRight[0] + extraTiles; x++) {
      images[`${x}`] = {}
      satImages[`${x}`] = {}
      for (let y = topLeft[1] - extraTiles; y <= bottomRight[1] + extraTiles; y++) {
          let img = new Image();
          let satImg = new Image();
          img.src = getTileURL(x, y, zoom);
          satImg.src = getSatelliteTileURL(x, y, zoom);
          images[`${x}`][`${y}`] = img;
          satImages[`${x}`][`${y}`] = satImg;
      }
  }
}

// -------------------- GeoJSON data --------------------

// Fetch GeoJSON data from the /data/features/features.geojson file in the server
// And parse the data into a JSON object
async function fetchGeoJSONData() {
  try {
    const response = await fetch("/processed.geojson");
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

// Remove coords outside bbox
function removeCoordsOutsideBbox(target_bbox,coords) {
  return coords.filter(coord => {
    return coord[0] >= target_bbox[0] && coord[0] <= target_bbox[2] && coord[1] >= target_bbox[1] && coord[1] <= target_bbox[3];
  });
}

// Reorder GeoJSON data, first with Polygons and then LineStrings
function reorderGeoJSON(geoJSON) {
  if (!geoJSON || !geoJSON.features) {
    console.error("Invalid GeoJSON data");
    return geoJSON;
  }

  const polygons = geoJSON.features.filter(feature => feature.geometry.type === 'Polygon' || feature.geometry.type === 'MultiPolygon');
  const lineStrings = geoJSON.features.filter(feature => feature.geometry.type === 'LineString');
  const points = geoJSON.features.filter(feature => feature.geometry.type === 'Point');
  
  return { ...geoJSON, features: polygons.concat(lineStrings, points) };
}

// Fetch and process the GeoJSON data
async function loadAndProcessGeoJSON() {
  geoJSON = await fetchGeoJSONData();
  if (geoJSON) {
    // Reorder GeoJSON data
    geoJSON = reorderGeoJSON(geoJSON);
    // Remove coords outside bbox for LineStrings
    // for(feature of geoJSON.features){
    //   if (feature.geometry.type === 'LineString') {
    //     feature.geometry.coordinates = removeCoordsOutsideBbox(bbox, feature.geometry.coordinates);
    //     // Remove feature if no coordinates left
    //     if (feature.geometry.coordinates.length === 0) {
    //       geoJSON.features = geoJSON.features.filter(f => f !== feature);
    //     }
    //   }
    // }
    console.log("GeoJSON data processed successfully");
  }
}
