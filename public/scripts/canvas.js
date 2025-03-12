// canvas.js

// -------------------- Constants --------------------

const PPC = 100; // Pixels per cell
const tileSize = 256; // Changed from 128 to match OSM standard

// Define the bounding box (min_lat, min_lon, max_lat, max_lon)
// const bbox = [55.328827, 10.528357, 55.343826, 10.558462]; // Example for Davinde Sjø

const COLORS = [
  '#8B5CF6', // Royal purple
  '#FF6B6B', // Vibrant coral (replaces red)
  '#10B981', // Emerald green (better than basic green)
  '#F59E0B', // Deep orange (more sophisticated)
  '#3B82F6', // Bright sapphire blue
  '#14B8A6', // Tropical teal (better than cyan)
  '#EC4899', // Raspberry pink (modern magenta alternative)
  '#EAB308'  // Gold yellow (less harsh than plain yellow)
];

// -------------------- Helper Functions for Shapes --------------------

// Compute vertices for an equilateral triangle (centered at 0,0)
function computeTriangleVertices(side) {
  const R = side / Math.sqrt(3);
  const vertices = [];
  for (let i = 0; i < 3; i++) {
    const angle = -Math.PI / 2 + i * (2 * Math.PI / 3);
    vertices.push([R * Math.cos(angle), R * Math.sin(angle)]);
  }
  return vertices;
}

// Compute vertices for a regular hexagon (centered at 0,0)
function computeHexagonVertices(r) {
  const vertices = [];
  for (let i = 0; i < 6; i++) {
    const angle = Math.PI / 6 + i * (Math.PI / 3);
    vertices.push([r * Math.cos(angle), r * Math.sin(angle)]);
  }
  return vertices;
}

// -------------------- Canvas Variables --------------------

const canvasContainer = document.getElementById('canvasContainer');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const resetButton = document.getElementById('resetViewButton');

// Show names and show shapes options
let showNames = true;
let showShapes = false;

// Canvas interaction variables
let zoomLevel = 1;
const zoomSensitivity = 0.5; // Adjust for zoom speed

let panOffsetX = 0;
let panOffsetY = 0;
let isDragging = false;
let dragStartX, dragStartY;

// -------------------- Canvas Sizing --------------------
function setCanvasSize() {
  const containerWidth = canvasContainer.clientWidth;
  const containerHeight = canvasContainer.clientHeight;
  
  // Original scaling logic
  if (containerWidth > containerHeight) {
    canvas.style.width = containerHeight + 'px';
    canvas.style.height = containerHeight + 'px';
  } else {
    canvas.style.width = containerWidth + 'px';
    canvas.style.height = containerWidth + 'px';
  }

  // Fixed positioning calculations
  const canvasDisplayWidth = parseInt(canvas.style.width);
  const canvasDisplayHeight = parseInt(canvas.style.height);
  
  canvas.style.position = 'absolute';
  canvas.style.left = `${(containerWidth - canvasDisplayWidth) / 2}px`;
  canvas.style.top = `${(containerHeight - canvasDisplayHeight) / 2}px`;

  // Keep your original render resolution (consider adding devicePixelRatio scaling)
  canvas.width = arena.width * PPC;
  canvas.height = arena.height * PPC;
}

// -------------------- OSM Background --------------------
// const bbox = [55.328827, 10.528357, 55.343826, 10.558462]; // Davinde Sjø
const orig_bbox = [55.39671, 10.544325, 55.411684, 10.567929]; // Rivers, lakes, roads and buildings
const extraTiles = 3;
let zoom = 17;
let topLeft = latLonToTile(orig_bbox[2], orig_bbox[1], zoom);
let bottomRight = latLonToTile(orig_bbox[0], orig_bbox[3], zoom);
let topLeftLatLon = tileToLatLon(topLeft.x, topLeft.y, zoom);
let bottomRightLatLon = tileToLatLon(bottomRight.x + 1, bottomRight.y + 1, zoom);
const bbox = [bottomRightLatLon.lat, topLeftLatLon.lon, topLeftLatLon.lat, bottomRightLatLon.lon];
console.log(bbox);
let tilesX = bottomRight.x - topLeft.x + 1;
let tilesY = bottomRight.y - topLeft.y + 1;
let images = [];

function latLonToTile(lat, lon, zoom) {
  const sin = Math.sin(lat * Math.PI / 180);
  const z2 = Math.pow(2, zoom);
  const x = Math.floor(z2 * (lon / 360 + 0.5));
  const y = Math.floor(z2 * (0.5 - Math.log((1 + sin) / (1 - sin)) / (4 * Math.PI)));
  return { x, y };
}

function tileToLatLon(x, y, zoom) {
  const n = Math.pow(2, zoom);
  const lon_deg = (x / n) * 360 - 180;  // Longitude calculation
  
  // Latitude calculation using inverse Gudermannian function
  const lat_rad = Math.atan(Math.sinh(Math.PI * (1 - 2 * y / n)));
  const lat_deg = lat_rad * (180 / Math.PI);

  return { lat: lat_deg, lon: lon_deg };
}

function getTileURL(x, y, zoom) {
    return `https://tile.openstreetmap.org/${zoom}/${x}/${y}.png`;
}

function loadTiles() {
    images = [];
    let loadedCount = 0;
    
    for (let x = topLeft.x - extraTiles; x <= bottomRight.x + extraTiles; x++) {
        images[x - topLeft.x] = [];
        for (let y = topLeft.y - extraTiles; y <= bottomRight.y + extraTiles; y++) {
            let img = new Image();
            img.src = getTileURL(x, y, zoom);
            img.onload = () => {
                loadedCount++;
                if (loadedCount === tilesX * tilesY) {
                    drawCanvas(topLeft, tilesX, tilesY);
                }
            };
            images[x - topLeft.x][y - topLeft.y] = img;
        }
    }
}

function drawCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let tileWidth = canvas.width / tilesX;
    let tileHeight = canvas.height / tilesY;
    
    ctx.strokeStyle = "black";
    for (let x = 0 - extraTiles; x < tilesX + extraTiles; x++) {
        for (let y = 0 - extraTiles; y < tilesY + extraTiles; y++) {
            ctx.drawImage(images[x][y], x * tileWidth, y * tileHeight, tileWidth, tileHeight);
            // ctx.strokeRect(x * tileWidth, y * tileHeight, tileWidth, tileHeight);
        }
    }

    if (geoJSON) {
        drawGeoJSON();
    } else {
        loadAndProcessGeoJSON();
    }

    // Draw a dashed bounding box from 0 to tiles X and Y
    drawDashedLine(ctx, [0, 0], [tilesX * tileWidth, 0], 10);
    drawDashedLine(ctx, [0, 0], [0, tilesY * tileHeight], 10);
    drawDashedLine(ctx, [tilesX * tileWidth, 0], [tilesX * tileWidth, tilesY * tileHeight], 10);
    drawDashedLine(ctx, [0, tilesY * tileHeight], [tilesX * tileWidth, tilesY * tileHeight], 10);
}

loadTiles();

// -------------------- GeoJSON Drawing Functions --------------------

function removeCoordsOutsideBbox(coords) {
  return coords.filter(coord => {
    return coord[1] >= bbox[0] && coord[1] <= bbox[2] && coord[0] >= bbox[1] && coord[0] <= bbox[3];
  });
}

function latLonToCanvas(lat, lon) {
  const west = bbox[1];
  const east = bbox[3];
  const south = bbox[0];
  const north = bbox[2];
  
  const x = ((lon - west) / (east - west)) * canvas.width;
  const y = ((north - lat) / (north - south)) * canvas.height;
  return { x, y };
}

function calculateCentroid(coords) {
  let sumLon = 0, sumLat = 0;
  coords.forEach(coord => {
      sumLon += coord[0];
      sumLat += coord[1];
  });
  return {
      lon: sumLon / coords.length,
      lat: sumLat / coords.length
  };
}

function drawPolygon(points, name) {
  try {
    // Get the first word in the name as the feature type
    featureType = name.split('_')[0];

    if (showShapes) {
      // Path
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      points.slice(1).forEach(point => ctx.lineTo(point.x, point.y));
      ctx.closePath();
      
      // Style
      ctx.fillStyle = colorMap[featureType];
      ctx.strokeStyle = 'black';
      ctx.lineWidth = 1;
      
      // Draw
      ctx.fill();
      ctx.stroke();
    }
    
    // if (showNames) {
    //   // Text
    //   ctx.fillStyle = 'black';
    //   ctx.font = '12px Arial';
    //   ctx.textAlign = 'center';
    //   ctx.textBaseline = 'middle';
    //   ctx.fillText(name, centroid.x, centroid.y);
    // }

  } catch (error) {
    console.log(error);
  }
}

function drawLineString(points, name) {
  try {
    // Get the first word in the name as the feature type
    featureType = name.split('_')[0];

    if (showShapes) {
      // Path
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      points.slice(1).forEach(point => ctx.lineTo(point.x, point.y));
      
      // Style
      ctx.strokeStyle = colorMap[featureType];
      ctx.lineWidth = 2;
      
      // Draw
      ctx.stroke();
    }

    // if (showNames) {
    //   // Choose text coords as median point
    //   const textX = points[Math.floor(points.length / 2)].x;
    //   const textY = points[Math.floor(points.length / 2)].y;
      
    //   // Text
    //   ctx.fillStyle = 'black';
    //   ctx.font = '12px Arial';
    //   ctx.textAlign = 'center';
    //   ctx.textBaseline = 'middle';
    //   ctx.fillText(name, textX, textY);
    // }
  } catch (error) {
    console.log(error);
  }
}

function drawGeoJSON() {
  // Names to draw list
  const namesToDraw = [];
  // Order features first with polygons, then linestrings
  geoJSON.features.forEach(feature => {
      const coords = feature.geometry.coordinates;
      const uniqueName = feature.properties.unique_name;
      
      // Convert coordinates to canvas points
      let points = [];
      let geoCoords = [];
      
      if (feature.geometry.type === 'Polygon') {
          geoCoords = coords[0];
      } else if (feature.geometry.type === 'LineString') {
          geoCoords = removeCoordsOutsideBbox(coords);
      }
      
      points = geoCoords.map(coord => {
          const [lon, lat] = coord;
          return latLonToCanvas(lat, lon);
      });
      
      // Draw feature
      if (feature.geometry.type === 'Polygon') {
          const centroidGeo = calculateCentroid(geoCoords);
          const centroidCanvas = latLonToCanvas(centroidGeo.lat, centroidGeo.lon);
          drawPolygon(points, uniqueName);
          namesToDraw.push(uniqueName, centroidCanvas);
      } else if (feature.geometry.type === 'LineString') {
          const textX = points[Math.floor(points.length / 2)].x;
          const textY = points[Math.floor(points.length / 2)].y;
          drawLineString(points, uniqueName);
          namesToDraw.push(uniqueName, { x: textX, y: textY });
      }
  });

  // Draw names
  if (showNames) {
    namesToDraw.forEach((name, idx) => {
      if (idx % 2 === 0) {
        const textX = namesToDraw[idx + 1].x;
        const textY = namesToDraw[idx + 1].y;
        ctx.fillStyle = 'black';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(name, textX, textY);
      }
    });
  }
}

// -------------------- Drawing Functions --------------------

// Draw a dashed line between two points
function drawDashedLine(ctx, start, end, dashLength = 5) {
  ctx.save();
  ctx.setLineDash([dashLength, dashLength]);
  ctx.beginPath();
  ctx.moveTo(start[0], start[1]);
  ctx.lineTo(end[0], end[1]);
  ctx.stroke();
  ctx.restore();
}

// Draw grid lines (each cell is PPC pixels)
function drawGrid() {
  ctx.save();
  ctx.strokeStyle = 'rgba(200,200,200,1)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= arena.width; i++) {
    ctx.beginPath();
    ctx.moveTo(i * PPC, 0);
    ctx.lineTo(i * PPC, arena.height * PPC);
    ctx.stroke();
  }
  for (let j = 0; j <= arena.height; j++) {
    ctx.beginPath();
    ctx.moveTo(0, j * PPC);
    ctx.lineTo(arena.width * PPC, j * PPC);
    ctx.stroke();
  }
  ctx.restore();
}

// Draw formation outline for a group (circle, square, triangle, hexagon)
function drawFormation(group) {
  const { formation_shape, formation_radius } = group.bhvr.params;
  const center = group.virtual_center;
  const color = group.robots.length > 1 ? COLORS[group.idx%COLORS.length] : '#000';
  const cx = center[0] * PPC;
  const cy = center[1] * PPC;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 5]);

  if (formation_shape === 'circle') {
    const radius = formation_radius * PPC;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
    ctx.stroke();
  } else if (formation_shape === 'square') {
    const size = formation_radius * 2 * PPC;
    ctx.strokeRect(cx - size / 2, cy - size / 2, size, size);
  } else if (formation_shape === 'triangle') {
    const vertices = computeTriangleVertices(formation_radius * 2);
    const points = vertices.map(v => [cx + v[0] * PPC, cy + v[1] * PPC]);
    for (let i = 0; i < 3; i++) {
      drawDashedLine(ctx, points[i], points[(i + 1) % 3]);
    }
  } else if (formation_shape === 'hexagon') {
    const vertices = computeHexagonVertices(formation_radius);
    const points = vertices.map(v => [cx + v[0] * PPC, cy + v[1] * PPC]);
    for (let i = 0; i < 6; i++) {
      drawDashedLine(ctx, points[i], points[(i + 1) % 6]);
    }
  }
  ctx.setLineDash([]);
  ctx.restore();
}

// Draw centered text
function drawCenteredText(text, x, y) {
  ctx.fillText(text, x * PPC, y * PPC);
}

// Draw map elements (river, lake, road, bridge, forest, field, town, farm) and labels
function drawMap() {
  ctx.save();
  
  // Configurable positions
  const riverPos = { x: 9.5, y: 0, width: 1, height: 15.5 };      // Vertical river
  const lakePos = { x: 6, y: 15.0, width: 8.0, height: 5.0 };     // Bottom lake
  const roadPos = { x: 0, y: 9.5, width: 20, height: 1 };         // Full-width road
  const bridgePos = { x: 8.5, y: 9, width: 3, height: 2 };     // Centered bridge
  const forestPos = { x: 11, y: 1, width: 5, height: 5 };    // Right-side forest
  const fieldPos = { x: 1.0, y: 0, width: 7.0, height: 5.0 };    // Left field
  const townPos = { x: 1.5, y: 6.5, width: 5, height: 7 };         // Central town
  const farmPos = { x: 15.0, y: 7.5, width: 4, height: 5 };      // Right farm


  // Drawing elements
  ctx.fillStyle = 'rgb(135,206,235)'; // light blue
  ctx.fillRect(riverPos.x * PPC, riverPos.y * PPC, riverPos.width * PPC, riverPos.height * PPC);

  ctx.fillStyle = 'rgb(103, 103, 255)'; // darker blue
  ctx.fillRect(lakePos.x * PPC, lakePos.y * PPC, lakePos.width * PPC, lakePos.height * PPC);

  ctx.fillStyle = 'rgb(100,100,100)'; // gray
  ctx.fillRect(roadPos.x * PPC, roadPos.y * PPC, roadPos.width * PPC, roadPos.height * PPC);

  ctx.fillStyle = 'rgb(100,100,100)'; // same as road
  ctx.fillRect(bridgePos.x * PPC, bridgePos.y * PPC, bridgePos.width * PPC, bridgePos.height * PPC);

  ctx.fillStyle = 'rgb(34,139,34)'; // forest green
  ctx.fillRect(forestPos.x * PPC, forestPos.y * PPC, forestPos.width * PPC, forestPos.height * PPC);

  ctx.fillStyle = 'rgb(144,238,144)'; // light green
  ctx.fillRect(fieldPos.x * PPC, fieldPos.y * PPC, fieldPos.width * PPC, fieldPos.height * PPC);

  ctx.fillStyle = 'rgb(211,211,211)'; // light gray
  ctx.fillRect(townPos.x * PPC, townPos.y * PPC, townPos.width * PPC, townPos.height * PPC);

  ctx.fillStyle = 'rgb(255,165,0)'; // light orange
  ctx.fillRect(farmPos.x * PPC, farmPos.y * PPC, farmPos.width * PPC, farmPos.height * PPC);

  // Labels
  ctx.fillStyle = '#000';
  ctx.font = '30px Arial';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  drawCenteredText("River", riverPos.x + riverPos.width / 2, riverPos.y + riverPos.height / 2);
  drawCenteredText("Lake",  lakePos.x + lakePos.width / 2, lakePos.y + lakePos.height / 2);
  // drawCenteredText("Road",  roadPos.x + roadPos.width / 2, roadPos.y + roadPos.height / 2);
  drawCenteredText("Bridge",  bridgePos.x + bridgePos.width / 2, bridgePos.y + bridgePos.height / 2);
  drawCenteredText("Forest",  forestPos.x + forestPos.width / 2, forestPos.y + forestPos.height / 2);
  drawCenteredText("Field",  fieldPos.x + fieldPos.width / 2, fieldPos.y + fieldPos.height / 2);
  drawCenteredText("Town",  townPos.x + townPos.width / 2, townPos.y + townPos.height / 2);
  drawCenteredText("Farm",  farmPos.x + farmPos.width / 2, farmPos.y + farmPos.height / 2);

  ctx.restore();
}

// Draw a destination marker (a cross) for the group
function drawDestination(group) {
  const trajectory = group.bhvr.params.trajectory;
  if (!trajectory) return;

  const scale = 1 / zoomLevel; // Inverse scaling to keep size constant

  for (let i = 0; i < trajectory.length; i++) {
    const dest = trajectory[i];
    const color = group.robots.length > 1 ? COLORS[group.idx % COLORS.length] : '#000';
    
    // Convert destination to pixel space
    const px = dest[0] * PPC;
    const py = dest[1] * PPC;
    
    // Scale the cross size
    const denom = 5;
    const size = (PPC / denom) * scale;
    
    ctx.save();

    // Draw cross shape
    ctx.strokeStyle = color;
    ctx.lineWidth = 3 * scale;
    ctx.beginPath();
    ctx.moveTo(px - size, py - size);
    ctx.lineTo(px + size, py + size);
    ctx.moveTo(px + size, py - size);
    ctx.lineTo(px - size, py + size);
    ctx.stroke();

    // Draw index label with a circle background
    ctx.fillStyle = color;
    ctx.font = `${30 * scale}px Arial`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(i, px, py - size * 2);

    ctx.restore();
  }
}


// Draw a robot with a directional line, index label, and target cross
function drawRobot(robot, color) {
  // Adjust scale based on zoom level
  const scale = 1 / zoomLevel;

  // Convert coordinates to pixels
  const px = robot.x * PPC;
  const py = robot.y * PPC;
  const endX = px + Math.cos(robot.angle) * (PPC / 2) * scale;
  const endY = py + Math.sin(robot.angle) * (PPC / 2) * scale;
  const targetX = robot.target_x * PPC;
  const targetY = robot.target_y * PPC;

  ctx.save();

  // Directional line
  ctx.strokeStyle = color;
  ctx.lineWidth = 4 * scale;
  ctx.beginPath();
  ctx.moveTo(px, py);
  ctx.lineTo(endX, endY);
  ctx.stroke();

  // Robot body (circle with battery level)
  const outerRadius = 25 * scale;
  const batteryRadius = 22 * scale;
  const innerRadius = 19 * scale;

  // Outer circle
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(px, py, outerRadius, 0, 2 * Math.PI);
  ctx.fill();

  // Battery level (pie chart)
  const batteryLevel = robot.battery_level;
  const batteryLevelAngle = batteryLevel * 2 * Math.PI;
  ctx.fillStyle = '#000';
  ctx.beginPath();
  ctx.moveTo(px, py);
  ctx.arc(px, py, batteryRadius, -Math.PI / 2, -Math.PI / 2 + batteryLevelAngle);
  ctx.closePath();
  ctx.fill();

  // Inner circle
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  ctx.arc(px, py, innerRadius, 0, 2 * Math.PI);
  ctx.fill();

  // Robot index label
  ctx.fillStyle = '#000';
  ctx.font = `${30 * scale}px Arial`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(robot.idx, px, py);

  // Target marker (small cross)
  const denom = 10;
  const size = (PPC / denom) * scale;
  ctx.strokeStyle = color;
  ctx.lineWidth = 3 * scale;
  ctx.beginPath();
  ctx.moveTo(targetX - size, targetY - size);
  ctx.lineTo(targetX + size, targetY + size);
  ctx.moveTo(targetX + size, targetY - size);
  ctx.lineTo(targetX - size, targetY + size);
  ctx.stroke();

  ctx.restore();
}


// Draw simulation status overlay (header and each group’s status)
function drawStatus() {
  ctx.save();
  
  // Reset transformations to ensure fixed positioning
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  ctx.fillStyle = '#000';
  ctx.font = 'bold 30px Arial';
  ctx.textAlign = 'left';
  
  let yOffset = 10;
  // ctx.fillText(`Simulation Step: ${current_step}`, 10, yOffset);
  yOffset += 30;

  let behaviorString = "";
  groups.forEach((group, idx) => {
    switch (group.bhvr.name) {
      case "random_walk":
        behaviorString = "Random Walk";
        break;
      case "form_and_follow_trajectory":
        let trajectoryStr = "" + group.bhvr.params.trajectory.length + " points";
        // let trajectoryStr = "";
        // for (let i = 0; i < group.bhvr.params.trajectory.length; i++) {
        //   const dest = group.bhvr.params.trajectory[i];
        //   trajectoryStr += `(${dest[0]}, ${dest[1]})`;
        //   if (i < group.bhvr.params.trajectory.length - 1) {
        //     trajectoryStr += " → ";
        //   }
        // }
        behaviorString = `Form & Follow Trajectory (${group.bhvr.params.formation_shape}) [${trajectoryStr}]`;
        break;
      default:
        behaviorString = "None";
    }

    const robotsInGroup = Array.from(group.robots).map(robot => robot.idx).join(',');
    const statusText = `G${group.idx} [${robotsInGroup}] -> ${behaviorString}`;
    const color = group.robots.length > 1 ? COLORS[group.idx % COLORS.length] : '#000';

    ctx.fillStyle = color;
    ctx.fillText(statusText, 10, yOffset);
    yOffset += 30;
  });

  ctx.restore();
}


// Main update function: clear canvas and redraw every element
function updateDisplay() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.save(); // Save the initial state

  ctx.translate(panOffsetX, panOffsetY); // Apply pan
  ctx.scale(zoomLevel, zoomLevel); // Apply zoom

  drawCanvas(topLeft, tilesX, tilesY);
  // drawMap();
  // drawGrid();
  groups.forEach(group => {
    drawDestination(group);
    group.robots.forEach(robot => {
      const color = group.robots.length > 1 ? COLORS[group.idx % COLORS.length] : '#000';
      drawRobot(robot, color);
    });
  });
  drawStatus();

  ctx.restore(); // Restore the original state
}

// -------------------- Canvas Interaction Functions --------------------

function resetView() {
  zoomLevel = 1;
  panOffsetX = 0;
  panOffsetY = 0;
  updateDisplay();
}

// Event listeners for zoom and pan
canvasContainer.addEventListener('wheel', (e) => {
  e.preventDefault();

  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  const preZoomMouseX = (mouseX - panOffsetX) / zoomLevel;
  const preZoomMouseY = (mouseY - panOffsetY) / zoomLevel;

  const zoomChange = e.deltaY > 0 ? -zoomSensitivity : zoomSensitivity;
  const newZoomLevel = Math.max(1.0, zoomLevel + zoomChange);

  panOffsetX = mouseX - preZoomMouseX * newZoomLevel;
  panOffsetY = mouseY - preZoomMouseY * newZoomLevel;

  zoomLevel = newZoomLevel;
  updateDisplay();
});

canvasContainer.addEventListener('mousedown', (e) => {
  isDragging = true;
  dragStartX = e.clientX - panOffsetX;
  dragStartY = e.clientY - panOffsetY;
});

canvasContainer.addEventListener('mousemove', (e) => {
  if (!isDragging) return;
  panOffsetX = e.clientX - dragStartX;
  panOffsetY = e.clientY - dragStartY;
  
  updateDisplay();
});

canvasContainer.addEventListener('mouseup', () => {
  isDragging = false;
});

canvasContainer.addEventListener('mouseleave', () => {
    isDragging = false;
});

// Reset view button event listener
resetButton.addEventListener('click', () => {
  resetView();
});