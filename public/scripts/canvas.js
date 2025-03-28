// canvas.js

// -------------------- Constants --------------------

const PPC = 200; // Pixels per cell
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

function hexToRgba(hex, opacity) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${opacity})`;
}

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

const arenaWidth = 10;
const arenaHeight = 10;
let visibleArenaMinX = 0;
let visibleArenaMinY = 0;
let visibleArenaMaxX = arenaWidth;
let visibleArenaMaxY = arenaHeight;

const canvasContainer = document.getElementById('canvasContainer');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const resetButton = document.getElementById('resetViewButton');

// Show names and show shapes options
let showNames = true;
let showShapes = false;
let showOverpass = true;
let showHardcoded = true;
let useSatellite = false;

// Canvas interaction variables
let zoomLevel = 1;
const zoomSensitivity = 0.5; // Adjust for zoom speed
let lastMousePoint = null;
let lastMouseCoords = null;
let closestFeatures = [];

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
  canvas.width = arenaWidth * PPC;
  canvas.height = arenaHeight * PPC;
}

// -------------------- OSM Background --------------------
function drawBbox(target_bbox) {
  const topLeftCanvas = lonLatToCanvas(target_bbox[0], target_bbox[3]);
  const bottomRightCanvas = lonLatToCanvas(target_bbox[2], target_bbox[1]);
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
  ctx.lineWidth = 2;
  ctx.strokeRect(topLeftCanvas.x, topLeftCanvas.y, bottomRightCanvas.x - topLeftCanvas.x, bottomRightCanvas.y - topLeftCanvas.y);
  ctx.fillStyle = 'rgba(255, 0, 0, 0.0)';
  ctx.fillRect(topLeftCanvas.x, topLeftCanvas.y, bottomRightCanvas.x - topLeftCanvas.x, bottomRightCanvas.y - topLeftCanvas.y);
  ctx.fillStyle = 'rgba(255, 0, 0, 0)';
  ctx.font = '12px Arial';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(`(${target_bbox[0]}, ${target_bbox[3]})`, topLeftCanvas.x + 5, topLeftCanvas.y + 5);
  ctx.fillText(`(${target_bbox[2]}, ${target_bbox[1]})`, bottomRightCanvas.x - 5, bottomRightCanvas.y - 5);
}

function drawTile(x, y) {
  const tileWidth = canvas.width / tilesX;
  const tileHeight = canvas.height / tilesY;
  const tileX = (x - topLeft[0]) * tileWidth;
  const tileY = (y - topLeft[1]) * tileHeight;
  if (useSatellite) {
    ctx.drawImage(satImages[`${x}`][`${y}`], tileX, tileY, tileWidth, tileHeight);
  } else {
    ctx.drawImage(images[`${x}`][`${y}`], tileX, tileY, tileWidth, tileHeight);
  }

  // Draw tile borders and coordinates (optional)
  // ctx.strokeStyle = "black";
  // ctx.strokeRect(tileX, tileY, tileWidth, tileHeight);
  // ctx.fillStyle = 'black';
  // ctx.font = '12px Arial';
  // ctx.textAlign = 'center';
  // ctx.textBaseline = 'middle';
  // ctx.fillText(`(${x},${y})`, tileX + tileWidth / 2, tileY + tileHeight / 2);
}

function drawCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw the OSM tiles
    for (let x = topLeft[0] - extraTiles; x <= bottomRight[0] + extraTiles; x++) {
        for (let y = topLeft[1] - extraTiles; y <= bottomRight[1] + extraTiles; y++) {
            drawTile(x, y);
        }
    }

    // Draw the GeoJSON shapes
    drawGeoJSON();

    // Draw the given bbox
    drawBbox(bbox);
    // Draw the original bbox
    // drawBbox(orig_bbox);
    // Draw the original center
    // center_canvas_coords = lonLatToCanvas(center[0], center[1])
    // ctx.beginPath();
    // ctx.arc(center_canvas_coords.x, center_canvas_coords.y, 20, 0, 2 * Math.PI); // Radius of 5 pixels
    // ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
    // ctx.fill();
    // ctx.closePath();
}

// -------------------- GeoJSON Drawing Functions --------------------

function lonLatToCanvas(lon, lat) {
  const min_lon = bbox[0];
  const min_lat = bbox[1];
  const max_lon = bbox[2];
  const max_lat = bbox[3];

  const x = ((lon - min_lon) / (max_lon - min_lon)) * arenaWidth * PPC;
  const y = ((max_lat - lat) / (max_lat - min_lat)) * arenaHeight * PPC;

  return { x, y };
}

function canvasToLonLat(x, y) {
  const min_lon = bbox[0];
  const min_lat = bbox[1];
  const max_lon = bbox[2];
  const max_lat = bbox[3];
  const lon = min_lon + (x / (arenaWidth * PPC)) * (max_lon - min_lon);
  const lat = max_lat - (y / (arenaHeight * PPC)) * (max_lat - min_lat);
  return { lon, lat };
}

function calculateCentroid(points) {
  let x = 0;
  let y = 0;
  points.forEach(point => {
    x += point.x;
    y += point.y;
  });
  return { x: x / points.length, y: y / points.length };
}

function drawPolygon(feature) {
  try {
    const featureColor = hexToRgba(feature.properties.color, 0.8) || 'rgba(0, 0, 0, 0.8)';
    if (feature.geometry.type === 'Polygon') {
      const points = feature.geometry.points;

      // Path
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      points.slice(1).forEach(point => ctx.lineTo(point.x, point.y));
      ctx.closePath();
      
      // Style
      ctx.fillStyle = featureColor || 'rgba(0, 0, 0, 0.5)';
      ctx.strokeStyle = 'black';
      ctx.lineWidth = 1;

      // Draw
      ctx.fill();
      ctx.stroke();
    } else if (feature.geometry.type === 'MultiPolygon') {
      const polygons = feature.geometry.points;

      // Draw first polygon representing the outer boundary
      ctx.beginPath();
      ctx.moveTo(polygons[0][0].x, polygons[0][0].y);
      polygons[0].slice(1).forEach(point => ctx.lineTo(point.x, point.y));
      ctx.closePath();
      ctx.fillStyle = featureColor || 'rgba(0, 0, 0, 0.5)';
      ctx.strokeStyle = 'black';
      ctx.lineWidth = 1;
      ctx.fill();
      ctx.stroke();

      // Draw inner polygons representing holes
      polygons.slice(1).forEach(polygon => {
        ctx.beginPath();
        ctx.moveTo(polygon[0].x, polygon[0].y);
        polygon.slice(1).forEach(point => ctx.lineTo(point.x, point.y));
        ctx.closePath();
        ctx.fillStyle = 'rgba(255, 255, 255, 0.5)'; // White for holes
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 1;
        ctx.fill();
        ctx.stroke();
      });
    }
  } catch (error) {
    // console.log(error);
  }
}

function drawLineString(feature) {
  try {
    const featureColor = hexToRgba(feature.properties.color, 1.0) || 'rgba(0, 0, 0, 1.0)';
    const points = feature.geometry.points;

    // Path
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    points.slice(1).forEach(point => ctx.lineTo(point.x, point.y));
    
    // Style
    ctx.strokeStyle = featureColor || 'rgba(0, 0, 0, 1.0)';
    ctx.lineWidth = 3;
    
    // Draw
    ctx.stroke();
  } catch (error) {
    // console.log(error);
  }
}

function drawPoint(feature) {
  try {
    const point = feature.geometry.points[0];
    
    // Define marker properties
    const radius = 15; // Larger, more visible size
    const borderColor = "black";
    const borderWidth = 1;
    const fillColor = feature.properties.color || "rgba(0, 0, 255, 0.7)";

    // Draw shadow for better visibility
    ctx.shadowColor = "rgba(0,0,0,0.3)";
    ctx.shadowBlur = 5;

    // Draw circular marker
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = fillColor;
    ctx.fill();

    // Draw border
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = borderWidth;
    ctx.stroke();

    // Optional: Add a small white center dot
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius * 0.1, 0, 2 * Math.PI);
    ctx.fillStyle = "white";
    ctx.fill();
  } catch (error) {
    console.error("Error drawing point:", error);
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
      let geoCoords = [];
      
      if (feature.geometry.type === 'MultiPolygon') {
          geoCoords = coords[0];
          if (!feature.geometry.points) {
            feature.geometry.points = [];
            for (let i = 0; i < geoCoords.length; i++) {
              feature.geometry.points[i] = [];
              for (let j = 0; j < geoCoords[i].length; j++) {
                const coord = geoCoords[i][j];
                const [lon, lat] = coord;
                feature.geometry.points[i].push(lonLatToCanvas(lon, lat));
              }
            }
          }
      } else if (feature.geometry.type === 'Polygon') {
          geoCoords = coords[0];
          if (!feature.geometry.points) {
            feature.geometry.points = geoCoords.map(coord => {
                const [lon, lat] = coord;
                return lonLatToCanvas(lon, lat);
            });
          }
      } else if (feature.geometry.type === 'LineString') {
          geoCoords = coords;
          if (!feature.geometry.points) {
            feature.geometry.points = geoCoords.map(coord => {
                const [lon, lat] = coord;
                return lonLatToCanvas(lon, lat);
            });
          }
      } else if (feature.geometry.type === 'Point') {
          geoCoords = [coords];
          if (!feature.geometry.points) {
            feature.geometry.points = geoCoords.map(coord => {
                const [lon, lat] = coord;
                return lonLatToCanvas(lon, lat);
            });
          }
      }
      
      // Draw feature
      try {
        const showFeature = (showHardcoded && feature.id < 1000) || (showOverpass && feature.id > 1000);
        const points = feature.geometry.points;
        if (showFeature) {
          if (feature.geometry.type === 'Polygon' || feature.geometry.type === 'MultiPolygon') {
            if (feature.geometry.type === 'MultiPolygon') {
              namesToDraw.push(uniqueName, calculateCentroid(points[0]));
            } else {
              namesToDraw.push(uniqueName, calculateCentroid(points));
            }
            if (showShapes) {
              drawPolygon(feature);
            }
          } else if (feature.geometry.type === 'LineString') {
            const textX = points[Math.floor(points.length / 2)].x;
            const textY = points[Math.floor(points.length / 2)].y;
            
            namesToDraw.push(uniqueName, { x: textX, y: textY });
            if (showShapes) {
              drawLineString(feature);
            }
          } else if (feature.geometry.type === 'Point') {
            const textX = points[0].x;
            const textY = points[0].y;
            namesToDraw.push(uniqueName, { x: textX, y: textY });
            if (showShapes) {
              drawPoint(feature);
            }
          }
        }
      } catch (error) {
        console.log(feature);
        console.log(error);
      }
  });

  // Draw names
  const scaledFontSize = Math.floor(Math.max(6, Math.min(40, 30 / zoomLevel)));
  const lineWidth = scaledFontSize / 10;
  
  if (showNames) {
    namesToDraw.forEach((name, idx) => {
      if (idx % 2 === 0) {
        const textX = namesToDraw[idx + 1].x;
        const textY = namesToDraw[idx + 1].y;
        if (useSatellite || showShapes) {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
          ctx.strokeStyle = 'rgba(0, 0, 0, 0.8)';
        } else {
          ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
        }
        ctx.lineWidth = lineWidth;
        ctx.font = `${scaledFontSize}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.strokeText(name, textX, textY);
        ctx.fillText(name, textX, textY);
      }
    });
  }

  // Draw the closest features
  if (closestFeatures.length > 0) {
    closestFeatures.forEach(feature => {
      const closestPoint = feature.closestPoint;
      const textX = closestPoint.x;
      const textY = closestPoint.y;
      const text = feature.name;
      // Draw a cross at the closest coordinates
      const denom = 20;
      const size = (PPC / denom) * (1 / zoomLevel);
      drawCross(ctx, textX, textY, size, 'rgba(255, 0, 0, 1)');
      // Draw the name
      ctx.fillStyle = 'rgba(255, 0, 0, 1)';
      // ctx.strokeStyle = 'rgba(0, 0, 0, 1)';
      ctx.lineWidth = lineWidth;
      ctx.font = `${scaledFontSize*0.75}px Arial`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      // ctx.strokeText(text, textX, textY);
      ctx.fillText(text, textX, textY - size * 4);
    });
  }

  // Draw the last mouse point as a big cross
  if (lastMousePoint) {
    drawCross(ctx, lastMousePoint.x, lastMousePoint.y, 30/zoomLevel, 'rgba(255, 0, 0, 1)');
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

// Draw a cross shape
function drawCross(ctx, x, y, size, color) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2 / zoomLevel;
  ctx.beginPath();
  ctx.moveTo(x - size, y - size);
  ctx.lineTo(x + size, y + size);
  ctx.moveTo(x + size, y - size);
  ctx.lineTo(x - size, y + size);
  ctx.stroke();
  ctx.restore();
}

// Draw grid lines (each cell is PPC pixels)
function drawGrid() {
  ctx.save();
  ctx.strokeStyle = 'rgba(200,200,200,1)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= arenaWidth; i++) {
    ctx.beginPath();
    ctx.moveTo(i * PPC, 0);
    ctx.lineTo(i * PPC, ENV['arena_height'] * PPC);
    ctx.stroke();
  }
  for (let j = 0; j <= ENV['arena_height']; j++) {
    ctx.beginPath();
    ctx.moveTo(0, j * PPC);
    ctx.lineTo(arenaHeight * PPC, j * PPC);
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

// Draw a destination marker (a cross) for the group
function drawDestination(group) {
  const trajectory = group.bhvr.params.trajectory;
  if (!trajectory) return;

  const scale = 1 / zoomLevel; // Inverse scaling to keep size constant

  for (let i = 0; i < trajectory.length; i++) {
    const dest = lonLatToCanvas(trajectory[i][0], trajectory[i][1]);
    const color = group.robots.length > 1 ? COLORS[group.idx % COLORS.length] : '#000';
    
    // Convert destination to pixel space
    const px = dest.x;
    const py = dest.y;
    
    // Scale the cross size
    const denom = 10;
    const size = (PPC / denom) * scale;
    
    ctx.save();

    // Draw cross shape
    drawCross(ctx, px, py, size, color);

    // Draw index label with a circle background
    ctx.fillStyle = color;
    ctx.font = `${30 * scale}px Arial`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(i, px, py - size * 2);

    // Draw line connecting destinations
    // if (i > 0) {
    //   const prevDest = lonLatToCanvas(trajectory[i - 1][0], trajectory[i - 1][1]);
    //   const prevPx = prevDest.x;
    //   const prevPy = prevDest.y;
    //   ctx.beginPath();
    //   ctx.moveTo(prevPx, prevPy);
    //   ctx.lineTo(px, py);
    //   ctx.strokeStyle = color;
    //   ctx.lineWidth = 2 * scale;
    //   ctx.stroke();
    // }

    ctx.restore();
  }
}


// Draw a robot with a directional line, index label, and target cross
function drawRobot(group, robot, color) {
  // Adjust scale based on zoom level
  const scale = 1 / zoomLevel;

  // Convert robot coordinates to canvas coordinates
  const canvasCoords = lonLatToCanvas(robot.x, robot.y);
  const canvasCoordsTarget = lonLatToCanvas(robot.target_x, robot.target_y);

  // Convert angle since y-axis is inverted
  // Source angle is in degrees from 0 to 360 where 0 is north and 90 is east
  // Canvas angle is in radians from 0 to 2*PI where 0 is east and PI/2 is north
  const canvasAngle =  (robot.angle * Math.PI / 180) - Math.PI / 2;
  const canvasTargetAngle = (robot.target_angle * Math.PI / 180) - Math.PI / 2;

  // Convert coordinates to pixels
  const px = canvasCoords.x;
  const py = canvasCoords.y;
  const endX = px + Math.cos(canvasAngle) * (PPC / 4) * scale;
  const endY = py + Math.sin(canvasAngle) * (PPC / 4) * scale;
  const targetX = canvasCoordsTarget.x;
  const targetY = canvasCoordsTarget.y;

  // // Draw a dashed linevery far from robot in the direction of the target
  // ctx.setLineDash([5, 5]);
  // ctx.strokeStyle = color;
  // ctx.lineWidth = 1 * scale;
  // ctx.beginPath();
  // ctx.moveTo(px, py);
  // ctx.lineTo(px + Math.cos(canvasTargetAngle) * (PPC * 100 * scale), py + Math.sin(canvasTargetAngle) * (PPC * 100 * scale));
  // ctx.stroke();
  // ctx.setLineDash([]);

  // // Draw a dashed line from robot in the direction of the angle 
  // ctx.setLineDash([5, 5]);
  // ctx.strokeStyle = color;
  // ctx.lineWidth = 1 * scale;
  // ctx.beginPath();
  // ctx.moveTo(px, py);
  // ctx.lineTo(px + Math.cos(canvasAngle) * (PPC * 100 * scale), py + Math.sin(canvasAngle) * (PPC * 100 * scale));
  // ctx.stroke();
  // ctx.setLineDash([]);

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

  if (group.bhvr.name === "form_and_follow_trajectory" || group.bhvr.name === "random_walk") {
    // Target marker (small cross)
    const denom = 20;
    const size = (PPC / denom) * scale;
    ctx.strokeStyle = color;
    ctx.lineWidth = 3 * scale;
    ctx.beginPath();
    ctx.moveTo(targetX - size, targetY - size);
    ctx.lineTo(targetX + size, targetY + size);
    ctx.moveTo(targetX + size, targetY - size);
    ctx.lineTo(targetX - size, targetY + size);
    ctx.stroke();
  } else if (group.bhvr.name === "cover_shape") {
    // Get the segment and segment index
    const segment = group.bhvr.data.segments[robot.idx];
    const segmentIdx = group.bhvr.data.segment_indexes[robot.idx];
    // Draw the segment
    ctx.strokeStyle = color;
    ctx.lineWidth = 2 * scale;
    ctx.beginPath();
    for (let i = 0; i < segment.length; i++) {
      const point = lonLatToCanvas(segment[i][0], segment[i][1]);
      const pointX = point.x;
      const pointY = point.y;
      if (i === 0) {
        ctx.moveTo(pointX, pointY);
      } else {
        ctx.lineTo(pointX, pointY);
      }
    }
    ctx.stroke();

    // Show current and next point
    if (segmentIdx > 0 && segmentIdx < group.bhvr.data.segments[robot.idx].length) {
      const currentPointCoords = group.bhvr.data.segments[robot.idx][segmentIdx];
      const currentPoint = lonLatToCanvas(currentPointCoords[0], currentPointCoords[1]);
      const denom = 20;
      const size = (PPC / denom) * scale;
      drawCross(ctx, currentPoint.x, currentPoint.y, size, color);
    }
    if (segmentIdx > 0 && segmentIdx < group.bhvr.data.segments[robot.idx].length - 1) {
      const nextPointCoords = group.bhvr.data.segments[robot.idx][segmentIdx + 1];
      const nextPoint = lonLatToCanvas(nextPointCoords[0], nextPointCoords[1]);
      const denom = 20;
      const size = (PPC / denom) * scale;
      drawCross(ctx, nextPoint.x, nextPoint.y, size, color);
    }
  }

  ctx.restore();
}


// Draw simulation status overlay (header and each group’s status)
function drawStatus() {
  ctx.save();
  
  // Reset transformations to ensure fixed positioning
  ctx.setTransform(1, 0, 0, 1, 0, 0);

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
      case "cover_shape":
        behaviorString = `Cover Shape`;
        break;
      default:
        behaviorString = "None";
    }

    const robotsInGroup = Array.from(group.robots).map(robot => robot.idx).join(',');
    const statusText = `G${group.idx} [${robotsInGroup}] -> ${behaviorString}`;
    const color = group.robots.length > 1 ? COLORS[group.idx % COLORS.length] : '#000';

    ctx.fillStyle = color;
    // ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
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

  drawCanvas();
  // drawGrid();
  groups.forEach(group => {
    drawDestination(group);
    group.robots.forEach(robot => {
      const color = group.robots.length > 1 ? COLORS[group.idx % COLORS.length] : '#000';
      drawRobot(group, robot, color);
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

function calculateVisibleArena() {
  // Compute visible arena bounds
  visibleArenaMinX = -panOffsetX / (PPC * zoomLevel);
  visibleArenaMinY = -panOffsetY / (PPC * zoomLevel);
  visibleArenaMaxX = (-panOffsetX + canvas.width) / (PPC * zoomLevel);
  visibleArenaMaxY = (-panOffsetY + canvas.height) / (PPC * zoomLevel);
  // console.log('Visible arena:', visibleArenaMinX, visibleArenaMinY, visibleArenaMaxX, visibleArenaMaxY);
}

// Is coord inside polygon?
// Check if a point is inside a polygon given by its vertices
function checkIfPointInPolygon(point, perimeterPoints) {
  const [x, y] = point; // Extract point coordinates
  let inside = false;

  for (let i = 0, j = perimeterPoints.length - 1; i < perimeterPoints.length; j = i++) {
      const [xi, yi] = [perimeterPoints[i].x, perimeterPoints[i].y]; // Current vertex
      const [xj, yj] = [perimeterPoints[j].x, perimeterPoints[j].y]; // Previous vertex

      // Check if the point is between the y-values of two edges and compute intersection
      const intersect = ((yi > y) !== (yj > y)) &&
                        (x < (xj - xi) * (y - yi) / (yj - yi) + xi);

      if (intersect) {
          inside = !inside;
      }
  }
  return inside;
}

// Get closest features to canvas coords
function getClosestFeatures(x, y) {
  // Compute distances and store in an array
  let distances = [];

  geoJSON.features.forEach(feature => {
    let points = feature.geometry.points;
    let holes = [];
    let closestDistance = Infinity;
    let closestPoint = null;

    if (feature.geometry.type === 'MultiPolygon' || feature.geometry.type === 'Polygon') {
      if (feature.geometry.type === 'MultiPolygon') {
        holes = points.slice(1);
        points = points[0];
      } else {
        points = points;
      }
      // Check if x and y are in the polygon
      const isPointInPolygon = checkIfPointInPolygon([x, y], points);
      if (isPointInPolygon) {
        // Check if x and y are in any of the holes
        let pointHole = null;
        holes.forEach(hole => {
          // console.log('Checking hole:', hole);
          if (checkIfPointInPolygon([x, y], hole)) {
            pointHole = hole;
          }
        });
        // console.log('Is point in ' + feature.properties.unique_name + ' polygon:', isPointInPolygon);
        if (pointHole) {
          // console.log('Is point in hole:', pointHole);
          pointHole.forEach(point => {
            const distance = Math.sqrt(Math.pow(x - point.x, 2) + Math.pow(y - point.y, 2));
            if (distance < closestDistance) {
              closestDistance = distance;
              closestPoint = point;
            }
          });
        } else {
          closestDistance = 0;
          closestPoint = { x, y };
        }
      } else {
        points.forEach(point => {
          const distance = Math.sqrt(Math.pow(x - point.x, 2) + Math.pow(y - point.y, 2));
          if (distance < closestDistance) {
            closestDistance = distance;
            closestPoint = point;
          }
        });
      }
    }

    if (feature.geometry.type === 'LineString') {
      points.forEach(point => {
        const distance = Math.sqrt(Math.pow(x - point.x, 2) + Math.pow(y - point.y, 2));
        if (distance < closestDistance) {
          closestDistance = distance;
          closestPoint = point;
        }
      });
    }

    if (feature.geometry.type === 'Point') {
      const point = points[0];
      const distance = Math.sqrt(Math.pow(x - point.x, 2) + Math.pow(y - point.y, 2));
      if (distance < closestDistance) {
        closestDistance = distance;
        closestPoint = point;
      }
    }

    if (closestDistance < Infinity) {
      distances.push({ 
        name: feature.properties.unique_name, 
        distance: closestDistance, 
        closestPoint: closestPoint
      });
    }
  });

  // Sort by distance and return the top 3 closest feature names
  return distances.sort((a, b) => a.distance - b.distance).slice(0, 5);
}

// -------------------- Event Listeners --------------------

function addCanvasEventListeners() {
  const canvasContainer = document.getElementById('canvasContainer');
  const resetButton = document.getElementById('resetViewButton');

  // Event listeners for zoom and pan
  canvasContainer.addEventListener('wheel', (e) => {
    e.preventDefault();

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Convert mouse position relative to the zoomed/panned space
    const preZoomMouseX = (mouseX - panOffsetX) / zoomLevel;
    const preZoomMouseY = (mouseY - panOffsetY) / zoomLevel;

    // Zooming logic
    const zoomChange = e.deltaY > 0 ? -zoomSensitivity : zoomSensitivity;
    const newZoomLevel = Math.max(0.5, zoomLevel + zoomChange);

    // Adjust pan offset to keep the zoom centered on the cursor
    panOffsetX = mouseX - preZoomMouseX * newZoomLevel;
    panOffsetY = mouseY - preZoomMouseY * newZoomLevel;

    zoomLevel = newZoomLevel;

    // Compute visible arena bounds
    calculateVisibleArena();

    updateDisplay();
  });


  // Double click event
  canvas.addEventListener('dblclick', (e) => {
    // Prevent default double-click behavior
    e.preventDefault();

    // Get the mouse position relative to the canvas
    const rect = canvas.getBoundingClientRect();
    // Take into account the zoom level and pan offset
    const arenaX = ((e.clientX - rect.left) / rect.width) * (visibleArenaMaxX - visibleArenaMinX) + visibleArenaMinX;
    const arenaY = ((e.clientY - rect.top) / rect.height) * (visibleArenaMaxY - visibleArenaMinY) + visibleArenaMinY;
    const px = arenaX * PPC;
    const py = arenaY * PPC;
    // console.log('Double clicked at:', px, py);
    lastMousePoint = { x: px, y: py };
    lastMouseCoords = canvasToLonLat(px, py);

    // Get the closest feature to the clicked point
    closestFeatures = getClosestFeatures(px, py);
  });

  // Reset dblclick variables when double-clicking outside the canvas
  document.addEventListener('dblclick', (e) => {
    if (!canvas.contains(e.target)) {
      lastMousePoint = null;
      lastMouseCoords = null;
      closestFeatures = [];
    }
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

    // Compute visible arena bounds
    calculateVisibleArena();
    
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
}

// -------------------- Initialization --------------------
addCanvasEventListeners();