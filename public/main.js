// main.js

// -------------------- Helper Functions --------------------
function sleep(ms) {
  return new Promise(resolve => {
    setTimeout(resolve, ms);
  });
}

// -------------------- Constants --------------------
const PPC = 100; // Pixels per cell
const TIME_STEP = 10; // Time step in milliseconds


// -------------------- Visualization constants --------------------
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

// -------------------- Canvas and Simulation Data --------------------
const canvasContainer = document.getElementById('canvasContainer');
const canvas = document.getElementById('canvas');
const resetButton = document.getElementById('resetViewButton');
const ctx = canvas.getContext('2d');

// Canvas interaction variables
let zoomLevel = 1;
const zoomSensitivity = 0.1; // Adjust for zoom speed

let panOffsetX = 0;
let panOffsetY = 0;
let isDragging = false;
let dragStartX, dragStartY;

// Simulation variables
let arena = {};
let n_updates = 0;
let current_step = 0;
let running = false;
let simData = {};
let groups = [];

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

  for (let i = 0; i < trajectory.length; i++) {
    const dest = trajectory[i];
    const color = group.robots.length > 1 ? COLORS[group.idx%COLORS.length] : '#000';
    const px = dest[0] * PPC;
    const py = dest[1] * PPC;
    const denom = 5;
    const size = PPC / denom

    // Draw cross shape
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(px - size, py - size);
    ctx.lineTo(px + size, py + size);
    ctx.moveTo(px + size, py - size);
    ctx.lineTo(px - size, py + size);
    ctx.stroke();
    ctx.restore();

    // Draw index label with a circle background
    ctx.save();
    ctx.fillStyle = color;
    ctx.font = '30px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(i, px, py - size*2);
    ctx.restore();
  }
}

// Draw a robot with a directional line, index label, and target cross
function drawRobot(robot, color) {
  const px = robot.x * PPC;
  const py = robot.y * PPC;
  const endX = px + Math.cos(robot.angle) * PPC/2;
  const endY = py + Math.sin(robot.angle) * PPC/2;
  const targetX = robot.target_x * PPC;
  const targetY = robot.target_y * PPC;

  ctx.save();
  // Directional line
  ctx.strokeStyle = color;
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(px, py);
  ctx.lineTo(endX, endY);
  ctx.stroke();

  // Robot body (circle with inner white circle and battery level circle)
  // Outer circle
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(px, py, 25, 0, 2 * Math.PI);
  ctx.fill();
  // Battery level circle (pie chart)
  const batteryLevel = robot.battery_level;
  const batteryLevelAngle = batteryLevel * 2 * Math.PI;
  ctx.fillStyle = '#000'; // Or any color you want for the battery
  ctx.beginPath();
  ctx.moveTo(px, py); // Move to the center of the circle
  ctx.arc(px, py, 22, -Math.PI / 2, -Math.PI / 2 + batteryLevelAngle); // Draw the arc from the top
  ctx.closePath(); // Close the path to create a pie slice
  ctx.fill();
  // Inner circle
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  ctx.arc(px, py, 19, 0, 2 * Math.PI);
  ctx.fill();

  // Robot index label
  ctx.fillStyle = '#000';
  ctx.font = '30px Arial';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(robot.idx, px, py);

  // Target marker (small cross)
  const denom = 10;
  const size = PPC / denom;
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
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
  ctx.fillStyle = '#000';
  ctx.font = 'bold 30px Arial';
  ctx.textAlign = 'left';
  let yOffset = 10;
  // ctx.fillText(`Simulation Step: ${current_step}`, 10, yOffset);
  yOffset += 30;

  // Create behavior description string
  let behaviorString = ""
  groups.forEach((group, idx) => {
    switch (group.bhvr.name) {
      case "random_walk":
        behaviorString = "Random Walk";
        break;
      case "form_and_follow_trajectory":
        trajectoryStr = "";
        for (let i = 0; i < group.bhvr.params.trajectory.length; i++) {
          const dest = group.bhvr.params.trajectory[i];
          trajectoryStr += `(${dest[0]}, ${dest[1]})`;
          if (i < group.bhvr.params.trajectory.length - 1) {
            trajectoryStr += " → ";
          }
        }
        behaviorString = "Form & Follow Trajectory (" + group.bhvr.params.formation_shape + ")" + " [" + trajectoryStr + "]";
        break;
      default:
        behaviorString = "None";
    }

    const robotsInGroup = Array.from(group.robots).map(robot => robot.idx).join(',');
    const statusText = `G${group.idx} [${robotsInGroup}] -> ${behaviorString}`;
    const color = group.robots.length > 1 ? COLORS[group.idx%COLORS.length] : '#000';
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

  drawMap();
  drawGrid();
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
  const newZoomLevel = Math.max(0.1, zoomLevel + zoomChange);

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

// -------------------- Expandable Tree View Functions --------------------
const nodeStates = {};
const elementMap = new Map();
const defaultState = "expanded";

function reconcileTree(parentEl, data, parentPath = "") {
  const existingNodes = new Map();
  Array.from(parentEl.children).forEach(li => {
    const path = li.dataset.path;
    if (path) existingNodes.set(path, li);
  });

  const keys = Object.keys(data).sort((keyA, keyB) => {
    const isParent = value => typeof value === "object" && value !== null;
    const parentStatusDiff = Number(isParent(data[keyA])) - Number(isParent(data[keyB]));
  
    // Natural sort for alphanumeric keys using Intl.Collator
    const collator = new Intl.Collator(undefined, {
      numeric: true,
      sensitivity: 'base'
    });
  
    return parentStatusDiff || collator.compare(keyA, keyB);
  });  

  // Process current nodes (first over ones without children)
  const usedPaths = new Set();
  for (const key of keys) {
    if (!data.hasOwnProperty(key)) continue;
    
    const currentPath = parentPath ? `${parentPath}.${key}` : key;
    usedPaths.add(currentPath);
    
    let li = existingNodes.get(currentPath) || document.createElement('li');
    if (!li.parentElement) parentEl.appendChild(li);
    
    li.dataset.path = currentPath;
    elementMap.set(currentPath, li);

    if (typeof data[key] === "object" && data[key] !== null) {
      updateObjectNode(li, key, data[key], currentPath);
    } else {
      updateLeafNode(li, key, data[key], currentPath);
    }
  }

  // Remove deleted nodes
  existingNodes.forEach((li, path) => {
    if (!usedPaths.has(path)) {
      li.remove();
      elementMap.delete(path);
      delete nodeStates[path];
    }
  });
}

function updateObjectNode(li, key, value, currentPath) {
  let toggleSpan = li.querySelector('.keytoggle');
  let labelSpan = li.querySelector('.keylabel');
  let childUl = li.querySelector('ul');

  // Initialize if new node
  if (!toggleSpan) {
    toggleSpan = document.createElement('span');
    toggleSpan.className = 'keytoggle';
    li.prepend(toggleSpan);
  }

  if (!labelSpan) {
    labelSpan = document.createElement('span');
    labelSpan.className = 'keylabel';
    toggleSpan.after(labelSpan);
  }

  if (!childUl) {
    childUl = document.createElement('ul');
    li.append(childUl);
  }

  // Update state
  const isExpanded = nodeStates[currentPath] ?? (defaultState === "expanded");
  toggleSpan.textContent = isExpanded ? "[-] " : "[+] ";
  labelSpan.textContent = `${key}: `;
  childUl.classList.toggle('is-hidden', !isExpanded);

  // Update toggle handler
  toggleSpan.onclick = (e) => {
    const wasExpanded = childUl.classList.toggle('is-hidden');
    nodeStates[currentPath] = !wasExpanded;
    toggleSpan.textContent = wasExpanded ? "[+] " : "[-] ";
    e.stopPropagation();
  };

  // Same handler for label
  labelSpan.onclick = (e) => {
    const wasExpanded = childUl.classList.toggle('is-hidden');
    nodeStates[currentPath] = !wasExpanded;
    toggleSpan.textContent = wasExpanded ? "[+] " : "[-] ";
    e.stopPropagation();
  };

  // Recurse with empty object protection
  reconcileTree(childUl, value || {}, currentPath);
}

function formatValue(value) {
  if (typeof value === "number") {
    let formatted = value.toFixed(3); // Format to 3 decimal places

    // Remove trailing zeros and decimal point if unnecessary
    if (formatted.endsWith('.000')) {
      return parseInt(formatted); // Convert to integer if no decimal part
    } else {
      return parseFloat(formatted); // Convert to float, removes trailing zeros
    }

  } else if (typeof value === "object" && value !== null) {
    return JSON.stringify(value);
  } else {
    return value; // Return original value for other types
  }
}

function updateLeafNode(li, key, value, currentPath) {
  // Remove any nested elements
  li.querySelectorAll('.keytoggle, .keylabel, ul').forEach(el => el.remove());
  value = formatValue(value);
  li.innerHTML= `<b>${key}</b>: ${value}`;
}

function updateTreeView(data) {
  const treeContainer = document.getElementById("treeView");
  
  if (!treeContainer.firstElementChild || 
      treeContainer.firstElementChild.tagName !== 'UL') {
    treeContainer.innerHTML = '<ul></ul>';
  }
  
  reconcileTree(treeContainer.querySelector('ul'), data);
}

// -------------------- Speech Recognition Functionality --------------------
let recordingOngoing = false;
const recordButton = document.getElementById('recordButton');
const chatInput = document.getElementById('chatInput');

if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognition = new SpeechRecognition();

  recognition.lang = 'en-US';
  recognition.continuous = false;
  recognition.interimResults = false;

  recognition.onstart = () => {
    console.log('Speech recognition started');
    recordingOngoing = true;
    recordButton.innerHTML = '<i class="fas fa-spinner fa-spin has-text-link"></i>';
    chatInput.value = '';
    chatInput.placeholder = 'Listening...';
  };

  recognition.onspeechend = () => {
    console.log('Speech recognition ended');
    recognition.stop();
    recordingOngoing = false;
    recordButton.innerHTML = '<i class="fas fa-microphone has-text-link"></i>';
    chatInput.placeholder = 'Type message or press CTRL+R to record...';
    chatInput.focus();
  };

  recognition.onerror = (event) => {
    console.error('Speech recognition error:', event.error);
    recordingOngoing = false;
    recordButton.innerHTML = '<i class="fas fa-microphone has-text-link"></i>';
    chatInput.placeholder = 'Type message or press CTRL+R to record...';
  };

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    console.log('Transcript:', transcript);
    chatInput.value = transcript;
    recordButton.innerHTML = '<i class="fas fa-microphone has-text-link"></i>';
    chatInput.placeholder = 'Type message or press CTRL+R to record...';
    recordingOngoing = false;
  };

  // Event listener for the record button
  recordButton.addEventListener('click', () => {
    if (!recordingOngoing) {
      console.log('Starting speech recognition');
      recognition.start();
    } else {
      console.log('Stopping speech recognition');
      recognition.stop();
    }
  });

  // Event listenner for key "CTRL+R" to start/stop recording
  document.addEventListener('keydown', (event) => {
    if ((event.ctrlKey && event.key === 'r') || (event.ctrlKey && event.key === 'R')) {
      if (!recordingOngoing) {
        event.preventDefault();
        console.log('Starting speech recognition');
        recognition.start();
      } else {
        event.preventDefault();
        console.log('Stopping speech recognition');
        recognition.stop();
      }
    }
  });
} else {
  console.error("Speech recognition is not supported in this browser.");
  document.getElementById('result').textContent = "Speech recognition is not supported in this browser.";
}

// -------------------- Chat Functionality --------------------

async function sendMessage(message) {
  try {
    await fetch('/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message })
    });
  } catch (err) {
    console.error("Message error:", err);
  }
}

function sendChat() {
  const chatInput = document.getElementById("chatInput");
  const chatLog = document.getElementById("chatLog");
  const message = chatInput.value.trim();
  if (message) {
    // Append the user message to the chat log
    const userMessage = document.createElement("div");
    userMessage.classList.add("chat-message", "user-message");
    userMessage.textContent = message;
    chatLog.appendChild(userMessage);
    chatInput.value = "";

    // Scroll to the bottom of the chat log
    chatLog.scrollTop = chatLog.scrollHeight;
    
    // Send the message to the server
    sendMessage(message);
  }
}

chatData = [];
function fetchChat() {
  fetch('/chat')
    .then(response => response.json())
    .then(data => {
      if (JSON.stringify(data) === JSON.stringify(chatData)) {
        return;
      }

      chatData = data;

      const chatLog = document.getElementById("chatLog");
      chatLog.innerHTML = ""; // Clear chat log

      data.forEach(message => {
        const messageElement = document.createElement("div");
        messageElement.classList.add("chat-message", message.role === "user" ? "user-message" : "ai-message");

        // Convert Markdown to HTML
        messageElement.innerHTML = marked.parse(message.content, breaks=true);
        
        chatLog.appendChild(messageElement);
      });

      chatLog.scrollTop = chatLog.scrollHeight;
    })
    .catch(error => console.error("Error fetching chat:", error));
}

// -------------------- Server Communication & Control Functions --------------------
async function fetchState() {
  // Get current timestamp
  const start_time = Date.now();
  try {
    // Fetch current chat
    fetchChat();

    // Fetch current state
    const response = await fetch('/state');
    simData = await response.json();
    arena = simData.arena;
    running = simData.running;
    current_step = simData.current_step;
    groups = simData.groups;
    delete simData.arena;
    delete simData.running;
    delete simData.current_step;
    delete simData.groups;
    

    // Update step counter
    document.getElementById('stepCounter').textContent = `Step: ${current_step}`;

    // If simulation is running
    if (running || n_updates === 0) {
      if (n_updates !== 0) {
        // Modify the pause button
        const pauseButton = document.getElementById("pause-button");
        pauseButton.querySelector(".text").textContent = "Pause";
        pauseButton.querySelector(".icon").innerHTML = '<i class="fas fa-pause"></i>';
      }
      
      // Update canvas, display and tree view
      setCanvasSize();
      updateDisplay();
      updateTreeView(groups);
      // Increase number of updates
      n_updates += 1;
    } else {
      // Modify the pause button
      const pauseButton = document.getElementById("pause-button");
      pauseButton.querySelector(".text").textContent = "Resume";
      pauseButton.querySelector(".icon").innerHTML = '<i class="fas fa-play"></i>';
    }
  } catch (error) {
    console.error("Error fetching state:", error);
  }
  // Request a new animation frame after waiting for at least TIME_STEP milliseconds
  const end_time = Date.now();
  await sleep(Math.max(0, TIME_STEP - (end_time - start_time)));
  requestAnimationFrame(fetchState);
}

async function sendCommand(command) {
  try {
    await fetch('/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command })
    });
  } catch (err) {
    console.error("Command error:", err);
  }
}

// Event listener functions
function togglePause() { sendCommand('pause'); }
function toggleReset() { 
  // Set number of updates to 0
  n_updates = 0;
  // Reset simulation state
  sendCommand('reset'); 
  // Reset view
  resetView();
}


// -------------------- Initialization --------------------

// Add event listener to send button
document.getElementById("sendButton").onclick = sendChat;

// Add event listener to chat input field
document.getElementById("chatInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) { // Avoid Shift+Enter issue
    event.preventDefault();  // Stop default new line
    sendChat();
}
});

// Add event listeners to control buttons
document.getElementById("pause-button").onclick = togglePause;
document.getElementById("reset-button").onclick = toggleReset;

// Add canvas sizing event listener
window.addEventListener('resize', () => {
  setCanvasSize();
  updateDisplay();
});

// Fetch initial state
fetchState();
