// main.js

// -------------------- Helper Functions --------------------
function sleep(ms) {
  return new Promise(resolve => {
    setTimeout(resolve, ms);
  });
}

// -------------------- Constants --------------------
const TIME_STEP = 25; // Time step in milliseconds

// -------------------- Simulation Variables --------------------

let n_updates = 0;
let current_step = 0;
let running = false;
let simData = {};
let groups = [];

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
    running = simData.running;
    current_step = simData.current_step;
    groups = simData.groups;
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

function addEventListeners() {
  // Focus on the chat input field when the page loads
  document.getElementById("chatInput").focus();
  // Add event listener to the chat input field
  document.getElementById("chatInput").addEventListener("focus", () => {
    document.getElementById("chatInput").select();
  });
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

  // Add event listener to showOverpassButton and showHardcodedButton
  document.getElementById("showOverpassButton").addEventListener("click", () => {
    const iconEl = document.getElementById("showOverpassButton").querySelector(".icon");
    const textEl = document.getElementById("showOverpassButton").querySelector(".text");
    textEl.innerHTML = showOverpass ? "Overpass" : "<b>Overpass</b>";
    iconEl.innerHTML = showOverpass ? '<i class="fas fa-eye"></i>' : '<i class="fas fa-eye-slash"></i>';
    showOverpass = !showOverpass;
    updateDisplay();
  });
  document.getElementById("showHardcodedButton").addEventListener("click", () => {
    const iconEl = document.getElementById("showHardcodedButton").querySelector(".icon");
    iconEl.innerHTML = showHardcoded ? '<i class="fas fa-eye"></i>' : '<i class="fas fa-eye-slash"></i>';
    showHardcoded = !showHardcoded;
    updateDisplay();
  });

  // Add event listener to showNamesButton and showShapesButton
  document.getElementById("showNamesButton").addEventListener("click", () => {
    const iconEl = document.getElementById("showNamesButton").querySelector(".icon");
    iconEl.innerHTML = showNames ? '<i class="fas fa-eye"></i>' : '<i class="fas fa-eye-slash"></i>';
    showNames = !showNames;
    updateDisplay();
  });
  document.getElementById("showShapesButton").addEventListener("click", () => {
    const iconEl = document.getElementById("showShapesButton").querySelector(".icon");
    iconEl.innerHTML = showShapes ? '<i class="fas fa-eye"></i>' : '<i class="fas fa-eye-slash"></i>';
    showShapes = !showShapes;
    updateDisplay();
  });

  // Add event listener to the satellite button
  document.getElementById("satellite-button").onclick = () => {
    const iconEl = document.getElementById("satellite-button").querySelector(".icon");
    iconEl.innerHTML = useSatellite ? '<i class="fas fa-satellite-dish"></i>' : '<i class="fas fa-eye-slash"></i>';
    useSatellite = !useSatellite;
    updateDisplay();
  };

  // Add event listener to click pause button on ctrl + space
  document.addEventListener('keydown', (event) => {
    if (event.ctrlKey && event.key === ' ') {
      event.preventDefault(); // Prevent scrolling
      togglePause();
    }
  });

  // Add canvas sizing event listener
  window.addEventListener('resize', () => {
    setCanvasSize();
    updateDisplay();
  });
}

// Wait until ENV, CONFIG and geoJSON are loaded (different from null values)
if (ENV === null || CONFIG === null || geoJSON === null) {
  console.log("Fetching initialization data...");
  // Fetch simulation config and GeoJSON data (once both are resolved, proceed with initialization)
  Promise.all([fetchConfig(), loadAndProcessGeoJSON()])
    .then(() => {
      console.log("Initialization complete");
      addEventListeners();
      fetchState();
    });   
}
// If ENV, CONFIG and geoJSON are already loaded
else {
  // Fetch initial state
  fetchState();
}
