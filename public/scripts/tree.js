// tree.js

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