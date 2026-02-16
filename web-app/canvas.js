// Get DOM elements
const imageInput = document.getElementById('imageInput');
const imageCanvas = document.getElementById('imageCanvas');
const svgOverlay = document.getElementById('svgOverlay');

// Store current SVG elements for reference
let currentSvgElements = new Map();

// Initialize canvas
function initCanvas() {
    // Set up message listener from parent window
    window.addEventListener('message', (event) => {
        const { type, data } = event.data;
        
        switch(type) {
            case 'TRIGGER_IMAGE_UPLOAD':
                // Trigger file input click
                imageInput.click();
                break;
                
            case 'DISPLAY_SVG':
                // Display SVG objects
                displaySvgObjects(data);
                break;
                
            case 'ERASE_SVG':
                // Erase all SVG objects
                eraseSvgObjects();
                break;
                
            default:
                console.log('Unknown message type:', type);
        }
    });
}

// Handle image upload
imageInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        
        reader.onload = (e) => {
            // Load image to canvas
            imageCanvas.src = e.target.result;
            imageCanvas.onload = () => {
                console.log('Image loaded successfully');
                // Clear any existing SVG overlays when new image is loaded
                eraseSvgObjects();
            };
        };
        
        reader.readAsDataURL(file);
    }
});

// Function to display SVG objects
function displaySvgObjects(svgObjects) {
    // Clear existing SVG
    eraseSvgObjects();
    
    // Create a new SVG element
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    
    // Set SVG attributes to match canvas size
    svg.setAttribute('width', imageCanvas.width || 800);
    svg.setAttribute('height', imageCanvas.height || 600);
    svg.setAttribute('viewBox', `0 0 ${imageCanvas.width || 800} ${imageCanvas.height || 600}`);
    
    // Add each SVG object to the SVG element
    svgObjects.forEach((obj, index) => {
        let element;
        
        // Create different SVG elements based on type
        switch(obj.type) {
            case 'rect':
                element = createRectangle(obj);
                break;
            case 'circle':
                element = createCircle(obj);
                break;
            case 'path':
                element = createPath(obj);
                break;
            default:
                console.warn('Unknown SVG type:', obj.type);
                return;
        }
        
        // Add unique ID and make it selectable
        const objId = obj.id || `svg-obj-${Date.now()}-${index}`;
        element.setAttribute('id', objId);
        element.setAttribute('data-type', obj.type);
        element.setAttribute('data-selected', 'false');
        
        // Make SVG element clickable for selection
        element.addEventListener('click', (e) => {
            e.stopPropagation();
            selectSvgObject(objId, obj.type, obj);
        });
        
        // Add hover effect
        element.addEventListener('mouseenter', (e) => {
            e.target.style.cursor = 'pointer';
            e.target.style.opacity = '0.8';
        });
        
        element.addEventListener('mouseleave', (e) => {
            e.target.style.opacity = '1';
        });
        
        svg.appendChild(element);
        
        // Store reference
        currentSvgElements.set(objId, element);
    });
    
    // Append SVG to overlay
    svgOverlay.innerHTML = '';
    svgOverlay.appendChild(svg);
}

// Helper functions to create SVG elements
function createRectangle(obj) {
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', obj.x || 0);
    rect.setAttribute('y', obj.y || 0);
    rect.setAttribute('width', obj.width || 50);
    rect.setAttribute('height', obj.height || 50);
    rect.setAttribute('fill', obj.fill || 'rgba(255, 0, 0, 0.3)');
    rect.setAttribute('stroke', obj.stroke || 'red');
    rect.setAttribute('stroke-width', obj.strokeWidth || 2);
    return rect;
}

function createCircle(obj) {
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', obj.cx || 50);
    circle.setAttribute('cy', obj.cy || 50);
    circle.setAttribute('r', obj.r || 25);
    circle.setAttribute('fill', obj.fill || 'rgba(0, 255, 0, 0.3)');
    circle.setAttribute('stroke', obj.stroke || 'green');
    circle.setAttribute('stroke-width', obj.strokeWidth || 2);
    return circle;
}

function createPath(obj) {
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', obj.d || 'M10 10 L50 10 L50 50 Z');
    path.setAttribute('fill', obj.fill || 'rgba(0, 0, 255, 0.3)');
    path.setAttribute('stroke', obj.stroke || 'blue');
    path.setAttribute('stroke-width', obj.strokeWidth || 2);
    return path;
}

// Function to select an SVG object
function selectSvgObject(id, type, data) {
    // Deselect all objects first
    currentSvgElements.forEach((element, elementId) => {
        element.setAttribute('data-selected', 'false');
        element.setAttribute('stroke', element.getAttribute('stroke') || 'black');
        element.setAttribute('stroke-width', '2');
    });
    
    // Select the clicked object
    const selectedElement = currentSvgElements.get(id);
    if (selectedElement) {
        selectedElement.setAttribute('data-selected', 'true');
        selectedElement.setAttribute('stroke', 'gold');
        selectedElement.setAttribute('stroke-width', '4');
    }
    
    // Send selection info to parent window
    window.parent.postMessage({
        type: 'SVG_OBJECT_SELECTED',
        data: {
            id: id,
            type: type,
            ...data
        }
    }, '*'); // In production, replace '*' with specific origin
}

// Function to erase all SVG objects
function eraseSvgObjects() {
    svgOverlay.innerHTML = '';
    currentSvgElements.clear();
    
    // Notify parent that selection is cleared
    window.parent.postMessage({
        type: 'SVG_OBJECT_SELECTED',
        data: {
            id: null,
            type: null
        }
    }, '*');
}

// Initialize canvas
initCanvas();