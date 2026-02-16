// Global variable to track the currently selected SVG object
// This can be accessed from both index.js and canvas.js
window.selectedSvgObject = {
    id: null,           // ID of the selected SVG object
    type: null,         // Type of object (rectangle, circle, etc.)
    data: null          // Additional data about the object
};

// Get reference to the iframe
const canvasFrame = document.getElementById('canvasFrame');

// Function to send messages to the canvas iframe
function sendMessageToCanvas(type, data) {
    if (canvasFrame && canvasFrame.contentWindow) {
        canvasFrame.contentWindow.postMessage({
            type: type,
            data: data
        }, '*'); // In production, replace '*' with specific origin
    }
}

// Load Image Button
document.getElementById('loadImageBtn').addEventListener('click', () => {
    console.log('Load image button clicked');
    // Trigger file input in canvas
    sendMessageToCanvas('TRIGGER_IMAGE_UPLOAD', null);
});

// Get SVG Objects Button
document.getElementById('getSvgBtn').addEventListener('click', async () => {
    console.log('Get SVG objects button clicked');
    try {
        // Send GET request to backend
        const response = await fetch('/api/get-svg-objects');
        const data = await response.json();
        
        if (data.svgObjects) {
            // Send SVG objects to canvas for display
            sendMessageToCanvas('DISPLAY_SVG', data.svgObjects);
            console.log('SVG objects sent to canvas:', data.svgObjects);
        }
    } catch (error) {
        console.error('Error fetching SVG objects:', error);
    }
});

// Refresh SVG Objects Button (Erase and get new)
document.getElementById('refreshSvgBtn').addEventListener('click', async () => {
    console.log('Refresh SVG objects button clicked');
    try {
        // First, erase existing SVG objects
        sendMessageToCanvas('ERASE_SVG', null);
        
        // Then fetch new SVG objects
        const response = await fetch('/api/get-new-svg-objects');
        const data = await response.json();
        
        if (data.svgObjects) {
            // Send new SVG objects to canvas
            sendMessageToCanvas('DISPLAY_SVG', data.svgObjects);
            console.log('New SVG objects sent to canvas:', data.svgObjects);
        }
    } catch (error) {
        console.error('Error fetching new SVG objects:', error);
    }
});

// Listen for messages from canvas (including selection updates)
window.addEventListener('message', (event) => {
    const { type, data } = event.data;
    
    if (type === 'SVG_OBJECT_SELECTED') {
        // Update global variable with selected object info
        window.selectedSvgObject = {
            id: data.id,
            type: data.type,
            data: data
        };
        
        // Update the UI to show selected object
        document.getElementById('selectedObjectId').textContent = 
            `ID: ${data.id} | Type: ${data.type}`;
        
        console.log('SVG object selected:', window.selectedSvgObject);
    }
});