// Leaflet map instance
let map;
let heatLayer;
let clickMarker;

// Risk calculation function
function calculateRiskCategory(heatmapArray) {
  if (!Array.isArray(heatmapArray) || heatmapArray.length === 0) {
    return { category: "Unknown", maxRisk: 0, meanRisk: 0 };
  }

  const maxRisk = Math.max(...heatmapArray);
  const meanRisk =
    heatmapArray.reduce((sum, risk) => sum + risk, 0) / heatmapArray.length;

  // Risk category determination based on maximum risk value
  let category;
  if (maxRisk < 0.2) {
    category = "Very Low";
  } else if (maxRisk < 0.4) {
    category = "Low";
  } else if (maxRisk < 0.6) {
    category = "Moderate";
  } else if (maxRisk < 0.8) {
    category = "High";
  } else {
    category = "Very High";
  }

  return {
    category: category,
    maxRisk: maxRisk,
    meanRisk: meanRisk,
  };
}

// Initialize Leaflet map
function initializeMap() {
  console.log("Initializing Leaflet map...");

  // Create map centered on California
  map = L.map("kepler-container").setView([37.7749, -119.4194], 7);

  // Add Stadia Maps tile layer with fallback
  const stadiaLayer = L.tileLayer(
    "https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}.png?api_key=f119a85a-8ab4-429d-b035-788972d2c0e2",
    {
      attribution:
        "© Stadia Maps, © Stamen Design, © OpenMapTiles © OpenStreetMap contributors",
    }
  );

  // Fallback tile layer (OpenStreetMap)
  const osmLayer = L.tileLayer(
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    {
      attribution: "© OpenStreetMap contributors",
    }
  );

  // Try Stadia Maps first, fallback to OSM if it fails
  stadiaLayer.addTo(map);

  // Add fallback layer
  const layerControl = L.control
    .layers({
      "Stadia Terrain": stadiaLayer,
      OpenStreetMap: osmLayer,
    })
    .addTo(map);

  // Add click event listener to the map
  map.on("click", function (e) {
    const lat = e.latlng.lat;
    const lng = e.latlng.lng;

    console.log("Map clicked at:", lat, lng);

    // Remove existing marker if it exists
    if (clickMarker) {
      map.removeLayer(clickMarker);
    }

    // Add a marker at the clicked location
    clickMarker = L.marker([lat, lng])
      .addTo(map)
      .bindPopup(
        `Getting wildfire prediction for:<br>Lat: ${lat.toFixed(
          4
        )}<br>Lng: ${lng.toFixed(4)}`
      )
      .openPopup();

    // Get prediction for the clicked location
    getPredictionForLocation(lat, lng);
  });

  console.log("Leaflet map initialized successfully");
}

// Function to visualize heatmap data
function visualizeHeatmap(apiResponse) {
  console.log("Received API response:", apiResponse);

  // Extract data from the API response
  const { heatmap, location } = apiResponse;

  if (!Array.isArray(heatmap) || heatmap.length !== 4096) {
    console.error("Invalid heatmap data. Expected array of 4096 values.");
    alert("Error: Invalid heatmap data received from backend.");
    return;
  }

  // Use dynamic center coordinates from the API response
  const centerLat = location.latitude;
  const centerLon = location.longitude;

  console.log("Using dynamic center coordinates:", { centerLat, centerLon });

  // Grid parameters - much smaller for focused heatmap
  const gridSize = 64;
  const latSpacing = 0.01; // ~1.1 km per grid cell (reduced from 0.02)
  const lonSpacing = 0.01; // ~1.1 km per grid cell (reduced from 0.02)

  // Calculate grid bounds
  const latStart = centerLat - (gridSize / 2) * latSpacing;
  const lonStart = centerLon - (gridSize / 2) * lonSpacing;

  // Convert flat array to heat points
  const heatPoints = [];

  for (let i = 0; i < gridSize; i++) {
    for (let j = 0; j < gridSize; j++) {
      const index = i * gridSize + j;
      const fire_risk = heatmap[index];

      // Filter out very low risk areas (< 0.05) and scale intensity
      if (fire_risk >= 0.05) {
        const lat = latStart + i * latSpacing;
        const lon = lonStart + j * lonSpacing;

        // Scale the fire risk to a reasonable intensity (0-1 range)
        const intensity = Math.min(fire_risk * 2, 1.0); // Scale up but cap at 1

        heatPoints.push([lat, lon, intensity]);
      }
    }
  }

  console.log(`Created ${heatPoints.length} heat points (filtered from 4096)`);

  // Remove existing heat layer if it exists
  if (heatLayer) {
    map.removeLayer(heatLayer);
  }

  // Create new heat layer with much smaller radius for very focused visualization
  heatLayer = L.heatLayer(heatPoints, {
    radius: 8, // Much smaller radius for tighter heat points
    blur: 4, // Minimal blur for sharp edges
    maxZoom: 15, // Higher zoom for better detail
    gradient: {
      0.1: "blue",
      0.3: "cyan",
      0.5: "lime",
      0.7: "yellow",
      0.9: "orange",
      1.0: "red",
    },
  }).addTo(map);

  // Center the map on the prediction location
  map.setView([centerLat, centerLon], 10);

  // Calculate risk category
  const riskAssessment = calculateRiskCategory(heatmap);
  console.log("Risk assessment:", riskAssessment);

  // Update the UI with risk information
  updateRiskDisplay(riskAssessment, location);

  console.log("Heatmap successfully added to map");
}

// Function to update the risk display in the UI
function updateRiskDisplay(riskAssessment, location) {
  const riskCategoryElement = document.getElementById("risk-category");
  const riskDetailsElement = document.getElementById("risk-details");
  const predictionResultsElement =
    document.getElementById("prediction-results");

  if (
    !riskCategoryElement ||
    !riskDetailsElement ||
    !predictionResultsElement
  ) {
    console.error("Risk display elements not found in DOM");
    return;
  }

  // Update risk category with color coding
  riskCategoryElement.textContent = riskAssessment.category;
  riskCategoryElement.className = riskAssessment.category
    .toLowerCase()
    .replace(" ", "-");

  // Update risk details
  const maxRiskPercent = (riskAssessment.maxRisk * 100).toFixed(1);
  const meanRiskPercent = (riskAssessment.meanRisk * 100).toFixed(1);

  // Handle location display with fallback
  const locationText =
    location && location.city && location.state
      ? `${location.city}, ${location.state}`
      : location && location.latitude && location.longitude
      ? `Lat: ${location.latitude.toFixed(
          4
        )}, Lon: ${location.longitude.toFixed(4)}`
      : "Unknown Location";

  riskDetailsElement.innerHTML = `
    <strong>Max Risk:</strong> ${maxRiskPercent}%<br>
    <strong>Average Risk:</strong> ${meanRiskPercent}%<br>
    <strong>Location:</strong> ${locationText}
  `;

  // Show the prediction results
  predictionResultsElement.style.display = "block";
}

// Function to get prediction for a specific location
function getPredictionForLocation(latitude, longitude) {
  const button = document.getElementById("predict-button");

  // Disable button during request
  button.disabled = true;
  button.textContent = "Generating Prediction...";

  console.log("Sending coordinates to backend:", {
    latitude: latitude,
    longitude: longitude,
  });

  // Send POST request to backend with correct format
  fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      latitude: latitude,
      longitude: longitude,
    }),
    mode: "cors",
  })
    .then((response) => {
      console.log("Response status:", response.status);

      if (!response.ok) {
        return response.text().then((text) => {
          console.error("Error response body:", text);
          throw new Error(
            `HTTP error! status: ${response.status}, body: ${text}`
          );
        });
      }
      return response.json();
    })
    .then((data) => {
      console.log("Prediction successful:", data);

      // Call visualize function with the full API response
      visualizeHeatmap(data);

      // Re-enable button
      button.disabled = false;
      button.textContent = "Generate Wildfire Prediction";
    })
    .catch((error) => {
      console.error("Error fetching prediction:", error);
      alert(
        "Failed to get prediction from backend. Make sure the Flask server is running on http://127.0.0.1:5000"
      );

      // Re-enable button
      button.disabled = false;
      button.textContent = "Generate Wildfire Prediction";
    });
}

// Handle predict button click (now uses sample location)
function handlePredictClick() {
  // Sample coordinates for California (wildfire-prone area)
  const sampleLatitude = 37.7749; // San Francisco latitude
  const sampleLongitude = -119.4194; // Central California longitude

  getPredictionForLocation(sampleLatitude, sampleLongitude);
}

// Initialize the application when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  // Initialize Leaflet map
  initializeMap();

  // Add event listener to predict button
  const predictButton = document.getElementById("predict-button");
  predictButton.addEventListener("click", handlePredictClick);

  console.log("Wildfire Prediction App initialized");
});
