// Use bare module imports (will be resolved by importmap)
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { scene } from "./canva.js";

// Initial reference joint positions - will be updated with received keypoints
let jointPositions = {
  Hips: [0, 0, 0],
  Spine1: [0.004583, 0.085716, 0.018183],
  Spine2: [0.009167, 0.171433, 0.036367],
  Neck: [0.01375, 0.25715, 0.05455],
  Head: [0.0104, 0.37315, 0.28845],
  LeftArm: [0.0598, 0.24255, 0.09535],
  LeftForeArm: [0.1087, 0.15465, 0.04125],
  LeftHand: [0.0612, 0.07755, 0.08645],
  RightArm: [-0.0323, 0.27175, 0.01375],
  RightForeArm: [-0.0662, 0.18295, -0.05235],
  RightHand: [-0.0492, 0.11295, 0.02055],
  LeftUpLeg: [0.0275, -0.00145, 0.02595],
  LeftLeg: [0.0207, -0.20325, 0.01945],
  LeftFoot: [0.0119, -0.38055, -0.14385],
  RightUpLeg: [-0.0275, 0.00145, -0.02595],
  RightLeg: [-0.0431, -0.19215, -0.03485],
  RightFoot: [-0.06, -0.36455, -0.18615],
};

const hierarchy = {
  Spine1: "Hips",
  Spine2: "Spine1",
  Neck: "Spine2",
  Head: "Neck",
  LeftForeArm: "LeftArm",
  LeftHand: "LeftForeArm",
  RightArm: "Neck",
  RightForeArm: "RightArm",
  RightHand: "RightForeArm",
  LeftUpLeg: "Hips",
  LeftLeg: "LeftUpLeg",
  LeftFoot: "LeftLeg",
  RightUpLeg: "Hips",
  RightLeg: "RightUpLeg",
  RightFoot: "RightLeg",
};

const modelPath = "https://models.readyplayer.me/67be034c9fab1c21c486eb14.glb";

let model = null;
let skeletonHelper = null;
let keypointsFromPython = null;
let websocket = null;
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;

// Load the 3D model
loadModel();

// Set up WebSocket connection
setupWebSocket();

// Function to connect to WebSocket server
function setupWebSocket() {
  if (reconnectAttempts >= maxReconnectAttempts) {
    console.error(
      "Max reconnect attempts reached. Please check if the backend server is running."
    );
    return;
  }

  // Try to connect to the WebSocket server
  console.log("Connecting to WebSocket server...");

  // Make sure we properly close any existing connection
  if (websocket) {
    websocket.close();
  }

  try {
    websocket = new WebSocket("ws://localhost:8765");

    // Connection opened
    websocket.addEventListener("open", (event) => {
      console.log("Connected to WebSocket server");
      reconnectAttempts = 0; // Reset the reconnect counter on successful connection

      // Dispatch event for UI updates
      document.dispatchEvent(
        new CustomEvent("websocketStatusChanged", {
          detail: { connected: true },
        })
      );
    });

    // Listen for messages from the server
    websocket.addEventListener("message", (event) => {
      try {
        const data = JSON.parse(event.data);

        // Log the keypoints data to console (uncomment for debugging)
        // console.log("Received 17 keypoints from Python:", data);

        // Store the keypoints for reference
        keypointsFromPython = data;

        // Update joint positions with the received keypoints, applying coordinate transformation (-x, y, -z)
        if (Object.keys(data).length > 0) {
          for (const key in data) {
            if (key in jointPositions) {
              const point = data[key];
              // Transform coordinates: (-x, y, -z)
              jointPositions[key] = [-point[0], point[1], -point[2]];
            }
          }

          // Update the model with newly calculated quaternions
          if (model) {
            updateModelWithCalculatedQuaternions();
          }
        }
      } catch (error) {
        console.error("Error parsing WebSocket message:", error);
      }
    });

    // Handle errors
    websocket.addEventListener("error", (event) => {
      console.error("WebSocket error:", event);
      console.warn(
        "No keypoints will be received. Make sure the Python script is running."
      );

      // Dispatch event for UI updates
      document.dispatchEvent(
        new CustomEvent("websocketStatusChanged", {
          detail: { connected: false },
        })
      );
    });

    // Handle connection close
    websocket.addEventListener("close", (event) => {
      console.log(
        `WebSocket connection closed (code: ${event.code}). Attempting to reconnect in 3 seconds...`
      );

      // Dispatch event for UI updates
      document.dispatchEvent(
        new CustomEvent("websocketStatusChanged", {
          detail: { connected: false },
        })
      );

      reconnectAttempts++;
      setTimeout(setupWebSocket, 3000);
    });
  } catch (e) {
    console.error("Error setting up WebSocket:", e);
    reconnectAttempts++;
    setTimeout(setupWebSocket, 3000);
  }
}

// Calculate and apply quaternions based on current joint positions
function updateModelWithCalculatedQuaternions() {
  if (!model) return;

  model.traverse((bone) => {
    if (bone.isBone) {
      const boneName = bone.name;
      const childBoneName = Object.keys(hierarchy).find(
        (key) => hierarchy[key] === boneName
      );

      if (
        jointPositions[boneName] &&
        jointPositions[childBoneName] &&
        bone.children.length > 0
      ) {
        // Calculate target direction based on current joint positions
        const targetDir = new THREE.Vector3()
          .fromArray(jointPositions[childBoneName])
          .sub(new THREE.Vector3().fromArray(jointPositions[boneName]))
          .normalize();

        const worldPos = new THREE.Vector3();
        bone.getWorldPosition(worldPos);

        const childWorldPos = new THREE.Vector3();
        bone.children[0].getWorldPosition(childWorldPos);

        const currentDir = new THREE.Vector3()
          .subVectors(childWorldPos, worldPos)
          .normalize();

        // Calculate quaternion for rotation
        const qWorld = new THREE.Quaternion().setFromUnitVectors(
          currentDir,
          targetDir
        );

        if (bone.parent) {
          bone.parent.updateMatrixWorld(true);
          const parentWorldQuat = new THREE.Quaternion();
          bone.parent.getWorldQuaternion(parentWorldQuat);

          const invParentWorldQuat = parentWorldQuat.clone().invert();
          const localRotationAdjustment = invParentWorldQuat
            .multiply(qWorld)
            .multiply(parentWorldQuat);

          bone.quaternion.premultiply(localRotationAdjustment);
        }
      }
    }
  });

  // Update the model matrices to reflect quaternion changes
  model.updateMatrixWorld(true);

  // The skeleton helper automatically updates when the bones update
  if (skeletonHelper) {
    skeletonHelper.update();
  }

  // Log the calculated quaternions (uncomment for debugging)
  // console.log("Updated joint positions:", jointPositions);
}

function loadModel() {
  // Show loading message or indicator
  console.log("Loading 3D model...");

  // Show loading indicator if exists
  const loadingOverlay = document.querySelector(".loading-overlay");
  if (loadingOverlay) {
    loadingOverlay.style.display = "flex";
  }

  const loader = new GLTFLoader();
  loader.load(
    modelPath,
    (gltf) => {
      model = gltf.scene;

      // Set appropriate scale
      model.scale.set(3, 3, 3);

      // Position model on the ground
      const bbox = new THREE.Box3().setFromObject(model);
      const height = bbox.max.y - bbox.min.y;
      const centerY = (bbox.max.y + bbox.min.y) / 2;

      // Adjust model position so it stands on the ground (y=0)
      model.position.y = -bbox.min.y;

      // Enable shadows
      model.traverse((node) => {
        if (node.isMesh) {
          node.castShadow = true;
          node.receiveShadow = true;

          // Improve material quality if needed
          if (node.material) {
            node.material.metalness = 0.2;
            node.material.roughness = 0.7;
          }
        }
      });

      scene.add(model);
      console.log("Model loaded successfully!");

      // Hide loading indicator
      if (loadingOverlay) {
        loadingOverlay.style.display = "none";
      }

      // Add skeleton helper for visualization
      model.traverse((object) => {
        if (object.isSkinnedMesh && object.skeleton) {
          if (!skeletonHelper) {
            skeletonHelper = new THREE.SkeletonHelper(
              object.skeleton.bones[0].parent
            );
            skeletonHelper.visible = true; // Make skeleton visible
            // scene.add(skeletonHelper);
          }
        }
      });

      // Set up initial poses
      updateModelWithCalculatedQuaternions();

      // Dispatch an event to notify that the model is loaded
      document.dispatchEvent(new CustomEvent("modelLoaded"));
    },
    // Progress callback
    (xhr) => {
      const percentComplete = (xhr.loaded / xhr.total) * 100;
      console.log(`Loading: ${Math.round(percentComplete)}% complete`);
    },
    // Error callback
    (error) => {
      console.error("Error loading model:", error);

      // Hide loading indicator if exists
      if (loadingOverlay) {
        loadingOverlay.style.display = "none";
      }

      // Show error message to user
      const canvasElement = document.getElementById("main-canvas");
      if (canvasElement) {
        const ctx = canvasElement.getContext("2d");
        if (ctx) {
          ctx.fillStyle = "#1f2937";
          ctx.fillRect(0, 0, canvasElement.width, canvasElement.height);
          ctx.font = "16px Arial";
          ctx.fillStyle = "white";
          ctx.textAlign = "center";
          ctx.fillText(
            "Error loading 3D model. Please check console for details.",
            canvasElement.width / 2,
            canvasElement.height / 2
          );
        }
      }
    }
  );
}
