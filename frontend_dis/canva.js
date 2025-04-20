// Use bare module imports (will be resolved by importmap)
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TextGeometry } from "three/examples/jsm/geometries/TextGeometry.js";
import { FontLoader } from "three/examples/jsm/loaders/FontLoader.js";

export let scene, camera, renderer, controls;

initCanvas();

function initCanvas() {
  // Get the canvas element
  const canvas = document.getElementById("main-canvas");

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111827); // Darker blue-gray background

  // Camera setup
  camera = new THREE.PerspectiveCamera(
    50,
    16 / 9, // Default aspect ratio, will be updated later
    0.1,
    1000
  );
  camera.position.set(3, 2, 10); // Adjusted camera height to align with ground at y=0

  // Renderer setup
  renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    canvas: canvas, // Use the existing canvas element
  });

  // Set initial size - this will be updated by the main script
  renderer.setSize(canvas.clientWidth || 1280, canvas.clientHeight || 720);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  // Enable physically correct lighting
  renderer.physicallyCorrectLights = true;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.3; // Increased exposure for overall brightness

  // Add environment
  setupEnvironment();

  // Lighting setup
  setupLighting();

  // Add ground
  createGround();

  // Comment out coordinate axes
  // createSimpleAxes();

  // Controls setup
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;
  controls.target.set(0, 0, 0); // Look at the origin where the hips and ground intersect

  // Initial render to show something on the screen
  renderer.render(scene, camera);

  animate();
}

function setupEnvironment() {
  // Enhanced ambient occlusion for better environment lighting
  const ambientLight = new THREE.AmbientLight(0x8090a0, 0.8); // Significantly brighter ambient light
  scene.add(ambientLight);

  // Add subtle fog for depth
  scene.fog = new THREE.FogExp2(0x111827, 0.015); // Further reduced fog for better visibility

  // Add environment lighting with an HDRI-like effect (simulated)
  const hemiLight = new THREE.HemisphereLight(0x90b0ff, 0x606060, 0.7); // Brighter hemisphere light
  hemiLight.position.set(0, 20, 0);
  scene.add(hemiLight);

  // Add environment spheres for visual interest
  const sphereGeometry = new THREE.SphereGeometry(0.2, 32, 32);
  const sphereMaterial = new THREE.MeshStandardMaterial({
    color: 0x3b82f6,
    metalness: 0.7,
    roughness: 0.2,
    emissive: 0x102a5a,
    emissiveIntensity: 0.4, // Increased emissive intensity for more glow
  });

  // Create decorative spheres around the scene
  const spherePositions = [
    [-6, 0, -5],
    [6, 1, -7],
    [-5, 2, -8],
    [7, 2, -3],
  ];

  spherePositions.forEach((position) => {
    const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
    sphere.position.set(...position);
    sphere.castShadow = true;
    scene.add(sphere);
  });

  // Add decorative blocks in the background
  addDecorativeBlocks();
}

// Function to add decorative blocks in the background
function addDecorativeBlocks() {
  // Create materials for blocks
  const materials = [
    new THREE.MeshStandardMaterial({
      color: 0x2563eb, // Blue
      metalness: 0.3,
      roughness: 0.7,
    }),
    new THREE.MeshStandardMaterial({
      color: 0x059669, // Green
      metalness: 0.2,
      roughness: 0.6,
    }),
    new THREE.MeshStandardMaterial({
      color: 0x7c3aed, // Purple
      metalness: 0.4,
      roughness: 0.5,
    }),
    new THREE.MeshStandardMaterial({
      color: 0xd97706, // Amber
      metalness: 0.2,
      roughness: 0.8,
    }),
  ];

  // Box configurations: [width, height, depth, x, y, z, material index, rotation y]
  const boxConfigs = [
    [2, 6, 1, -10, 0, -15, 0, 0.2],
    [1, 4, 1, -7, -1, -12, 1, 0],
    [3, 2, 1, -13, 0, -14, 2, -0.3],
    [1.5, 8, 1, 12, 1, -16, 3, 0.1],
    [2, 5, 1, 8, -1.5, -13, 0, -0.2],
    [4, 3, 1, 15, 0, -18, 1, 0.4],
    [1, 7, 1, 5, -1, -15, 2, 0],
    [2.5, 4, 1, -15, -2, -18, 3, -0.5],
    [3, 9, 1, 18, -3, -20, 0, 0.3],
  ];

  boxConfigs.forEach((config) => {
    const [width, height, depth, x, y, z, materialIndex, rotY] = config;
    const geometry = new THREE.BoxGeometry(width, height, depth);
    const block = new THREE.Mesh(geometry, materials[materialIndex]);
    block.position.set(x, y, z);
    block.rotation.y = rotY;
    block.castShadow = true;
    block.receiveShadow = true;
    scene.add(block);
  });
}

function setupLighting() {
  // Key light - main light source with shadows (simulates sun)
  const keyLight = new THREE.DirectionalLight(0xffffff, 1.1); // Brighter key light
  keyLight.position.set(-5, 12, 10);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.width = 2048;
  keyLight.shadow.mapSize.height = 2048;
  keyLight.shadow.camera.near = 0.5;
  keyLight.shadow.camera.far = 50;
  keyLight.shadow.camera.left = -15;
  keyLight.shadow.camera.right = 15;
  keyLight.shadow.camera.top = 15;
  keyLight.shadow.camera.bottom = -15;
  keyLight.shadow.bias = -0.0005;
  scene.add(keyLight);

  // Fill light - softer light from opposite side
  const fillLight = new THREE.DirectionalLight(0x9090ff, 0.7); // Slightly brighter fill light
  fillLight.position.set(10, 8, -10);
  scene.add(fillLight);

  // Back light - rim lighting from behind
  const backLight = new THREE.DirectionalLight(0xc0c0ff, 0.6); // Slightly brighter back light
  backLight.position.set(0, 6, -15);
  scene.add(backLight);

  // Front light - new light source from the front
  const frontLight = new THREE.DirectionalLight(0xfff0e0, 0.9); // Warm white front light
  frontLight.position.set(0, 3, 15); // Positioned in front of the model
  frontLight.castShadow = true;
  frontLight.shadow.mapSize.width = 1024;
  frontLight.shadow.mapSize.height = 1024;
  frontLight.shadow.camera.near = 0.5;
  frontLight.shadow.camera.far = 30;
  frontLight.shadow.camera.left = -10;
  frontLight.shadow.camera.right = 10;
  frontLight.shadow.camera.top = 10;
  frontLight.shadow.camera.bottom = -10;
  frontLight.shadow.bias = -0.0005;
  scene.add(frontLight);

  // Enhanced spotlight on the character
  const spotLight = new THREE.SpotLight(0xffffff, 1.2); // Brighter spotlight
  spotLight.position.set(0, 15, 5);
  spotLight.angle = Math.PI / 7; // Wider angle
  spotLight.penumbra = 0.4; // Softer edge
  spotLight.decay = 1.5;
  spotLight.distance = 50;
  spotLight.castShadow = true;
  spotLight.shadow.bias = -0.0001;
  spotLight.shadow.mapSize.width = 1024;
  spotLight.shadow.mapSize.height = 1024;
  spotLight.target.position.set(0, 0, 0);
  scene.add(spotLight);
  scene.add(spotLight.target);

  // Add a subtle second spotlight for drama
  const accentLight = new THREE.SpotLight(0x6495ed, 0.7); // Slightly brighter accent light
  accentLight.position.set(-8, 10, 5);
  accentLight.angle = Math.PI / 8;
  accentLight.penumbra = 0.5;
  accentLight.decay = 1.5;
  accentLight.distance = 40;
  accentLight.castShadow = false; // No shadow to avoid complexity
  accentLight.target.position.set(0, 3, 0); // Aimed at upper body
  scene.add(accentLight);
  scene.add(accentLight.target);

  // Face spotlight - focused specifically on the model's face
  const faceSpotlight = new THREE.SpotLight(0xfffaf0, 1.4); // Even brighter face light
  faceSpotlight.position.set(0, 8, 8); // Moved more to the front for better face illumination
  faceSpotlight.angle = Math.PI / 10; // Slightly wider angle
  faceSpotlight.penumbra = 0.3; // Defined edge
  faceSpotlight.decay = 1.2;
  faceSpotlight.distance = 30;
  faceSpotlight.castShadow = false; // No shadow to avoid complexity with multiple lights

  // Target the face area (slightly higher than body center)
  faceSpotlight.target.position.set(0, 4, 0); // Assuming model's face is around y=4
  scene.add(faceSpotlight);
  scene.add(faceSpotlight.target);
}

function createGround() {
  // Calculate ground position - place it at the origin (0.0) to match the hips position
  const groundY = 0.0;

  // Create a larger, nicer ground
  const groundGeometry = new THREE.CircleGeometry(30, 64); // Increased from 20 to 30
  const groundMaterial = new THREE.MeshStandardMaterial({
    color: 0x202030,
    roughness: 0.7,
    metalness: 0.1,
  });
  const ground = new THREE.Mesh(groundGeometry, groundMaterial);
  ground.rotation.x = -Math.PI / 2; // Rotate to be horizontal
  ground.position.y = groundY;
  ground.receiveShadow = true;
  scene.add(ground);

  // Add a grid helper with more refined appearance
  const gridHelper = new THREE.GridHelper(30, 60, 0x555555, 0x333333); // Increased from 20/40 to 30/60
  gridHelper.position.y = groundY + 0.01; // Slightly above ground to prevent z-fighting
  gridHelper.material.opacity = 0.4; // More transparent
  gridHelper.material.transparent = true;
  scene.add(gridHelper);

  // Add a subtle point light at ground level for bounce light effect
  const groundLight = new THREE.PointLight(0x3040a0, 0.7, 12); // Brighter ground light with larger radius
  groundLight.position.set(0, groundY + 0.5, 0);
  scene.add(groundLight);

  // Add a ring highlight on the floor under the character
  const ringGeometry = new THREE.RingGeometry(1.5, 2.5, 32);
  const ringMaterial = new THREE.MeshBasicMaterial({
    color: 0x3b82f6,
    transparent: true,
    opacity: 0.25, // Increased opacity for more visibility
    side: THREE.DoubleSide,
  });
  const ring = new THREE.Mesh(ringGeometry, ringMaterial);
  ring.rotation.x = -Math.PI / 2;
  ring.position.set(0, groundY + 0.02, 0);
  scene.add(ring);
}

// Function to create coordinate axes with labels
function createSimpleAxes() {
  // Create a thin axes helper (just lines, no blocks)
  const axesHelper = new THREE.AxesHelper(5);

  // Make the lines a bit thinner by using a custom material for each axis
  const materials = axesHelper.material;
  if (Array.isArray(materials)) {
    materials.forEach((material) => {
      material.linewidth = 1; // Set a thin line width (note: this may not work in all browsers)
    });
  } else if (materials) {
    materials.linewidth = 1;
  }

  scene.add(axesHelper);

  // Add small line labels at the end of each axis - very simple text
  addSimpleAxisLabel(5.2, 0, 0, "X", 0xff0000);
  addSimpleAxisLabel(0, 5.2, 0, "Y", 0x00ff00);
  addSimpleAxisLabel(0, 0, 5.2, "Z", 0x0000ff);
}

function addSimpleAxisLabel(x, y, z, text, color) {
  // Create a simple canvas-based sprite for the text label
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;

  const ctx = canvas.getContext("2d");
  ctx.fillStyle = `#${color.toString(16).padStart(6, "0")}`;
  ctx.font = "Bold 48px Arial";
  ctx.fillText(text, 20, 44);

  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture });
  const sprite = new THREE.Sprite(material);

  sprite.position.set(x, y, z);
  sprite.scale.set(0.5, 0.5, 0.5);
  scene.add(sprite);
}

function onWindowResize() {
  const canvas = renderer.domElement;
  //make the camera a little bit higher;
  controls.target.set(0, 0, 0);
  camera.aspect = canvas.clientWidth / canvas.clientHeight;

  camera.updateProjectionMatrix();
  renderer.setSize(canvas.clientWidth, canvas.clientHeight);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

// Handle window resizing
window.addEventListener("resize", onWindowResize);
