// This file acts as a shim to make bare "three" imports work
import * as THREE from "https://unpkg.com/three@0.175.0/build/three.module.js";

// Re-export everything from THREE
export default THREE;
export * from "https://unpkg.com/three@0.175.0/build/three.module.js";
