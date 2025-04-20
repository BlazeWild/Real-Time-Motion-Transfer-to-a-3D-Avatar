// Define the importmap
const importMap = {
  imports: {
    three: "./three-shim.js",
    "three/": "https://unpkg.com/three@0.175.0/",
  },
};

// Create and append the importmap to the document
const importMapScript = document.createElement("script");
importMapScript.type = "importmap";
importMapScript.textContent = JSON.stringify(importMap);
document.currentScript.after(importMapScript);

console.log("Import map installed");
