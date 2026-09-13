// // import React, { useState, Suspense, useEffect } from "react";
// // import { Canvas } from "@react-three/fiber";
// // import { OrbitControls } from "@react-three/drei";
// // import * as THREE from "three";
// // import { API_URL } from "../config/apiConfig";
// // import { Box, Upload, RotateCw, Settings2, Zap } from "lucide-react";
// // import "../styles/heightmap3d.css";

// // // Import GLTFLoader for GLB files
// // import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

// // // 3D Model Component (supports both STL and GLB)
// // function Model3D({ url, isGLB = false }) {
// //   const [model, setModel] = useState(null);
// //   const [error, setError] = useState(null);

// //   useEffect(() => {
// //     if (!url) return;

// //     const loadModel = async () => {
// //       try {
// //         if (isGLB) {
// //           // Load GLB file
// //           const loader = new GLTFLoader();
// //           loader.load(
// //             url,
// //             (gltf) => {
// //               setModel(gltf.scene);
// //               setError(null);
// //             },
// //             undefined,
// //             (error) => {
// //               console.error("GLB loading error:", error);
// //               setError("Failed to load 3D model");
// //             }
// //           );
// //         } else {
// //           // Load STL file
// //           const response = await fetch(url);
// //           const arrayBuffer = await response.arrayBuffer();
// //           const geometry = parseSTL(arrayBuffer);
// //           geometry.center();
// //           geometry.computeVertexNormals();
          
// //           const material = new THREE.MeshPhongMaterial({
// //             color: "#4f46e5",
// //             wireframe: false,
// //             shininess: 100,
// //             side: THREE.DoubleSide,
// //           });
          
// //           const mesh = new THREE.Mesh(geometry, material);
// //           const group = new THREE.Group();
// //           group.add(mesh);
// //           setModel(group);
// //           setError(null);
// //         }
// //       } catch (err) {
// //         console.error("Error loading model:", err);
// //         setError("Failed to load 3D model");
// //       }
// //     };

// //     loadModel();
// //   }, [url, isGLB]);

// //   if (error) {
// //     return null;
// //   }

// //   if (!model) {
// //     return null;
// //   }

// //   return <primitive object={model} />;
// // }

// // function parseSTL(arrayBuffer) {
// //   const view = new DataView(arrayBuffer);
// //   const isASCII = isASCIIFile(arrayBuffer);

// //   if (isASCII) {
// //     return parseASCIISTL(new TextDecoder().decode(arrayBuffer));
// //   } else {
// //     return parseBinarySTL(view);
// //   }
// // }

// // function isASCIIFile(arrayBuffer) {
// //   const view = new Uint8Array(arrayBuffer);
// //   const header = new TextDecoder().decode(view.slice(0, 5));
// //   return header.toLowerCase() === "solid";
// // }

// // function parseBinarySTL(view) {
// //   const triangles = view.getUint32(80, true);
// //   const geometry = new THREE.BufferGeometry();
// //   const vertices = [];
// //   const normals = [];

// //   let offset = 84;
// //   for (let i = 0; i < triangles; i++) {
// //     const nx = view.getFloat32(offset, true); offset += 4;
// //     const ny = view.getFloat32(offset, true); offset += 4;
// //     const nz = view.getFloat32(offset, true); offset += 4;

// //     for (let j = 0; j < 3; j++) {
// //       vertices.push(view.getFloat32(offset, true)); offset += 4;
// //       vertices.push(view.getFloat32(offset, true)); offset += 4;
// //       vertices.push(view.getFloat32(offset, true)); offset += 4;
// //       normals.push(nx, ny, nz);
// //     }

// //     offset += 2;
// //   }

// //   geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(vertices), 3));
// //   geometry.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(normals), 3));

// //   return geometry;
// // }

// // function parseASCIISTL(data) {
// //   const geometry = new THREE.BufferGeometry();
// //   const vertexPattern = /vertex\s+([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)/g;
// //   const vertices = [];

// //   let vertexMatch;
// //   while ((vertexMatch = vertexPattern.exec(data)) !== null) {
// //     vertices.push(parseFloat(vertexMatch[1]), parseFloat(vertexMatch[3]), parseFloat(vertexMatch[5]));
// //   }

// //   geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(vertices), 3));
// //   geometry.computeVertexNormals();

// //   return geometry;
// // }

// // function LoadingSpinner() {
// //   return null;
// // }

// // export default function Heightmap3D() {
// //   const [modelUrl, setModelUrl] = useState(null);
// //   const [modelType, setModelType] = useState("glb");
// //   const [loading, setLoading] = useState(false);
// //   const [error, setError] = useState(null);
// //   const [fileName, setFileName] = useState(null);
// //   const [showAdvanced, setShowAdvanced] = useState(false);
// //   const [settings, setSettings] = useState({
// //     resize_to: 300,
// //     height_scale: 12.0,
// //     smooth_sigma: 1.2,
// //   });

// //   async function handleUpload(e, useGLB = true) {
// //     const file = e.target.files[0];
// //     if (!file) return;

// //     setLoading(true);
// //     setError(null);
// //     setFileName(file.name);

// //     try {
// //       const formData = new FormData();
// //       formData.append("image", file);

// //       const endpoint = useGLB ? "/api/generate-3d-glb" : "/api/generate-3d-heightmap";
      
// //       let url = `http://localhost:5002${endpoint}`;
// //       if (useGLB) {
// //         const params = new URLSearchParams({
// //           resize_to: settings.resize_to,
// //           height_scale: settings.height_scale,
// //           smooth_sigma: settings.smooth_sigma,
// //         });
// //         url += `?${params.toString()}`;
// //       }

// //       const res = await fetch(url, {
// //         method: "POST",
// //         body: formData,
// //       });

// //       if (!res.ok) {
// //         throw new Error(`Server error: ${res.status}`);
// //       }

// //       const blob = await res.blob();
// //       const blobUrl = URL.createObjectURL(blob);
// //       setModelUrl(blobUrl);
// //       setModelType(useGLB ? "glb" : "stl");
// //     } catch (err) {
// //       setError(`Failed to generate 3D model: ${err.message}`);
// //       console.error("Upload error:", err);
// //     } finally {
// //       setLoading(false);
// //     }
// //   }

// //   return (
// //     <div style={{ padding: "2rem", minHeight: "100vh", background: "var(--bg-gradient)" }}>
// //       <div style={{ maxWidth: "1200px", margin: "0 auto" }}>
// //         {/* Header */}
// //         <div style={{ marginBottom: "2rem" }}>
// //           <h1 style={{ display: "flex", alignItems: "center", gap: "0.75rem", margin: 0, color: "var(--dark)", fontSize: "2rem", fontWeight: 700 }}>
// //             <Box size={32} style={{ color: "#4f46e5" }} />
// //             3D Heightmap Generator
// //           </h1>
// //           <p style={{ margin: "0.5rem 0 0 0", color: "var(--secondary)", fontSize: "1rem" }}>
// //             Convert 2D structural images into interactive 3D models (GLB textured or STL) for advanced visualization
// //           </p>
// //         </div>

// //         {/* Upload Card */}
// //         <div className="card" style={{ marginBottom: "2rem" }}>
// //           <div className="card-header">
// //             <h2 style={{ margin: 0, display: "flex", alignItems: "center", gap: "0.6rem" }}>
// //               <Upload size={24} />
// //               Upload Structural Image
// //             </h2>
// //             <p style={{ margin: "0.25rem 0 0 0", color: "var(--secondary)", fontSize: "0.875rem" }}>
// //               Select a 2D image (JPG, PNG) to convert into a 3D model
// //             </p>
// //           </div>
// //           <div className="card-body">
// //             {/* Format & Settings */}
// //             <div style={{ marginBottom: "1.5rem", display: "flex", gap: "1rem", flexWrap: "wrap", alignItems: "center" }}>
// //               <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
// //                 <label style={{ fontWeight: 600, color: "var(--dark)" }}>Format:</label>
// //                 <select
// //                   style={{
// //                     padding: "0.5rem 1rem",
// //                     borderRadius: "var(--border-radius)",
// //                     border: "1px solid var(--glass-border)",
// //                     backgroundColor: "white",
// //                     color: "var(--dark)",
// //                     cursor: "pointer",
// //                     fontWeight: 600,
// //                   }}
// //                 >
// //                   <option>🎨 GLB (Textured) - Recommended</option>
// //                   <option>⚪ STL (Monochrome)</option>
// //                 </select>
// //               </div>
              
// //               <button
// //                 onClick={() => setShowAdvanced(!showAdvanced)}
// //                 style={{
// //                   marginLeft: "auto",
// //                   padding: "0.5rem 1rem",
// //                   backgroundColor: showAdvanced ? "#4f46e5" : "#e5e7eb",
// //                   color: showAdvanced ? "white" : "var(--dark)",
// //                   border: "none",
// //                   borderRadius: "var(--border-radius)",
// //                   cursor: "pointer",
// //                   fontWeight: 600,
// //                   display: "flex",
// //                   alignItems: "center",
// //                   gap: "0.5rem",
// //                 }}
// //               >
// //                 <Settings2 size={18} />
// //                 Advanced
// //               </button>
// //             </div>

// //             {/* Advanced Settings */}
// //             {showAdvanced && (
// //               <div style={{
// //                 marginBottom: "1.5rem",
// //                 padding: "1rem",
// //                 backgroundColor: "rgba(79, 70, 229, 0.05)",
// //                 border: "1px solid var(--glass-border)",
// //                 borderRadius: "var(--border-radius)",
// //               }}>
// //                 <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem" }}>
// //                   <div>
// //                     <label style={{ display: "block", fontWeight: 600, color: "var(--dark)", marginBottom: "0.5rem", fontSize: "0.875rem" }}>
// //                       Resolution: {settings.resize_to}×{settings.resize_to}
// //                     </label>
// //                     <input
// //                       type="range"
// //                       min="100"
// //                       max="500"
// //                       step="50"
// //                       value={settings.resize_to}
// //                       onChange={(e) => setSettings({ ...settings, resize_to: parseInt(e.target.value) })}
// //                       style={{ width: "100%" }}
// //                     />
// //                   </div>

// //                   <div>
// //                     <label style={{ display: "block", fontWeight: 600, color: "var(--dark)", marginBottom: "0.5rem", fontSize: "0.875rem" }}>
// //                       Height Scale: {settings.height_scale.toFixed(1)}
// //                     </label>
// //                     <input
// //                       type="range"
// //                       min="2"
// //                       max="30"
// //                       step="0.5"
// //                       value={settings.height_scale}
// //                       onChange={(e) => setSettings({ ...settings, height_scale: parseFloat(e.target.value) })}
// //                       style={{ width: "100%" }}
// //                     />
// //                   </div>

// //                   <div>
// //                     <label style={{ display: "block", fontWeight: 600, color: "var(--dark)", marginBottom: "0.5rem", fontSize: "0.875rem" }}>
// //                       Smoothing (σ): {settings.smooth_sigma.toFixed(1)}
// //                     </label>
// //                     <input
// //                       type="range"
// //                       min="0"
// //                       max="5"
// //                       step="0.1"
// //                       value={settings.smooth_sigma}
// //                       onChange={(e) => setSettings({ ...settings, smooth_sigma: parseFloat(e.target.value) })}
// //                       style={{ width: "100%" }}
// //                     />
// //                   </div>
// //                 </div>
// //               </div>
// //             )}

// //             {/* Upload Area */}
// //             <div
// //               style={{
// //                 display: "flex",
// //                 flexDirection: "column",
// //                 alignItems: "center",
// //                 gap: "1.5rem",
// //                 padding: "2rem",
// //                 border: "2px dashed var(--glass-border)",
// //                 borderRadius: "var(--border-radius)",
// //                 backgroundColor: "rgba(79, 70, 229, 0.05)",
// //                 cursor: "pointer",
// //               }}
// //               onDragOver={(e) => {
// //                 e.preventDefault();
// //                 e.currentTarget.style.borderColor = "#4f46e5";
// //               }}
// //               onDragLeave={(e) => {
// //                 e.currentTarget.style.borderColor = "var(--glass-border)";
// //               }}
// //               onDrop={(e) => {
// //                 e.preventDefault();
// //                 if (e.dataTransfer.files.length > 0) {
// //                   const fileInput = document.querySelector('input[type="file"]');
// //                   fileInput.files = e.dataTransfer.files;
// //                   handleUpload({ target: { files: e.dataTransfer.files } });
// //                 }
// //               }}
// //             >
// //               <Upload size={48} style={{ color: "#4f46e5", opacity: 0.7 }} />
// //               <div style={{ textAlign: "center" }}>
// //                 <p style={{ margin: "0 0 0.5rem 0", fontWeight: 600, color: "var(--dark)" }}>
// //                   Drop your image here or click to browse
// //                 </p>
// //                 <p style={{ margin: 0, fontSize: "0.875rem", color: "var(--secondary)" }}>
// //                   Supported: JPG, PNG, GIF, BMP
// //                 </p>
// //               </div>
// //               <input
// //                 type="file"
// //                 accept="image/*"
// //                 onChange={(e) => handleUpload(e, true)}
// //                 style={{ display: "none" }}
// //                 id="imageUpload"
// //               />
// //               <label
// //                 htmlFor="imageUpload"
// //                 style={{
// //                   padding: "0.75rem 1.5rem",
// //                   backgroundColor: "#4f46e5",
// //                   color: "white",
// //                   borderRadius: "var(--border-radius)",
// //                   cursor: "pointer",
// //                   fontWeight: 600,
// //                 }}
// //               >
// //                 Choose File
// //               </label>
// //             </div>

// //             {fileName && (
// //               <div style={{ marginTop: "1rem", padding: "1rem", backgroundColor: "rgba(79, 70, 229, 0.1)", borderRadius: "var(--border-radius)" }}>
// //                 <p style={{ margin: 0, color: "var(--dark)", fontWeight: 600 }}>📄 {fileName}</p>
// //               </div>
// //             )}

// //             {loading && (
// //               <div style={{ marginTop: "1rem", padding: "1rem", textAlign: "center", color: "#666" }}>
// //                 <RotateCw size={24} style={{ animation: "spin 1s linear infinite", marginBottom: "0.5rem" }} />
// //                 <p style={{ margin: 0, fontWeight: 600 }}>Generating 3D model...</p>
// //               </div>
// //             )}

// //             {error && (
// //               <div style={{ marginTop: "1rem", padding: "1rem", backgroundColor: "#fee2e2", border: "1px solid #fecaca", borderRadius: "var(--border-radius)", color: "#991b1b" }}>
// //                 <p style={{ margin: 0, fontWeight: 600 }}>❌ {error}</p>
// //               </div>
// //             )}

// //             {modelUrl && !loading && (
// //               <div style={{ marginTop: "1rem", padding: "1rem", backgroundColor: "#dcfce7", border: "1px solid #bbf7d0", borderRadius: "var(--border-radius)", color: "#166534" }}>
// //                 <p style={{ margin: 0, fontWeight: 600 }}>✅ 3D model generated successfully!</p>
// //               </div>
// //             )}
// //           </div>
// //         </div>

// //         {/* 3D Viewer */}
// //         {modelUrl && (
// //           <div className="card">
// //             <div className="card-header">
// //               <h2 style={{ margin: 0, display: "flex", alignItems: "center", gap: "0.6rem" }}>
// //                 <Box size={24} />
// //                 Interactive 3D {modelType === "glb" ? "Textured" : "STL"} Viewer
// //               </h2>
// //             </div>
// //             <div className="card-body">
// //               <div style={{ height: "700px", borderRadius: "var(--border-radius)", overflow: "hidden", border: "1px solid var(--glass-border)" }}>
// //                 <Canvas camera={{ position: [150, 150, 150], fov: 50 }}>
// //                   <ambientLight intensity={0.6} />
// //                   <directionalLight position={[10, 10, 10]} intensity={0.8} />
// //                   <pointLight position={[-10, -10, -10]} intensity={0.3} />
                  
// //                   <Suspense fallback={<LoadingSpinner />}>
// //                     <Model3D url={modelUrl} isGLB={modelType === "glb"} />
// //                   </Suspense>
                  
// //                   <OrbitControls autoRotate={false} enableZoom={true} enablePan={true} enableRotate={true} />
// //                 </Canvas>
// //               </div>

// //               <div style={{ marginTop: "1.5rem", padding: "1rem", backgroundColor: "var(--light)", borderRadius: "var(--border-radius)", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "1rem" }}>
// //                 <div>
// //                   <p style={{ margin: 0, fontSize: "0.875rem", color: "var(--secondary)" }}>Type</p>
// //                   <p style={{ margin: "0.25rem 0 0 0", fontWeight: 600 }}>{modelType === "glb" ? "GLB" : "STL"}</p>
// //                 </div>
// //                 <div>
// //                   <p style={{ margin: 0, fontSize: "0.875rem", color: "var(--secondary)" }}>Resolution</p>
// //                   <p style={{ margin: "0.25rem 0 0 0", fontWeight: 600 }}>{settings.resize_to}×{settings.resize_to}</p>
// //                 </div>
// //                 <div>
// //                   <p style={{ margin: 0, fontSize: "0.875rem", color: "var(--secondary)" }}>Height</p>
// //                   <p style={{ margin: "0.25rem 0 0 0", fontWeight: 600 }}>{settings.height_scale.toFixed(1)}</p>
// //                 </div>
// //               </div>

// //               <div style={{ marginTop: "1.5rem", display: "flex", gap: "1rem" }}>
// //                 <button
// //                   onClick={() => {
// //                     const link = document.createElement("a");
// //                     link.href = modelUrl;
// //                     link.download = `heightmap.${modelType}`;
// //                     link.click();
// //                   }}
// //                   style={{
// //                     padding: "0.75rem 1.5rem",
// //                     backgroundColor: "#4f46e5",
// //                     color: "white",
// //                     border: "none",
// //                     borderRadius: "var(--border-radius)",
// //                     cursor: "pointer",
// //                     fontWeight: 600,
// //                   }}
// //                 >
// //                   📥 Download
// //                 </button>
// //                 <button
// //                   onClick={() => {
// //                     setModelUrl(null);
// //                     setFileName(null);
// //                   }}
// //                   style={{
// //                     padding: "0.75rem 1.5rem",
// //                     backgroundColor: "#e5e7eb",
// //                     color: "var(--dark)",
// //                     border: "none",
// //                     borderRadius: "var(--border-radius)",
// //                     cursor: "pointer",
// //                     fontWeight: 600,
// //                   }}
// //                 >
// //                   🔄 New Model
// //                 </button>
// //               </div>
// //             </div>
// //           </div>
// //         )}

// //         {/* Features */}
// //         <div style={{ marginTop: "3rem", padding: "2rem", backgroundColor: "var(--light)", borderRadius: "var(--border-radius)", border: "1px solid var(--glass-border)" }}>
// //           <h3 style={{ margin: "0 0 1.5rem 0", color: "var(--dark)", fontSize: "1.25rem", fontWeight: 600 }}>✨ Features</h3>
// //           <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "1.5rem" }}>
// //             <div>
// //               <p style={{ margin: 0, fontWeight: 600, color: "var(--dark)", display: "flex", alignItems: "center", gap: "0.5rem" }}>
// //                 <Zap size={18} /> 🎨 Textured GLB
// //               </p>
// //               <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.875rem", color: "var(--secondary)" }}>Advanced generation with heatmap colors and edge detection</p>
// //             </div>
// //             <div>
// //               <p style={{ margin: 0, fontWeight: 600, color: "var(--dark)" }}>⚪ Monochrome STL</p>
// //               <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.875rem", color: "var(--secondary)" }}>Traditional format for 3D printing and CAD</p>
// //             </div>
// //             <div>
// //               <p style={{ margin: 0, fontWeight: 600, color: "var(--dark)" }}>🔍 Interactive</p>
// //               <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.875rem", color: "var(--secondary)" }}>Real-time 3D visualization with full controls</p>
// //             </div>
// //             <div>
// //               <p style={{ margin: 0, fontWeight: 600, color: "var(--dark)" }}>⚙️ Customizable</p>
// //               <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.875rem", color: "var(--secondary)" }}>Adjust resolution, height, and smoothing</p>
// //             </div>
// //             <div>
// //               <p style={{ margin: 0, fontWeight: 600, color: "var(--dark)" }}>💾 Export Options</p>
// //               <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.875rem", color: "var(--secondary)" }}>Download GLB or STL files</p>
// //             </div>
// //             <div>
// //               <p style={{ margin: 0, fontWeight: 600, color: "var(--dark)" }}>🏗️ Infrastructure Ready</p>
// //               <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.875rem", color: "var(--secondary)" }}>Perfect for structural health monitoring</p>
// //             </div>
// //           </div>
// //         </div>
// //       </div>

// //       <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
// //     </div>
// //   );
// // }

// import React, { useState, Suspense, useEffect } from "react";
// import { Canvas } from "@react-three/fiber";
// import { OrbitControls } from "@react-three/drei";
// import * as THREE from "three";
// import { Box, Upload, RotateCw, Settings2, Zap } from "lucide-react";
// import "../styles/heightmap3d.css";

// // Import GLTFLoader for GLB files
// import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

// // 3D Model Component (supports both STL and GLB)
// function Model3D({ url, isGLB = false }) {
//   const [model, setModel] = useState(null);
//   const [error, setError] = useState(null);

//   useEffect(() => {
//     if (!url) return;

//     const loadModel = async () => {
//       try {
//         if (isGLB) {
//           // Load GLB file
//           const loader = new GLTFLoader();
//           loader.load(
//             url,
//             (gltf) => {
//               // Enable vertex colors for all meshes in the scene
//               gltf.scene.traverse((node) => {
//                 if (node.isMesh && node.geometry) {
//                   // Compute normals for proper lighting on both sides
//                   node.geometry.computeVertexNormals();
                  
//                   // Ensure geometry has color attribute
//                   if (node.geometry.attributes.color) {
//                     // Material already has colors, just enable them
//                     if (node.material) {
//                       node.material.vertexColors = true;
//                       node.material.side = THREE.DoubleSide;
//                       node.material.needsUpdate = true;
                      
//                       if (Array.isArray(node.material)) {
//                         node.material.forEach(mat => {
//                           mat.vertexColors = true;
//                           mat.side = THREE.DoubleSide;
//                           mat.needsUpdate = true;
//                         });
//                       }
//                     }
//                   } else {
//                     // No colors in geometry - create a new material that forces vertex colors
//                     const newMaterial = new THREE.MeshStandardMaterial({
//                       vertexColors: true,
//                       side: THREE.DoubleSide,
//                       metalness: 0.0,
//                       roughness: 0.8,
//                     });
//                     node.material = newMaterial;
//                   }
//                 }
//               });
              
//               setModel(gltf.scene);
//               setError(null);
//             },
//             undefined,
//             (error) => {
//               console.error("GLB loading error:", error);
//               setError("Failed to load 3D model");
//             }
//           );
//         } else {
//           // Load STL file
//           const response = await fetch(url);
//           const arrayBuffer = await response.arrayBuffer();
//           const geometry = parseSTL(arrayBuffer);
//           geometry.center();
//           geometry.computeVertexNormals();
          
//           const material = new THREE.MeshPhongMaterial({
//             color: "#4f46e5",
//             wireframe: false,
//             shininess: 100,
//             side: THREE.DoubleSide,
//           });
          
//           const mesh = new THREE.Mesh(geometry, material);
//           const group = new THREE.Group();
//           group.add(mesh);
//           setModel(group);
//           setError(null);
//         }
//       } catch (err) {
//         console.error("Error loading model:", err);
//         setError("Failed to load 3D model");
//       }
//     };

//     loadModel();
//   }, [url, isGLB]);

//   if (error) {
//     return null;
//   }

//   if (!model) {
//     return null;
//   }

//   return <primitive object={model} />;
// }

// function parseSTL(arrayBuffer) {
//   const view = new DataView(arrayBuffer);
//   const isASCII = isASCIIFile(arrayBuffer);

//   if (isASCII) {
//     return parseASCIISTL(new TextDecoder().decode(arrayBuffer));
//   } else {
//     return parseBinarySTL(view);
//   }
// }

// function isASCIIFile(arrayBuffer) {
//   const view = new Uint8Array(arrayBuffer);
//   const header = new TextDecoder().decode(view.slice(0, 5));
//   return header.toLowerCase() === "solid";
// }

// function parseBinarySTL(view) {
//   const triangles = view.getUint32(80, true);
//   const geometry = new THREE.BufferGeometry();
//   const vertices = [];
//   const normals = [];

//   let offset = 84;
//   for (let i = 0; i < triangles; i++) {
//     const nx = view.getFloat32(offset, true); offset += 4;
//     const ny = view.getFloat32(offset, true); offset += 4;
//     const nz = view.getFloat32(offset, true); offset += 4;

//     for (let j = 0; j < 3; j++) {
//       vertices.push(view.getFloat32(offset, true)); offset += 4;
//       vertices.push(view.getFloat32(offset, true)); offset += 4;
//       vertices.push(view.getFloat32(offset, true)); offset += 4;
//       normals.push(nx, ny, nz);
//     }

//     offset += 2;
//   }

//   geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(vertices), 3));
//   geometry.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(normals), 3));

//   return geometry;
// }

// function parseASCIISTL(data) {
//   const geometry = new THREE.BufferGeometry();
//   const vertexPattern = /vertex\s+([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)/g;
//   const vertices = [];

//   let vertexMatch;
//   while ((vertexMatch = vertexPattern.exec(data)) !== null) {
//     vertices.push(parseFloat(vertexMatch[1]), parseFloat(vertexMatch[3]), parseFloat(vertexMatch[5]));
//   }

//   geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(vertices), 3));
//   geometry.computeVertexNormals();

//   return geometry;
// }

// function LoadingSpinner() {
//   return null;
// }

// export default function Heightmap3D() {
//   const [modelUrl, setModelUrl] = useState(null);
//   const [modelType, setModelType] = useState("glb");
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState(null);
//   const [fileName, setFileName] = useState(null);
//   const [showAdvanced, setShowAdvanced] = useState(false);
//   const [format, setFormat] = useState("glb"); // Track selected format
//   const [settings, setSettings] = useState({
//     resize_to: 300,
//     height_scale: 12.0,
//     smooth_sigma: 1.2,
//   });

//   async function handleUpload(e, useGLB = null) {
//     // If useGLB is not provided, use the format state
//     if (useGLB === null) {
//       useGLB = format === "glb";
//     }
    
//     const file = e.target.files[0];
//     if (!file) return;

//     setLoading(true);
//     setError(null);
//     setFileName(file.name);

//     try {
//       const formData = new FormData();
//       formData.append("image", file);

//       const endpoint = useGLB ? "/api/generate-3d-glb" : "/api/generate-3d-heightmap";
      
//       let url = `http://localhost:5002${endpoint}`;
//       if (useGLB) {
//         const params = new URLSearchParams({
//           resize_to: settings.resize_to,
//           height_scale: settings.height_scale,
//           smooth_sigma: settings.smooth_sigma,
//         });
//         url += `?${params.toString()}`;
//       }

//       const res = await fetch(url, {
//         method: "POST",
//         body: formData,
//       });

//       if (!res.ok) {
//         throw new Error(`Server error: ${res.status}`);
//       }

//       const blob = await res.blob();
//       const blobUrl = URL.createObjectURL(blob);
//       setModelUrl(blobUrl);
//       setModelType(useGLB ? "glb" : "stl");
//     } catch (err) {
//       setError(`Failed to generate 3D model: ${err.message}`);
//       console.error("Upload error:", err);
//     } finally {
//       setLoading(false);
//     }
//   }

//   return (
//     <div style={{ padding: "2rem", minHeight: "100vh", background: "var(--bg-gradient)" }}>
//       <div style={{ maxWidth: "1200px", margin: "0 auto" }}>
//         {/* Header */}
//         <div style={{ marginBottom: "2rem" }}>
//           <h1 style={{ display: "flex", alignItems: "center", gap: "0.75rem", margin: 0, color: "var(--dark)", fontSize: "2rem", fontWeight: 700 }}>
//             <Box size={32} style={{ color: "#4f46e5" }} />
//             3D Heightmap Generator
//           </h1>
//           <p style={{ margin: "0.5rem 0 0 0", color: "var(--secondary)", fontSize: "1rem" }}>
//             Convert 2D structural images into interactive 3D models (GLB textured or STL) for advanced visualization
//           </p>
//         </div>

//         {/* Upload Card */}
//         <div className="card" style={{ marginBottom: "2rem" }}>
//           <div className="card-header">
//             <h2 style={{ margin: 0, display: "flex", alignItems: "center", gap: "0.6rem" }}>
//               <Upload size={24} />
//               Upload Structural Image
//             </h2>
//             <p style={{ margin: "0.25rem 0 0 0", color: "var(--secondary)", fontSize: "0.875rem" }}>
//               Select a 2D image (JPG, PNG) to convert into a 3D model
//             </p>
//           </div>
//           <div className="card-body">
//             {/* Format & Settings */}
//             <div style={{ marginBottom: "1.5rem", display: "flex", gap: "1rem", flexWrap: "wrap", alignItems: "center" }}>
//               <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
//                 <label style={{ fontWeight: 600, color: "var(--dark)" }}>Format:</label>
//                 <select
//                   value={format}
//                   onChange={(e) => setFormat(e.target.value)}
//                   style={{
//                     padding: "0.5rem 1rem",
//                     borderRadius: "var(--border-radius)",
//                     border: "1px solid var(--glass-border)",
//                     backgroundColor: "white",
//                     color: "var(--dark)",
//                     cursor: "pointer",
//                     fontWeight: 600,
//                   }}
//                 >
//                   <option value="glb">🎨 GLB (Textured) - Recommended</option>
//                   <option value="stl">⚪ STL (Monochrome)</option>
//                 </select>
//               </div>
              
//               <button
//                 onClick={() => setShowAdvanced(!showAdvanced)}
//                 style={{
//                   marginLeft: "auto",
//                   padding: "0.5rem 1rem",
//                   backgroundColor: showAdvanced ? "#4f46e5" : "#e5e7eb",
//                   color: showAdvanced ? "white" : "var(--dark)",
//                   border: "none",
//                   borderRadius: "var(--border-radius)",
//                   cursor: "pointer",
//                   fontWeight: 600,
//                   display: "flex",
//                   alignItems: "center",
//                   gap: "0.5rem",
//                 }}
//               >
//                 <Settings2 size={18} />
//                 Advanced
//               </button>
//             </div>

//             {/* Advanced Settings */}
//             {showAdvanced && (
//               <div style={{
//                 marginBottom: "1.5rem",
//                 padding: "1rem",
//                 backgroundColor: "rgba(79, 70, 229, 0.05)",
//                 border: "1px solid var(--glass-border)",
//                 borderRadius: "var(--border-radius)",
//               }}>
//                 <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem" }}>
//                   <div>
//                     <label style={{ display: "block", fontWeight: 600, color: "var(--dark)", marginBottom: "0.5rem", fontSize: "0.875rem" }}>
//                       Resolution: {settings.resize_to}×{settings.resize_to}
//                     </label>
//                     <input
//                       type="range"
//                       min="100"
//                       max="500"
//                       step="50"
//                       value={settings.resize_to}
//                       onChange={(e) => setSettings({ ...settings, resize_to: parseInt(e.target.value) })}
//                       style={{ width: "100%" }}
//                     />
//                   </div>

//                   <div>
//                     <label style={{ display: "block", fontWeight: 600, color: "var(--dark)", marginBottom: "0.5rem", fontSize: "0.875rem" }}>
//                       Height Scale: {settings.height_scale.toFixed(1)}
//                     </label>
//                     <input
//                       type="range"
//                       min="2"
//                       max="30"
//                       step="0.5"
//                       value={settings.height_scale}
//                       onChange={(e) => setSettings({ ...settings, height_scale: parseFloat(e.target.value) })}
//                       style={{ width: "100%" }}
//                     />
//                   </div>

//                   <div>
//                     <label style={{ display: "block", fontWeight: 600, color: "var(--dark)", marginBottom: "0.5rem", fontSize: "0.875rem" }}>
//                       Smoothing (σ): {settings.smooth_sigma.toFixed(1)}
//                     </label>
//                     <input
//                       type="range"
//                       min="0"
//                       max="5"
//                       step="0.1"
//                       value={settings.smooth_sigma}
//                       onChange={(e) => setSettings({ ...settings, smooth_sigma: parseFloat(e.target.value) })}
//                       style={{ width: "100%" }}
//                     />
//                   </div>
//                 </div>
//               </div>
//             )}

//             {/* Upload Area */}
//             <div
//               style={{
//                 display: "flex",
//                 flexDirection: "column",
//                 alignItems: "center",
//                 gap: "1.5rem",
//                 padding: "2rem",
//                 border: "2px dashed var(--glass-border)",
//                 borderRadius: "var(--border-radius)",
//                 backgroundColor: "rgba(79, 70, 229, 0.05)",
//                 cursor: "pointer",
//               }}
//               onDragOver={(e) => {
//                 e.preventDefault();
//                 e.currentTarget.style.borderColor = "#4f46e5";
//               }}
//               onDragLeave={(e) => {
//                 e.currentTarget.style.borderColor = "var(--glass-border)";
//               }}
//               onDrop={(e) => {
//                 e.preventDefault();
//                 if (e.dataTransfer.files.length > 0) {
//                   const fileInput = document.querySelector('input[type="file"]');
//                   fileInput.files = e.dataTransfer.files;
//                   handleUpload({ target: { files: e.dataTransfer.files } });
//                 }
//               }}
//             >
//               <Upload size={48} style={{ color: "#4f46e5", opacity: 0.7 }} />
//               <div style={{ textAlign: "center" }}>
//                 <p style={{ margin: "0 0 0.5rem 0", fontWeight: 600, color: "var(--dark)" }}>
//                   Drop your image here or click to browse
//                 </p>
//                 <p style={{ margin: 0, fontSize: "0.875rem", color: "var(--secondary)" }}>
//                   Supported: JPG, PNG, GIF, BMP
//                 </p>
//               </div>
//               <input
//                 type="file"
//                 accept="image/*"
//                 onChange={(e) => handleUpload(e)}
//                 style={{ display: "none" }}
//                 id="imageUpload"
//               />
//               <label
//                 htmlFor="imageUpload"
//                 style={{
//                   padding: "0.75rem 1.5rem",
//                   backgroundColor: "#4f46e5",
//                   color: "white",
//                   borderRadius: "var(--border-radius)",
//                   cursor: "pointer",
//                   fontWeight: 600,
//                 }}
//               >
//                 Choose File
//               </label>
//             </div>

//             {fileName && (
//               <div style={{ marginTop: "1rem", padding: "1rem", backgroundColor: "rgba(79, 70, 229, 0.1)", borderRadius: "var(--border-radius)" }}>
//                 <p style={{ margin: 0, color: "var(--dark)", fontWeight: 600 }}>📄 {fileName}</p>
//               </div>
//             )}

//             {loading && (
//               <div style={{ marginTop: "1rem", padding: "1rem", textAlign: "center", color: "#666" }}>
//                 <RotateCw size={24} style={{ animation: "spin 1s linear infinite", marginBottom: "0.5rem" }} />
//                 <p style={{ margin: 0, fontWeight: 600 }}>Generating 3D model...</p>
//               </div>
//             )}

//             {error && (
//               <div style={{ marginTop: "1rem", padding: "1rem", backgroundColor: "#fee2e2", border: "1px solid #fecaca", borderRadius: "var(--border-radius)", color: "#991b1b" }}>
//                 <p style={{ margin: 0, fontWeight: 600 }}>❌ {error}</p>
//               </div>
//             )}

//             {modelUrl && !loading && (
//               <div style={{ marginTop: "1rem", padding: "1rem", backgroundColor: "#dcfce7", border: "1px solid #bbf7d0", borderRadius: "var(--border-radius)", color: "#166534" }}>
//                 <p style={{ margin: 0, fontWeight: 600 }}>✅ 3D model generated successfully!</p>
//               </div>
//             )}
//           </div>
//         </div>

//         {/* 3D Viewer */}
//         {modelUrl && (
//           <div className="card">
//             <div className="card-header">
//               <h2 style={{ margin: 0, display: "flex", alignItems: "center", gap: "0.6rem" }}>
//                 <Box size={24} />
//                 Interactive 3D {modelType === "glb" ? "Textured" : "STL"} Viewer
//               </h2>
//             </div>
//             <div className="card-body">
//               <div style={{ height: "600px", borderRadius: "var(--border-radius)", overflow: "hidden", border: "1px solid var(--glass-border)" }}>
//                 <Canvas camera={{ position: [150, 150, 150], fov: 50 }}>
//                   <ambientLight intensity={1.5} />
//                   <directionalLight position={[10, 10, 10]} intensity={1.2} />
//                   <directionalLight position={[-10, -10, -10]} intensity={0.8} />
//                   <pointLight position={[0, 0, 50]} intensity={0.6} />
                  
//                   <Suspense fallback={<LoadingSpinner />}>
//                     <Model3D url={modelUrl} isGLB={modelType === "glb"} />
//                   </Suspense>
                  
//                   <OrbitControls autoRotate={false} enableZoom={true} enablePan={true} enableRotate={true} />
//                 </Canvas>
//               </div>

//               <div style={{ marginTop: "1.5rem", padding: "1rem", backgroundColor: "var(--light)", borderRadius: "var(--border-radius)", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "1rem" }}>
//                 <div>
//                   <p style={{ margin: 0, fontSize: "0.875rem", color: "var(--secondary)" }}>Type</p>
//                   <p style={{ margin: "0.25rem 0 0 0", fontWeight: 600 }}>{modelType === "glb" ? "GLB" : "STL"}</p>
//                 </div>
//                 <div>
//                   <p style={{ margin: 0, fontSize: "0.875rem", color: "var(--secondary)" }}>Resolution</p>
//                   <p style={{ margin: "0.25rem 0 0 0", fontWeight: 600 }}>{settings.resize_to}×{settings.resize_to}</p>
//                 </div>
//                 <div>
//                   <p style={{ margin: 0, fontSize: "0.875rem", color: "var(--secondary)" }}>Height</p>
//                   <p style={{ margin: "0.25rem 0 0 0", fontWeight: 600 }}>{settings.height_scale.toFixed(1)}</p>
//                 </div>
//               </div>

//               <div style={{ marginTop: "1.5rem", display: "flex", gap: "1rem" }}>
//                 <button
//                   onClick={() => {
//                     const link = document.createElement("a");
//                     link.href = modelUrl;
//                     link.download = `heightmap.${modelType}`;
//                     link.click();
//                   }}
//                   style={{
//                     padding: "0.75rem 1.5rem",
//                     backgroundColor: "#4f46e5",
//                     color: "white",
//                     border: "none",
//                     borderRadius: "var(--border-radius)",
//                     cursor: "pointer",
//                     fontWeight: 600,
//                   }}
//                 >
//                   📥 Download
//                 </button>
//                 <button
//                   onClick={() => {
//                     setModelUrl(null);
//                     setFileName(null);
//                   }}
//                   style={{
//                     padding: "0.75rem 1.5rem",
//                     backgroundColor: "#e5e7eb",
//                     color: "var(--dark)",
//                     border: "none",
//                     borderRadius: "var(--border-radius)",
//                     cursor: "pointer",
//                     fontWeight: 600,
//                   }}
//                 >
//                   🔄 New Model
//                 </button>
//               </div>
//             </div>
//           </div>
//         )}

//         {/* Features */}
//         <div style={{ marginTop: "3rem", padding: "2rem", backgroundColor: "var(--light)", borderRadius: "var(--border-radius)", border: "1px solid var(--glass-border)" }}>
//           <h3 style={{ margin: "0 0 1.5rem 0", color: "var(--dark)", fontSize: "1.25rem", fontWeight: 600 }}>✨ Features</h3>
//           <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "1.5rem" }}>
//             <div>
//               <p style={{ margin: 0, fontWeight: 600, color: "var(--dark)", display: "flex", alignItems: "center", gap: "0.5rem" }}>
//                 <Zap size={18} /> 🎨 Textured GLB
//               </p>
//               <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.875rem", color: "var(--secondary)" }}>Advanced generation with heatmap colors and edge detection</p>
//             </div>
//             <div>
//               <p style={{ margin: 0, fontWeight: 600, color: "var(--dark)" }}>⚪ Monochrome STL</p>
//               <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.875rem", color: "var(--secondary)" }}>Traditional format for 3D printing and CAD</p>
//             </div>
//             <div>
//               <p style={{ margin: 0, fontWeight: 600, color: "var(--dark)" }}>🔍 Interactive</p>
//               <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.875rem", color: "var(--secondary)" }}>Real-time 3D visualization with full controls</p>
//             </div>
//             <div>
//               <p style={{ margin: 0, fontWeight: 600, color: "var(--dark)" }}>⚙️ Customizable</p>
//               <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.875rem", color: "var(--secondary)" }}>Adjust resolution, height, and smoothing</p>
//             </div>
//             <div>
//               <p style={{ margin: 0, fontWeight: 600, color: "var(--dark)" }}>💾 Export Options</p>
//               <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.875rem", color: "var(--secondary)" }}>Download GLB or STL files</p>
//             </div>
//             <div>
//               <p style={{ margin: 0, fontWeight: 600, color: "var(--dark)" }}>🏗️ Infrastructure Ready</p>
//               <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.875rem", color: "var(--secondary)" }}>Perfect for structural health monitoring</p>
//             </div>
//           </div>
//         </div>
//       </div>

//       <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
//     </div>
//   );
// }

import React, { useState, Suspense, useEffect } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { API_URL } from "../config/apiConfig";
import { Box, Upload, RotateCw, Settings2, Zap, Ruler, Calculator } from "lucide-react";
import "../styles/heightmap3d.css";

// Import GLTFLoader for GLB files
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

// 3D Model Component (supports both STL and GLB)
function Model3D({ url, isGLB = false }) {
  const [model, setModel] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!url) return;

    const loadModel = async () => {
      try {
        if (isGLB) {
          // Load GLB file
          const loader = new GLTFLoader();
          loader.load(
            url,
            (gltf) => {
              // Enable vertex colors for all meshes in the scene
              gltf.scene.traverse((node) => {
                if (node.isMesh && node.geometry) {
                  // Compute normals for proper lighting on both sides
                  node.geometry.computeVertexNormals();
                  
                  // Ensure geometry has color attribute
                  if (node.geometry.attributes.color) {
                    // Material already has colors, just enable them
                    if (node.material) {
                      node.material.vertexColors = true;
                      node.material.side = THREE.DoubleSide;
                      node.material.needsUpdate = true;
                      
                      if (Array.isArray(node.material)) {
                        node.material.forEach(mat => {
                          mat.vertexColors = true;
                          mat.side = THREE.DoubleSide;
                          mat.needsUpdate = true;
                        });
                      }
                    }
                  } else {
                    // No colors in geometry - create a new material that forces vertex colors
                    const newMaterial = new THREE.MeshStandardMaterial({
                      vertexColors: true,
                      side: THREE.DoubleSide,
                      metalness: 0.0,
                      roughness: 0.8,
                    });
                    node.material = newMaterial;
                  }
                }
              });
              
              setModel(gltf.scene);
              setError(null);
            },
            undefined,
            (error) => {
              console.error("GLB loading error:", error);
              setError("Failed to load 3D model");
            }
          );
        } else {
          // Load STL file
          const response = await fetch(url);
          const arrayBuffer = await response.arrayBuffer();
          const geometry = parseSTL(arrayBuffer);
          geometry.center();
          geometry.computeVertexNormals();
          
          const material = new THREE.MeshPhongMaterial({
            color: "#4f46e5",
            wireframe: false,
            shininess: 100,
            side: THREE.DoubleSide,
          });
          
          const mesh = new THREE.Mesh(geometry, material);
          const group = new THREE.Group();
          group.add(mesh);
          setModel(group);
          setError(null);
        }
      } catch (err) {
        console.error("Error loading model:", err);
        setError("Failed to load 3D model");
      }
    };

    loadModel();
  }, [url, isGLB]);

  if (error) {
    return null;
  }

  if (!model) {
    return null;
  }

  return <primitive object={model} />;
}

function parseSTL(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  const isASCII = isASCIIFile(arrayBuffer);

  if (isASCII) {
    return parseASCIISTL(new TextDecoder().decode(arrayBuffer));
  } else {
    return parseBinarySTL(view);
  }
}

function isASCIIFile(arrayBuffer) {
  const view = new Uint8Array(arrayBuffer);
  const header = new TextDecoder().decode(view.slice(0, 5));
  return header.toLowerCase() === "solid";
}

function parseBinarySTL(view) {
  const triangles = view.getUint32(80, true);
  const geometry = new THREE.BufferGeometry();
  const vertices = [];
  const normals = [];

  let offset = 84;
  for (let i = 0; i < triangles; i++) {
    const nx = view.getFloat32(offset, true); offset += 4;
    const ny = view.getFloat32(offset, true); offset += 4;
    const nz = view.getFloat32(offset, true); offset += 4;

    for (let j = 0; j < 3; j++) {
      vertices.push(view.getFloat32(offset, true)); offset += 4;
      vertices.push(view.getFloat32(offset, true)); offset += 4;
      vertices.push(view.getFloat32(offset, true)); offset += 4;
      normals.push(nx, ny, nz);
    }

    offset += 2;
  }

  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(vertices), 3));
  geometry.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(normals), 3));

  return geometry;
}

function parseASCIISTL(data) {
  const geometry = new THREE.BufferGeometry();
  const vertexPattern = /vertex\s+([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)/g;
  const vertices = [];

  let vertexMatch;
  while ((vertexMatch = vertexPattern.exec(data)) !== null) {
    vertices.push(parseFloat(vertexMatch[1]), parseFloat(vertexMatch[3]), parseFloat(vertexMatch[5]));
  }

  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(vertices), 3));
  geometry.computeVertexNormals();

  return geometry;
}

function LoadingSpinner() {
  return null;
}

export default function Heightmap3D() {
  const [modelUrl, setModelUrl] = useState(null);
  const [modelType, setModelType] = useState("glb");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [fileName, setFileName] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [format, setFormat] = useState("glb"); // Track selected format
  const [settings, setSettings] = useState({
    resize_to: 300,
    height_scale: 12.0,
    smooth_sigma: 1.2,
  });
  const [scaleInputs, setScaleInputs] = useState({
    focalMm: "50",
    sensorMm: "36",
    imageWidthPx: "1000",
    distanceMm: "1000",
    crackPixels: "250",
  });

  const focalPixels = Number(scaleInputs.focalMm) * Number(scaleInputs.imageWidthPx) /
    Number(scaleInputs.sensorMm);
  const mmPerPixel = focalPixels > 0
    ? Number(scaleInputs.distanceMm) / focalPixels
    : 0;
  const estimatedLengthMm = mmPerPixel * Number(scaleInputs.crackPixels);

  async function handleUpload(e, useGLB = null) {
    // If useGLB is not provided, use the format state
    if (useGLB === null) {
      useGLB = format === "glb";
    }
    
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    setFileName(file.name);

    try {
      const formData = new FormData();
      formData.append("image", file);

      const endpoint = useGLB ? "/api/generate-3d-glb" : "/api/generate-3d-heightmap";
      
      let url = `${API_URL}${endpoint}`;
      if (useGLB) {
        const params = new URLSearchParams({
          resize_to: settings.resize_to,
          height_scale: settings.height_scale,
          smooth_sigma: settings.smooth_sigma,
        });
        url += `?${params.toString()}`;
      }

      const res = await fetch(url, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }

      const blob = await res.blob();
      const blobUrl = URL.createObjectURL(blob);
      setModelUrl(blobUrl);
      setModelType(useGLB ? "glb" : "stl");
    } catch (err) {
      setError(`Failed to generate 3D model: ${err.message}`);
      console.error("Upload error:", err);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ padding: "2rem", minHeight: "100vh", background: "var(--bg-gradient)" }}>
      <div style={{ maxWidth: "1200px", margin: "0 auto" }}>
        {/* Header */}
        <div style={{ marginBottom: "2rem" }}>
          <h1 style={{ display: "flex", alignItems: "center", gap: "0.75rem", margin: 0, color: "var(--dark)", fontSize: "2rem", fontWeight: 700 }}>
            <Box size={32} style={{ color: "#4f46e5" }} />
            3D Heightmap Generator
          </h1>
          <p style={{ margin: "0.5rem 0 0 0", color: "var(--secondary)", fontSize: "1rem" }}>
            Convert 2D structural images into interactive 3D models (GLB textured or STL) for advanced visualization
          </p>
        </div>

        <div className="card" style={{ marginBottom: "2rem", borderLeft: "4px solid #0f766e" }}>
          <div className="card-header">
            <h2 style={{ margin: 0, display: "flex", alignItems: "center", gap: "0.6rem" }}>
              <Ruler size={24} />
              Camera Scale Recovery
            </h2>
            <p style={{ margin: "0.25rem 0 0 0", color: "var(--secondary)", fontSize: "0.875rem" }}>
              Calculate physical scale before crack-length estimation
            </p>
          </div>
          <div className="card-body">
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: "1rem" }}>
              {[
                ["focalMm", "Focal length (mm)"],
                ["sensorMm", "Sensor width (mm)"],
                ["imageWidthPx", "Image width (px)"],
                ["distanceMm", "Surface distance (mm)"],
                ["crackPixels", "Crack length (px)"],
              ].map(([key, label]) => (
                <label key={key} style={{ display: "grid", gap: "0.4rem", color: "var(--dark)", fontWeight: 600, fontSize: "0.875rem" }}>
                  {label}
                  <input
                    type="number"
                    min="0.001"
                    step="any"
                    value={scaleInputs[key]}
                    onChange={(event) => setScaleInputs({ ...scaleInputs, [key]: event.target.value })}
                    style={{ padding: "0.65rem", border: "1px solid var(--glass-border)", borderRadius: "var(--border-radius)", fontSize: "1rem" }}
                  />
                </label>
              ))}
            </div>
            <div style={{ marginTop: "1.25rem", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem" }}>
              <div style={{ padding: "1rem", background: "#ecfdf5", border: "1px solid #99f6e4", borderRadius: "var(--border-radius)" }}>
                <span style={{ display: "block", color: "#115e59", fontSize: "0.8rem" }}>Effective focal length</span>
                <strong style={{ fontSize: "1.35rem", color: "#134e4a" }}>{Number.isFinite(focalPixels) ? focalPixels.toFixed(2) : "-"} px</strong>
              </div>
              <div style={{ padding: "1rem", background: "#ecfdf5", border: "1px solid #99f6e4", borderRadius: "var(--border-radius)" }}>
                <span style={{ display: "block", color: "#115e59", fontSize: "0.8rem" }}>Recovered scale</span>
                <strong style={{ fontSize: "1.35rem", color: "#134e4a" }}>{Number.isFinite(mmPerPixel) ? mmPerPixel.toFixed(4) : "-"} mm/px</strong>
              </div>
              <div style={{ padding: "1rem", background: "#fff7ed", border: "1px solid #fed7aa", borderRadius: "var(--border-radius)" }}>
                <span style={{ display: "block", color: "#9a3412", fontSize: "0.8rem" }}>Estimated crack length</span>
                <strong style={{ fontSize: "1.35rem", color: "#7c2d12" }}>{Number.isFinite(estimatedLengthMm) ? estimatedLengthMm.toFixed(2) : "-"} mm</strong>
              </div>
            </div>
            <p style={{ margin: "1rem 0 0", color: "#7c2d12", fontSize: "0.8rem", display: "flex", gap: "0.4rem", alignItems: "center" }}>
              <Calculator size={15} />
              Pinhole estimate for a fronto-parallel surface. Use a coplanar reference or metric depth for perspective-correct measurement.
            </p>
          </div>
        </div>

        {/* Upload Card */}
        <div className="card" style={{ marginBottom: "2rem" }}>
          <div className="card-header">
            <h2 style={{ margin: 0, display: "flex", alignItems: "center", gap: "0.6rem" }}>
              <Upload size={24} />
              Upload Structural Image
            </h2>
            <p style={{ margin: "0.25rem 0 0 0", color: "var(--secondary)", fontSize: "0.875rem" }}>
              Select a 2D image (JPG, PNG) to convert into a 3D model
            </p>
          </div>
          <div className="card-body">
            {/* Format & Settings */}
            <div style={{ marginBottom: "1.5rem", display: "flex", gap: "1rem", flexWrap: "wrap", alignItems: "center" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <label style={{ fontWeight: 600, color: "var(--dark)" }}>Format:</label>
                <select
                  value={format}
                  onChange={(e) => setFormat(e.target.value)}
                  style={{
                    padding: "0.5rem 1rem",
                    borderRadius: "var(--border-radius)",
                    border: "1px solid var(--glass-border)",
                    backgroundColor: "white",
                    color: "var(--dark)",
                    cursor: "pointer",
                    fontWeight: 600,
                  }}
                >
                  <option value="glb">🎨 GLB (Textured) - Recommended</option>
                  <option value="stl">⚪ STL (Monochrome)</option>
                </select>
              </div>
              
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                style={{
                  marginLeft: "auto",
                  padding: "0.5rem 1rem",
                  backgroundColor: showAdvanced ? "#4f46e5" : "#e5e7eb",
                  color: showAdvanced ? "white" : "var(--dark)",
                  border: "none",
                  borderRadius: "var(--border-radius)",
                  cursor: "pointer",
                  fontWeight: 600,
                  display: "flex",
                  alignItems: "center",
                  gap: "0.5rem",
                }}
              >
                <Settings2 size={18} />
                Advanced
              </button>
            </div>

            {/* Advanced Settings */}
            {showAdvanced && (
              <div style={{
                marginBottom: "1.5rem",
                padding: "1rem",
                backgroundColor: "rgba(79, 70, 229, 0.05)",
                border: "1px solid var(--glass-border)",
                borderRadius: "var(--border-radius)",
              }}>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem" }}>
                  <div>
                    <label style={{ display: "block", fontWeight: 600, color: "var(--dark)", marginBottom: "0.5rem", fontSize: "0.875rem" }}>
                      Resolution: {settings.resize_to}×{settings.resize_to}
                    </label>
                    <input
                      type="range"
                      min="100"
                      max="500"
                      step="50"
                      value={settings.resize_to}
                      onChange={(e) => setSettings({ ...settings, resize_to: parseInt(e.target.value) })}
                      style={{ width: "100%" }}
                    />
                  </div>

                  <div>
                    <label style={{ display: "block", fontWeight: 600, color: "var(--dark)", marginBottom: "0.5rem", fontSize: "0.875rem" }}>
                      Height Scale: {settings.height_scale.toFixed(1)}
                    </label>
                    <input
                      type="range"
                      min="2"
                      max="30"
                      step="0.5"
                      value={settings.height_scale}
                      onChange={(e) => setSettings({ ...settings, height_scale: parseFloat(e.target.value) })}
                      style={{ width: "100%" }}
                    />
                  </div>

                  <div>
                    <label style={{ display: "block", fontWeight: 600, color: "var(--dark)", marginBottom: "0.5rem", fontSize: "0.875rem" }}>
                      Smoothing (σ): {settings.smooth_sigma.toFixed(1)}
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="5"
                      step="0.1"
                      value={settings.smooth_sigma}
                      onChange={(e) => setSettings({ ...settings, smooth_sigma: parseFloat(e.target.value) })}
                      style={{ width: "100%" }}
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Upload Area */}
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: "1.5rem",
                padding: "2rem",
                border: "2px dashed var(--glass-border)",
                borderRadius: "var(--border-radius)",
                backgroundColor: "rgba(79, 70, 229, 0.05)",
                cursor: "pointer",
              }}
              onDragOver={(e) => {
                e.preventDefault();
                e.currentTarget.style.borderColor = "#4f46e5";
              }}
              onDragLeave={(e) => {
                e.currentTarget.style.borderColor = "var(--glass-border)";
              }}
              onDrop={(e) => {
                e.preventDefault();
                if (e.dataTransfer.files.length > 0) {
                  const fileInput = document.querySelector('input[type="file"]');
                  fileInput.files = e.dataTransfer.files;
                  handleUpload({ target: { files: e.dataTransfer.files } });
                }
              }}
            >
              <Upload size={48} style={{ color: "#4f46e5", opacity: 0.7 }} />
              <div style={{ textAlign: "center" }}>
                <p style={{ margin: "0 0 0.5rem 0", fontWeight: 600, color: "var(--dark)" }}>
                  Drop your image here or click to browse
                </p>
                <p style={{ margin: 0, fontSize: "0.875rem", color: "var(--secondary)" }}>
                  Supported: JPG, PNG, GIF, BMP
                </p>
              </div>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => handleUpload(e)}
                style={{ display: "none" }}
                id="imageUpload"
              />
              <label
                htmlFor="imageUpload"
                style={{
                  padding: "0.75rem 1.5rem",
                  backgroundColor: "#4f46e5",
                  color: "white",
                  borderRadius: "var(--border-radius)",
                  cursor: "pointer",
                  fontWeight: 600,
                }}
              >
                Choose File
              </label>
            </div>

            {fileName && (
              <div style={{ marginTop: "1rem", padding: "1rem", backgroundColor: "rgba(79, 70, 229, 0.1)", borderRadius: "var(--border-radius)" }}>
                <p style={{ margin: 0, color: "var(--dark)", fontWeight: 600 }}>📄 {fileName}</p>
              </div>
            )}

            {loading && (
              <div style={{ marginTop: "1rem", padding: "1rem", textAlign: "center", color: "#666" }}>
                <RotateCw size={24} style={{ animation: "spin 1s linear infinite", marginBottom: "0.5rem" }} />
                <p style={{ margin: 0, fontWeight: 600 }}>Generating 3D model...</p>
              </div>
            )}

            {error && (
              <div style={{ marginTop: "1rem", padding: "1rem", backgroundColor: "#fee2e2", border: "1px solid #fecaca", borderRadius: "var(--border-radius)", color: "#991b1b" }}>
                <p style={{ margin: 0, fontWeight: 600 }}>❌ {error}</p>
              </div>
            )}

            {modelUrl && !loading && (
              <div style={{ marginTop: "1rem", padding: "1rem", backgroundColor: "#dcfce7", border: "1px solid #bbf7d0", borderRadius: "var(--border-radius)", color: "#166534" }}>
                <p style={{ margin: 0, fontWeight: 600 }}>✅ 3D model generated successfully!</p>
              </div>
            )}
          </div>
        </div>

        {/* 3D Viewer */}
        {modelUrl && (
          <div className="card">
            <div className="card-header">
              <h2 style={{ margin: 0, display: "flex", alignItems: "center", gap: "0.6rem" }}>
                <Box size={24} />
                Interactive 3D {modelType === "glb" ? "Textured" : "STL"} Viewer
              </h2>
            </div>
            <div className="card-body">
              <div style={{ height: "600px", borderRadius: "var(--border-radius)", overflow: "hidden", border: "1px solid var(--glass-border)" }}>
                <Canvas camera={{ position: [150, 150, 150], fov: 50 }}>
                  <ambientLight intensity={1.5} />
                  <directionalLight position={[10, 10, 10]} intensity={1.2} />
                  <directionalLight position={[-10, -10, -10]} intensity={0.8} />
                  <pointLight position={[0, 0, 50]} intensity={0.6} />
                  
                  <Suspense fallback={<LoadingSpinner />}>
                    <Model3D url={modelUrl} isGLB={modelType === "glb"} />
                  </Suspense>
                  
                  <OrbitControls autoRotate={false} enableZoom={true} enablePan={true} enableRotate={true} />
                </Canvas>
              </div>

              <div style={{ marginTop: "1.5rem", padding: "1rem", backgroundColor: "var(--light)", borderRadius: "var(--border-radius)", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "1rem" }}>
                <div>
                  <p style={{ margin: 0, fontSize: "0.875rem", color: "var(--secondary)" }}>Type</p>
                  <p style={{ margin: "0.25rem 0 0 0", fontWeight: 600 }}>{modelType === "glb" ? "GLB" : "STL"}</p>
                </div>
                <div>
                  <p style={{ margin: 0, fontSize: "0.875rem", color: "var(--secondary)" }}>Resolution</p>
                  <p style={{ margin: "0.25rem 0 0 0", fontWeight: 600 }}>{settings.resize_to}×{settings.resize_to}</p>
                </div>
                <div>
                  <p style={{ margin: 0, fontSize: "0.875rem", color: "var(--secondary)" }}>Height</p>
                  <p style={{ margin: "0.25rem 0 0 0", fontWeight: 600 }}>{settings.height_scale.toFixed(1)}</p>
                </div>
              </div>

              <div style={{ marginTop: "1.5rem", display: "flex", gap: "1rem" }}>
                <button
                  onClick={() => {
                    const link = document.createElement("a");
                    link.href = modelUrl;
                    link.download = `heightmap.${modelType}`;
                    link.click();
                  }}
                  style={{
                    padding: "0.75rem 1.5rem",
                    backgroundColor: "#4f46e5",
                    color: "white",
                    border: "none",
                    borderRadius: "var(--border-radius)",
                    cursor: "pointer",
                    fontWeight: 600,
                  }}
                >
                  📥 Download
                </button>
                <button
                  onClick={() => {
                    setModelUrl(null);
                    setFileName(null);
                  }}
                  style={{
                    padding: "0.75rem 1.5rem",
                    backgroundColor: "#e5e7eb",
                    color: "var(--dark)",
                    border: "none",
                    borderRadius: "var(--border-radius)",
                    cursor: "pointer",
                    fontWeight: 600,
                  }}
                >
                  🔄 New Model
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Features */}
        <div style={{ marginTop: "3rem", padding: "2rem", backgroundColor: "var(--light)", borderRadius: "var(--border-radius)", border: "1px solid var(--glass-border)" }}>
          <h3 style={{ margin: "0 0 1.5rem 0", color: "var(--dark)", fontSize: "1.25rem", fontWeight: 600 }}>✨ Features</h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "1.5rem" }}>
            <div>
              <p style={{ margin: 0, fontWeight: 600, color: "var(--dark)", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <Zap size={18} /> 🎨 Textured GLB
              </p>
              <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.875rem", color: "var(--secondary)" }}>Advanced generation with heatmap colors and edge detection</p>
            </div>
            <div>
              <p style={{ margin: 0, fontWeight: 600, color: "var(--dark)" }}>⚪ Monochrome STL</p>
              <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.875rem", color: "var(--secondary)" }}>Traditional format for 3D printing and CAD</p>
            </div>
            <div>
              <p style={{ margin: 0, fontWeight: 600, color: "var(--dark)" }}>🔍 Interactive</p>
              <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.875rem", color: "var(--secondary)" }}>Real-time 3D visualization with full controls</p>
            </div>
            <div>
              <p style={{ margin: 0, fontWeight: 600, color: "var(--dark)" }}>⚙️ Customizable</p>
              <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.875rem", color: "var(--secondary)" }}>Adjust resolution, height, and smoothing</p>
            </div>
            <div>
              <p style={{ margin: 0, fontWeight: 600, color: "var(--dark)" }}>💾 Export Options</p>
              <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.875rem", color: "var(--secondary)" }}>Download GLB or STL files</p>
            </div>
            <div>
              <p style={{ margin: 0, fontWeight: 600, color: "var(--dark)" }}>🏗️ Infrastructure Ready</p>
              <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.875rem", color: "var(--secondary)" }}>Perfect for structural health monitoring</p>
            </div>
          </div>
        </div>
      </div>

      <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}