import React, { useState, useRef, useEffect } from 'react';
import { useAnalysis } from '../contexts/AnalysisContext';
import { API_URL } from '../config/apiConfig';
import { Camera, Upload, Play, Loader, PieChart, Target, Activity, TrendingUp, CheckCircle, AlertTriangle, BarChart3, Wind, Droplet, Zap, Leaf, Download, Shield } from 'lucide-react';

const ImageAnalysis = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [depthFile, setDepthFile] = useState(null);
  const [depthPreview, setDepthPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [outputImages, setOutputImages] = useState(null);
  const [tabSwitched, setTabSwitched] = useState(false);
  const fileInputRef = useRef(null);
  const { updateAnalysis, clearAnalysis, lastAnalysis, outputImages: contextOutputImages } = useAnalysis();
  
  const [settings, setSettings] = useState({
    confidenceThreshold: 0.3,
    pixelToCmRatio: 0.1
  });

  // Load saved image and results on component mount and when context changes
  useEffect(() => {
    // Restore from context when tab is switched back - PRIORITY
    if (lastAnalysis && contextOutputImages) {
      setResults(lastAnalysis);
      setOutputImages(contextOutputImages);
      setTabSwitched(true);
    }
    
    // Restore from sessionStorage if available
    const savedImage = sessionStorage.getItem('lastUploadedImage');
    if (savedImage && !preview) {
      setPreview(savedImage);
    }
  }, [lastAnalysis, contextOutputImages]);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
        // Save image to sessionStorage for persistence across tabs
        sessionStorage.setItem('lastUploadedImage', reader.result);
      };
      reader.readAsDataURL(selectedFile);
      setResults(null);
      setOutputImages(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setLoading(true);
    setError(null);
    setProgress(0);
    
    try {
      const reader = new FileReader();
      const fileBase64Promise = new Promise((resolve, reject) => {
        reader.onload = () => resolve(reader.result);
        reader.onerror = (error) => reject(error);
      });
      reader.readAsDataURL(file);
      const fileBase64 = await fileBase64Promise;

      let depthBase64 = null;
      if (depthFile) {
        const depthReader = new FileReader();
        const depthPromise = new Promise((resolve, reject) => {
          depthReader.onload = () => resolve(depthReader.result);
          depthReader.onerror = (error) => reject(error);
        });
        depthReader.readAsDataURL(depthFile);
        depthBase64 = await depthPromise;
      }

      // Simulate progress
      const progressInterval = setInterval(() => {
        setProgress(prev => Math.min(prev + 10, 90));
      }, 300);

      const response = await fetch(`${API_URL}/api/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: fileBase64,
          depth: depthBase64,
          px_to_cm_ratio: settings.pixelToCmRatio,
          confidence_threshold: settings.confidenceThreshold
        })
      });

      clearInterval(progressInterval);
      
      if (!response.ok) {
        throw new Error('Analysis failed');
      }

      const data = await response.json();
      setResults(data.results);
      setOutputImages(data.output_images);
      updateAnalysis(data.results, data.output_images); // Store BOTH results and images in context for persistence
      setProgress(100);
      setLoading(false);
    } catch (err) {
      console.error('Analysis error:', err);
      setError(`Analysis failed. Make sure the backend is running and accessible at ${API_URL}`);
      setLoading(false);
      setProgress(0);
    }
  };

  const handleDownloadPDF = async () => {
    if (!results) return;
    
    try {
      const response = await fetch(`${API_URL}/api/download-pdf`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          analysis_date: new Date().toISOString(),
          site_analysis: results,
        })
      });
      
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', 'heritage-analysis-report.pdf');
        document.body.appendChild(link);
        link.click();
        link.parentNode.removeChild(link);
      }
    } catch (err) {
      console.error('PDF download error:', err);
      alert('PDF generation failed. Feature may not be available.');
    }
  };

  const MetricCard = ({ icon: Icon, title, value, subtitle, color = 'primary' }) => (
    <div className="metric-card">
      <div className={`metric-icon bg-${color}`}>
        <Icon size={24} />
      </div>
      <div className="metric-content">
        <div className="metric-title">{title}</div>
        <div className="metric-value" style={{ fontSize: '0.95rem', fontWeight: '700', color: '#000000' }}>{value}</div>
        {subtitle && <div className="metric-subtitle">{subtitle}</div>}
      </div>
    </div>
  );

  const SeverityBadge = ({ severity }) => {
    const colors = {
      'Minor': 'success',
      'Moderate': 'warning',
      'Severe': 'danger',
      'Critical': 'danger'
    };
    return <span className={`badge bg-${colors[severity] || 'secondary'}`}>{severity}</span>;
  };

  const ImageInsights = ({ lastAnalysis }) => {
    if (!lastAnalysis) {
      return (
        <div className="content-area">
          <div className="alert alert-warning">
            No analysis data available. Please analyze an image first.
          </div>
        </div>
      );
    }

    const { images, metrics } = lastAnalysis;

    return (
      <div className="content-area">
        <h2>Image Insights</h2>
        <div className="image-grid">
          {images.map((image, index) => (
            <div key={index} className="image-card">
              <img src={image} alt={`Processed ${index}`} />
            </div>
          ))}
        </div>
        <div className="metrics">
          <h3>Metrics</h3>
          <pre>{JSON.stringify(metrics, null, 2)}</pre>
        </div>
      </div>
    );
  };

  return (
    <div className="content-area">
      {error && (
        <div className="alert alert-error" style={{ marginBottom: '2rem' }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {!file && !loading && !results && (
        <div className="card">
          <div className="card-header">
            <h2 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <Upload size={28} />
              🏗️ Structural Health Upload & Analysis
            </h2>
            <p className="mb-1" style={{ color: 'var(--secondary)', marginTop: '0.25rem' }}>
              Comprehensive AI-powered analysis: crack detection, biological growth assessment, material classification, severity scoring, environmental impact & durability prediction
            </p>
          </div>
          <div className="card-body">
            <div className="upload-zone" onClick={() => fileInputRef.current?.click()}>
              <Upload size={64} style={{ color: 'var(--primary)', marginBottom: '1rem' }} />
              <h3 style={{ marginBottom: '0.5rem', color: '#ffffff' }}>Click to Upload Image</h3>
              <p style={{ color: 'rgba(255,255,255,0.8)', marginBottom: '1rem' }}>
                PNG, JPG, JPEG • Max 10MB
              </p>
              <p style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.875rem' }}>
                Comprehensive structural analysis including crack detection, biological growth,<br />
                material classification, and environmental impact assessment
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                style={{ display: 'none' }}
              />
            </div>
          </div>
        </div>
      )}

      {file && !loading && !results && (
        <div className="card">
          <div className="card-header">
            <h2 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <Camera size={28} />
              📷 Image Preview & Analysis Settings
            </h2>
          </div>
          <div className="card-body">
            <div style={{ position: 'relative', marginBottom: '2rem' }}>
              <img src={preview} alt="Preview" className="preview-image" />
            </div>

            {/* Analysis Settings */}
            <div className="settings-section" style={{ marginBottom: '2rem', padding: '1.5rem', background: 'var(--light)', borderRadius: 'var(--border-radius)', border: '1px solid var(--glass-border)' }}>
              <h4 style={{ margin: '0 0 1rem 0', color: 'var(--dark)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Activity size={20} />
                Analysis Settings
              </h4>

              <div className="settings-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1.5rem' }}>
                <div className="setting-item">
                  <label style={{ display: 'block', fontWeight: 600, color: 'var(--dark)', marginBottom: '0.5rem' }}>
                    Pixel to CM Ratio
                  </label>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <input
                      type="range"
                      min="0.01"
                      max="1.0"
                      step="0.01"
                      value={settings.pixelToCmRatio}
                      onChange={(e) => setSettings(prev => ({ ...prev, pixelToCmRatio: parseFloat(e.target.value) }))}
                      style={{ flex: 1 }}
                    />
                    <span style={{ fontSize: '0.875rem', color: 'var(--secondary)', minWidth: '50px', textAlign: 'right' }}>
                      {settings.pixelToCmRatio.toFixed(2)}
                    </span>
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--secondary)', marginTop: '0.25rem' }}>
                    Lower values = larger measurements
                  </div>
                </div>

                <div className="setting-item">
                  <label style={{ display: 'block', fontWeight: 600, color: 'var(--dark)', marginBottom: '0.5rem' }}>
                    Confidence Threshold
                  </label>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <input
                      type="range"
                      min="0.1"
                      max="0.9"
                      step="0.05"
                      value={settings.confidenceThreshold}
                      onChange={(e) => setSettings(prev => ({ ...prev, confidenceThreshold: parseFloat(e.target.value) }))}
                      style={{ flex: 1 }}
                    />
                    <span style={{ fontSize: '0.875rem', color: 'var(--secondary)', minWidth: '50px', textAlign: 'right' }}>
                      {(settings.confidenceThreshold * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--secondary)', marginTop: '0.25rem' }}>
                    Minimum confidence for detections
                  </div>
                </div>

                <div className="setting-item">
                  <label style={{ display: 'block', fontWeight: 600, color: 'var(--dark)', marginBottom: '0.5rem' }}>
                    Optional Depth Map
                  </label>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={(e) => {
                        const f = e.target.files[0];
                        if (f) {
                          setDepthFile(f);
                          const r = new FileReader();
                          r.onloadend = () => setDepthPreview(r.result);
                          r.readAsDataURL(f);
                        } else {
                          setDepthFile(null);
                          setDepthPreview(null);
                        }
                      }}
                      style={{ fontSize: '0.82rem', padding: '0.35rem', border: '1px solid var(--glass-border)', borderRadius: '4px', background: '#fff', width: '100%' }}
                    />
                    {depthFile && (
                      <button
                        type="button"
                        onClick={() => { setDepthFile(null); setDepthPreview(null); }}
                        style={{ background: '#fee2e2', color: '#dc2626', border: 'none', borderRadius: '4px', padding: '4px 8px', fontSize: '0.75rem', cursor: 'pointer', whiteSpace: 'nowrap' }}
                      >
                        Clear
                      </button>
                    )}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--secondary)', marginTop: '0.25rem' }}>
                    {depthFile ? `Loaded: ${depthFile.name}` : 'Optional 8/16-bit depth map. If omitted, monocular contrast depth is synthesized.'}
                  </div>
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
              <button 
                className="btn-primary"
                onClick={handleAnalyze}
              >
                <Play size={20} style={{ marginRight: '0.5rem' }} />
                Start Analysis
              </button>
            </div>
          </div>
        </div>
      )}

      {loading && (
        <div className="card">
          <div className="card-body">
            <div className="progress-container" style={{ marginBottom: '2rem' }}>
              <div className="progress-bar-wrapper">
                <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
              </div>
              <div className="progress-text">
                <Loader size={16} className="spinning" style={{ display: 'inline-block', marginRight: '0.5rem' }} />
                Processing... {progress}%
              </div>
            </div>
          </div>
        </div>
      )}

      {results && !loading && (
        <div className="results-section">
          {/* VISUALIZATIONS FIRST - Priority Display */}
          {outputImages && (
            <div className="card" style={{ marginBottom: '2rem' }}>
              <div className="card-header">
                <h2 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                  <PieChart size={24} />
                  📊 Comprehensive Analysis Visualizations
                </h2>
                <p style={{ margin: '0.25rem 0 0 0', color: 'var(--secondary)', fontSize: '0.875rem' }}>
                  AI-powered image analysis with 6 detailed processing outputs: Original, Cracks, Growth, Segmentation, Depth, Edges
                </p>
              </div>
              <div className="card-body">
                {/* UNIFIED 3x3 IMAGE GRID - ALL 9 IMAGES SAME FORMAT */}
                <div className="image-grid-unified" style={{ 
                  display: 'grid', 
                  gridTemplateColumns: 'repeat(3, 1fr)', 
                  gap: '1.5rem'
                }}>
                  {/* ROW 1: Original 3 Images */}
                  <div className="image-card" style={{ padding: '1rem', background: 'var(--light)', borderRadius: 'var(--border-radius)', border: '1px solid var(--glass-border)' }}>
                    <div className="image-card-title" style={{ fontWeight: 600, marginBottom: '1rem', color: 'var(--dark)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      📸 Original Image
                    </div>
                    {outputImages?.original ? (
                      <img src={outputImages.original} alt="Original" style={{ width: '100%', height: 'auto', objectFit: 'contain', minHeight: '300px', borderRadius: 'var(--border-radius)', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }} />
                    ) : (
                      <div style={{ width: '100%', height: '300px', background: 'rgba(209, 213, 219, 0.5)', borderRadius: 'var(--border-radius)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#a0a0a0', fontSize: '0.9rem' }}>No image loaded</div>
                    )}
                  </div>

                  <div className="image-card" style={{ padding: '1rem', background: 'var(--light)', borderRadius: 'var(--border-radius)', border: '1px solid var(--glass-border)' }}>
                    <div className="image-card-title" style={{ fontWeight: 600, marginBottom: '1rem', color: 'var(--dark)' }}>
                      🟣 Crack MST Graph
                    </div>
                    {outputImages?.crack_mst_graph ? (
                      <img src={outputImages.crack_mst_graph} alt="Crack MST graph" style={{ width: '100%', height: '300px', objectFit: 'cover', borderRadius: 'var(--border-radius)' }} />
                    ) : (
                      <div style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#a0a0a0' }}>Available when segmentation masks are returned</div>
                    )}
                  </div>

                  <div className="image-card" style={{ padding: '1rem', background: 'var(--light)', borderRadius: 'var(--border-radius)', border: '1px solid var(--glass-border)' }}>
                    <div className="image-card-title" style={{ fontWeight: 600, marginBottom: '1rem', color: 'var(--dark)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      🔍 Crack Detection
                      <span style={{ fontSize: '0.875rem', color: 'var(--secondary)', fontWeight: 'normal' }}>
                        ({results?.crack_detection?.count || 0})
                      </span>
                    </div>
                    {outputImages?.crack_detection ? (
                      <img src={outputImages.crack_detection} alt="Crack Detection" style={{ width: '100%', height: 'auto', objectFit: 'contain', minHeight: '320px', borderRadius: 'var(--border-radius)', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }} />
                    ) : (
                      <div style={{ width: '100%', height: '320px', background: 'rgba(209, 213, 219, 0.5)', borderRadius: 'var(--border-radius)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#a0a0a0', fontSize: '0.9rem' }}>Processing...</div>
                    )}
                  </div>

                  <div className="image-card" style={{ padding: '1rem', background: 'var(--light)', borderRadius: 'var(--border-radius)', border: '1px solid var(--glass-border)' }}>
                    <div className="image-card-title" style={{ fontWeight: 600, marginBottom: '1rem', color: 'var(--dark)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      🌿 Biological Growth
                      <span style={{ fontSize: '0.875rem', color: 'var(--secondary)', fontWeight: 'normal' }}>
                        ({results?.biological_growth?.growth_percentage?.toFixed(1) || 0}%)
                      </span>
                    </div>
                    {outputImages?.biological_growth ? (
                      <img src={outputImages.biological_growth} alt="Biological Growth" style={{ width: '100%', height: 'auto', objectFit: 'contain', minHeight: '300px', borderRadius: 'var(--border-radius)', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }} />
                    ) : (
                      <div style={{ width: '100%', height: '300px', background: 'rgba(209, 213, 219, 0.5)', borderRadius: 'var(--border-radius)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#a0a0a0', fontSize: '0.9rem' }}>No image loaded</div>
                    )}
                  </div>

                  {/* ROW 2: Original 3 Images */}
                  <div className="image-card" style={{ padding: '1rem', background: 'var(--light)', borderRadius: 'var(--border-radius)', border: '1px solid var(--glass-border)' }}>
                    <div className="image-card-title" style={{ fontWeight: 600, marginBottom: '1rem', color: 'var(--dark)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      🎯 AI Segmentation
                    </div>
                    {outputImages?.segmentation ? (
                      <img src={outputImages.segmentation} alt="Segmentation" style={{ width: '100%', height: '300px', objectFit: 'cover', borderRadius: 'var(--border-radius)', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }} />
                    ) : (
                      <div style={{ width: '100%', height: '300px', background: 'rgba(209, 213, 219, 0.5)', borderRadius: 'var(--border-radius)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#a0a0a0', fontSize: '0.9rem' }}>Processing...</div>
                    )}
                  </div>

                  <div className="image-card" style={{ padding: '1rem', background: 'var(--light)', borderRadius: 'var(--border-radius)', border: '1px solid var(--glass-border)' }}>
                    <div className="image-card-title" style={{ fontWeight: 600, marginBottom: '1rem', color: 'var(--dark)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      📊 Depth Analysis
                    </div>
                    {outputImages?.depth_estimation ? (
                      <div>
                        <img src={outputImages.depth_estimation} alt="Depth Estimation" style={{ width: '100%', height: '260px', objectFit: 'cover', borderRadius: 'var(--border-radius)', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }} />
                        {(results?.crack_quantification?.depth_recess?.predicted_relative_depth ?? results?.depth_recess?.predicted_relative_depth) !== undefined && (
                          <div style={{ marginTop: '0.6rem', padding: '0.5rem 0.75rem', background: '#f0fdfa', border: '1px solid #ccfbf1', borderRadius: '6px', fontSize: '0.85rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <span style={{ fontWeight: 600, color: '#0f766e' }}>Relative Crevice Depth (ΔZ):</span>
                              <strong style={{ fontSize: '1rem', color: '#0f766e' }}>
                                {(results?.crack_quantification?.depth_recess?.predicted_relative_depth ?? results?.depth_recess?.predicted_relative_depth).toFixed(1)} mm
                              </strong>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.25rem', fontSize: '0.75rem', color: '#64748b' }}>
                              <span>{(results?.crack_quantification?.depth_recess?.reliability || results?.depth_recess?.reliability || 'ESTIMATED').replace('_', ' ')}</span>
                              <span>90% CI: [{(results?.crack_quantification?.depth_recess?.interval_90 || results?.depth_recess?.interval_90 || [0,0])[0]?.toFixed(1)}, {(results?.crack_quantification?.depth_recess?.interval_90 || results?.depth_recess?.interval_90 || [0,0])[1]?.toFixed(1)}] mm</span>
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div style={{ width: '100%', height: '300px', background: 'rgba(209, 213, 219, 0.5)', borderRadius: 'var(--border-radius)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#a0a0a0', fontSize: '0.9rem' }}>Processing...</div>
                    )}
                  </div>

                  <div className="image-card" style={{ padding: '1rem', background: 'var(--light)', borderRadius: 'var(--border-radius)', border: '1px solid var(--glass-border)' }}>
                    <div className="image-card-title" style={{ fontWeight: 600, marginBottom: '1rem', color: 'var(--dark)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      ⚡ Edge Detection
                    </div>
                    {outputImages?.edge_detection ? (
                      <img src={outputImages.edge_detection} alt="Edge Detection" style={{ width: '100%', height: '300px', objectFit: 'cover', borderRadius: 'var(--border-radius)', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }} />
                    ) : (
                      <div style={{ width: '100%', height: '300px', background: 'rgba(209, 213, 219, 0.5)', borderRadius: 'var(--border-radius)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#a0a0a0', fontSize: '0.9rem' }}>Processing...</div>
                    )}
                  </div>

                  {/* ROW 3: NEW Advanced Analysis - SAME FORMAT as others */}
                  <div className="image-card" style={{ padding: '1rem', background: 'var(--light)', borderRadius: 'var(--border-radius)', border: '1px solid var(--glass-border)' }}>
                    <div className="image-card-title" style={{ fontWeight: 600, marginBottom: '1rem', color: 'var(--dark)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      💧 Moisture/Dampness Heatmap
                    </div>
                    {outputImages?.moisture_dampness_heatmap ? (
                      <img src={outputImages.moisture_dampness_heatmap} alt="Moisture Heatmap" style={{ width: '100%', height: '300px', objectFit: 'cover', borderRadius: 'var(--border-radius)', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }} />
                    ) : (
                      <div style={{ width: '100%', height: '300px', background: 'rgba(209, 213, 219, 0.5)', borderRadius: 'var(--border-radius)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#a0a0a0', fontSize: '0.9rem' }}>Processing...</div>
                    )}
                  </div>

                  <div className="image-card" style={{ padding: '1rem', background: 'var(--light)', borderRadius: 'var(--border-radius)', border: '1px solid var(--glass-border)' }}>
                    <div className="image-card-title" style={{ fontWeight: 600, marginBottom: '1rem', color: 'var(--dark)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      🔴 Structural Stress Map
                    </div>
                    {outputImages?.structural_stress_map ? (
                      <img src={outputImages.structural_stress_map} alt="Stress Map" style={{ width: '100%', height: '300px', objectFit: 'cover', borderRadius: 'var(--border-radius)', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }} />
                    ) : (
                      <div style={{ width: '100%', height: '300px', background: 'rgba(209, 213, 219, 0.5)', borderRadius: 'var(--border-radius)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#a0a0a0', fontSize: '0.9rem' }}>Processing...</div>
                    )}
                  </div>

                  <div className="image-card" style={{ padding: '1rem', background: 'var(--light)', borderRadius: 'var(--border-radius)', border: '1px solid var(--glass-border)' }}>
                    <div className="image-card-title" style={{ fontWeight: 600, marginBottom: '1rem', color: 'var(--dark)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      🔥 Thermal/Infrared Simulation
                    </div>
                    {outputImages?.thermal_infrared_simulation ? (
                      <img src={outputImages.thermal_infrared_simulation} alt="Thermal Map" style={{ width: '100%', height: '300px', objectFit: 'cover', borderRadius: 'var(--border-radius)', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }} />
                    ) : (
                      <div style={{ width: '100%', height: '300px', background: 'rgba(209, 213, 219, 0.5)', borderRadius: 'var(--border-radius)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#a0a0a0', fontSize: '0.9rem' }}>Processing...</div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}
          {results.crack_quantification && (
            <div className="card" style={{ marginBottom: '2rem', borderLeft: '4px solid #0f766e' }}>
              <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
                <div>
                  <h2 style={{ margin: 0 }}>Crack Centreline Length & 3D Depth Recess</h2>
                  <p style={{ margin: '0.25rem 0 0', color: 'var(--secondary)', fontSize: '0.875rem' }}>
                    Multi-modal quantification: Centreline length from segmentation mask & relative crevice depth (ΔZ)
                  </p>
                </div>
                {(results.crack_quantification?.depth_recess?.reliability || results.depth_recess?.reliability) && (
                  <span style={{
                    padding: '4px 12px',
                    borderRadius: '20px',
                    fontSize: '0.78rem',
                    fontWeight: 700,
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                    background: (results.crack_quantification?.depth_recess?.reliability || results.depth_recess?.reliability) === 'HIGH_RELIABILITY' ? 'rgba(34, 197, 94, 0.15)' : (results.crack_quantification?.depth_recess?.reliability || results.depth_recess?.reliability) === 'MEDIUM_RELIABILITY' ? 'rgba(234, 179, 8, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                    color: (results.crack_quantification?.depth_recess?.reliability || results.depth_recess?.reliability) === 'HIGH_RELIABILITY' ? '#16a34a' : (results.crack_quantification?.depth_recess?.reliability || results.depth_recess?.reliability) === 'MEDIUM_RELIABILITY' ? '#ca8a04' : '#dc2626',
                    border: `1px solid ${(results.crack_quantification?.depth_recess?.reliability || results.depth_recess?.reliability) === 'HIGH_RELIABILITY' ? '#16a34a' : (results.crack_quantification?.depth_recess?.reliability || results.depth_recess?.reliability) === 'MEDIUM_RELIABILITY' ? '#ca8a04' : '#dc2626'}`
                  }}>
                    Depth: {(results.crack_quantification?.depth_recess?.reliability || results.depth_recess?.reliability).replace('_', ' ')}
                  </span>
                )}
              </div>
              <div className="card-body">
                {results.crack_quantification.available ? (
                  <>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem', marginBottom: '1rem' }}>
                      <div><span>Pixel network length</span><strong style={{ display: 'block', fontSize: '1.4rem' }}>{results.crack_quantification.total_pixel_length?.toFixed(2)} px</strong></div>
                      <div><span>Scale</span><strong style={{ display: 'block', fontSize: '1.4rem' }}>{results.crack_quantification.scale?.mm_per_px ? `${results.crack_quantification.scale.mm_per_px.toFixed(4)} mm/px` : 'Unavailable'}</strong></div>
                      <div><span>Physical length</span><strong style={{ display: 'block', fontSize: '1.4rem' }}>{results.crack_quantification.total_length_mm ? `${results.crack_quantification.total_length_mm.toFixed(2)} mm` : 'Pixel-only'}</strong></div>
                      <div>
                        <span>3D Depth Recess (ΔZ)</span>
                        <strong style={{ display: 'block', fontSize: '1.4rem', color: '#0f766e' }}>
                          {(results.crack_quantification?.depth_recess?.predicted_relative_depth ?? results.depth_recess?.predicted_relative_depth) !== undefined ? `${(results.crack_quantification?.depth_recess?.predicted_relative_depth ?? results.depth_recess?.predicted_relative_depth).toFixed(1)} mm` : 'N/A'}
                        </strong>
                        {(results.crack_quantification?.depth_recess?.interval_90 || results.depth_recess?.interval_90) && (
                          <span style={{ fontSize: '0.75rem', color: 'var(--secondary)' }}>
                            90% CI: [{(results.crack_quantification?.depth_recess?.interval_90 || results.depth_recess?.interval_90)[0]?.toFixed(1)}, {(results.crack_quantification?.depth_recess?.interval_90 || results.depth_recess?.interval_90)[1]?.toFixed(1)}] mm
                          </span>
                        )}
                      </div>
                    </div>
                    <p style={{ marginBottom: 0, color: '#9a3412', fontSize: '0.85rem' }}>
                      Method: {results.crack_quantification.path_method} + network graph. Scale: {results.crack_quantification.scale?.method || 'not recovered'} ({results.crack_quantification.scale?.confidence || 'low confidence'}). Depth model: ExtraTrees Champion (Crevice depth in mm).
                    </p>
                  </>
                ) : (
                  <p style={{ margin: 0, color: '#9a3412' }}>{results.crack_quantification.reason}</p>
                )}
              </div>
            </div>
          )}
          <div className="card" style={{ marginBottom: '2rem' }}>
            <div className="card-header">
              <h2 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                <BarChart3 size={24} />
                📈 Comprehensive Structural Assessment Dashboard
              </h2>
              <p style={{ margin: '0.25rem 0 0 0', color: 'var(--secondary)', fontSize: '0.875rem' }}>
                Real-time analysis metrics: Health Score, Damage Detection, Material Classification, Severity Assessment & Environmental Impact
              </p>
            </div>
            <div className="card-body">
              {/* Key Performance Indicators */}
              <div className="kpi-section" style={{ marginBottom: '2rem' }}>
                <h4 style={{ marginBottom: '1rem', color: 'var(--dark)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <TrendingUp size={20} />
                  Key Performance Indicators
                </h4>
                <div className="metrics-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
                  <MetricCard
                    icon={CheckCircle}
                    title="Structural Health"
                    value={`${results.data_science_insights?.statistical_summary?.structural_health_score?.toFixed(1) || 'N/A'}/100`}
                    subtitle="Overall condition"
                    color={results.data_science_insights?.statistical_summary?.structural_health_score > 70 ? 'success' : results.data_science_insights?.statistical_summary?.structural_health_score > 40 ? 'warning' : 'danger'}
                  />
                  <MetricCard
                    icon={AlertTriangle}
                    title="Risk Level"
                    value={results.data_science_insights?.predictive_analytics?.risk_assessment || 'Low'}
                    subtitle="Based on analysis"
                    color={results.data_science_insights?.predictive_analytics?.risk_assessment === 'Critical' ? 'danger' : results.data_science_insights?.predictive_analytics?.risk_assessment === 'Moderate' ? 'warning' : 'success'}
                  />
                  <MetricCard
                    icon={Leaf}
                    title="Sustainability"
                    value={`${results.environmental_impact_assessment?.sustainability_score?.toFixed(1) || 'N/A'}/10`}
                    subtitle="Environmental rating"
                    color={results.environmental_impact_assessment?.sustainability_score > 7 ? 'success' : results.environmental_impact_assessment?.sustainability_score > 4 ? 'warning' : 'danger'}
                  />
                  <MetricCard
                    icon={Activity}
                    title="Maintenance Urgency"
                    value={results.data_science_insights?.statistical_summary?.maintenance_urgency || 'Low'}
                    subtitle="Recommended action"
                    color={results.data_science_insights?.statistical_summary?.maintenance_urgency === 'High' ? 'danger' : results.data_science_insights?.statistical_summary?.maintenance_urgency === 'Medium' ? 'warning' : 'success'}
                  />
                  <MetricCard
                    icon={Activity}
                    title="3D Crevice Depth"
                    value={(results.crack_quantification?.depth_recess?.predicted_relative_depth ?? results.depth_recess?.predicted_relative_depth) !== undefined ? `${(results.crack_quantification?.depth_recess?.predicted_relative_depth ?? results.depth_recess?.predicted_relative_depth).toFixed(1)} mm` : 'N/A'}
                    subtitle={(results.crack_quantification?.depth_recess?.reliability || results.depth_recess?.reliability) ? `${(results.crack_quantification?.depth_recess?.reliability || results.depth_recess?.reliability).replace('_', ' ')} (90% CI)` : '3D Crevice Depth'}
                    color="warning"
                  />
                </div>
              </div>

              {/* Material Analysis with Dynamic Confidence Chart */}
              <div className="material-analysis-section" style={{ marginBottom: '2rem' }}>
                <h4 style={{ marginBottom: '1rem', color: 'var(--dark)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Target size={20} />
                  🏗️ Material Classification & Durability Properties
                </h4>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
                  <MetricCard
                    icon={Activity}
                    title="Primary Material"
                    value={results.material_analysis?.predicted_material || 'Unknown'}
                    subtitle={`AI Confidence: ${(results.material_analysis?.probabilities ? Math.max(...(typeof results.material_analysis.probabilities === 'object' ? Object.values(results.material_analysis.probabilities) : [])) * 100 : 0).toFixed(1)}%`}
                    color="primary"
                  />
                  <MetricCard
                    icon={BarChart3}
                    title="Material Density"
                    value={`${results.material_analysis?.material_properties?.density_kg_m3 || 'N/A'}`}
                    subtitle="kg/m³"
                    color="info"
                  />
                  <MetricCard
                    icon={TrendingUp}
                    title="Durability"
                    value={`${results.material_analysis?.material_properties?.durability_score || 'N/A'}/10`}
                    subtitle="Durability Score"
                    color="success"
                  />
                  <MetricCard
                    icon={Leaf}
                    title="Material Quality"
                    value={results.material_analysis?.predicted_material?.includes('Stone') || results.material_analysis?.predicted_material?.includes('Concrete') ? 'Excellent' : results.material_analysis?.predicted_material?.includes('Brick') || results.material_analysis?.predicted_material?.includes('Marble') ? 'Very Good' : 'Good'}
                    subtitle="Estimated Grade"
                    color="warning"
                  />
                </div>

                {/* Severity Classification Card */}
                <div style={{ 
                  background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(245, 158, 11, 0.1))', 
                  border: '1px solid rgba(239, 68, 68, 0.2)', 
                  borderRadius: 'var(--border-radius)', 
                  padding: '1.5rem',
                  marginBottom: '1.5rem'
                }}>
                  <h5 style={{ margin: '0 0 1rem 0', color: 'var(--dark)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <AlertTriangle size={18} />
                    ⚠️ Damage Severity Distribution & Assessment
                  </h5>
                  {results.crack_detection?.details?.length > 0 ? (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '1rem' }}>
                      {results.crack_detection.statistics?.severity_distribution && Object.entries(results.crack_detection.statistics.severity_distribution).map(([severity, count]) => {
                        const severityColors = {
                          'Critical': { bg: '#dc2626', light: 'rgba(220, 38, 38, 0.1)' },
                          'Severe': { bg: '#ea580c', light: 'rgba(234, 88, 12, 0.1)' },
                          'Moderate': { bg: '#ca8a04', light: 'rgba(202, 138, 4, 0.1)' },
                          'Minor': { bg: '#16a34a', light: 'rgba(22, 163, 74, 0.1)' }
                        };
                        const colors = severityColors[severity] || { bg: '#6b7280', light: 'rgba(107, 114, 128, 0.1)' };
                        return (
                          <div key={severity} style={{
                            background: colors.light,
                            border: `2px solid ${colors.bg}`,
                            borderRadius: '8px',
                            padding: '1rem',
                            textAlign: 'center'
                          }}>
                            <div style={{ fontSize: '0.75rem', color: 'var(--secondary)', marginBottom: '0.5rem', fontWeight: '600' }}>
                              {severity}
                            </div>
                            <div style={{ fontSize: '1.5rem', fontWeight: '700', color: colors.bg, marginBottom: '0.25rem' }}>
                              {count}
                            </div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--secondary)' }}>
                              {count === 1 ? 'crack' : 'cracks'}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div style={{ color: 'var(--secondary)', fontStyle: 'italic' }}>
                      ✅ No structural cracks detected - excellent condition!
                    </div>
                  )}
                </div>
              </div>

              {/* Environmental Impact with Dynamic Values */}
              <div className="environmental-impact-section">
                <h4 style={{ marginBottom: '1rem', color: 'var(--dark)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Wind size={20} />
                  Environmental Impact Assessment
                </h4>
                <div className="metrics-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
                  <MetricCard
                    icon={Wind}
                    title="Carbon Footprint"
                    value={`${results.environmental_impact_assessment?.carbon_footprint_kg?.toFixed(2)} kg`}
                    subtitle="CO₂ Equivalent"
                    color="danger"
                  />
                  <MetricCard
                    icon={Droplet}
                    title="Water Footprint"
                    value={`${results.environmental_impact_assessment?.water_footprint_liters?.toFixed(2)} L`}
                    color="info"
                  />
                  <MetricCard
                    icon={Zap}
                    title="Energy Consumption"
                    value={`${results.environmental_impact_assessment?.energy_consumption_kwh?.toFixed(2)} kWh`}
                    color="warning"
                  />
                  <MetricCard
                    icon={Leaf}
                    title="Eco-Efficiency"
                    value={`${results.environmental_impact_assessment?.eco_efficiency_rating?.toFixed(1)}/10`}
                    subtitle="Efficiency rating"
                    color="success"
                  />
                </div>

                {/* Environmental Recommendations */}
                {results.environmental_impact_assessment?.recommendations && (
                  <div className="recommendations-section" style={{ marginTop: '2rem' }}>
                    <h6 style={{ marginBottom: '1rem', fontWeight: 700, color: 'var(--dark)' }}>Environmental Recommendations</h6>
                    <div className="recommendations-list" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem' }}>
                      {results.environmental_impact_assessment.recommendations.map((rec, idx) => (
                        <div key={idx} style={{
                          padding: '1rem',
                          background: 'var(--light)',
                          borderRadius: 'var(--border-radius)',
                          border: '1px solid var(--glass-border)',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.75rem'
                        }}>
                          <CheckCircle size={18} style={{ color: 'var(--primary)', flexShrink: 0 }} />
                          <span style={{ fontSize: '0.875rem', color: 'var(--dark)' }}>{rec}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Detailed Analysis Sections */}
          <div className="detailed-analysis" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '2rem' }}>
            {/* Crack Detection Details */}
            <div className="card">
              <div className="card-header">
                <h3 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                  <AlertTriangle size={20} />
                  🔴 Structural Crack Detection & Analysis
                </h3>
              </div>
              <div className="card-body">
                <div className="metrics-grid" style={{ marginBottom: '1.5rem' }}>
                  <MetricCard
                    icon={AlertTriangle}
                    title="Total Cracks"
                    value={results.crack_detection?.count || 0}
                    color="danger"
                  />
                  <MetricCard
                    icon={BarChart3}
                    title="Total Area"
                    value={`${results.crack_detection?.statistics?.total_area_cm2?.toFixed(2) || 0} cm²`}
                    color="warning"
                  />
                  <MetricCard
                    icon={TrendingUp}
                    title="Average Size"
                    value={`${results.crack_detection?.statistics?.average_size_cm2?.toFixed(2) || 0} cm²`}
                    color="info"
                  />
                </div>

                {results.crack_detection?.details?.length > 0 && (
                  <div className="crack-details">
                    <h6 style={{ marginBottom: '1rem', fontWeight: 700, color: 'var(--dark)' }}>
                      Detected Cracks ({results.crack_detection.details.length})
                    </h6>
                    <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                      {results.crack_detection.details.slice(0, 10).map((crack, idx) => (
                        <div key={idx} className="crack-item" style={{
                          marginBottom: '0.75rem',
                          padding: '0.75rem',
                          background: 'var(--light)',
                          borderRadius: 'var(--border-radius)',
                          border: '1px solid #e5e7eb',
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center'
                        }}>
                          <span style={{ fontWeight: 600, color: 'var(--dark)' }}>
                            Crack {idx + 1}: {crack.width_cm?.toFixed(2)} × {crack.length_cm?.toFixed(2)} cm
                          </span>
                          <SeverityBadge severity={crack.severity} />
                        </div>
                      ))}
                      {results.crack_detection.details.length > 10 && (
                        <div style={{ textAlign: 'center', padding: '1rem', color: 'var(--secondary)', fontSize: '0.875rem' }}>
                          ... and {results.crack_detection.details.length - 10} more cracks detected
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Biological Growth Details */}
            <div className="card">
              <div className="card-header">
                <h3 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                  <Leaf size={20} />
                  🌱 Biological Growth & Contamination Analysis
                </h3>
              </div>
              <div className="card-body">
                <div className="metrics-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
                  <MetricCard
                    icon={Leaf}
                    title="Growth Detected"
                    value={results.biological_growth?.growth_detected ? 'Yes ✓' : 'No ✓'}
                    color={results.biological_growth?.growth_detected ? 'warning' : 'success'}
                  />
                  <MetricCard
                    icon={BarChart3}
                    title="Coverage Area"
                    value={`${results.biological_growth?.growth_percentage?.toFixed(2)}%`}
                    color={results.biological_growth?.growth_percentage > 10 ? 'danger' : results.biological_growth?.growth_percentage > 5 ? 'warning' : 'success'}
                  />
                  <MetricCard
                    icon={TrendingUp}
                    title="Affected Area"
                    value={`${results.biological_growth?.affected_area_cm2?.toFixed(2)} cm²`}
                    color="info"
                  />
                  <MetricCard
                    icon={Activity}
                    title="Growth Type"
                    value={results.biological_growth?.growth_percentage > 0 ? 'Moss/Algae/Lichen' : 'None'}
                    color={results.biological_growth?.growth_percentage > 0 ? 'warning' : 'success'}
                  />
                </div>

                {/* Biological Growth Classification */}
                <div style={{ 
                  background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(5, 150, 105, 0.1))', 
                  border: '1px solid rgba(16, 185, 129, 0.2)', 
                  borderRadius: 'var(--border-radius)', 
                  padding: '1.5rem',
                  marginBottom: '1.5rem'
                }}>
                  <h5 style={{ margin: '0 0 1rem 0', color: 'var(--dark)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Leaf size={18} />
                    Biological Growth Assessment
                  </h5>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '1rem' }}>
                    <div style={{
                      background: 'rgba(16, 185, 129, 0.1)',
                      border: '2px solid #10b981',
                      borderRadius: '8px',
                      padding: '1rem',
                      textAlign: 'center'
                    }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--secondary)', marginBottom: '0.5rem', fontWeight: '600' }}>
                        Moss/Lichen Growth
                      </div>
                      <div style={{ fontSize: '1.25rem', fontWeight: '700', color: '#10b981' }}>
                        {results.biological_growth?.growth_percentage > 5 ? '⚠ Present' : '✓ None'}
                      </div>
                    </div>
                    <div style={{
                      background: 'rgba(34, 197, 94, 0.1)',
                      border: '2px solid #22c55e',
                      borderRadius: '8px',
                      padding: '1rem',
                      textAlign: 'center'
                    }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--secondary)', marginBottom: '0.5rem', fontWeight: '600' }}>
                        Algae Growth
                      </div>
                      <div style={{ fontSize: '1.25rem', fontWeight: '700', color: '#22c55e' }}>
                        {results.biological_growth?.growth_percentage > 8 ? '⚠ Present' : '✓ None'}
                      </div>
                    </div>
                    <div style={{
                      background: 'rgba(34, 211, 238, 0.1)',
                      border: '2px solid #06b6d4',
                      borderRadius: '8px',
                      padding: '1rem',
                      textAlign: 'center'
                    }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--secondary)', marginBottom: '0.5rem', fontWeight: '600' }}>
                        Vegetation
                      </div>
                      <div style={{ fontSize: '1.25rem', fontWeight: '700', color: '#06b6d4' }}>
                        {results.biological_growth?.growth_percentage > 12 ? '⚠ Present' : '✓ None'}
                      </div>
                    </div>
                  </div>
                  
                  {/* Growth Trend Visualization */}
                  {results.biological_growth?.growth_percentage > 0 && (
                    <div style={{ marginTop: '1.5rem' }}>
                      <h6 style={{ marginBottom: '1rem', fontWeight: '700', color: 'var(--dark)' }}>Growth Coverage Intensity</h6>
                      <div style={{
                        width: '100%',
                        height: '60px',
                        background: 'linear-gradient(90deg, var(--success), var(--warning), var(--danger))',
                        borderRadius: '30px',
                        position: 'relative',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}>
                        <div style={{
                          position: 'absolute',
                          left: `${Math.min(results.biological_growth.growth_percentage, 100)}%`,
                          transform: 'translateX(-50%)',
                          width: '4px',
                          height: '40px',
                          background: 'var(--dark)',
                          borderRadius: '2px'
                        }} />
                        <span style={{
                          color: 'white',
                          fontWeight: 'bold',
                          textShadow: '0 0 4px rgba(0,0,0,0.5)',
                          zIndex: 1
                        }}>
                          {results.biological_growth.growth_percentage.toFixed(1)}%
                        </span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.5rem', fontSize: '0.75rem', color: 'var(--secondary)' }}>
                        <span>Low (0%)</span>
                        <span>Moderate (50%)</span>
                        <span>High (100%)</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="card" style={{ marginTop: '2rem' }}>
            <div className="card-body">
              <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
                <button
                  className="btn-primary"
                  onClick={() => {
                    setFile(null);
                    setPreview(null);
                    setResults(null);
                    setOutputImages(null);
                    setError(null);
                    clearAnalysis(); // Clear from context as well
                  }}
                >
                  <Upload size={18} style={{ marginRight: '0.5rem' }} />
                  Analyze Another Image
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {preview && !results && !loading && (
        <div className="card">
          <div className="card-header">
            <h2 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <Camera size={24} />
              Image Preview
            </h2>
          </div>
          <div className="card-body">
            <img src={preview} alt="Preview" className="preview-image" />
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageAnalysis;