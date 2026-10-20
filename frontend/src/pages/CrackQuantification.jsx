import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { useAnalysis } from '../contexts/AnalysisContext';
import { API_URL } from '../config/apiConfig';
import {
  Ruler,
  Camera,
  Upload,
  Play,
  Loader,
  Layers,
  Box,
  AlertTriangle,
  CheckCircle,
  RefreshCw,
  Sparkles,
  ArrowLeft,
  ArrowRight,
  Download,
  Info,
  Sliders,
  Maximize2,
  Shield
} from 'lucide-react';
import {
  QuantificationMetrics,
  StagePipelineGallery,
  TriMethodCard,
  RestorationCard,
  WidthVolumeViewer,
  XaiAttributionCard,
  DepthRecessCard
} from '../components/quantification';

const CrackQuantification = () => {
  const {
    uploadedImage,
    currentFile,
    quantificationResults,
    setQuantificationResults,
    navigateToSegmentation,
    setActiveTab,
    setUploadedImage,
    setCurrentFile,
    activeQuantSubTab,
    setActiveQuantSubTab
  } = useAnalysis();

  const currentSubTab = activeQuantSubTab || 'all';

  const [imagePreview, setImagePreview] = useState(uploadedImage || null);
  const [file, setFile] = useState(currentFile || null);
  const [depthFile, setDepthFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  // Settings
  const [mmPerPx, setMmPerPx] = useState(0.277);
  const [calibrationMode, setCalibrationMode] = useState('user'); // 'user', 'auto', 'camera'
  const [applyRestoration, setApplyRestoration] = useState(true);
  const [distanceMm, setDistanceMm] = useState(1000.0);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.25);

  const fileInputRef = useRef(null);
  const depthInputRef = useRef(null);

  // Synchronize when uploadedImage in context changes
  useEffect(() => {
    if (uploadedImage && !imagePreview) {
      setImagePreview(uploadedImage);
    }
    if (currentFile && !file) {
      setFile(currentFile);
    }
  }, [uploadedImage, currentFile]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setCurrentFile(selectedFile);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
        setUploadedImage(reader.result);
        sessionStorage.setItem('lastUploadedImage', reader.result);
      };
      reader.readAsDataURL(selectedFile);
      setError(null);
    }
  };

  const handleRunQuantification = async () => {
    if (!imagePreview && !file) {
      setError('Please upload an image or import one from Image Analysis.');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(false);

    try {
      const formData = new FormData();
      if (file) {
        formData.append('image', file);
      } else if (imagePreview) {
        // Convert base64 dataURL to blob
        const res = await fetch(imagePreview);
        const blob = await res.blob();
        formData.append('image', blob, 'crack_inspection.jpg');
      }

      formData.append('mm_per_px', mmPerPx.toString());
      formData.append('calibration_mode', calibrationMode);
      formData.append('apply_restoration', applyRestoration ? 'true' : 'false');
      formData.append('distance_mm', distanceMm.toString());
      formData.append('confidence_threshold', confidenceThreshold.toString());

      if (depthFile) {
        formData.append('depth', depthFile);
      }

      const response = await axios.post(`${API_URL}/api/crack-length`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 90000
      });

      if (response.data && response.data.status === 'success') {
        setQuantificationResults(response.data);
        setSuccess(true);
      } else {
        setError(response.data?.reason || 'Quantification failed to return valid data.');
      }
    } catch (err) {
      console.error('Quantification API error:', err);
      const errMsg = err.response?.data?.reason || err.message || 'Server connection error during quantification.';
      setError(`Quantification failed: ${errMsg}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', width: '100%' }}>
      {/* Navigation Bridge between Segmentation and Quantification */}
      <div style={{
        background: '#fff',
        border: '1px solid #e2e8f0',
        borderRadius: '14px',
        padding: '0.85rem 1.25rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '0.75rem',
        boxShadow: '0 4px 16px rgba(0,0,0,0.03)'
      }}>
        <button
          onClick={navigateToSegmentation}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '8px',
            background: '#f1f5f9',
            border: '1px solid #cbd5e1',
            borderRadius: '8px',
            padding: '7px 14px',
            color: '#334155',
            fontSize: '0.85rem',
            fontWeight: '600',
            cursor: 'pointer',
            transition: 'all 0.15s ease'
          }}
        >
          <ArrowLeft size={16} /> Back to Image Segmentation & Material Analysis
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.82rem', color: '#64748b' }}>
          <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#22c55e' }}></span>
          {imagePreview ? (
            <span style={{ color: '#0f172a', fontWeight: '600' }}>
              Image active: Ready for Physical Quantification
            </span>
          ) : (
            <span>Upload an infrastructure photo or select from Segmentation</span>
          )}
        </div>
      </div>

      {/* Main Controls Card */}
      <div style={{
        background: '#fff',
        border: '1px solid #e2e8f0',
        borderRadius: '16px',
        padding: '1.5rem',
        boxShadow: '0 8px 30px rgba(0,0,0,0.04)',
        display: 'flex',
        flexDirection: 'column',
        gap: '1.25rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem', borderBottom: '1px solid #f1f5f9', paddingBottom: '1rem' }}>
          <div>
            <h2 style={{ margin: 0, fontSize: '1.35rem', fontWeight: '800', color: '#0f172a', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Ruler size={24} color="#0284c7" />
              Physical Crack Quantification Platform
            </h2>
            <p style={{ margin: '4px 0 0 0', fontSize: '0.88rem', color: '#64748b' }}>
              Scientific multi-method benchmark: Length (mm), Opening Width, 3D Crevice Recess Volume (cm³), and ACI 224R Structural Code Assessment.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '0.6rem', flexWrap: 'wrap' }}>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="image/*"
              style={{ display: 'none' }}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                padding: '9px 16px',
                background: '#f8fafc',
                border: '1px solid #cbd5e1',
                borderRadius: '10px',
                color: '#334155',
                fontSize: '0.85rem',
                fontWeight: '600',
                cursor: 'pointer'
              }}
            >
              <Upload size={16} /> Choose Image
            </button>

            <button
              onClick={() => {
                fetch('/sample_crack.jpg')
                  .then(res => res.blob())
                  .then(blob => {
                    const sampleFile = new File([blob], 'concrete_crack_sample.jpg', { type: 'image/jpeg' });
                    setFile(sampleFile);
                    setCurrentFile(sampleFile);
                    const reader = new FileReader();
                    reader.onloadend = () => {
                      setImagePreview(reader.result);
                      setUploadedImage(reader.result);
                      sessionStorage.setItem('lastUploadedImage', reader.result);
                    };
                    reader.readAsDataURL(blob);
                  })
                  .catch(err => console.error('Error loading sample image:', err));
              }}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                padding: '9px 16px',
                background: '#f0fdf4',
                border: '1px solid #86efac',
                borderRadius: '10px',
                color: '#15803d',
                fontSize: '0.85rem',
                fontWeight: '600',
                cursor: 'pointer'
              }}
              title="Load standard concrete crack inspection sample image"
            >
              <Sparkles size={16} /> Load Demo Photo
            </button>

            <input
              type="file"
              ref={depthInputRef}
              onChange={(e) => setDepthFile(e.target.files[0])}
              accept="image/*"
              style={{ display: 'none' }}
            />
            <button
              onClick={() => depthInputRef.current?.click()}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                padding: '9px 16px',
                background: depthFile ? 'rgba(15, 118, 110, 0.1)' : '#f8fafc',
                border: depthFile ? '1px solid #0f766e' : '1px solid #cbd5e1',
                borderRadius: '10px',
                color: depthFile ? '#0f766e' : '#334155',
                fontSize: '0.85rem',
                fontWeight: '600',
                cursor: 'pointer'
              }}
              title="Attach optional depth map file (otherwise automatic monocular depth is used)"
            >
              <Box size={16} /> {depthFile ? 'Depth Map Attached' : 'Attach Depth Map'}
            </button>

            <button
              onClick={handleRunQuantification}
              disabled={loading || (!imagePreview && !file)}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                padding: '9px 22px',
                background: loading ? '#94a3b8' : 'linear-gradient(135deg, #0284c7, #2563eb)',
                border: 'none',
                borderRadius: '10px',
                color: '#fff',
                fontSize: '0.88rem',
                fontWeight: '700',
                cursor: loading ? 'not-allowed' : 'pointer',
                boxShadow: '0 4px 14px rgba(2, 132, 199, 0.3)'
              }}
            >
              {loading ? (
                <>
                  <Loader size={18} className="animate-spin" /> Processing Stages...
                </>
              ) : (
                <>
                  <Play size={18} /> Run Physical Quantification
                </>
              )}
            </button>
          </div>
        </div>

        {/* Quick Settings Toolbar */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '1rem',
          background: '#f8fafc',
          padding: '1rem',
          borderRadius: '12px',
          border: '1px solid #e2e8f0',
          fontSize: '0.85rem'
        }}>
          <div>
            <label style={{ display: 'block', fontWeight: '600', color: '#475569', marginBottom: '4px' }}>
              Metric Scale (mm/px):
            </label>
            <input
              type="number"
              step="0.001"
              value={mmPerPx}
              onChange={(e) => setMmPerPx(parseFloat(e.target.value) || 0.277)}
              style={{
                width: '100%',
                padding: '6px 10px',
                borderRadius: '6px',
                border: '1px solid #cbd5e1',
                fontSize: '0.85rem',
                fontWeight: '600',
                background: '#fff'
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', fontWeight: '600', color: '#475569', marginBottom: '4px' }}>
              Calibration Pathway:
            </label>
            <select
              value={calibrationMode}
              onChange={(e) => setCalibrationMode(e.target.value)}
              style={{
                width: '100%',
                padding: '6px 10px',
                borderRadius: '6px',
                border: '1px solid #cbd5e1',
                fontSize: '0.85rem',
                background: '#fff'
              }}
            >
              <option value="user">Known Metric Scale (mm/px)</option>
              <option value="auto">Auto EXIF & Sensor DB</option>
              <option value="marker">ArUco Calibration Marker</option>
              <option value="planar_quad">Interactive 4-Point Quad</option>
            </select>
          </div>

          <div>
            <label style={{ display: 'block', fontWeight: '600', color: '#475569', marginBottom: '4px' }}>
              Working Distance (mm):
            </label>
            <input
              type="number"
              step="50"
              value={distanceMm}
              onChange={(e) => setDistanceMm(parseFloat(e.target.value) || 1000)}
              style={{
                width: '100%',
                padding: '6px 10px',
                borderRadius: '6px',
                border: '1px solid #cbd5e1',
                fontSize: '0.85rem',
                background: '#fff'
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', fontWeight: '600', color: '#475569', marginBottom: '4px' }}>
              Detection Confidence:
            </label>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={confidenceThreshold}
              onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
              style={{ width: '100%', marginTop: '6px' }}
            />
            <div style={{ textAlign: 'right', fontSize: '0.75rem', color: '#64748b' }}>
              {(confidenceThreshold * 100).toFixed(0)}%
            </div>
          </div>
        </div>

        {error && (
          <div style={{ padding: '12px 16px', background: '#fee2e2', border: '1px solid #f87171', borderRadius: '10px', color: '#991b1b', fontSize: '0.88rem' }}>
            <strong>Error:</strong> {error}
          </div>
        )}
      </div>

      {/* Dedicated Quantification & 3D Depth Sub-Navigation Bar */}
      <div style={{
        position: 'sticky',
        top: '10px',
        zIndex: 40,
        background: '#0f172a',
        borderRadius: '14px',
        padding: '8px 12px',
        boxShadow: '0 10px 25px -5px rgba(15, 23, 42, 0.4)',
        border: '1px solid #1e293b',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '8px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
          {[
            { id: 'all', label: 'All Modules (Overview)', icon: Layers, badge: 'Full Board' },
            { id: 'depth', label: '3D Depth & Recess Engine', icon: Box, badge: 'ExtraTrees ML', color: '#06b6d4', glow: true },
            { id: 'width', label: 'Width & ACI Severity', icon: Ruler, badge: 'Calipers', color: '#f59e0b' },
            { id: 'restoration', label: 'AI Restoration Comparison', icon: Sparkles, badge: 'Enhanced', color: '#38bdf8' },
            { id: 'xai', label: 'Explainable AI (XAI)', icon: Shield, badge: 'Attribution', color: '#a855f7' },
            { id: 'gallery', label: '16 Diagnostic Layers', icon: Maximize2, badge: 'Gallery', color: '#10b981' }
          ].map(tab => {
            const active = currentSubTab === tab.id;
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveQuantSubTab(tab.id)}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '8px 14px',
                  borderRadius: '10px',
                  fontSize: '0.84rem',
                  fontWeight: active ? '700' : '600',
                  color: active ? '#ffffff' : '#94a3b8',
                  background: active
                    ? (tab.color ? `linear-gradient(135deg, ${tab.color}, #0284c7)` : 'linear-gradient(135deg, #0284c7, #2563eb)')
                    : 'transparent',
                  border: active
                    ? '1px solid rgba(255,255,255,0.3)'
                    : '1px solid transparent',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                  boxShadow: active && tab.glow ? '0 0 16px rgba(6, 182, 212, 0.45)' : 'none'
                }}
              >
                <Icon size={16} color={active ? '#fff' : (tab.color || '#94a3b8')} />
                <span>{tab.label}</span>
                {tab.badge && (
                  <span style={{
                    fontSize: '0.68rem',
                    padding: '2px 6px',
                    borderRadius: '5px',
                    background: active ? 'rgba(255,255,255,0.25)' : 'rgba(255,255,255,0.08)',
                    color: active ? '#fff' : '#cbd5e1',
                    fontWeight: '700'
                  }}>
                    {tab.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {quantificationResults && (
          <div style={{ fontSize: '0.78rem', color: '#38bdf8', fontWeight: '700', padding: '0 8px' }}>
            ● Analysis Results Active
          </div>
        )}
      </div>

      {/* Results Section By Sub-Tab */}
      {quantificationResults && (
        <>
          {/* Key Metric Summary Cards (Always visible on all and depth tabs) */}
          {(currentSubTab === 'all' || currentSubTab === 'depth' || currentSubTab === 'width') && (
            <QuantificationMetrics data={quantificationResults} />
          )}

          {/* VIEW: 3D DEPTH ENGINE (DEDICATED SUB-NAV TAB) */}
          {currentSubTab === 'depth' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              {/* 3D Depth Header Banner */}
              <div style={{
                background: 'linear-gradient(135deg, #042f2e 0%, #0f172a 100%)',
                border: '1px solid #14b8a6',
                borderRadius: '16px',
                padding: '1.25rem 1.5rem',
                color: '#fff',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                flexWrap: 'wrap',
                gap: '1rem',
                boxShadow: '0 8px 30px rgba(20, 184, 166, 0.15)'
              }}>
                <div>
                  <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', background: '#134e4a', padding: '4px 10px', borderRadius: '8px', color: '#2dd4bf', fontSize: '0.75rem', fontWeight: '800', marginBottom: '8px' }}>
                    <Box size={14} /> EXTRATREES ML CHAMPION REGRESSOR • SCALED CONFORMAL PREDICTION
                  </div>
                  <h3 style={{ margin: 0, fontSize: '1.35rem', fontWeight: '800' }}>
                    3D Crevice Recess & Depth Estimation Suite
                  </h3>
                  <p style={{ margin: '6px 0 0 0', color: '#99f6e4', fontSize: '0.85rem' }}>
                    Integrated from depth-estimation module: estimates relative fracture depth (ΔZ in mm), 90% & 95% conformal prediction intervals, and calculates true 3D crevice recess volume (cm³).
                  </p>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    onClick={() => depthInputRef.current?.click()}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '6px',
                      padding: '8px 14px',
                      background: '#0d9488',
                      border: 'none',
                      borderRadius: '8px',
                      color: '#fff',
                      fontSize: '0.82rem',
                      fontWeight: '700',
                      cursor: 'pointer'
                    }}
                  >
                    <Upload size={14} /> {depthFile ? 'Change Depth Map' : 'Upload External Depth Map'}
                  </button>
                </div>
              </div>

              {/* 3D Depth Recess ExtraTrees Prediction & Uncertainty Bounds */}
              {quantificationResults.depth_recess && (
                <DepthRecessCard
                  depthData={quantificationResults.depth_recess}
                  depthImage={quantificationResults.stage_images?.volume_marked || quantificationResults.stage_images?.stage4_depth_3d}
                  depthHeatmap={quantificationResults.stage_images?.stage4_depth_3d || quantificationResults.stage_images?.depth_map}
                  volumeImage={quantificationResults.stage_images?.volume_marked || quantificationResults.stage_images?.recess_volume_map}
                  originalImage={quantificationResults.stage_images?.original || imagePreview}
                />
              )}

              {/* 3D Recess Volume Detailed Views & Colormaps */}
              <WidthVolumeViewer
                data={quantificationResults}
                stageImages={quantificationResults.stage_images}
              />

              {/* Tri-Method Benchmark & 3D Arc-Length */}
              <TriMethodCard
                measurements={quantificationResults.measurements}
                planarity={quantificationResults.planarity}
                calibrationStatus={quantificationResults.calibration_status}
              />
            </div>
          )}

          {/* VIEW: ALL MODULES (EXECUTIVE OVERVIEW) */}
          {currentSubTab === 'all' && (
            <>
              <StagePipelineGallery
                stageImages={quantificationResults.stage_images}
                activeStageId="all_in_one_dashboard"
              />

              {quantificationResults.depth_recess && (
                <DepthRecessCard
                  depthData={quantificationResults.depth_recess}
                  depthImage={quantificationResults.stage_images?.volume_marked || quantificationResults.stage_images?.stage4_depth_3d}
                  depthHeatmap={quantificationResults.stage_images?.stage4_depth_3d || quantificationResults.stage_images?.depth_map}
                  volumeImage={quantificationResults.stage_images?.volume_marked || quantificationResults.stage_images?.recess_volume_map}
                  originalImage={quantificationResults.stage_images?.original || imagePreview}
                />
              )}

              <XaiAttributionCard
                xaiData={quantificationResults.xai}
                xaiImage={quantificationResults.stage_images?.xai_saliency_attribution}
              />

              <TriMethodCard
                measurements={quantificationResults.measurements}
                planarity={quantificationResults.planarity}
                calibrationStatus={quantificationResults.calibration_status}
              />

              <WidthVolumeViewer
                data={quantificationResults}
                stageImages={quantificationResults.stage_images}
              />
            </>
          )}

          {/* VIEW: WIDTH & ACI SEVERITY */}
          {currentSubTab === 'width' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              <WidthVolumeViewer
                data={quantificationResults}
                stageImages={quantificationResults.stage_images}
              />
              <TriMethodCard
                measurements={quantificationResults.measurements}
                planarity={quantificationResults.planarity}
                calibrationStatus={quantificationResults.calibration_status}
              />
            </div>
          )}

          {/* VIEW: RESTORATION COMPARISON */}
          {currentSubTab === 'restoration' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              <RestorationCard
                applyRestoration={applyRestoration}
                onToggleRestoration={() => setApplyRestoration(!applyRestoration)}
                restorationInfo={quantificationResults?.restoration}
                stageImages={quantificationResults?.stage_images}
              />
            </div>
          )}

          {/* VIEW: EXPLAINABLE AI (XAI) */}
          {currentSubTab === 'xai' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              <XaiAttributionCard
                xaiData={quantificationResults.xai}
                xaiImage={quantificationResults.stage_images?.xai_saliency_attribution}
              />
            </div>
          )}

          {/* VIEW: 16 DIAGNOSTIC LAYERS GALLERY */}
          {currentSubTab === 'gallery' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              <StagePipelineGallery
                stageImages={quantificationResults.stage_images}
                activeStageId="all_in_one_dashboard"
              />
            </div>
          )}
        </>
      )}

      {/* Pre-quantification Preview State when Depth Tab is clicked without results */}
      {!quantificationResults && currentSubTab === 'depth' && (
        <div style={{
          background: '#fff',
          border: '1px solid #e2e8f0',
          borderRadius: '16px',
          padding: '1.8rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.25rem'
        }}>
          <div style={{
            background: 'linear-gradient(135deg, #042f2e 0%, #0f172a 100%)',
            borderRadius: '12px',
            padding: '1.5rem',
            color: '#fff'
          }}>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', background: '#134e4a', padding: '4px 10px', borderRadius: '8px', color: '#2dd4bf', fontSize: '0.75rem', fontWeight: '800', marginBottom: '8px' }}>
              <Box size={14} /> 3D DEPTH ESTIMATION MODULE (depth-estimation-integration)
            </div>
            <h3 style={{ margin: 0, fontSize: '1.35rem', fontWeight: '800' }}>
              3D Crevice Recess & Conformal Depth Engine
            </h3>
            <p style={{ margin: '6px 0 0 0', color: '#99f6e4', fontSize: '0.85rem' }}>
              Uses ExtraTreesRegressor champion model + conformal prediction intervals to estimate crack crevice relative depth (ΔZ in mm) and volumetric recess capacity.
            </p>
          </div>

          <div style={{ textAlign: 'center', padding: '1rem' }}>
            {imagePreview ? (
              <div>
                <p style={{ fontWeight: '600', color: '#334155', marginBottom: '10px' }}>Loaded Image for 3D Depth Analysis:</p>
                <img
                  src={imagePreview}
                  alt="Preview"
                  style={{ maxHeight: '380px', maxWidth: '100%', borderRadius: '12px', border: '1px solid #cbd5e1' }}
                />
              </div>
            ) : (
              <p style={{ color: '#64748b' }}>Please upload an inspection image to run 3D depth and crevice recess analysis.</p>
            )}

            <div style={{ marginTop: '1.25rem' }}>
              <button
                onClick={handleRunQuantification}
                disabled={loading || (!imagePreview && !file)}
                style={{
                  padding: '10px 28px',
                  background: 'linear-gradient(135deg, #0d9488, #0284c7)',
                  border: 'none',
                  borderRadius: '10px',
                  color: '#fff',
                  fontSize: '0.92rem',
                  fontWeight: '700',
                  cursor: (loading || (!imagePreview && !file)) ? 'not-allowed' : 'pointer'
                }}
              >
                {loading ? 'Running 3D Depth Engine...' : 'Run 3D Crevice Depth Quantification'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Initial Preview State if not yet quantified and not in depth tab */}
      {!quantificationResults && currentSubTab !== 'depth' && imagePreview && (
        <div style={{
          background: '#fff',
          border: '1px solid #e2e8f0',
          borderRadius: '16px',
          padding: '1.5rem',
          textAlign: 'center'
        }}>
          <h4 style={{ margin: '0 0 0.8rem 0', color: '#0f172a' }}>Preview Image Ready</h4>
          <img
            src={imagePreview}
            alt="Preview"
            style={{ maxHeight: '420px', maxWidth: '100%', borderRadius: '12px', border: '1px solid #cbd5e1' }}
          />
          <div style={{ marginTop: '1.25rem' }}>
            <button
              onClick={handleRunQuantification}
              disabled={loading}
              style={{
                padding: '10px 28px',
                background: 'linear-gradient(135deg, #0284c7, #2563eb)',
                border: 'none',
                borderRadius: '10px',
                color: '#fff',
                fontSize: '0.92rem',
                fontWeight: '700',
                cursor: 'pointer'
              }}
            >
              Analyze & Quantify Physical Metrics Now
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default CrackQuantification;
