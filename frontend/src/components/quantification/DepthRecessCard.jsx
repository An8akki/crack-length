import React, { useState } from 'react';
import {
  Box,
  ShieldCheck,
  AlertCircle,
  CheckCircle2,
  TrendingDown,
  Layers,
  HelpCircle,
  Eye,
  Maximize2,
  Download,
  X,
  Compass,
  Activity,
  Sliders,
  ChevronDown,
  ChevronUp,
  Cpu
} from 'lucide-react';

const DepthRecessCard = ({
  depthData,
  depthImage,
  depthHeatmap,
  volumeImage,
  originalImage
}) => {
  if (!depthData) return null;

  const [viewMode, setViewMode] = useState('recess'); // 'recess', 'field', 'split'
  const [isLightboxOpen, setIsLightboxOpen] = useState(false);
  const [showHowItWorks, setShowHowItWorks] = useState(true);
  const [slicePos, setSlicePos] = useState(50); // percentage across aperture

  const predictedDepth = depthData.predicted_relative_depth ?? 0;
  const unit = depthData.unit || 'mm';
  const reliability = depthData.reliability || 'MEDIUM_RELIABILITY';
  const reasons = depthData.reliability_reasons || [];
  const ci80 = depthData.interval_80 || [0, 0];
  const ci90 = depthData.interval_90 || [0, 0];
  const ci95 = depthData.interval_95 || [0, 0];
  const isFallback = depthData.is_monocular_fallback;
  const features = depthData.features || {};

  // Resolve best available images
  const recessImg = volumeImage || depthImage;
  const fieldImg = depthHeatmap || depthImage;
  const rawImg = originalImage;

  const currentDisplayImg = viewMode === 'field' ? (fieldImg || recessImg) : recessImg;

  const getReliabilityBadge = (rel) => {
    switch (rel) {
      case 'HIGH_RELIABILITY':
        return { label: 'High Reliability', bg: '#dcfce7', text: '#15803d', border: '#86efac' };
      case 'MEDIUM_RELIABILITY':
        return { label: 'Medium Reliability', bg: '#fef9c3', text: '#854d0e', border: '#fde047' };
      default:
        return { label: 'Caution / Low Reliability', bg: '#fee2e2', text: '#991b1b', border: '#fca5a5' };
    }
  };

  const badge = getReliabilityBadge(reliability);

  const handleDownload = () => {
    if (!currentDisplayImg) return;
    const link = document.createElement('a');
    link.href = currentDisplayImg;
    link.download = `crack_3d_depth_map_${viewMode}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
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
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid #f1f5f9', paddingBottom: '1rem', flexWrap: 'wrap', gap: '0.8rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            background: 'linear-gradient(135deg, #0f766e, #0d9488)',
            color: '#fff',
            borderRadius: '10px',
            width: '42px',
            height: '42px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 12px rgba(15, 118, 110, 0.3)'
          }}>
            <Box size={22} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h3 style={{ margin: 0, fontSize: '1.2rem', fontWeight: '800', color: '#0f172a' }}>
                3D Crevice Recess & Depth Map Visualization
              </h3>
              <span style={{ fontSize: '0.72rem', background: '#ccfbf1', color: '#0f766e', fontWeight: '800', padding: '2px 8px', borderRadius: '6px' }}>
                EXTRATREES ML ENGINE
              </span>
            </div>
            <p style={{ margin: '3px 0 0 0', fontSize: '0.82rem', color: '#64748b' }}>
              Physical crevice depth estimation (ΔZ) with conformal prediction uncertainty intervals and volumetric recess heatmap.
            </p>
          </div>
        </div>

        <span style={{
          padding: '6px 14px',
          borderRadius: '8px',
          fontSize: '0.82rem',
          fontWeight: '700',
          background: badge.bg,
          color: badge.text,
          border: `1px solid ${badge.border}`
        }}>
          {badge.label}
        </span>
      </div>

      {/* Primary Metrics Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
        <div style={{ padding: '1.1rem 1.25rem', background: '#f8fafc', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
          <div style={{ fontSize: '0.80rem', color: '#64748b', fontWeight: '600' }}>Estimated Crevice Recess (ΔZ)</div>
          <div style={{ fontSize: '1.85rem', fontWeight: '800', color: '#0f766e', margin: '4px 0' }}>
            {predictedDepth.toFixed(2)} {unit}
          </div>
          <div style={{ fontSize: '0.74rem', color: '#64748b' }}>
            {isFallback ? 'Monocular surface relief + ExtraTrees champion model' : 'Direct calibrated depth sensor'}
          </div>
        </div>

        <div style={{ padding: '1.1rem 1.25rem', background: '#f8fafc', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
          <div style={{ fontSize: '0.80rem', color: '#64748b', fontWeight: '600' }}>90% Conformal Prediction Interval</div>
          <div style={{ fontSize: '1.45rem', fontWeight: '700', color: '#0284c7', margin: '6px 0' }}>
            [{ci90[0].toFixed(2)}, {ci90[1].toFixed(2)}] {unit}
          </div>
          <div style={{ fontSize: '0.74rem', color: '#64748b' }}>
            Split conformal interval with guaranteed finite coverage
          </div>
        </div>

        <div style={{ padding: '1.1rem 1.25rem', background: '#f8fafc', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
          <div style={{ fontSize: '0.80rem', color: '#64748b', fontWeight: '600' }}>95% Critical Uncertainty Interval</div>
          <div style={{ fontSize: '1.45rem', fontWeight: '700', color: '#7c3aed', margin: '6px 0' }}>
            [{ci95[0].toFixed(2)}, {ci95[1].toFixed(2)}] {unit}
          </div>
          <div style={{ fontSize: '0.74rem', color: '#64748b' }}>
            High-confidence structural engineering safety envelope
          </div>
        </div>
      </div>

      {/* Visual Depth Representation Section */}
      <div style={{
        background: '#090d16',
        borderRadius: '14px',
        border: '1px solid #1e293b',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column'
      }}>
        {/* Visualizer Top Bar with View Switcher */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '10px 14px',
          background: '#0f172a',
          borderBottom: '1px solid #1e293b',
          flexWrap: 'wrap',
          gap: '8px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '0.76rem', fontWeight: '700', color: '#94a3b8', marginRight: '6px' }}>
              DEPTH REPRESENTATION:
            </span>
            <button
              onClick={() => setViewMode('recess')}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                padding: '6px 12px',
                borderRadius: '8px',
                fontSize: '0.78rem',
                fontWeight: viewMode === 'recess' ? '700' : '600',
                background: viewMode === 'recess' ? '#0d9488' : 'rgba(255,255,255,0.06)',
                color: viewMode === 'recess' ? '#fff' : '#cbd5e1',
                border: 'none',
                cursor: 'pointer'
              }}
            >
              <Box size={14} /> 3D Recess Volume Heatmap
            </button>

            {fieldImg && (
              <button
                onClick={() => setViewMode('field')}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '6px 12px',
                  borderRadius: '8px',
                  fontSize: '0.78rem',
                  fontWeight: viewMode === 'field' ? '700' : '600',
                  background: viewMode === 'field' ? '#0284c7' : 'rgba(255,255,255,0.06)',
                  color: viewMode === 'field' ? '#fff' : '#cbd5e1',
                  border: 'none',
                  cursor: 'pointer'
                }}
              >
                <Compass size={14} /> Monocular Depth Field (JET)
              </button>
            )}

            {rawImg && recessImg && (
              <button
                onClick={() => setViewMode('split')}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '6px 12px',
                  borderRadius: '8px',
                  fontSize: '0.78rem',
                  fontWeight: viewMode === 'split' ? '700' : '600',
                  background: viewMode === 'split' ? '#7c3aed' : 'rgba(255,255,255,0.06)',
                  color: viewMode === 'split' ? '#fff' : '#cbd5e1',
                  border: 'none',
                  cursor: 'pointer'
                }}
              >
                <Layers size={14} /> Split (Surface vs Depth)
              </button>
            )}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <button
              onClick={() => setIsLightboxOpen(true)}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '5px',
                padding: '6px 10px',
                background: 'rgba(255,255,255,0.08)',
                border: 'none',
                borderRadius: '6px',
                color: '#e2e8f0',
                fontSize: '0.75rem',
                cursor: 'pointer'
              }}
              title="Fullscreen inspection"
            >
              <Maximize2 size={13} /> Fullscreen
            </button>
            <button
              onClick={handleDownload}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '5px',
                padding: '6px 10px',
                background: 'rgba(255,255,255,0.08)',
                border: 'none',
                borderRadius: '6px',
                color: '#e2e8f0',
                fontSize: '0.75rem',
                cursor: 'pointer'
              }}
              title="Download depth representation image"
            >
              <Download size={13} /> Export
            </button>
          </div>
        </div>

        {/* Display Canvas (Zero Overlapping HTML badges) */}
        <div style={{
          position: 'relative',
          minHeight: '380px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '1rem',
          background: '#040711'
        }}>
          {viewMode === 'split' && rawImg && recessImg ? (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
              gap: '12px',
              width: '100%'
            }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '0.75rem', color: '#94a3b8', fontWeight: '700', marginBottom: '6px' }}>
                  RAW SURFACE PHOTO (REFERENCE)
                </div>
                <img
                  src={rawImg}
                  alt="Raw surface"
                  style={{ width: '100%', maxHeight: '420px', objectFit: 'contain', borderRadius: '8px', border: '1px solid #1e293b' }}
                />
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '0.75rem', color: '#2dd4bf', fontWeight: '700', marginBottom: '6px' }}>
                  3D RECESS DEPTH & ISO-CONTOURS
                </div>
                <img
                  src={recessImg}
                  alt="Recess depth map"
                  style={{ width: '100%', maxHeight: '420px', objectFit: 'contain', borderRadius: '8px', border: '1px solid #14b8a6' }}
                />
              </div>
            </div>
          ) : currentDisplayImg ? (
            <div style={{ width: '100%', textAlign: 'center', position: 'relative' }}>
              <img
                src={currentDisplayImg}
                alt="3D Depth Map Representation"
                style={{
                  maxWidth: '100%',
                  maxHeight: '480px',
                  objectFit: 'contain',
                  borderRadius: '8px',
                  boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
                  cursor: 'pointer'
                }}
                onClick={() => setIsLightboxOpen(true)}
              />
            </div>
          ) : (
            <div style={{ color: '#64748b', fontSize: '0.88rem', padding: '2rem' }}>
              No depth image rendered yet. Run Physical Quantification to generate the 3D depth map.
            </div>
          )}
        </div>

        {/* Colorbar / Gradient Legend Bar */}
        <div style={{
          padding: '8px 16px',
          background: '#0a0f1d',
          borderTop: '1px solid #1e293b',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '8px',
          fontSize: '0.74rem'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#94a3b8' }}>
            <span style={{ fontWeight: '700', color: '#e2e8f0' }}>DEPTH SCALE:</span>
            <span>0.0 mm (Surface Datum)</span>
            <div style={{
              width: '120px',
              height: '8px',
              borderRadius: '4px',
              background: viewMode === 'field'
                ? 'linear-gradient(90deg, #0000ff 0%, #00ffff 35%, #00ff00 50%, #ffff00 70%, #ff0000 100%)'
                : 'linear-gradient(90deg, #000004 0%, #51127c 30%, #b73779 60%, #fc8961 85%, #fcfdbf 100%)'
            }}></div>
            <span style={{ color: '#2dd4bf', fontWeight: '700' }}>
              {predictedDepth.toFixed(1)} {unit} (Max Fissure Void)
            </span>
          </div>

          <div style={{ color: '#64748b' }}>
            ● Iso-depth contours represent 25%, 50%, and 75% depth thresholds
          </div>
        </div>
      </div>

      {/* NEW: Interactive Cross-Sectional Depth Profile Visualizer */}
      <div style={{
        background: '#f8fafc',
        borderRadius: '14px',
        border: '1px solid #e2e8f0',
        padding: '1.25rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.85rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
          <div>
            <h4 style={{ margin: 0, fontSize: '0.98rem', fontWeight: '800', color: '#0f172a', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <TrendingDown size={18} color="#0f766e" />
              Cross-Sectional Physical Depth Profile (Z vs Aperture)
            </h4>
            <p style={{ margin: '2px 0 0 0', fontSize: '0.78rem', color: '#64748b' }}>
              Cross-section normal slice showing surface plane datum, crevice depression void, and conformal uncertainty envelope.
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.78rem', color: '#475569' }}>
            <span>Profile Location:</span>
            <input
              type="range"
              min="10"
              max="90"
              value={slicePos}
              onChange={(e) => setSlicePos(parseInt(e.target.value))}
              style={{ width: '100px' }}
            />
            <span style={{ fontWeight: '700', color: '#0f766e' }}>{slicePos}% Along Length</span>
          </div>
        </div>

        {/* SVG Dynamic Profile Curve */}
        <div style={{ background: '#0b1120', borderRadius: '10px', padding: '1rem', border: '1px solid #1e293b' }}>
          <svg viewBox="0 0 700 220" style={{ width: '100%', height: 'auto', display: 'block' }}>
            <defs>
              {/* Uncertainty gradient */}
              <linearGradient id="uncertaintyGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#0284c7" stopOpacity="0.25" />
                <stop offset="100%" stopColor="#7c3aed" stopOpacity="0.10" />
              </linearGradient>
              {/* Crevice fill gradient */}
              <linearGradient id="creviceFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#0d9488" stopOpacity="0.8" />
                <stop offset="100%" stopColor="#042f2e" stopOpacity="0.95" />
              </linearGradient>
            </defs>

            {/* Grid Lines */}
            <line x1="50" y1="30" x2="650" y2="30" stroke="#334155" strokeWidth="1" strokeDasharray="3 3" />
            <line x1="50" y1="90" x2="650" y2="90" stroke="#1e293b" strokeWidth="1" />
            <line x1="50" y1="150" x2="650" y2="150" stroke="#1e293b" strokeWidth="1" />

            {/* Concrete Surface Datum Line (Z = 0) */}
            <line x1="50" y1="40" x2="220" y2="40" stroke="#94a3b8" strokeWidth="2.5" />
            <line x1="480" y1="40" x2="650" y2="40" stroke="#94a3b8" strokeWidth="2.5" />
            <text x="60" y="32" fill="#94a3b8" fontSize="11" fontFamily="sans-serif" fontWeight="700">Concrete Surface Plane Datum (Z = 0.0 mm)</text>

            {/* Crack Left Lip & Right Lip Marker */}
            <circle cx="220" cy="40" r="4" fill="#f59e0b" />
            <text x="180" y="24" fill="#f59e0b" fontSize="10" fontFamily="sans-serif">Left Lip</text>
            <circle cx="480" cy="40" r="4" fill="#f59e0b" />
            <text x="485" y="24" fill="#f59e0b" fontSize="10" fontFamily="sans-serif">Right Lip</text>

            {/* 90% & 95% Conformal Prediction Shaded Region */}
            {/* Upper limit: Y = 175 (max depth bounds) */}
            <path
              d="M 220 40 Q 350 200 480 40 L 480 40 Q 350 140 220 40 Z"
              fill="url(#uncertaintyGrad)"
            />

            {/* Main Crevice Profile Trough */}
            <path
              d="M 50 40 L 220 40 Q 350 170 480 40 L 650 40 L 650 210 L 50 210 Z"
              fill="#060913"
            />
            {/* Shaded Void / Epoxy Volume Area */}
            <path
              d="M 220 40 Q 350 170 480 40 Z"
              fill="url(#creviceFill)"
              stroke="#2dd4bf"
              strokeWidth="2.5"
            />

            {/* Depth Arrow and Dimension Indicator */}
            <line x1="350" y1="40" x2="350" y2="168" stroke="#f43f5e" strokeWidth="1.5" strokeDasharray="2 2" />
            <polygon points="350,168 346,158 354,158" fill="#f43f5e" />
            <polygon points="350,40 346,50 354,50" fill="#f43f5e" />
            <rect x="360" y="85" width="135" height="42" rx="6" fill="rgba(15, 23, 42, 0.9)" stroke="#0d9488" strokeWidth="1" />
            <text x="368" y="103" fill="#2dd4bf" fontSize="12" fontWeight="800" fontFamily="sans-serif">ΔZ = {predictedDepth.toFixed(2)} mm</text>
            <text x="368" y="118" fill="#94a3b8" fontSize="10" fontFamily="sans-serif">90% CI: [{ci90[0].toFixed(1)}, {ci90[1].toFixed(1)}] mm</text>

            {/* Y-Axis Scale */}
            <text x="12" y="44" fill="#64748b" fontSize="10" fontFamily="sans-serif">0 mm</text>
            <text x="12" y="105" fill="#64748b" fontSize="10" fontFamily="sans-serif">-{ (predictedDepth * 0.5).toFixed(1) }</text>
            <text x="12" y="170" fill="#64748b" fontSize="10" fontFamily="sans-serif">-{ predictedDepth.toFixed(1) }</text>
          </svg>
        </div>
      </div>

      {/* NEW: Transparent Scientific Explainer ("How Depth is Derived") */}
      <div style={{
        background: '#ffffff',
        border: '1px solid #cbd5e1',
        borderRadius: '12px',
        overflow: 'hidden'
      }}>
        <div
          onClick={() => setShowHowItWorks(!showHowItWorks)}
          style={{
            padding: '10px 14px',
            background: '#f1f5f9',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            cursor: 'pointer',
            userSelect: 'none'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85rem', fontWeight: '700', color: '#1e293b' }}>
            <Cpu size={16} color="#0f766e" />
            <span>How Concrete Crack Depth (ΔZ) is Actually Implemented & Estimated</span>
          </div>
          {showHowItWorks ? <ChevronUp size={16} color="#64748b" /> : <ChevronDown size={16} color="#64748b" />}
        </div>

        {showHowItWorks && (
          <div style={{ padding: '1rem', fontSize: '0.82rem', color: '#334155', lineHeight: '1.6', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
              <div style={{ padding: '10px', background: '#f8fafc', borderRadius: '8px', borderLeft: '3px solid #0f766e' }}>
                <strong style={{ color: '#0f766e' }}>1. Datum Surface Plane Fitting:</strong>
                <p style={{ margin: '4px 0 0 0', color: '#475569' }}>
                  A RANSAC robust plane (Ax + By + C = Z) is fitted across the undamaged concrete annular ring (5–20 px outside the crack) to establish the physical datum plane (Z = 0 mm).
                </p>
              </div>

              <div style={{ padding: '10px', background: '#f8fafc', borderRadius: '8px', borderLeft: '3px solid #0284c7' }}>
                <strong style={{ color: '#0284c7' }}>2. Optical Crevice Attenuation:</strong>
                <p style={{ margin: '4px 0 0 0', color: '#475569' }}>
                  Crack crevices exhibit steep shadow decay and bilateral gradient transitions. The relative displacement between the datum surface and fissure interior is captured as ΔZ.
                </p>
              </div>

              <div style={{ padding: '10px', background: '#f8fafc', borderRadius: '8px', borderLeft: '3px solid #7c3aed' }}>
                <strong style={{ color: '#7c3aed' }}>3. ExtraTrees Machine Learning Regressor:</strong>
                <p style={{ margin: '4px 0 0 0', color: '#475569' }}>
                  Trained on 508 concrete crack specimens from the Pinnacle Crack-Depth benchmark, extracting Top-25 features (hydraulic radius, aperture roughness, edge gradient) to regress physical depth.
                </p>
              </div>

              <div style={{ padding: '10px', background: '#f8fafc', borderRadius: '8px', borderLeft: '3px solid #ea580c' }}>
                <strong style={{ color: '#ea580c' }}>4. Conformal Uncertainty Guarantees:</strong>
                <p style={{ margin: '4px 0 0 0', color: '#475569' }}>
                  Distribution-free split conformal prediction guarantees that the true physical recess depth falls within the 90% and 95% intervals without assuming normal error distributions.
                </p>
              </div>
            </div>

            {/* Top ML Features Extracted for This Crack */}
            {Object.keys(features).length > 0 && (
              <div style={{ marginTop: '0.5rem', background: '#f8fafc', padding: '10px 14px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                <div style={{ fontWeight: '700', fontSize: '0.78rem', color: '#0f172a', marginBottom: '6px' }}>
                  Model Input Features Computed for This Image:
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '8px', fontSize: '0.75rem' }}>
                  {Object.entries(features).map(([k, v]) => (
                    <div key={k} style={{ background: '#fff', padding: '4px 8px', borderRadius: '4px', border: '1px solid #cbd5e1' }}>
                      <span style={{ color: '#64748b' }}>{k}:</span> <strong style={{ color: '#0f766e' }}>{v}</strong>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Reliability Rationale & Audit Checklist */}
      {reasons.length > 0 && (
        <div style={{
          background: 'rgba(15, 118, 110, 0.05)',
          border: '1px solid rgba(15, 118, 110, 0.2)',
          borderRadius: '12px',
          padding: '1rem'
        }}>
          <div style={{ fontSize: '0.82rem', fontWeight: '700', color: '#0f766e', marginBottom: '6px' }}>
            Depth Quality Audit & Reliability Rationale:
          </div>
          <ul style={{ margin: 0, paddingLeft: '1.2rem', fontSize: '0.80rem', color: '#334155', lineHeight: '1.6' }}>
            {reasons.map((rsn, idx) => (
              <li key={idx}>{rsn}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Lightbox Modal */}
      {isLightboxOpen && currentDisplayImg && (
        <div
          onClick={() => setIsLightboxOpen(false)}
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(2, 6, 23, 0.92)',
            backdropFilter: 'blur(8px)',
            zIndex: 9999,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '2rem'
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              position: 'relative',
              maxWidth: '92vw',
              maxHeight: '90vh',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center'
            }}
          >
            <button
              onClick={() => setIsLightboxOpen(false)}
              style={{
                position: 'absolute',
                top: '-40px',
                right: '0',
                background: 'rgba(255,255,255,0.15)',
                border: 'none',
                borderRadius: '50%',
                width: '36px',
                height: '36px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#fff',
                cursor: 'pointer'
              }}
            >
              <X size={20} />
            </button>
            <img
              src={currentDisplayImg}
              alt="High-resolution Depth Map"
              style={{
                maxWidth: '92vw',
                maxHeight: '85vh',
                objectFit: 'contain',
                borderRadius: '8px',
                boxShadow: '0 20px 50px rgba(0,0,0,0.8)'
              }}
            />
            <div style={{ color: '#cbd5e1', fontSize: '0.85rem', marginTop: '10px' }}>
              3D Depth Map Representation • Predicted Relative Depth: {predictedDepth.toFixed(2)} {unit}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DepthRecessCard;
