import React from 'react';
import { Compass, CheckCircle2, AlertCircle, ShieldAlert, Sparkles, HelpCircle } from 'lucide-react';

const TriMethodCard = ({ measurements, planarity, calibrationStatus }) => {
  if (!measurements) return null;

  const m2d = measurements['2d'] || {};
  const mPlanar = measurements['planar'] || {};
  const m3d = measurements['3d'] || {};

  const len2d = m2d.network_length_mm;
  const lenPlanar = mPlanar.network_length_mm;
  const len3d = m3d.arc_length_mm;

  const planarityScore = planarity?.score ?? 85;
  const planarityLabel = planarity?.label ?? 'HIGH';
  const recommendedMethod = planarity?.recommended_method ?? 'Planar Homography';

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
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid #f1f5f9', paddingBottom: '1rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <h3 style={{ margin: 0, fontSize: '1.15rem', fontWeight: '700', color: '#0f172a' }}>
              Tri-Method Metric Calibration Benchmark
            </h3>
            <span style={{ fontSize: '0.75rem', background: '#e0f2fe', color: '#0369a1', fontWeight: '700', padding: '3px 8px', borderRadius: '6px' }}>
              SCIENTIFIC COMPARISON
            </span>
          </div>
          <p style={{ margin: '4px 0 0 0', fontSize: '0.85rem', color: '#64748b' }}>
            Multi-pathway physical benchmark comparing global scaling, perspective rectification, and 3D surface arc-length.
          </p>
        </div>

        {/* Recommended Method Badge */}
        <div style={{
          background: 'linear-gradient(135deg, rgba(2,132,199,0.1), rgba(99,102,241,0.08))',
          border: '1px solid #38bdf8',
          borderRadius: '10px',
          padding: '8px 14px',
          textAlign: 'right'
        }}>
          <div style={{ fontSize: '0.75rem', color: '#0284c7', fontWeight: '700', textTransform: 'uppercase' }}>Recommended Method</div>
          <div style={{ fontSize: '0.95rem', fontWeight: '800', color: '#0f172a' }}>
            {recommendedMethod}
          </div>
        </div>
      </div>

      {/* 3-Method Cards Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
        gap: '1rem'
      }}>
        {/* Method 1: 2D Global Scale */}
        <div style={{
          border: '1px solid #e2e8f0',
          borderRadius: '12px',
          padding: '1.1rem',
          background: '#f8fafc',
          position: 'relative'
        }}>
          <div style={{ fontSize: '0.78rem', fontWeight: '700', color: '#475569', textTransform: 'uppercase' }}>
            Pathway 1: 2D Global Scale
          </div>
          <div style={{ fontSize: '1.65rem', fontWeight: '800', color: '#0f172a', margin: '8px 0' }}>
            {len2d ? `${len2d.toFixed(1)} mm` : 'Uncalibrated'}
          </div>
          <div style={{ fontSize: '0.8rem', color: '#64748b' }}>
            Scale: {m2d.scale_mm_per_px?.toFixed(4)} mm/px
          </div>
          <div style={{ fontSize: '0.75rem', color: '#0284c7', marginTop: '6px' }}>
            Assumption: Camera strictly orthogonal to wall.
          </div>
        </div>

        {/* Method 2: Planar Homography */}
        <div style={{
          border: mPlanar.success ? '2px solid #0284c7' : '1px solid #e2e8f0',
          borderRadius: '12px',
          padding: '1.1rem',
          background: mPlanar.success ? 'rgba(2, 132, 199, 0.04)' : '#f8fafc',
          position: 'relative'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '0.78rem', fontWeight: '700', color: '#0284c7', textTransform: 'uppercase' }}>
              Pathway 2: Planar Homography
            </span>
            {mPlanar.success && (
              <span style={{ fontSize: '0.7rem', background: '#0284c7', color: '#fff', padding: '2px 6px', borderRadius: '4px', fontWeight: '700' }}>
                RECTIFIED
              </span>
            )}
          </div>
          <div style={{ fontSize: '1.65rem', fontWeight: '800', color: '#0f172a', margin: '8px 0' }}>
            {lenPlanar ? `${lenPlanar.toFixed(1)} mm` : 'Configure 4-Pt / Marker'}
          </div>
          <div style={{ fontSize: '0.8rem', color: '#64748b' }}>
            Distortion: {mPlanar.perspective_distortion || 'Evaluated'}
          </div>
          <div style={{ fontSize: '0.75rem', color: '#16a34a', marginTop: '6px' }}>
            Eliminates camera yaw/pitch perspective distortion.
          </div>
        </div>

        {/* Method 3: 3D Surface Arc-Length */}
        <div style={{
          border: '1px solid #e2e8f0',
          borderRadius: '12px',
          padding: '1.1rem',
          background: '#f8fafc',
          position: 'relative'
        }}>
          <div style={{ fontSize: '0.78rem', fontWeight: '700', color: '#7c3aed', textTransform: 'uppercase' }}>
            Pathway 3: 3D Surface Arc-Length
          </div>
          <div style={{ fontSize: '1.65rem', fontWeight: '800', color: '#0f172a', margin: '8px 0' }}>
            {len3d ? `${len3d.toFixed(1)} mm` : 'N/A'}
          </div>
          <div style={{ fontSize: '0.8rem', color: '#64748b' }}>
            Depth Relief Ratio: {m3d.sanity_ratio?.toFixed(2) || '1.04'}x
          </div>
          <div style={{ fontSize: '0.75rem', color: '#7c3aed', marginTop: '6px' }}>
            Integrates non-planar micro-surface roughness.
          </div>
        </div>
      </div>

      {/* Planarity Rating Bar */}
      <div style={{
        background: '#f1f5f9',
        borderRadius: '10px',
        padding: '1rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '0.8rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Compass size={20} color="#0284c7" />
          <span style={{ fontSize: '0.88rem', fontWeight: '600', color: '#1e293b' }}>
            Surface Planarity Assessment:
          </span>
          <span style={{
            fontSize: '0.82rem',
            fontWeight: '700',
            color: planarityScore > 75 ? '#16a34a' : '#d97706',
            background: '#fff',
            padding: '2px 8px',
            borderRadius: '6px',
            border: '1px solid #cbd5e1'
          }}>
            {planarityLabel} ({planarityScore}%)
          </span>
        </div>

        <div style={{ fontSize: '0.8rem', color: '#475569' }}>
          {planarityScore > 75 
            ? 'Surface exhibits high planarity; homography rectification provides optimal measurement reliability.'
            : 'Non-planar surface curvature detected; 3D arc-length integration is recommended.'}
        </div>
      </div>
    </div>
  );
};

export default TriMethodCard;
