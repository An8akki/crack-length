import React, { useState } from 'react';
import { Activity, Box, ShieldAlert, CheckCircle2, ChevronRight, BarChart2, Eye, Download, Info } from 'lucide-react';

const WidthVolumeViewer = ({ data, stageImages }) => {
  const [activeTab, setActiveTab] = useState('width');

  if (!data) return null;

  const widthMm = data.width_mm ?? 0;
  const widthMaxMm = data.width_max_mm ?? 0;
  const widthMinMm = data.width_min_mm ?? 0;
  const widthCi = data.width_confidence_interval_95_mm;

  const volMm3 = data.volume_mm3 ?? 0;
  const volCm3 = data.volume_cm3 ?? (volMm3 / 1000.0);
  const meanDepthMm = data.mean_recess_depth_mm ?? 0;
  const maxDepthMm = data.max_recess_depth_mm ?? 0;
  const volUnc = data.volume_uncertainty || {};

  const aci = data.aci_classification || {};

  // Marked visualizer references
  const widthImg = stageImages?.width_marked || stageImages?.inspection_master_marked;
  const volumeImg = stageImages?.volume_marked || stageImages?.inspection_master_marked;
  const severityImg = stageImages?.severity_marked || stageImages?.inspection_master_marked;

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
      {/* Sub-Tabs */}
      <div style={{ display: 'flex', gap: '0.75rem', borderBottom: '1px solid #f1f5f9', paddingBottom: '0.75rem', flexWrap: 'wrap' }}>
        <button
          onClick={() => setActiveTab('width')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '8px 18px',
            borderRadius: '10px',
            border: activeTab === 'width' ? '2px solid #f59e0b' : '1px solid #e2e8f0',
            background: activeTab === 'width' ? 'rgba(245, 158, 11, 0.08)' : '#f8fafc',
            color: activeTab === 'width' ? '#b45309' : '#64748b',
            fontWeight: '700',
            fontSize: '0.88rem',
            cursor: 'pointer',
            transition: 'all 0.15s ease'
          }}
        >
          <Activity size={18} /> Physical Width & Calipers Marked
        </button>

        <button
          onClick={() => setActiveTab('volume')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '8px 18px',
            borderRadius: '10px',
            border: activeTab === 'volume' ? '2px solid #8b5cf6' : '1px solid #e2e8f0',
            background: activeTab === 'volume' ? 'rgba(139, 92, 246, 0.08)' : '#f8fafc',
            color: activeTab === 'volume' ? '#6d28d9' : '#64748b',
            fontWeight: '700',
            fontSize: '0.88rem',
            cursor: 'pointer',
            transition: 'all 0.15s ease'
          }}
        >
          <Box size={18} /> 3D Recess Volume & Depth Marked
        </button>

        <button
          onClick={() => setActiveTab('severity')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '8px 18px',
            borderRadius: '10px',
            border: activeTab === 'severity' ? '2px solid #ef4444' : '1px solid #e2e8f0',
            background: activeTab === 'severity' ? 'rgba(239, 68, 68, 0.08)' : '#f8fafc',
            color: activeTab === 'severity' ? '#b91c1c' : '#64748b',
            fontWeight: '700',
            fontSize: '0.88rem',
            cursor: 'pointer',
            transition: 'all 0.15s ease'
          }}
        >
          <ShieldAlert size={18} /> ACI 224R Severity Risk Zoning
        </button>
      </div>

      {/* Tab Content: Width */}
      {activeTab === 'width' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1.5rem', alignItems: 'start' }}>
          {/* Marked Width Image */}
          {widthImg ? (
            <div style={{
              background: '#090d16',
              borderRadius: '12px',
              overflow: 'hidden',
              border: '1px solid #1e293b',
              position: 'relative'
            }}>
              <img
                src={widthImg}
                alt="Crack Width Marked Diagnostic"
                style={{ width: '100%', maxHeight: '420px', objectFit: 'contain', display: 'block' }}
              />
            </div>
          ) : (
            <div style={{ padding: '2rem', textAlign: 'center', background: '#f8fafc', borderRadius: '12px' }}>
              Width visualization not available.
            </div>
          )}

          {/* Width Metrics & Guidelines */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div>
              <h4 style={{ margin: '0 0 0.4rem 0', fontSize: '1.05rem', color: '#0f172a' }}>
                Orthogonal Normal Ray Width Profiling
              </h4>
              <p style={{ margin: '0 0 0.8rem 0', fontSize: '0.82rem', color: '#64748b' }}>
                Width is measured along perpendicular normal rays cast orthogonal to the local skeleton tangent across opposing crack boundaries.
              </p>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 14px', background: '#f8fafc', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                <span style={{ color: '#475569', fontSize: '0.85rem' }}>Maximum Opening (W_max):</span>
                <strong style={{ color: '#ef4444', fontSize: '1.05rem' }}>{widthMaxMm.toFixed(2)} mm</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 14px', background: '#f8fafc', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                <span style={{ color: '#475569', fontSize: '0.85rem' }}>Average Opening (W_avg):</span>
                <strong style={{ color: '#0f172a', fontSize: '0.95rem' }}>{widthMm.toFixed(2)} mm</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 14px', background: '#f8fafc', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                <span style={{ color: '#475569', fontSize: '0.85rem' }}>Minimum Opening (W_min):</span>
                <strong style={{ color: '#16a34a', fontSize: '0.95rem' }}>{widthMinMm.toFixed(2)} mm</strong>
              </div>
              {widthCi && (
                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 14px', background: 'rgba(2,132,199,0.06)', borderRadius: '8px', border: '1px solid rgba(2,132,199,0.2)' }}>
                  <span style={{ color: '#0369a1', fontSize: '0.85rem', fontWeight: '600' }}>95% Confidence Interval:</span>
                  <strong style={{ color: '#0284c7', fontSize: '0.95rem' }}>[{widthCi[0].toFixed(2)} - {widthCi[1].toFixed(2)}] mm</strong>
                </div>
              )}
            </div>

            <div style={{ background: '#f8fafc', borderRadius: '10px', padding: '1rem', border: '1px solid #e2e8f0' }}>
              <h5 style={{ margin: '0 0 0.4rem 0', fontSize: '0.85rem', color: '#0f172a' }}>ACI 224R Structural Thresholds</h5>
              <div style={{ fontSize: '0.80rem', color: '#475569', lineHeight: '1.6' }}>
                <div>• <strong>Hairline (&lt; 0.10 mm):</strong> Cosmetic only, no structural repair required.</div>
                <div>• <strong>Fine (0.10 - 0.20 mm):</strong> Tolerable in dry interior service environments.</div>
                <div>• <strong>Moderate (0.20 - 0.30 mm):</strong> Risk of moisture penetration and corrosion.</div>
                <div>• <strong>Severe (&ge; 0.30 mm):</strong> Structural integrity compromised; immediate epoxy injection.</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tab Content: Volume */}
      {activeTab === 'volume' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1.5rem', alignItems: 'start' }}>
          {/* Marked Volume Image */}
          {volumeImg ? (
            <div style={{
              background: '#090d16',
              borderRadius: '12px',
              overflow: 'hidden',
              border: '1px solid #1e293b',
              position: 'relative'
            }}>
              <img
                src={volumeImg}
                alt="Crack Volume Marked Diagnostic"
                style={{ width: '100%', maxHeight: '420px', objectFit: 'contain', display: 'block' }}
              />
            </div>
          ) : (
            <div style={{ padding: '2rem', textAlign: 'center', background: '#f8fafc', borderRadius: '12px' }}>
              Volume visualization not available.
            </div>
          )}

          {/* Volume Metrics & Epoxy Repair Estimation */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div>
              <h4 style={{ margin: '0 0 0.4rem 0', fontSize: '1.05rem', color: '#0f172a' }}>
                Discrete 3D Crevice Recess Integration
              </h4>
              <p style={{ margin: '0 0 0.8rem 0', fontSize: '0.82rem', color: '#64748b' }}>
                Calculated by integrating depth recess profiles across crack surface pixels: V = ∑ [Z_recess(p)] × (scale)².
              </p>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 14px', background: 'rgba(139,92,246,0.06)', borderRadius: '8px', border: '1px solid rgba(139,92,246,0.2)' }}>
                <span style={{ color: '#5b21b6', fontSize: '0.85rem', fontWeight: '600' }}>3D Recess Volume:</span>
                <strong style={{ color: '#7c3aed', fontSize: '1.15rem' }}>{volCm3.toFixed(3)} cm³</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 14px', background: '#f8fafc', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                <span style={{ color: '#475569', fontSize: '0.85rem' }}>Recess Volume (mm³):</span>
                <strong style={{ color: '#0f172a', fontSize: '0.95rem' }}>{volMm3.toLocaleString()} mm³</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 14px', background: '#f8fafc', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                <span style={{ color: '#475569', fontSize: '0.85rem' }}>Mean Crevice Depth:</span>
                <strong style={{ color: '#0f172a', fontSize: '0.95rem' }}>{meanDepthMm.toFixed(2)} mm</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 14px', background: '#f8fafc', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                <span style={{ color: '#475569', fontSize: '0.85rem' }}>Maximum Crevice Depth:</span>
                <strong style={{ color: '#ef4444', fontSize: '0.95rem' }}>{maxDepthMm.toFixed(2)} mm</strong>
              </div>
            </div>

            <div style={{ background: '#f8fafc', borderRadius: '10px', padding: '1rem', border: '1px solid #e2e8f0' }}>
              <h5 style={{ margin: '0 0 0.4rem 0', fontSize: '0.85rem', color: '#0f172a' }}>Epoxy Grout Remediation Estimation</h5>
              <div style={{ padding: '8px 12px', background: 'rgba(139, 92, 246, 0.08)', borderRadius: '8px', border: '1px solid rgba(139, 92, 246, 0.2)' }}>
                <div style={{ fontSize: '0.78rem', color: '#4c1d95', fontWeight: '600' }}>Estimated Structural Epoxy Grout Required:</div>
                <div style={{ fontSize: '1.2rem', fontWeight: '800', color: '#6d28d9', margin: '3px 0' }}>
                  {(volCm3 * 1.15).toFixed(2)} mL <span style={{ fontSize: '0.78rem', fontWeight: '500', color: '#6b7280' }}>(incl. 15% safety factor)</span>
                </div>
                <div style={{ fontSize: '0.72rem', color: '#64748b' }}>
                  Complies with ASTM C881 Type I/IV structural pressure injection specifications.
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tab Content: Severity */}
      {activeTab === 'severity' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1.5rem', alignItems: 'start' }}>
          {/* Marked Severity Image */}
          {severityImg ? (
            <div style={{
              background: '#090d16',
              borderRadius: '12px',
              overflow: 'hidden',
              border: '1px solid #1e293b',
              position: 'relative'
            }}>
              <img
                src={severityImg}
                alt="Crack Severity Marked Diagnostic"
                style={{ width: '100%', maxHeight: '420px', objectFit: 'contain', display: 'block' }}
              />
            </div>
          ) : (
            <div style={{ padding: '2rem', textAlign: 'center', background: '#f8fafc', borderRadius: '12px' }}>
              Severity visualization not available.
            </div>
          )}

          {/* Severity Classification Details */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div style={{
              background: 'linear-gradient(135deg, rgba(239,68,68,0.08), rgba(245,158,11,0.05))',
              borderRadius: '12px',
              border: '1px solid #f87171',
              padding: '1.25rem'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                <span style={{
                  padding: '3px 10px',
                  borderRadius: '6px',
                  background: '#ef4444',
                  color: '#fff',
                  fontSize: '0.75rem',
                  fontWeight: '800'
                }}>
                  {aci.category || 'Severe'}
                </span>
                <h4 style={{ margin: 0, fontSize: '1rem', color: '#991b1b' }}>
                  {aci.code_standard || 'ACI 224R-01'} Structural Code Compliance
                </h4>
              </div>
              <p style={{ margin: '0 0 0.8rem 0', fontSize: '0.85rem', color: '#334155', lineHeight: '1.6' }}>
                {aci.action || 'Structural integrity compromised; immediate epoxy injection / reinforcement repair advised.'}
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.6rem', fontSize: '0.82rem', background: '#fff', padding: '10px', borderRadius: '8px', border: '1px solid #fee2e2' }}>
                <div><strong>Dominant Orientation:</strong> {data.orientation_deg ?? 110}° (Shear Angle)</div>
                <div><strong>Branching Junctions:</strong> {data.junctions ?? 0} points</div>
                <div><strong>Network Endpoints:</strong> {data.endpoints ?? 0} terminals</div>
                <div><strong>Connected Components:</strong> {data.components ?? 1} clusters</div>
              </div>
            </div>

            <div style={{ background: '#f8fafc', borderRadius: '10px', padding: '1rem', border: '1px solid #e2e8f0' }}>
              <h5 style={{ margin: '0 0 0.4rem 0', fontSize: '0.85rem', color: '#0f172a' }}>Remediation Workflow</h5>
              <ol style={{ margin: 0, paddingLeft: '1.2rem', fontSize: '0.80rem', color: '#475569', lineHeight: '1.6' }}>
                <li>Surface preparation: Clean loose concrete particles via wire brushing or low-pressure air blast.</li>
                <li>Mount injection ports at 1.5× depth intervals along the marked skeleton centerline.</li>
                <li>Seal surface crack face between injection ports with rapid-curing epoxy paste.</li>
                <li>Inject low-viscosity structural epoxy grout starting from lowest elevation port until bleed-through.</li>
              </ol>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default WidthVolumeViewer;
