import React from 'react';
import { Ruler, Activity, Layers, AlertTriangle, ShieldCheck, Box, Compass, Sparkles } from 'lucide-react';

const QuantificationMetrics = ({ data }) => {
  if (!data) return null;

  const lengthMm = data.length_mm ?? (data.measurements?.['2d']?.network_length_mm ?? 0);
  const lengthPx = data.length_px ?? (data.measurements?.['2d']?.network_length_px ?? 0);
  const mainPathMm = data.main_path_mm ?? (data.measurements?.['2d']?.main_path_mm ?? 0);
  const mstLengthMm = data.mst_length_mm ?? (data.measurements?.['2d']?.mst_length_mm ?? 0);

  const meanWidthMm = data.width_mm ?? (data.measurements?.['2d']?.mean_width_mm ?? 0);
  const maxWidthMm = data.width_max_mm ?? 0;
  const minWidthMm = data.width_min_mm ?? 0;
  const widthCi = data.width_confidence_interval_95_mm;

  const volumeMm3 = data.volume_mm3 ?? 0;
  const volumeCm3 = data.volume_cm3 ?? (volumeMm3 / 1000.0);
  const meanDepthMm = data.mean_recess_depth_mm ?? 0;
  const maxDepthMm = data.max_recess_depth_mm ?? 0;

  const areaMm2 = data.area_mm2 ?? (data.measurements?.['2d']?.area_mm2 ?? 0);
  const areaUncMm2 = data.area_uncertainty_mm2 ?? 0;

  const aciObj = data.aci_classification || {};
  const aciCategory = aciObj.category || 'Severe';
  const aciAction = aciObj.action || 'Structural intervention required.';
  const aciStandard = aciObj.code_standard || 'ACI 224R-01';

  const scaleVal = data.scale?.mm_per_px ?? (data.measurements?.['2d']?.scale_mm_per_px ?? 0.277);
  const scaleSource = data.scale?.source ?? 'user_or_auto';
  const planarityLabel = data.planarity?.label || 'MEDIUM';

  // Severity color styles
  const getSeverityStyle = (cat) => {
    switch (cat?.toLowerCase()) {
      case 'hairline':
        return { bg: 'rgba(34, 197, 94, 0.12)', border: '#22c55e', text: '#15803d', label: 'HAIRLINE (SAFE)' };
      case 'fine':
        return { bg: 'rgba(59, 130, 246, 0.12)', border: '#3b82f6', text: '#1d4ed8', label: 'FINE (MONITOR)' };
      case 'moderate':
        return { bg: 'rgba(245, 158, 11, 0.12)', border: '#f59e0b', text: '#b45309', label: 'MODERATE (REPAIR)' };
      case 'severe':
      default:
        return { bg: 'rgba(239, 68, 68, 0.12)', border: '#ef4444', text: '#b91c1c', label: 'SEVERE (CRITICAL)' };
    }
  };

  const sevStyle = getSeverityStyle(aciCategory);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', width: '100%' }}>
      {/* Structural Severity Banner */}
      <div style={{
        background: sevStyle.bg,
        border: `2px solid ${sevStyle.border}`,
        borderRadius: '12px',
        padding: '1.25rem 1.5rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '1rem',
        boxShadow: '0 4px 16px rgba(0,0,0,0.04)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{
            background: sevStyle.border,
            color: '#fff',
            borderRadius: '10px',
            width: '44px',
            height: '44px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <AlertTriangle size={26} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <span style={{ fontWeight: '800', fontSize: '1.15rem', color: sevStyle.text, letterSpacing: '0.5px' }}>
                {sevStyle.label}
              </span>
              <span style={{ fontSize: '0.8rem', background: '#fff', padding: '2px 8px', borderRadius: '6px', border: `1px solid ${sevStyle.border}`, color: '#475569', fontWeight: '600' }}>
                {aciStandard}
              </span>
            </div>
            <p style={{ margin: '4px 0 0 0', fontSize: '0.88rem', color: '#334155', fontWeight: '500' }}>
              <strong>Remediation:</strong> {aciAction}
            </p>
          </div>
        </div>

        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: '0.78rem', color: '#64748b', fontWeight: '600', textTransform: 'uppercase' }}>Physical Calibration</div>
          <div style={{ fontSize: '1rem', fontWeight: '700', color: '#0f172a' }}>
            {scaleVal.toFixed(4)} <span style={{ fontSize: '0.8rem', color: '#64748b' }}>mm/px</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: '#0284c7', fontWeight: '600' }}>
            {scaleSource.replace('_', ' ').toUpperCase()} • Planarity: {planarityLabel}
          </div>
        </div>
      </div>

      {/* Metric Cards Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
        gap: '1.25rem'
      }}>
        {/* Length Card */}
        <div className="metric-card" style={{
          background: 'linear-gradient(135deg, rgba(255,255,255,0.95), rgba(248,250,252,0.9))',
          border: '1px solid rgba(59, 130, 246, 0.2)',
          borderRadius: '12px',
          padding: '1.25rem',
          boxShadow: '0 4px 20px rgba(0,0,0,0.04)',
          position: 'relative'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.82rem', fontWeight: '700', color: '#64748b', textTransform: 'uppercase' }}>Network Length</span>
            <div style={{ background: 'rgba(59, 130, 246, 0.1)', color: '#2563eb', borderRadius: '8px', padding: '6px' }}>
              <Ruler size={18} />
            </div>
          </div>
          <div style={{ fontSize: '1.85rem', fontWeight: '800', color: '#0f172a' }}>
            {lengthMm.toFixed(1)} <span style={{ fontSize: '0.95rem', fontWeight: '600', color: '#64748b' }}>mm</span>
          </div>
          <div style={{ fontSize: '0.82rem', color: '#64748b', marginTop: '0.4rem', borderTop: '1px solid #f1f5f9', paddingTop: '0.5rem' }}>
            <div>Main Path: <strong>{mainPathMm.toFixed(1)} mm</strong></div>
            <div>MST Branch: <strong>{mstLengthMm.toFixed(1)} mm</strong> ({lengthPx.toFixed(0)} px)</div>
          </div>
        </div>

        {/* Width Card */}
        <div className="metric-card" style={{
          background: 'linear-gradient(135deg, rgba(255,255,255,0.95), rgba(248,250,252,0.9))',
          border: '1px solid rgba(245, 158, 11, 0.25)',
          borderRadius: '12px',
          padding: '1.25rem',
          boxShadow: '0 4px 20px rgba(0,0,0,0.04)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.82rem', fontWeight: '700', color: '#64748b', textTransform: 'uppercase' }}>Opening Width</span>
            <div style={{ background: 'rgba(245, 158, 11, 0.1)', color: '#d97706', borderRadius: '8px', padding: '6px' }}>
              <Activity size={18} />
            </div>
          </div>
          <div style={{ fontSize: '1.85rem', fontWeight: '800', color: '#0f172a' }}>
            {maxWidthMm.toFixed(2)} <span style={{ fontSize: '0.95rem', fontWeight: '600', color: '#64748b' }}>mm (Max)</span>
          </div>
          <div style={{ fontSize: '0.82rem', color: '#64748b', marginTop: '0.4rem', borderTop: '1px solid #f1f5f9', paddingTop: '0.5rem' }}>
            <div>Average Width: <strong>{meanWidthMm.toFixed(2)} mm</strong></div>
            <div>Min Width: <strong>{minWidthMm.toFixed(2)} mm</strong> {widthCi && `(95% CI: ±${((widthCi[1] - widthCi[0])/2).toFixed(2)}mm)`}</div>
          </div>
        </div>

        {/* Volume & Depth Card */}
        <div className="metric-card" style={{
          background: 'linear-gradient(135deg, rgba(255,255,255,0.95), rgba(248,250,252,0.9))',
          border: '1px solid rgba(139, 92, 246, 0.25)',
          borderRadius: '12px',
          padding: '1.25rem',
          boxShadow: '0 4px 20px rgba(0,0,0,0.04)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.82rem', fontWeight: '700', color: '#64748b', textTransform: 'uppercase' }}>3D Recess Volume</span>
            <div style={{ background: 'rgba(139, 92, 246, 0.1)', color: '#7c3aed', borderRadius: '8px', padding: '6px' }}>
              <Box size={18} />
            </div>
          </div>
          <div style={{ fontSize: '1.85rem', fontWeight: '800', color: '#0f172a' }}>
            {volumeCm3.toFixed(3)} <span style={{ fontSize: '0.95rem', fontWeight: '600', color: '#64748b' }}>cm³</span>
          </div>
          <div style={{ fontSize: '0.82rem', color: '#64748b', marginTop: '0.4rem', borderTop: '1px solid #f1f5f9', paddingTop: '0.5rem' }}>
            <div>Volume: <strong>{volumeMm3.toLocaleString(undefined, { maximumFractionDigits: 1 })} mm³</strong></div>
            <div>Mean Depth: <strong>{meanDepthMm.toFixed(2)} mm</strong> (Max: {maxDepthMm.toFixed(1)} mm)</div>
          </div>
        </div>

        {/* Surface Area Card */}
        <div className="metric-card" style={{
          background: 'linear-gradient(135deg, rgba(255,255,255,0.95), rgba(248,250,252,0.9))',
          border: '1px solid rgba(16, 185, 129, 0.25)',
          borderRadius: '12px',
          padding: '1.25rem',
          boxShadow: '0 4px 20px rgba(0,0,0,0.04)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.82rem', fontWeight: '700', color: '#64748b', textTransform: 'uppercase' }}>Surface Area</span>
            <div style={{ background: 'rgba(16, 185, 129, 0.1)', color: '#059669', borderRadius: '8px', padding: '6px' }}>
              <Layers size={18} />
            </div>
          </div>
          <div style={{ fontSize: '1.85rem', fontWeight: '800', color: '#0f172a' }}>
            {areaMm2.toLocaleString(undefined, { maximumFractionDigits: 1 })} <span style={{ fontSize: '0.95rem', fontWeight: '600', color: '#64748b' }}>mm²</span>
          </div>
          <div style={{ fontSize: '0.82rem', color: '#64748b', marginTop: '0.4rem', borderTop: '1px solid #f1f5f9', paddingTop: '0.5rem' }}>
            <div>Bounds (95% CI): <strong>±{areaUncMm2.toFixed(1)} mm²</strong></div>
            <div>Junctions: <strong>{data.junctions ?? 0}</strong> • Endpoints: <strong>{data.endpoints ?? 0}</strong></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default QuantificationMetrics;
