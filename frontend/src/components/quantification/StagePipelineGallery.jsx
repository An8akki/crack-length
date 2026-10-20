import React, { useState } from 'react';
import {
  Eye,
  Layers,
  Compass,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Download,
  Maximize2,
  X,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Sparkles,
  Activity,
  Box,
  ShieldAlert,
  GitBranch,
  Grid,
  Cpu
} from 'lucide-react';

const StagePipelineGallery = ({ stageImages, activeStageId = 'all_in_one_dashboard' }) => {
  const [selectedStage, setSelectedStage] = useState(activeStageId);
  const [filterCategory, setFilterCategory] = useState('all');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);

  if (!stageImages || Object.keys(stageImages).length === 0) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center', background: '#f8fafc', borderRadius: '14px', border: '1px dashed #cbd5e1' }}>
        <p style={{ color: '#64748b' }}>No stage images available yet. Run quantification to generate pipeline stages.</p>
      </div>
    );
  }

  const allStages = [
    {
      id: 'all_in_one_dashboard',
      category: 'marked',
      title: 'Executive All-In-One Dashboard',
      shortTitle: 'All-In-One Board',
      badge: '6-PANEL COMPOSITE',
      desc: 'Unified 6-panel executive engineering board clubbing Restoration, AI Segmentation, Master HUD, Width Calipers, 3D Volume, and ACI 224R Severity.',
      color: '#0284c7',
      icon: Grid
    },
    {
      id: 'inspection_master_marked',
      category: 'marked',
      title: 'Master Inspection Marked HUD',
      shortTitle: 'Master HUD Overlay',
      badge: 'MASTER OVERLAY',
      desc: 'High-precision engineering inspection photograph annotated with centerline skeleton, normal calipers, crevice depth heatmap, and ACI severity badge.',
      color: '#06b6d4',
      icon: ShieldAlert
    },
    {
      id: 'xai_saliency_attribution',
      category: 'xai',
      title: 'Explainable AI (XAI) Saliency & Attribution Map',
      shortTitle: 'XAI Attribution Map',
      badge: 'EXPLAINABLE AI',
      desc: 'Feature saliency attribution map explaining the neural network decision drivers: Shadow Contrast (34%), Boundary Gradient (28%), Continuity (22%), and 3D Relief (16%).',
      color: '#eab308',
      icon: Cpu
    },
    {
      id: 'width_marked',
      category: 'marked',
      title: 'Orthogonal Normal-Ray Width Profile',
      shortTitle: 'Width Calipers',
      badge: 'WIDTH CALIPERS',
      desc: 'High-density orthogonal normal-ray calipers along skeleton stations with maximum opening callouts and dynamic width colormap.',
      color: '#f59e0b',
      icon: Activity
    },
    {
      id: 'volume_marked',
      category: 'marked',
      title: '3D Crevice Recess Depth & Volume Map',
      shortTitle: '3D Volume & Depth',
      badge: '3D RECESS MAP',
      desc: 'Continuous 3D crevice recess depth heatmap overlaid on damage with iso-depth contours and total volumetric integration in cm³.',
      color: '#8b5cf6',
      icon: Box
    },
    {
      id: 'severity_marked',
      category: 'marked',
      title: 'ACI 224R Structural Severity Zoning',
      shortTitle: 'Severity Zoning',
      badge: 'CODE ZONING',
      desc: 'Civil code classification categorizing crack points into Hairline (<0.10mm), Fine (0.10-0.20mm), Moderate (0.20-0.30mm), and Severe (≥0.30mm).',
      color: '#ef4444',
      icon: ShieldAlert
    },
    {
      id: 'restoration_comparison',
      category: 'restoration',
      title: 'RestorationNet Deep Enhancement (Raw vs Restored)',
      shortTitle: 'Restoration Compare',
      badge: 'RAW VS RESTORED',
      desc: 'Side-by-side diagnostic comparison of Raw concrete photograph versus 8-ResBlock deep restored image with sharpness and entropy metrics.',
      color: '#3b82f6',
      icon: Sparkles
    },
    {
      id: 'stage3_skeleton_comparison',
      category: 'skeletons',
      title: 'Tri-Algorithm Skeleton Benchmark',
      shortTitle: 'Skeleton Benchmark',
      badge: 'TRI-ALGORITHM',
      desc: 'Comparative multi-layer overlay of Guo-Hall (cyan), Zhang-Suen (lime green), and Morphological thinning (orange) with topology verification.',
      color: '#10b981',
      icon: GitBranch
    },
    {
      id: 'stage3_guo_hall',
      category: 'skeletons',
      title: 'Guo-Hall Thinning Algorithm',
      shortTitle: 'Guo-Hall Thinning',
      badge: 'PRIMARY CENTERLINE',
      desc: 'Bifurcation-preserving parallel thinning producing accurate 1-pixel branch centerlines.',
      color: '#06b6d4',
      icon: GitBranch
    },
    {
      id: 'stage3_zhang_suen',
      category: 'skeletons',
      title: 'Zhang-Suen Thinning Algorithm',
      shortTitle: 'Zhang-Suen Thinning',
      badge: 'PARALLEL SKELETON',
      desc: 'Classic iterative two-subiteration skeletonization maintaining 8-connectivity and line symmetry.',
      color: '#22c55e',
      icon: GitBranch
    },
    {
      id: 'stage3_morphological',
      category: 'skeletons',
      title: 'Morphological Medial Axis Thinning',
      shortTitle: 'Morphological Thinning',
      badge: 'MEDIAL AXIS',
      desc: 'Morphological hit-or-miss thinning algorithm preserving structural topological genus.',
      color: '#ea580c',
      icon: GitBranch
    },
    {
      id: 'stage5_mst_graph',
      category: 'skeletons',
      title: 'Stage 5: Minimum Spanning Tree (MST) Graph',
      shortTitle: 'MST Graph Network',
      badge: 'GRAPH THEORY',
      desc: 'Topological graph connecting crack endpoints and branching junctions along minimum-cost spatial edges.',
      color: '#ec4899',
      icon: Layers
    },
    {
      id: 'stage2_segmentation',
      category: 'restoration',
      title: 'Stage 2: YOLOv8-Seg Damage Mask',
      shortTitle: 'Segmentation Mask',
      badge: 'AI SEGMENTATION',
      desc: 'Sub-pixel instance segmentation mask highlighting concrete defect boundaries and surface contours.',
      color: '#14b8a6',
      icon: Layers
    },
    {
      id: 'stage1_detection',
      category: 'restoration',
      title: 'Stage 1: YOLOv8 Defect Localization',
      shortTitle: 'YOLO Detection',
      badge: 'BOUNDING BOXES',
      desc: 'Deep convolutional defect bounding boxes with model prediction confidence scores.',
      color: '#6366f1',
      icon: Eye
    },
    {
      id: 'stage4_depth_3d',
      category: 'marked',
      title: 'Stage 4: Monocular Depth Heatmap',
      shortTitle: 'Depth Heatmap',
      badge: 'MONOCULAR DEPTH',
      desc: 'Monocular relative depth relief representation computed via Laplacian surface reconstruction.',
      color: '#8b5cf6',
      icon: Box
    },
    {
      id: 'planar_rectified',
      category: 'restoration',
      title: 'Planar Perspective Rectification (Homography)',
      shortTitle: 'Planar Rectified',
      badge: 'HOMOGRAPHY',
      desc: 'Perspective-distortion-free orthogonal projection rectified to metric plane coordinate system.',
      color: '#8b5cf6',
      icon: Compass
    },
    {
      id: 'original',
      category: 'restoration',
      title: 'Original Input Image',
      shortTitle: 'Raw Photo',
      badge: 'INPUT',
      desc: 'Unprocessed baseline concrete surface photograph captured in the field.',
      color: '#64748b',
      icon: Eye
    }
  ];

  // Filter available stages
  const availableStages = allStages.filter(stg => stageImages[stg.id]);

  const filteredStages = filterCategory === 'all'
    ? availableStages
    : availableStages.filter(stg => stg.category === filterCategory);

  // Fallback if current selection not found
  const activeStage = availableStages.find(s => s.id === selectedStage) || availableStages[0];
  const activeImgSrc = activeStage ? stageImages[activeStage.id] : null;

  const currentIdx = availableStages.findIndex(s => s.id === activeStage?.id);

  const handleNext = () => {
    if (availableStages.length === 0) return;
    const nextIdx = (currentIdx + 1) % availableStages.length;
    setSelectedStage(availableStages[nextIdx].id);
  };

  const handlePrev = () => {
    if (availableStages.length === 0) return;
    const prevIdx = (currentIdx - 1 + availableStages.length) % availableStages.length;
    setSelectedStage(availableStages[prevIdx].id);
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
      {/* Gallery Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid #f1f5f9', paddingBottom: '1rem', flexWrap: 'wrap', gap: '0.8rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Layers size={22} color="#0284c7" />
            <h3 style={{ margin: 0, fontSize: '1.2rem', fontWeight: '800', color: '#0f172a' }}>
              Multi-Layer Marked Inspection Gallery
            </h3>
            <span style={{ fontSize: '0.75rem', fontWeight: '700', padding: '2px 8px', borderRadius: '6px', background: '#e0f2fe', color: '#0369a1' }}>
              {availableStages.length} Available Diagnostic Layers
            </span>
          </div>
          <p style={{ margin: '4px 0 0 0', fontSize: '0.85rem', color: '#64748b' }}>
            Unobstructed engineering view: Normal calipers, 3D depth recess, ACI severity zoning, and Explainable AI (XAI) feature attribution.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          {activeImgSrc && (
            <>
              <button
                onClick={() => { setIsModalOpen(true); setZoomLevel(1); }}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '7px 14px',
                  background: '#f8fafc',
                  borderRadius: '8px',
                  border: '1px solid #cbd5e1',
                  color: '#334155',
                  fontSize: '0.82rem',
                  fontWeight: '600',
                  cursor: 'pointer'
                }}
              >
                <Maximize2 size={14} /> Fullscreen Lightbox
              </button>

              <a
                href={activeImgSrc}
                download={`${activeStage?.id || 'stage_marked'}.png`}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '7px 14px',
                  background: '#0284c7',
                  borderRadius: '8px',
                  border: 'none',
                  color: '#fff',
                  fontSize: '0.82rem',
                  fontWeight: '700',
                  textDecoration: 'none',
                  cursor: 'pointer'
                }}
              >
                <Download size={14} /> Download Image
              </a>
            </>
          )}
        </div>
      </div>

      {/* Category Filter Pills */}
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
        {[
          { id: 'all', label: 'All Layers' },
          { id: 'marked', label: 'Engineering Marked' },
          { id: 'xai', label: 'Explainable AI (XAI)' },
          { id: 'skeletons', label: 'Skeletonization Layers' },
          { id: 'restoration', label: 'AI Restoration & Segmentation' }
        ].map(cat => (
          <button
            key={cat.id}
            onClick={() => setFilterCategory(cat.id)}
            style={{
              padding: '6px 14px',
              borderRadius: '8px',
              border: filterCategory === cat.id ? '1px solid #0284c7' : '1px solid #e2e8f0',
              background: filterCategory === cat.id ? '#0284c7' : '#f8fafc',
              color: filterCategory === cat.id ? '#fff' : '#475569',
              fontSize: '0.78rem',
              fontWeight: filterCategory === cat.id ? '700' : '500',
              cursor: 'pointer',
              transition: 'all 0.15s ease'
            }}
          >
            {cat.label}
          </button>
        ))}
      </div>

      {/* Layer Selector Chips */}
      <div style={{
        display: 'flex',
        gap: '0.55rem',
        overflowX: 'auto',
        paddingBottom: '8px',
        scrollbarWidth: 'thin'
      }}>
        {filteredStages.map((stage) => {
          const isSelected = activeStage?.id === stage.id;
          const IconComponent = stage.icon || Layers;
          return (
            <button
              key={stage.id}
              onClick={() => setSelectedStage(stage.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '8px 14px',
                borderRadius: '10px',
                border: isSelected ? `2px solid ${stage.color}` : '1px solid #e2e8f0',
                background: isSelected ? 'rgba(2, 132, 199, 0.08)' : '#f8fafc',
                color: isSelected ? stage.color : '#475569',
                fontWeight: isSelected ? '700' : '600',
                fontSize: '0.82rem',
                cursor: 'pointer',
                whiteSpace: 'nowrap',
                transition: 'all 0.15s ease'
              }}
            >
              <IconComponent size={14} color={isSelected ? stage.color : '#64748b'} />
              {stage.shortTitle}
            </button>
          );
        })}
      </div>

      {/* Main Interactive Stage Viewer Container (Zero Overlap) */}
      {activeStage && activeImgSrc && (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          borderRadius: '14px',
          overflow: 'hidden',
          background: '#090d16',
          border: '1px solid #1e293b',
          boxShadow: '0 4px 20px rgba(0,0,0,0.1)'
        }}>
          {/* Top Non-Overlapping Header Bar (Placed ABOVE the image so it NEVER covers OpenCV HUD text) */}
          <div style={{
            background: '#0f172a',
            padding: '10px 16px',
            borderBottom: `2px solid ${activeStage.color}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '10px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span style={{ fontWeight: '800', fontSize: '0.92rem', color: '#fff' }}>
                {activeStage.title}
              </span>
              <span style={{
                fontSize: '0.70rem',
                fontWeight: '800',
                padding: '2px 8px',
                borderRadius: '4px',
                background: activeStage.color,
                color: '#fff',
                letterSpacing: '0.5px'
              }}>
                {activeStage.badge}
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span style={{ fontSize: '0.78rem', color: '#94a3b8' }}>
                Layer {currentIdx + 1} of {availableStages.length}
              </span>
              <div style={{ display: 'flex', gap: '4px' }}>
                <button
                  onClick={handlePrev}
                  style={{
                    background: '#1e293b',
                    border: '1px solid #334155',
                    color: '#fff',
                    padding: '5px 12px',
                    borderRadius: '6px',
                    fontSize: '0.78rem',
                    cursor: 'pointer',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '4px'
                  }}
                  title="Previous Layer"
                >
                  <ChevronLeft size={14} /> Prev Layer
                </button>
                <button
                  onClick={handleNext}
                  style={{
                    background: '#1e293b',
                    border: '1px solid #334155',
                    color: '#fff',
                    padding: '5px 12px',
                    borderRadius: '6px',
                    fontSize: '0.78rem',
                    cursor: 'pointer',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '4px'
                  }}
                  title="Next Layer"
                >
                  Next Layer <ChevronRight size={14} />
                </button>
              </div>
            </div>
          </div>

          {/* Clean Image Viewport - ZERO HTML Overlays covering image pixels */}
          <div style={{
            position: 'relative',
            minHeight: '440px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: '#090d16',
            padding: '4px 0'
          }}>
            <img
              src={activeImgSrc}
              alt={activeStage.title}
              onClick={() => setIsModalOpen(true)}
              style={{
                width: '100%',
                maxHeight: '640px',
                objectFit: 'contain',
                display: 'block',
                cursor: 'zoom-in'
              }}
            />
          </div>

          {/* Bottom Non-Overlapping Description Bar */}
          <div style={{
            background: '#0f172a',
            padding: '10px 16px',
            borderTop: '1px solid #1e293b',
            color: '#cbd5e1',
            fontSize: '0.82rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '8px'
          }}>
            <span>{activeStage.desc}</span>
            <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
              Click image to inspect in fullscreen lightbox
            </span>
          </div>
        </div>
      )}

      {/* Lightbox Modal with Zoom Controls */}
      {isModalOpen && activeImgSrc && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.94)',
          zIndex: 9999,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '1.5rem'
        }}>
          {/* Lightbox Toolbar */}
          <div style={{
            position: 'absolute',
            top: '20px',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            background: 'rgba(15, 23, 42, 0.92)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '12px',
            padding: '8px 18px',
            color: '#fff',
            zIndex: 10000
          }}>
            <span style={{ fontWeight: '800', fontSize: '0.9rem' }}>{activeStage?.title}</span>
            <div style={{ display: 'flex', gap: '8px', marginLeft: '12px' }}>
              <button
                onClick={() => setZoomLevel(prev => Math.min(prev + 0.25, 3))}
                style={{ background: '#1e293b', border: 'none', color: '#fff', padding: '6px 10px', borderRadius: '6px', cursor: 'pointer' }}
                title="Zoom In"
              >
                <ZoomIn size={16} />
              </button>
              <button
                onClick={() => setZoomLevel(prev => Math.max(prev - 0.25, 0.5))}
                style={{ background: '#1e293b', border: 'none', color: '#fff', padding: '6px 10px', borderRadius: '6px', cursor: 'pointer' }}
                title="Zoom Out"
              >
                <ZoomOut size={16} />
              </button>
              <button
                onClick={() => setZoomLevel(1)}
                style={{ background: '#1e293b', border: 'none', color: '#fff', padding: '6px 10px', borderRadius: '6px', cursor: 'pointer' }}
                title="Reset Zoom"
              >
                <RotateCcw size={16} />
              </button>
            </div>
            <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>{(zoomLevel * 100).toFixed(0)}%</span>
            <button
              onClick={() => setIsModalOpen(false)}
              style={{
                background: '#ef4444',
                border: 'none',
                color: '#fff',
                padding: '6px 12px',
                borderRadius: '6px',
                cursor: 'pointer',
                marginLeft: '8px',
                fontWeight: '700'
              }}
            >
              <X size={16} />
            </button>
          </div>

          <div style={{
            maxWidth: '92vw',
            maxHeight: '84vh',
            overflow: 'auto',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginTop: '50px'
          }}>
            <img
              src={activeImgSrc}
              alt="Fullscreen Inspection"
              style={{
                transform: `scale(${zoomLevel})`,
                transition: 'transform 0.15s ease',
                maxWidth: '100%',
                maxHeight: '80vh',
                objectFit: 'contain',
                borderRadius: '8px'
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default StagePipelineGallery;
