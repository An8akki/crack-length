import React, { useState, useEffect } from 'react';
import { Camera, Ruler, Box, Activity, BarChart3, Leaf, Shield, Zap, Eye, Wifi, WifiOff, Clock, AlertTriangle } from 'lucide-react';
import './styles/main.css';
import './styles/mobile-responsive.css';

// Import pages
import ImageAnalysis from './pages/ImageAnalysis';
import CrackQuantification from './pages/CrackQuantification';
import Heightmap3D from './pages/Heightmap3D';
import { AnalysisProvider, useAnalysis } from './contexts/AnalysisContext';

function AppContent() {
  const { activeTab, setActiveTab, setActiveQuantSubTab } = useAnalysis();
  const [logos, setLogos] = useState({
    sdg: null,
    sdg9: null,
    sdg11: null,
    sdg13: null
  });

  // Fetch logos from Hugging Face Models
  useEffect(() => {
    const fetchLogos = async () => {
      try {
        const hfModelUrl = 'https://huggingface.co/RijjaExplore/InfraVision-Models/resolve/main';
        setLogos({
          sdg: `${hfModelUrl}/sdg.png`,
          sdg9: `${hfModelUrl}/sdg9.png`,
          sdg11: `${hfModelUrl}/sdg11.png`,
          sdg13: `${hfModelUrl}/sdg13.png`
        });
      } catch (error) {
        console.error('Error fetching logos:', error);
      }
    };
    fetchLogos();
  }, []);

  const routes = [
    {
      id: 'image-analysis',
      label: 'Image Analysis',
      Icon: Camera,
      Component: ImageAnalysis,
      description: 'AI-powered crack detection & material analysis'
    },
    {
      id: 'crack-quantification',
      label: 'Crack Quantification',
      Icon: Ruler,
      Component: CrackQuantification,
      description: 'Physical length, opening width, 3D volume & ACI severity'
    },
    {
      id: '3d-depth',
      label: '3D Depth Estimation',
      Icon: Box,
      Component: CrackQuantification,
      description: 'ExtraTrees ML crevice depth (ΔZ), conformal intervals & recess map'
    },
    {
      id: '3d-heightmap',
      label: '3D Heightmap',
      Icon: Box,
      Component: Heightmap3D,
      description: 'Convert 2D images to 3D heightmaps'
    }
  ];

  const [failedLogos, setFailedLogos] = useState({});

  const features = [
    { icon: Shield, title: 'AI-Powered Analysis', desc: 'Advanced ML models for infrastructure assessment' },
    { icon: Eye, title: 'Real-time Monitoring', desc: 'Continuous surveillance & instant alerts' },
    { icon: Zap, title: 'Instant Insights', desc: 'Rapid assessment & actionable reports' }
  ];

  return (
    <div className="app-container">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-overlay"></div>
        
        {/* SDG Logos - Corner Layout (Only rendered if successfully loaded) */}
        <div style={{ position: 'absolute', top: '20px', right: '20px', display: 'flex', gap: '1.25rem', zIndex: '20', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
          {logos.sdg && !failedLogos.sdg && (
            <div style={{ width: '100px', height: '100px', background: 'rgba(255,255,255,0.15)', borderRadius: '12px', border: '2px solid rgba(255,255,255,0.4)', display: 'flex', alignItems: 'center', justifyContent: 'center', backdropFilter: 'blur(10px)', boxShadow: '0 8px 32px rgba(0,0,0,0.2)' }}>
              <img src={logos.sdg} alt="SDG" style={{ width: '85px', height: '85px', objectFit: 'contain' }} onError={() => setFailedLogos(prev => ({ ...prev, sdg: true }))} />
            </div>
          )}
          {logos.sdg9 && !failedLogos.sdg9 && (
            <div style={{ width: '100px', height: '100px', background: 'rgba(255,255,255,0.15)', borderRadius: '12px', border: '2px solid rgba(255,255,255,0.4)', display: 'flex', alignItems: 'center', justifyContent: 'center', backdropFilter: 'blur(10px)', boxShadow: '0 8px 32px rgba(0,0,0,0.2)' }}>
              <img src={logos.sdg9} alt="SDG 9" style={{ width: '85px', height: '85px', objectFit: 'contain' }} onError={() => setFailedLogos(prev => ({ ...prev, sdg9: true }))} />
            </div>
          )}
          {logos.sdg11 && !failedLogos.sdg11 && (
            <div style={{ width: '100px', height: '100px', background: 'rgba(255,255,255,0.15)', borderRadius: '12px', border: '2px solid rgba(255,255,255,0.4)', display: 'flex', alignItems: 'center', justifyContent: 'center', backdropFilter: 'blur(10px)', boxShadow: '0 8px 32px rgba(0,0,0,0.2)' }}>
              <img src={logos.sdg11} alt="SDG 11" style={{ width: '85px', height: '85px', objectFit: 'contain' }} onError={() => setFailedLogos(prev => ({ ...prev, sdg11: true }))} />
            </div>
          )}
          {logos.sdg13 && !failedLogos.sdg13 && (
            <div style={{ width: '100px', height: '100px', background: 'rgba(255,255,255,0.15)', borderRadius: '12px', border: '2px solid rgba(255,255,255,0.4)', display: 'flex', alignItems: 'center', justifyContent: 'center', backdropFilter: 'blur(10px)', boxShadow: '0 8px 32px rgba(0,0,0,0.2)' }}>
              <img src={logos.sdg13} alt="SDG 13" style={{ width: '85px', height: '85px', objectFit: 'contain' }} onError={() => setFailedLogos(prev => ({ ...prev, sdg13: true }))} />
            </div>
          )}
        </div>

        <div className="hero-content">
          <div className="hero-text">
            <h1 className="hero-title">
              InfraVision AI
            </h1>
            <p className="hero-subtitle">
              Intelligent Structural Health Monitoring & Advanced Infrastructure Analysis
            </p>
            <div className="hero-features">
              {features.map((feature, index) => (
                <div key={index} className="hero-feature">
                  <feature.icon className="feature-icon" size={24} />
                  <div>
                    <h3>{feature.title}</h3>
                    <p>{feature.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="hero-visual">
            <div className="floating-elements">
              <div className="floating-card card-1" onClick={() => setActiveTab('image-analysis')} style={{ cursor: 'pointer' }} title="Open Image Analysis & Segmentation">
                <div className="floating-card-icon" style={{ background: 'rgba(56, 189, 248, 0.18)', color: '#38bdf8' }}>
                  <Camera size={26} />
                </div>
                <div className="floating-card-body">
                  <div className="floating-card-title">Segmentation</div>
                  <div className="floating-card-desc">Damage masks & crack classes</div>
                </div>
              </div>

              <div className="floating-card card-2" onClick={() => { setActiveQuantSubTab('all'); setActiveTab('crack-quantification'); }} style={{ cursor: 'pointer' }} title="Open Crack Quantification">
                <div className="floating-card-icon" style={{ background: 'rgba(245, 158, 11, 0.18)', color: '#f59e0b' }}>
                  <Ruler size={26} />
                </div>
                <div className="floating-card-body">
                  <div className="floating-card-title">Quantification</div>
                  <div className="floating-card-desc">Length, calipers & ACI severity</div>
                </div>
              </div>

              <div className="floating-card card-3" onClick={() => { setActiveQuantSubTab('depth'); setActiveTab('3d-depth'); }} style={{ cursor: 'pointer' }} title="Open 3D Depth & Recess Engine">
                <div className="floating-card-icon" style={{ background: 'rgba(6, 182, 212, 0.18)', color: '#06b6d4' }}>
                  <Box size={26} />
                </div>
                <div className="floating-card-body">
                  <div className="floating-card-title">3D Depth & Recess</div>
                  <div className="floating-card-desc">ExtraTrees ML crevice ΔZ in mm</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Main Dashboard */}
      <div className="dashboard-container">
        {/* Navigation */}
        <nav className="nav-tabs-container">
          <div className="nav-tabs">
            {routes.map(({ id, label, Icon, description }) => (
              <button
                key={id}
                className={`nav-tab ${activeTab === id ? 'active' : ''}`}
                onClick={() => {
                  if (id === '3d-depth') {
                    setActiveQuantSubTab('depth');
                    setActiveTab('3d-depth');
                  } else if (id === 'crack-quantification') {
                    setActiveQuantSubTab('all');
                    setActiveTab('crack-quantification');
                  } else {
                    setActiveTab(id);
                  }
                }}
                title={description}
              >
                <Icon size={20} />
                <span className="nav-label">{label}</span>
                <span className="nav-desc">{description}</span>
              </button>
            ))}
          </div>
        </nav>

        {/* Main Content */}
        <main className="main-content-frame">
          <div className="content-frame">
            {routes.map(({ id, Component }) =>
              activeTab === id && <Component key={id} />
            )}
          </div>
        </main>
      </div>

      {/* Footer */}
      <footer className="app-footer">
        <div className="footer-bottom">
          <p style={{ fontWeight: '500', marginBottom: '15px', color: '#000' }}>&copy; 2026 InfraVision AI - Intelligent Structural Health Monitoring & Advanced Infrastructure Analysis System</p>
          <p style={{ fontWeight: '500', color: '#000' }}>Rijja H, Rohith Varshighan S, Anagesh R J, and Hudson</p>
          <div className="sdg-badges" style={{ marginTop: '15px' }}>
            <div className="sdg-badge" title="SDG 9: Industry, Innovation & Infrastructure">
              <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                <rect width="100" height="100" fill="#DD1C3D"/>
                <text x="50" y="50" textAnchor="middle" dy=".3em" fontSize="24" fontWeight="bold" fill="white">9</text>
              </svg>
              <span>SDG 9</span>
            </div>
            <div className="sdg-badge" title="SDG 11: Sustainable Cities & Communities">
              <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                <rect width="100" height="100" fill="#FCC30B"/>
                <text x="50" y="50" textAnchor="middle" dy=".3em" fontSize="24" fontWeight="bold" fill="#000">11</text>
              </svg>
              <span>SDG 11</span>
            </div>
            <div className="sdg-badge" title="SDG 13: Climate Action">
              <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                <rect width="100" height="100" fill="#407D52"/>
                <text x="50" y="50" textAnchor="middle" dy=".3em" fontSize="24" fontWeight="bold" fill="white">13</text>
              </svg>
              <span>SDG 13</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

function App() {
  return (
    <AnalysisProvider>
      <AppContent />
    </AnalysisProvider>
  );
}

export default App;
