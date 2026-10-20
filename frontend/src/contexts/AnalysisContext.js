import React, { createContext, useContext, useState } from 'react';

const AnalysisContext = createContext();

export const useAnalysis = () => {
  const context = useContext(AnalysisContext);
  if (!context) {
    throw new Error('useAnalysis must be used within an AnalysisProvider');
  }
  return context;
};

export const AnalysisProvider = ({ children }) => {
  const [activeTab, setActiveTab] = useState('image-analysis');
  const [lastAnalysis, setLastAnalysis] = useState(null);
  const [outputImages, setOutputImages] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(() => {
    return sessionStorage.getItem('lastUploadedImage') || null;
  });
  const [currentFile, setCurrentFile] = useState(null);
  const [quantificationResults, setQuantificationResults] = useState(null);
  const [activeQuantSubTab, setActiveQuantSubTab] = useState('overview');

  const updateAnalysis = (analysisData, images) => {
    setLastAnalysis(analysisData);
    setOutputImages(images);
  };

  const clearAnalysis = () => {
    setLastAnalysis(null);
    setOutputImages(null);
    setQuantificationResults(null);
  };

  const navigateToTab = (tabId) => {
    setActiveTab(tabId);
  };

  const navigateToQuantification = (imageData, fileObj = null) => {
    if (imageData) {
      setUploadedImage(imageData);
      sessionStorage.setItem('lastUploadedImage', imageData);
    }
    if (fileObj) {
      setCurrentFile(fileObj);
    }
    setActiveTab('crack-quantification');
  };

  const navigateToSegmentation = () => {
    setActiveTab('image-analysis');
  };

  return (
    <AnalysisContext.Provider value={{
      activeTab,
      setActiveTab,
      lastAnalysis,
      outputImages,
      uploadedImage,
      setUploadedImage,
      currentFile,
      setCurrentFile,
      quantificationResults,
      setQuantificationResults,
      activeQuantSubTab,
      setActiveQuantSubTab,
      updateAnalysis,
      clearAnalysis,
      navigateToTab,
      navigateToQuantification,
      navigateToSegmentation
    }}>
      {children}
    </AnalysisContext.Provider>
  );
};