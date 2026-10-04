import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import LandingPage from './pages/LandingPage';
import UploadPage from './pages/UploadPage';
import ResultsPage from './pages/ResultsPage';
import HowItWorksPage from './pages/HowItWorksPage';
import MonitoringPage from './pages/MonitoringPage';
import ReportPage from './pages/ReportPage';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/how-it-works" element={<HowItWorksPage />} />
          <Route path="/monitoring" element={<MonitoringPage />} />
          <Route path="/report" element={<ReportPage />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
