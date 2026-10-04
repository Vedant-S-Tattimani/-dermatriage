import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity, ShieldAlert, HeartPulse } from 'lucide-react';
import GooeyNav from './GooeyNav';
import StarBorder from './StarBorder';

const Layout = ({ children }) => {
  const location = useLocation();

  const navLinks = [
    { name: 'Home', path: '/' },
    { name: 'Upload', path: '/upload' },
    { name: 'How It Works', path: '/how-it-works' },
    { name: 'Monitoring', path: '/monitoring' },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-dark-bg transition-colors relative">


      {/* Navbar */}
      <nav className="sticky top-0 z-50 bg-white/70 dark:bg-dark-bg/70 backdrop-blur-xl border-b border-slate-200 dark:border-dark-border transition-colors">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-20 items-center">
            <Link to="/" className="flex items-center gap-2 group">
              <div className="bg-medical-600 p-2 rounded-lg group-hover:rotate-12 transition-transform shadow-lg shadow-medical-500/20">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <span className="text-xl font-bold text-slate-900 dark:text-white tracking-tight">
                AI Skin <span className="text-medical-600">Triage</span>
              </span>
            </Link>

            <div className="hidden lg:flex items-center gap-8">
              <GooeyNav
                items={[
                  { label: "Home", href: "/" },
                  { label: "Upload", href: "/upload" },
                  { label: "How It Works", href: "/how-it-works" },
                  { label: "Monitoring", href: "/monitoring" },
                ]}
                particleCount={12}
                particleDistances={[60, 5]}
                particleR={80}
                animationTime={500}
                timeVariance={200}
              />
            </div>

            <div className="flex items-center gap-4">
              <div className="lg:hidden flex items-center gap-4">
              </div>
              <StarBorder
                as={Link}
                to="/upload"
                color="#ffffff"
                speed="4s"
                thickness={2}
                className="py-0 hidden sm:block"
              >
                Start Analysis
              </StarBorder>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-grow relative z-10">
        {children}
      </main>

      {/* Footer Disclaimer */}
      <footer className="sticky bottom-0 z-40 bg-slate-900 text-white py-4 shadow-[0_-4px_20px_rgba(0,0,0,0.1)]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center justify-center gap-3 text-center">
          <ShieldAlert className="w-5 h-5 text-amber-400 animate-pulse flex-shrink-0" />
          <p className="text-xs sm:text-sm font-medium text-slate-300">
            <span className="text-amber-400 font-bold uppercase tracking-widest text-[10px]">Disclaimer:</span> Patterns detected are NOT a diagnosis. Always consult a dermatologist.
          </p>
        </div>
      </footer>
      
      {/* Real Footer */}
      <footer className="bg-white dark:bg-slate-900 border-t border-slate-200 dark:border-dark-border py-16 pb-32 relative z-10 transition-colors">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 grid grid-cols-1 md:grid-cols-4 gap-12">
          <div className="md:col-span-2">
            <div className="flex items-center gap-2 mb-6">
              <HeartPulse className="w-6 h-6 text-medical-600" />
              <span className="text-xl font-black text-slate-900 dark:text-white uppercase tracking-tighter">NeuralSkin AI</span>
            </div>
            <p className="text-slate-500 dark:text-dark-muted text-sm leading-relaxed max-w-md font-medium">
              Empowering global health through instant, high-precision dermatological triage. Leveraging the latest in vision transformers and medical pattern recognition.
            </p>
          </div>
          <div>
            <h4 className="text-xs font-black text-slate-900 dark:text-white uppercase tracking-[0.2em] mb-6">Navigation</h4>
            <ul className="space-y-4">
              {navLinks.map(link => (
                <li key={link.path}>
                  <Link to={link.path} className="text-slate-500 dark:text-dark-muted hover:text-medical-600 transition-colors text-sm font-bold">{link.name}</Link>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h4 className="text-xs font-black text-slate-900 dark:text-white uppercase tracking-[0.2em] mb-6">Resources</h4>
            <ul className="space-y-4 font-bold text-sm">
              <li><span className="text-slate-500 dark:text-dark-muted hover:text-medical-600 transition-colors cursor-pointer">Clinical Papers</span></li>
              <li><span className="text-slate-500 dark:text-dark-muted hover:text-medical-600 transition-colors cursor-pointer">Privacy Protocol</span></li>
              <li><span className="text-slate-500 dark:text-dark-muted hover:text-medical-600 transition-colors cursor-pointer">API Documentation</span></li>
            </ul>
          </div>
        </div>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-16 pt-8 border-t border-slate-100 dark:border-dark-border text-center">
          <p className="text-slate-400 dark:text-dark-muted text-[10px] font-bold uppercase tracking-widest">© 2026 AI Skin Assistant. Production Prototype v1.1</p>
        </div>
      </footer>


    </div>
  );
};

export default Layout;
