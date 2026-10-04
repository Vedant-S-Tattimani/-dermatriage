import React, { useState, useRef, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Link, useLocation, Navigate } from 'react-router-dom';
import { 
  ShieldCheck, AlertTriangle, Eye, EyeOff, Info, 
  RotateCcw, MessageCircle, Send, Loader2, Bot, 
  User, Activity, Zap, ChevronRight, CheckCircle2,
  Clock, Thermometer, FlaskConical, Image as ImageIcon, Scan,
  Share2, Download, Printer, X, Calendar, MapPin, FileText
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ScrollReveal from '../components/ScrollReveal';

const ResultsPage = () => {
  const { t } = useTranslation();
  const [viewMode, setViewMode] = useState("original"); // original, xai, mask, focus
  const [xaiMethod, setXaiMethod] = useState("gradcampp");
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.6);
  const [mediaLoading, setMediaLoading] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef(null);
  const [feedbackSubmitted, setFeedbackStatus] = useState(false);



  const location = useLocation();
  const { result, originalImage, metadata } = location.state || {};

  useEffect(() => {
    if (chatOpen) {
      chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [chatMessages, chatOpen]);

  if (!result) {
    return <Navigate to="/upload" replace />;
  }

  const backendUrl = "http://127.0.0.1:8001";

  const submitFeedback = async (correctClass) => {
    try {
      await fetch(`${backendUrl}/api/v1/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          request_id: result.request_id,
          correct_class: correctClass
        }),
      });
      setFeedbackStatus(true);
    } catch (e) {
      console.error("Feedback failed", e);
    }
  };

  
  // Mapping for UI accents based on risk level
  const riskConfig = {
    "LOW": { 
      color: "bg-emerald-500", 
      light: "bg-emerald-50 dark:bg-emerald-900/10", 
      border: "border-emerald-200 dark:border-emerald-800/40", 
      text: "text-emerald-700 dark:text-emerald-400",
      icon: <ShieldCheck className="w-6 h-6" />
    },
    "MEDIUM": { 
      color: "bg-amber-500", 
      light: "bg-amber-50 dark:bg-amber-900/10", 
      border: "border-amber-200 dark:border-amber-800/40", 
      text: "text-amber-700 dark:text-amber-400",
      icon: <Activity className="w-6 h-6" />
    },
    "HIGH": { 
      color: "bg-rose-600", 
      light: "bg-rose-50 dark:bg-rose-900/10", 
      border: "border-rose-200 dark:border-rose-800/40", 
      text: "text-rose-700 dark:text-rose-400",
      icon: <AlertTriangle className="w-6 h-6" />
    },
    "UNCERTAIN": { 
      color: "bg-slate-500", 
      light: "bg-slate-50 dark:bg-slate-900/10", 
      border: "border-slate-200 dark:border-slate-800/40", 
      text: "text-slate-700 dark:text-slate-400",
      icon: <Info className="w-6 h-6" />
    }
  };

  const currentRisk = riskConfig[result.risk_level?.toUpperCase()] || riskConfig["UNCERTAIN"];
  
  // overlay = blended original + heatmap
  const getXaiOverlayUrl = () => {
    const path = result.xai_reports?.[xaiMethod] || result.gradcam_url;
    return path ? `${backendUrl}${path}?t=${Date.now()}` : null;
  };

  // pure heatmap = pure colormap, no original image mixed in
  const getPureHeatmapUrl = () => {
    const path = result.xai_heatmap_reports?.[xaiMethod];
    // Fall back to overlay if pure heatmap not present (older response)
    return path ? `${backendUrl}${path}?t=${Date.now()}` : getXaiOverlayUrl();
  };

  // Keep backward-compatible alias used below
  const getXaiUrl = getXaiOverlayUrl;

  const maskUrl = result.segmentation_mask_url ? `${backendUrl}${result.segmentation_mask_url}?t=${Date.now()}` : null;
  const focusUrl = result.cropped_image_url ? `${backendUrl}${result.cropped_image_url}?t=${Date.now()}` : null;

  const handleViewChange = (newMode) => {
    if (newMode !== viewMode) {
      setMediaLoading(true);
      setViewMode(newMode);
    }
  };

  const handleXaiMethodChange = (newMethod) => {
    if (newMethod !== xaiMethod) {
      setMediaLoading(true);
      setXaiMethod(newMethod);
    }
  };

  const viewModes = [
    { id: "original", label: "Original", icon: <ImageIcon className="w-4 h-4" /> },
    { id: "xai", label: "Heatmap", icon: <Zap className="w-4 h-4" />, disabled: !result.xai_reports },
  ];

  const xaiMethods = [
    { id: "gradcam", label: "Grad-CAM" },
    { id: "gradcampp", label: "Grad-CAM++" },
    { id: "eigencam", label: "Eigen-CAM" },
  ];

  const sendMessage = async () => {
    const trimmed = chatInput.trim();
    if (!trimmed || chatLoading) return;

    const userMsg = { role: "user", content: trimmed };
    const updatedMessages = [...chatMessages, userMsg];
    setChatMessages(updatedMessages);
    setChatInput("");
    setChatLoading(true);

    try {
      const res = await fetch(`${backendUrl}/api/v1/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: updatedMessages.filter(msg => msg && msg.role && typeof msg.content === 'string' && msg.content.trim() !== ""),
          context: `Prediction: ${result.full_name}, Confidence: ${(result.confidence * 100).toFixed(1)}%, Risk Level: ${result.risk_level}. Description: ${result.clinical_description}. Patient: ${metadata?.age} years old, at ${metadata?.location}. Symptoms: ${metadata?.rapid_change ? 'rapidly changing' : ''} ${metadata?.bleeding ? 'bleeding' : ''}`,
          language: window.localStorage.getItem('i18nextLng') || 'en'
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        let errMsg = "An error occurred.";
        if (data && data.detail) {
          if (typeof data.detail === 'string') {
            errMsg = data.detail;
          } else if (Array.isArray(data.detail)) {
            errMsg = data.detail.map(err => err.msg || JSON.stringify(err)).join(", ");
          } else if (typeof data.detail === 'object') {
            errMsg = data.detail.message || JSON.stringify(data.detail);
          }
        } else if (data && data.message) {
          errMsg = data.message;
        } else if (data && data.error) {
          errMsg = typeof data.error === 'string' ? data.error : JSON.stringify(data.error);
        }
        throw new Error(errMsg);
      }
      setChatMessages(prev => [...prev, { role: "assistant", content: data.reply }]);
    } catch (err) {
      setChatMessages(prev => [...prev, { role: "assistant", content: err.message || "I encountered a synchronization error with the clinical knowledge base. Please re-submit your query." }]);
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 md:py-12 bg-slate-50 dark:bg-dark-bg min-h-screen transition-colors duration-300">
      
      {/* ── Dashboard Header ────────────────────────────────────────────── */}
      <header className="mb-10 flex flex-col md:flex-row md:items-end justify-between gap-6">
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="space-y-1"
        >
          <div className="flex items-center gap-2 text-medical-600 dark:text-medical-400 font-black text-[10px] uppercase tracking-[0.3em] mb-2 text-gradient">
            <FlaskConical className="w-4 h-4" />
            {t('results.report')}
          </div>
          <h1 className="text-4xl md:text-5xl font-black text-slate-900 dark:text-white tracking-tighter">{t('results.analytics')}</h1>
          <p className="text-slate-500 dark:text-dark-muted font-medium flex items-center gap-2">
            <span className="bg-slate-200 dark:bg-dark-card px-2 py-0.5 rounded text-[10px] font-mono border border-slate-300 dark:border-dark-border">{result.request_id.split('-')[0]}</span>
            <span className="w-1 h-1 rounded-full bg-slate-300"></span>
            Vision Engine: <span className="text-medical-600 dark:text-medical-400 font-bold uppercase tracking-tighter">ConvNeXt</span>
          </p>
        </motion.div>
        
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-wrap gap-3"
        >
          <Link 
            to="/report" 
            state={{ result, originalImage, metadata }}
            className="btn-secondary bg-white text-slate-800 border-slate-200 hover:bg-slate-50 dark:bg-dark-card dark:border-dark-border dark:text-white dark:hover:bg-slate-800 flex items-center gap-2"
          >
            <FileText className="w-4 h-4 text-medical-600" />
            Generate Report
          </Link>
          <Link to="/upload" className="btn-secondary dark:bg-dark-card dark:border-dark-border dark:text-white dark:hover:bg-slate-800 flex items-center gap-2">
            <RotateCcw className="w-4 h-4 text-medical-600" />
            Restart Analysis
          </Link>
        </motion.div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* ── Left Column: Clinical Vision (5/12) ─────────────────────────── */}
        <div className="lg:col-span-5 space-y-6">
          <ScrollReveal direction="left">
            <div className="bg-white dark:bg-dark-card p-2 rounded-[2.5rem] shadow-2xl shadow-slate-200/50 dark:shadow-none border border-white dark:border-dark-border relative overflow-hidden group transition-colors duration-300">
              {/* View Selector Tabs */}
              <div className="absolute top-6 left-1/2 -translate-x-1/2 z-20 flex bg-white/90 dark:bg-dark-bg/90 backdrop-blur-md p-1.5 rounded-2xl border border-white dark:border-dark-border shadow-2xl">
                {viewModes.map((mode) => (
                  <button
                    key={mode.id}
                    disabled={mode.disabled}
                    onClick={() => handleViewChange(mode.id)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${
                      viewMode === mode.id 
                        ? 'bg-medical-600 text-white shadow-lg' 
                        : 'text-slate-500 dark:text-dark-muted hover:bg-slate-100 dark:hover:bg-slate-800 disabled:opacity-20'
                    }`}
                  >
                    {mode.id === viewMode ? mode.icon : null}
                    {mode.label}
                  </button>
                ))}
              </div>

              <div className="aspect-[4/5] relative rounded-[2.1rem] overflow-hidden bg-slate-100 dark:bg-slate-900 transition-colors">
                {/* Skeleton Loader Overlay */}
                <AnimatePresence>
                  {mediaLoading && (
                    <motion.div 
                      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                      className="absolute inset-0 z-10 bg-slate-100 dark:bg-slate-900"
                    >
                      <div className="w-full h-full skeleton" />
                    </motion.div>
                  )}
                </AnimatePresence>

                <AnimatePresence mode="wait">
                  {viewMode === "original" && (
                    <motion.img 
                      key="original"
                      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                      src={originalImage} 
                      onLoad={() => setMediaLoading(false)}
                      alt="Raw Scan" className="w-full h-full object-cover transition-transform duration-1000 group-hover:scale-105"
                    />
                  )}
                  {viewMode === "xai" && (
                    <motion.div
                      key="xai"
                      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                      className="w-full h-full relative"
                    >
                      {/* Base: original image */}
                      <img src={originalImage} className="absolute inset-0 w-full h-full object-cover" alt="Base" />
                      {/* Pure Grad-CAM heatmap blended on top at user-controlled opacity */}
                      <motion.img 
                        src={getPureHeatmapUrl()} 
                        onLoad={() => setMediaLoading(false)}
                        alt="Grad-CAM Heatmap" 
                        style={{ opacity: heatmapOpacity, mixBlendMode: 'multiply' }}
                        className="absolute inset-0 w-full h-full object-cover" 
                      />
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>

            {/* XAI Controls Card */}
            <AnimatePresence>
              {viewMode === "xai" && (
                <motion.div
                  initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 15 }}
                  className="mt-6 p-8 bg-white dark:bg-dark-card rounded-[2rem] border border-slate-200 dark:border-dark-border shadow-xl space-y-8"
                >
                  <div className="space-y-4">
                    <label className="text-[10px] font-black text-slate-400 dark:text-dark-muted uppercase tracking-[0.2em] block">{t('results.saliency_method')}</label>
                    <div className="grid grid-cols-3 gap-3">
                      {xaiMethods.map(m => (
                        <button
                          key={m.id}
                          onClick={() => handleXaiMethodChange(m.id)}
                          className={`px-3 py-4 rounded-2xl border-2 text-[10px] font-black uppercase tracking-widest transition-all ${
                            xaiMethod === m.id 
                              ? 'bg-medical-50 dark:bg-medical-900/20 border-medical-500 text-medical-600 dark:text-medical-400 shadow-md shadow-medical-500/10' 
                              : 'bg-slate-50 dark:bg-dark-bg border-transparent text-slate-400 dark:text-dark-muted hover:bg-slate-100 dark:hover:bg-slate-800'
                          }`}
                        >
                          {m.label}
                        </button>
                      ))}
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <label className="text-[10px] font-black text-slate-400 dark:text-dark-muted uppercase tracking-[0.2em]">{t('results.heatmap_opacity')}</label>
                      <span className="px-3 py-1 bg-slate-100 dark:bg-dark-bg rounded-lg text-[10px] font-bold text-slate-900 dark:text-white">{(heatmapOpacity * 100).toFixed(0)}%</span>
                    </div>
                    <input 
                      type="range" min="0" max="1" step="0.01" 
                      value={heatmapOpacity}
                      onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                      className="w-full h-1.5 bg-slate-100 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer accent-medical-600"
                    />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Timing Stats */}
            <div className="mt-6 grid grid-cols-2 gap-4">
              <div className="bg-white dark:bg-dark-card p-6 rounded-3xl border border-slate-100 dark:border-dark-border flex items-center gap-4 transition-colors shadow-sm">
                <div className="w-12 h-12 rounded-2xl bg-indigo-50 dark:bg-indigo-900/20 flex items-center justify-center text-indigo-600 dark:text-indigo-400">
                  <Clock className="w-6 h-6" />
                </div>
                <div>
                  <div className="text-[10px] font-black text-slate-400 dark:text-dark-muted uppercase tracking-widest leading-none mb-1">{t('results.latency')}</div>
                  <div className="text-lg font-black text-slate-900 dark:text-white">{result.inference_time_ms.toFixed(0)}ms</div>
                </div>
              </div>
              <div className="bg-white dark:bg-dark-card p-6 rounded-3xl border border-slate-100 dark:border-dark-border flex items-center gap-4 transition-colors shadow-sm">
                <div className="w-12 h-12 rounded-2xl bg-emerald-50 dark:bg-emerald-900/20 flex items-center justify-center text-emerald-600 dark:text-emerald-400">
                  <CheckCircle2 className="w-6 h-6" />
                </div>
                <div>
                  <div className="text-[10px] font-black text-slate-400 dark:text-dark-muted uppercase tracking-widest leading-none mb-1">{t('results.status')}</div>
                  <div className="text-lg font-black text-slate-900 dark:text-white uppercase">{t('results.verified')}</div>
                </div>
              </div>
            </div>
          </ScrollReveal>
        </div>

        {/* ── Right Column: Insights (7/12) ───────────────────────────────── */}
        <div className="lg:col-span-7 space-y-8">
          
          {/* Diagnostic Core Card */}
          <ScrollReveal direction="right">
            <div className="bg-white dark:bg-dark-card rounded-[2.5rem] border border-slate-200 dark:border-dark-border overflow-hidden shadow-xl transition-colors duration-300">
              <div className="p-8 md:p-12">
                <div className="flex flex-col md:flex-row md:items-start justify-between gap-10 mb-12">
                  <div className="space-y-6 flex-1">
                    <div className={`inline-flex items-center gap-2.5 px-5 py-2.5 rounded-full font-black text-[10px] uppercase tracking-[0.2em] border ${currentRisk.light} ${currentRisk.text} ${currentRisk.border}`}>
                      {currentRisk.icon}
                      {result.risk_level} Risk Level
                    </div>
                    <div>
                      <h2 className="text-5xl md:text-6xl font-black text-slate-900 dark:text-white tracking-tighter mb-4 leading-none">
                        {result.full_name}
                      </h2>
                      <p className="text-xl text-slate-500 dark:text-dark-muted font-medium max-w-lg leading-relaxed">
                        {result.clinical_description}
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex flex-col items-center bg-slate-50 dark:bg-dark-bg p-6 rounded-[2rem] border border-slate-100 dark:border-dark-border transition-colors">
                    <div className="relative w-32 h-32 flex items-center justify-center">
                      <svg className="w-full h-full transform -rotate-90">
                        <circle cx="64" cy="64" r="58" fill="none" stroke="currentColor" className="text-slate-200 dark:text-slate-800" strokeWidth="10" />
                        <motion.circle 
                          cx="64" cy="64" r="58" fill="none" 
                          stroke={currentRisk.color.replace('bg-', '') === 'emerald-500' ? '#10b981' : currentRisk.color.replace('bg-', '') === 'amber-500' ? '#f59e0b' : '#e11d48'} 
                          strokeWidth="10" 
                          strokeDasharray={364}
                          initial={{ strokeDashoffset: 364 }}
                          animate={{ strokeDashoffset: 364 - (364 * result.confidence) }}
                          transition={{ duration: 2, ease: "circOut", delay: 0.5 }}
                          strokeLinecap="round"
                        />
                      </svg>
                      <div className="absolute inset-0 flex flex-col items-center justify-center">
                        <span className="text-3xl font-black text-slate-900 dark:text-white">{(result.confidence * 100).toFixed(0)}%</span>
                        <span className="text-[8px] font-black text-slate-400 dark:text-dark-muted uppercase tracking-widest">{t('results.certainty')}</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className={`p-8 rounded-[2rem] ${currentRisk.light} border-2 ${currentRisk.border} border-dashed flex flex-col md:flex-row items-center md:items-start gap-8 transition-colors`}>
                  <div className={`w-16 h-16 rounded-[1.5rem] ${currentRisk.color} flex items-center justify-center text-white flex-shrink-0 shadow-2xl shadow-medical-500/40`}>
                    <CheckCircle2 className="w-8 h-8" />
                  </div>
                  <div className="text-center md:text-left">
                    <h4 className={`text-xl font-black mb-2 uppercase tracking-tighter ${currentRisk.text}`}>{t('results.recommendation')}</h4>
                    <p className="text-slate-700 dark:text-slate-300 leading-relaxed font-bold text-lg">
                      {result.recommendation}
                    </p>
                  </div>
                </div>
              </div>


              {/* Advanced Probability Distribution */}
              <div className="bg-slate-50/80 dark:bg-dark-bg/50 border-t border-slate-100 dark:border-dark-border p-8 md:p-12 transition-colors">
                <div className="flex justify-between items-center mb-10">
                   <h4 className="text-[10px] font-black text-slate-400 dark:text-dark-muted uppercase tracking-[0.3em]">{t('results.differential')}</h4>
                   {!feedbackSubmitted ? (
                     <div className="flex items-center gap-3">
                        <span className="text-[8px] font-black text-slate-400 uppercase tracking-widest">{t('results.verify_accuracy')}</span>
                        <button onClick={() => submitFeedback(result.class_name)} className="p-2 rounded-lg bg-emerald-50 text-emerald-600 hover:bg-emerald-100 transition-colors"><CheckCircle2 className="w-4 h-4" /></button>
                        <button onClick={() => setChatOpen(true)} className="p-2 rounded-lg bg-rose-50 text-rose-600 hover:bg-rose-100 transition-colors"><X className="w-4 h-4" /></button>
                     </div>
                   ) : (
                     <span className="text-[8px] font-black text-emerald-500 uppercase tracking-[0.2em] flex items-center gap-2"><CheckCircle2 className="w-3 h-3" /> Feedback Recorded</span>
                   )}
                </div>
                <div className="grid grid-cols-1 gap-10">
                  {result.top_3.map((pred, idx) => (
                    <div key={idx} className="space-y-4">
                      <div className="flex justify-between items-end">
                        <div className="flex items-center gap-3">
                          <span className={`flex items-center justify-center w-6 h-6 rounded-lg text-[10px] font-black ${idx === 0 ? 'bg-medical-600 text-white' : 'bg-slate-200 dark:bg-slate-800 text-slate-500 transition-colors'}`}>
                            {idx + 1}
                          </span>
                          <span className={`text-lg font-black tracking-tight ${idx === 0 ? 'text-slate-900 dark:text-white' : 'text-slate-500 dark:text-dark-muted transition-colors'}`}>
                            {pred.class_name}
                          </span>
                        </div>
                        <span className="text-sm font-mono font-black text-medical-600 dark:text-medical-400 transition-colors">{(pred.probability * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-3 bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden transition-colors">
                        <motion.div 
                          initial={{ width: 0 }}
                          animate={{ width: `${pred.probability * 100}%` }}
                          transition={{ duration: 1.5, delay: 0.8 + (idx * 0.2), ease: "circOut" }}
                          className={`h-full rounded-full shadow-lg ${idx === 0 ? currentRisk.color : 'bg-slate-400 dark:bg-slate-600 transition-colors'}`}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </ScrollReveal>

          {/* Multimodal Context & Consultation */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
            
            <ScrollReveal direction="up" delay={0.2}>
              <div className="bg-white dark:bg-dark-card rounded-[2.5rem] p-10 border border-slate-200 dark:border-dark-border shadow-xl h-full transition-colors duration-300">
                <h3 className="text-xl font-black text-slate-900 dark:text-white mb-8 flex items-center gap-3 tracking-tighter uppercase">
                  <User className="w-6 h-6 text-indigo-500" />
                  Intake Profile
                </h3>
                <div className="space-y-6">
                  {[
                    { label: "Patient Age", val: metadata?.age ? `${metadata.age}y` : "N/A", icon: <Calendar className="w-4 h-4" /> },
                    { label: "Location", val: metadata?.location || "N/A", icon: <MapPin className="w-4 h-4" /> },
                    { label: "Duration", val: metadata?.duration || "N/A", icon: <Clock className="w-4 h-4" /> }
                  ].map((item, i) => (
                    <div key={i} className="flex items-center justify-between py-1 border-b border-slate-50 dark:border-slate-800 transition-colors">
                      <div className="flex items-center gap-3 text-slate-400 dark:text-dark-muted font-bold text-[10px] uppercase tracking-widest transition-colors">
                        {item.icon} {item.label}
                      </div>
                      <span className="text-sm font-black text-slate-700 dark:text-slate-300 tracking-tight transition-colors">{item.val}</span>
                    </div>
                  ))}
                  
                  <div className="pt-4 space-y-4">
                    <label className="text-[10px] font-black text-slate-400 dark:text-dark-muted uppercase tracking-[0.2em] block transition-colors">{t('results.symptom_markers')}</label>
                    <div className="flex flex-wrap gap-3">
                      {metadata?.bleeding && <span className="px-4 py-2 bg-rose-50 dark:bg-rose-900/20 text-rose-600 dark:text-rose-400 rounded-2xl text-[10px] font-black border border-rose-100 dark:border-rose-900/30 uppercase tracking-widest shadow-sm transition-colors">Bleeding</span>}
                      {metadata?.itching && <span className="px-4 py-2 bg-amber-50 dark:bg-amber-900/20 text-amber-600 dark:text-amber-400 rounded-2xl text-[10px] font-black border border-amber-100 dark:border-amber-900/30 uppercase tracking-widest shadow-sm transition-colors">Itching</span>}
                      {metadata?.rapid_change && <span className="px-4 py-2 bg-indigo-50 dark:bg-indigo-900/20 text-indigo-600 dark:text-indigo-400 rounded-2xl text-[10px] font-black border border-indigo-100 dark:border-indigo-900/30 uppercase tracking-widest shadow-sm animate-pulse transition-colors">Rapid Change</span>}
                      {!metadata?.bleeding && !metadata?.itching && !metadata?.rapid_change && <span className="text-xs text-slate-400 italic">{t('results.no_symptoms')}</span>}
                    </div>
                  </div>
                </div>
              </div>
            </ScrollReveal>

            <ScrollReveal direction="up" delay={0.4}>
              <div className="bg-slate-900 dark:bg-indigo-950 rounded-[2.5rem] p-10 text-white h-full relative overflow-hidden flex flex-col justify-between shadow-2xl transition-colors duration-500">
                <div className="relative z-10">
                  <div className="flex items-center gap-3 mb-6">
                    <MessageCircle className="w-8 h-8 text-medical-400" />
                    <h3 className="text-2xl font-black tracking-tighter uppercase">{t('results.clinical_chat')}</h3>
                  </div>
                  <p className="text-slate-400 dark:text-indigo-200 text-base leading-relaxed mb-10 font-medium transition-colors">
                    Consult our medically-trained assistant for pattern insights, condition papers, and clarified next steps.
                  </p>
                </div>
                
                <button 
                  onClick={() => setChatOpen(true)}
                  className="w-full py-5 bg-medical-600 hover:bg-medical-500 text-white rounded-[2rem] font-black text-xl flex items-center justify-center gap-4 transition-all relative z-10 shadow-2xl shadow-medical-500/20 active:scale-95"
                >
                  Start Consultation
                  <ChevronRight className="w-6 h-6" />
                </button>

                <div className="absolute -right-16 -bottom-16 w-64 h-64 bg-medical-600/10 rounded-full blur-[80px]"></div>
              </div>
            </ScrollReveal>
          </div>
        </div>
      </div>

        <div className="lg:hidden fixed bottom-0 left-0 right-0 p-4 bg-white/80 dark:bg-dark-bg/80 backdrop-blur-xl border-t border-slate-200 dark:border-dark-border z-40 flex gap-2">
          <Link 
            to="/report" 
            state={{ result, originalImage, metadata }}
            className="flex-1 flex items-center justify-center gap-2 py-4 bg-slate-100 dark:bg-dark-card text-slate-800 dark:text-white rounded-2xl font-black text-lg transition-colors border border-slate-200 dark:border-dark-border"
          >
            <FileText className="w-5 h-5" /> Report
          </Link>
          <Link to="/upload" className="flex-1 flex items-center justify-center gap-2 py-4 bg-medical-600 text-white rounded-2xl font-black text-lg shadow-lg shadow-medical-500/30">
            <RotateCcw className="w-5 h-5" /> New Scan
          </Link>
        </div>

      {/* ── Production Chat Console ────────────────────────────────────── */}
      <AnimatePresence>
        {chatOpen && (
          <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 sm:p-8">
            <motion.div 
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              onClick={() => setChatOpen(false)}
              className="absolute inset-0 bg-slate-900/80 backdrop-blur-md"
            />
            
            <motion.div 
              initial={{ opacity: 0, scale: 0.9, y: 30 }} animate={{ opacity: 1, scale: 1, y: 0 }} exit={{ opacity: 0, scale: 0.9, y: 30 }}
              className="relative w-full max-w-3xl bg-white dark:bg-dark-card rounded-[3rem] shadow-[0_50px_100px_rgba(0,0,0,0.4)] overflow-hidden flex flex-col h-[85vh] max-h-[800px] border border-white dark:border-dark-border transition-colors duration-300"
            >
              {/* Console Header */}
              <div className="px-10 py-8 border-b border-slate-100 dark:border-dark-border flex items-center justify-between bg-slate-50/50 dark:bg-dark-bg/50 transition-colors">
                <div className="flex items-center gap-5">
                  <div className="w-14 h-14 rounded-3xl bg-medical-600 flex items-center justify-center text-white shadow-xl shadow-medical-500/30">
                    <Bot className="w-8 h-8" />
                  </div>
                  <div>
                    <h3 className="text-xl font-black text-slate-900 dark:text-white tracking-tight uppercase">{t('results.expert')}</h3>
                    <div className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400 text-[10px] font-black uppercase tracking-[0.2em]">
                      <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                      Connection Secure
                    </div>
                  </div>
                </div>
                <button 
                  onClick={() => setChatOpen(false)}
                  className="p-4 hover:bg-slate-200 dark:hover:bg-slate-800 rounded-3xl text-slate-400 transition-all active:scale-90"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              {/* Conversation Area */}
              <div className="flex-1 overflow-y-auto p-10 space-y-8 scrollbar-thin dark:bg-dark-card transition-colors duration-300">
                {chatMessages.length === 0 && (
                  <div className="h-full flex flex-col items-center justify-center text-center px-16 space-y-8">
                    <div className="w-24 h-24 bg-slate-50 dark:bg-dark-bg rounded-[2rem] flex items-center justify-center border border-slate-100 dark:border-dark-border transition-colors">
                      <MessageCircle className="w-12 h-12 text-slate-200 dark:text-slate-800" />
                    </div>
                    <div className="space-y-4">
                      <h4 className="text-xl font-black text-slate-900 dark:text-white uppercase tracking-tighter">{t('results.initialized')}</h4>
                      <p className="text-slate-400 dark:text-dark-muted font-medium max-w-sm leading-relaxed transition-colors">Ask specific questions about the {result.full_name} profile or clinical symptoms.</p>
                    </div>
                  </div>
                )}

                {chatMessages.map((msg, idx) => (
                  <div key={idx} className={`flex gap-5 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}>
                    <div className={`w-12 h-12 rounded-2xl flex items-center justify-center flex-shrink-0 shadow-lg ${
                      msg.role === "user" ? "bg-slate-900 dark:bg-indigo-600 text-white" : "bg-medical-100 dark:bg-medical-900/30 text-medical-600 dark:text-medical-400"
                    }`}>
                      {msg.role === "user" ? <User className="w-6 h-6" /> : <Bot className="w-6 h-6" />}
                    </div>
                    <div className={`max-w-[75%] px-8 py-5 rounded-[2rem] text-sm leading-relaxed font-bold shadow-sm ${
                      msg.role === "user" 
                        ? "bg-slate-900 dark:bg-indigo-600 text-white rounded-tr-none" 
                        : "bg-slate-100 dark:bg-dark-bg text-slate-700 dark:text-slate-300 rounded-tl-none border border-slate-200 dark:border-dark-border transition-colors"
                    }`}>
                      {msg.content}
                    </div>
                  </div>
                ))}

                {chatLoading && (
                  <div className="flex gap-5">
                    <div className="w-12 h-12 rounded-2xl bg-medical-100 dark:bg-medical-900/30 text-medical-600 dark:text-medical-400 flex items-center justify-center shadow-lg transition-colors">
                      <Bot className="w-6 h-6" />
                    </div>
                    <div className="bg-slate-100 dark:bg-dark-bg border border-slate-200 dark:border-dark-border text-slate-400 dark:text-dark-muted px-8 py-5 rounded-[2rem] rounded-tl-none text-sm font-bold flex items-center gap-4 transition-colors">
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Cross-referencing medical database...
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              {/* Console Input */}
              <div className="p-8 bg-slate-50/50 dark:bg-dark-bg/50 border-t border-slate-100 dark:border-dark-border transition-colors">
                <div className="relative flex items-center gap-4">
                  <input 
                    type="text" value={chatInput} onChange={(e) => setChatInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
                    placeholder={t('results.input_placeholder')} disabled={chatLoading}
                    className="flex-1 bg-white dark:bg-dark-card border border-slate-200 dark:border-dark-border text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-dark-muted px-8 py-5 rounded-[2rem] text-sm outline-none focus:ring-4 focus:ring-medical-500/10 focus:border-medical-500 transition-all shadow-inner disabled:opacity-50"
                  />
                  <button 
                    onClick={sendMessage} disabled={chatLoading || !chatInput.trim()}
                    className="bg-medical-600 hover:bg-medical-500 disabled:opacity-40 text-white p-5 rounded-[2rem] transition-all shadow-2xl shadow-medical-500/30 active:scale-90"
                  >
                    <Send className="w-6 h-6" />
                  </button>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* ── Footer Disclaimer ────────────────────────────────────────────── */}
      <footer className="mt-16 pt-10 border-t border-slate-200 dark:border-dark-border transition-colors duration-300">
        <div className="bg-amber-50 dark:bg-amber-900/10 border border-amber-200 dark:border-amber-800/40 rounded-[2.5rem] p-10 flex items-start gap-8 transition-colors shadow-sm">
          <div className="w-14 h-14 rounded-3xl bg-amber-500 flex items-center justify-center text-white flex-shrink-0 shadow-xl shadow-amber-500/20">
            <ShieldCheck className="w-8 h-8" />
          </div>
          <div className="space-y-3">
            <h4 className="text-sm font-black text-amber-900 dark:text-amber-400 uppercase tracking-[0.3em]">{t('results.disclaimer_title')}</h4>
            <p className="text-xs text-amber-800/80 dark:text-amber-200/60 leading-relaxed font-bold transition-colors">
              {result.disclaimer} This is an autonomous pattern recognition triage. Automated assessment cannot replace histological examination. 
              <span className="text-amber-900 dark:text-amber-200 block mt-2 transition-colors"> If a lesion is bleeding, growing, or changing color rapidly, it must be clinically biopsied regardless of AI certainty scores.</span>
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default ResultsPage;
