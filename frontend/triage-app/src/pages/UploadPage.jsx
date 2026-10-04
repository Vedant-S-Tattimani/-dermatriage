import React, { useState, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { 
  Upload, Image as ImageIcon, CheckCircle2, 
  AlertCircle, Info, Loader2, ShieldCheck, 
  Search, Scan, Activity, Microscope, User,
  Calendar, MapPin, Clock, AlertTriangle, X
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ScrollReveal from '../components/ScrollReveal';

const UploadPage = () => {
  const { t } = useTranslation();
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState(null);
  
  const fileInputRef = useRef(null);

  // Patient Metadata State
  const [metadata, setMetadata] = useState({
    age: "",
    location: "",
    duration: "",
    itching: false,
    bleeding: false,
    rapid_change: false
  });

  const navigate = useNavigate();

  const handleMetadataChange = (e) => {
    const { name, value, type, checked } = e.target;
    setMetadata(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const processFile = (file) => {
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError("Please upload a valid image file (JPEG, PNG, or WebP).");
        return;
      }
      if (file.size > 10 * 1024 * 1024) {
        setError("Image size exceeds 10MB limit.");
        return;
      }
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setError(null);
    }
  };

  const handleImageChange = (e) => {
    processFile(e.target.files[0]);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    processFile(e.dataTransfer.files[0]);
  };

  const handleSampleImage = () => {
    setPreviewUrl('https://images.unsplash.com/photo-1584622650111-993a426fbf0a?auto=format&fit=crop&q=80&w=800');
    setSelectedImage({ name: 'sample-skin-lesion.jpg' });
    setMetadata({
      age: "65",
      location: "Left Shoulder",
      duration: "3 months",
      itching: true,
      bleeding: false,
      rapid_change: true
    });
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!selectedImage) return;

    setIsAnalyzing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedImage);
      
      // Append multimodal metadata
      if (metadata.age) formData.append("age", metadata.age);
      if (metadata.location) formData.append("location", metadata.location);
      if (metadata.duration) formData.append("duration", metadata.duration);
      formData.append("itching", metadata.itching);
      formData.append("bleeding", metadata.bleeding);
      formData.append("rapid_change", metadata.rapid_change);

      let persistentImageUrl = previewUrl;
      if (selectedImage instanceof File) {
        persistentImageUrl = await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result);
          reader.readAsDataURL(selectedImage);
        });
      }

      const response = await fetch("http://127.0.0.1:8001/api/v1/triage", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Analysis failed. Please check the image quality.");
      }

      const result = await response.json();
      
      navigate('/results', { 
        state: { 
          result, 
          
          
          originalImage: persistentImageUrl,
          metadata // Pass back for display
        } 
      });
    } catch (err) {
      console.error("Analysis Error:", err);
      setError(err.message || "An unexpected error occurred during processing.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 md:py-20 bg-slate-50 dark:bg-dark-bg min-h-screen transition-colors">
      <div className="flex flex-col lg:flex-row gap-16 items-start">
        
        {/* ── Left Column: Interaction ────────────────────────────────────── */}
        <div className="flex-1 w-full space-y-10">
          <header>
            <motion.div 
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 rounded-full font-bold text-xs uppercase tracking-widest mb-4"
            >
              <Scan className="w-4 h-4" />
              {t('upload.intake')}
            </motion.div>
            <h1 className="text-5xl font-black text-slate-900 dark:text-white tracking-tight leading-[1.1]">
              {t('upload.title_multi')} <br />
              <span className="text-medical-600">{t('upload.title_ai')}</span>
            </h1>
            <p className="text-slate-500 dark:text-dark-muted text-lg mt-4 max-w-lg font-medium">
              {t('upload.subtitle')}
            </p>
          </header>

          <ScrollReveal>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
              {/* Image Upload Area */}
              <div className="space-y-4">
                <label className="text-sm font-black text-slate-400 dark:text-dark-muted uppercase tracking-widest block px-2">{t('upload.step1')}</label>
                <div 
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => !previewUrl && fileInputRef.current.click()}
                  className={`relative border-2 border-dashed rounded-[2.5rem] p-1 h-[420px] transition-all duration-500 cursor-pointer overflow-hidden ${
                    previewUrl 
                      ? 'border-medical-500 bg-white dark:bg-dark-card' 
                      : isDragging 
                        ? 'border-medical-600 bg-medical-50 dark:bg-indigo-900/20 scale-[1.02]' 
                        : 'border-slate-300 dark:border-dark-border hover:border-medical-400 bg-white dark:bg-dark-card'
                  }`}
                >
                  <input 
                    type="file" 
                    ref={fileInputRef}
                    className="hidden" 
                    onChange={handleImageChange} 
                    accept="image/*" 
                  />
                  
                  <div className="h-full rounded-[2.3rem] overflow-hidden bg-slate-50/50 dark:bg-slate-900/50 flex flex-col items-center justify-center relative transition-colors">
                    {!previewUrl ? (
                      <div className="p-8 flex flex-col items-center text-center">
                        <div className={`w-20 h-20 bg-white dark:bg-dark-bg rounded-3xl flex items-center justify-center mb-6 shadow-xl shadow-slate-200/50 dark:shadow-none transition-transform duration-500 border border-slate-100 dark:border-dark-border ${isDragging ? 'scale-110' : ''}`}>
                          <Upload className={`w-8 h-8 transition-colors ${isDragging ? 'text-medical-600' : 'text-slate-400'}`} />
                        </div>
                        <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-2">{t('upload.drop_here')}</h3>
                        <p className="text-xs text-slate-400 dark:text-dark-muted mb-8 max-w-[200px]">{t('upload.supports')}</p>
                        
                        <div className="btn-primary text-sm px-8 py-3">
                          {t('upload.browse')}
                        </div>
                      </div>
                    ) : (
                      <>
                        <img src={previewUrl} alt="Scan Preview" className="w-full h-full object-cover" />
                        <div className="absolute inset-0 bg-slate-900/40 opacity-0 hover:opacity-100 transition-opacity flex items-center justify-center backdrop-blur-sm">
                          <button 
                            onClick={(e) => {
                              e.stopPropagation();
                              setPreviewUrl(null); 
                              setSelectedImage(null);
                            }}
                            className="bg-white text-rose-600 px-6 py-2 rounded-xl font-bold hover:bg-rose-50 transition-all shadow-2xl flex items-center gap-2"
                          >
                            <X className="w-4 h-4" />
                            Discard Image
                          </button>
                        </div>
                      </>
                    )}
                    
                    {/* Dragging indicator */}
                    <AnimatePresence>
                      {isDragging && (
                        <motion.div 
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          exit={{ opacity: 0 }}
                          className="absolute inset-0 bg-medical-600/10 backdrop-blur-[2px] flex items-center justify-center pointer-events-none"
                        >
                          <div className="bg-white dark:bg-dark-bg p-6 rounded-3xl shadow-2xl flex flex-col items-center gap-4">
                            <Upload className="w-12 h-12 text-medical-600 animate-bounce" />
                            <span className="font-bold text-medical-700 dark:text-medical-400">{t('upload.release')}</span>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </div>
              </div>

              {/* Patient Metadata Form */}
              <div className="space-y-4">
                <label className="text-sm font-black text-slate-400 dark:text-dark-muted uppercase tracking-widest block px-2">{t('upload.step2')}</label>
                <div className="bg-white dark:bg-dark-card rounded-[2.5rem] p-8 border border-slate-200 dark:border-dark-border shadow-xl shadow-slate-200/30 dark:shadow-none space-y-6 h-[420px] overflow-y-auto scrollbar-thin transition-colors">
                  <div className="grid grid-cols-1 gap-6">
                    <div className="space-y-2">
                      <label className="text-[10px] font-black text-slate-400 dark:text-dark-muted uppercase tracking-widest flex items-center gap-2">
                        <Calendar className="w-3 h-3 text-indigo-500" /> Patient Age
                      </label>
                      <input 
                        type="number" name="age" value={metadata.age} onChange={handleMetadataChange}
                        placeholder="e.g. 65"
                        className="w-full bg-slate-50 dark:bg-dark-bg border border-slate-100 dark:border-dark-border rounded-xl px-4 py-3 text-sm outline-none focus:ring-2 focus:ring-medical-500 dark:text-white transition-all shadow-inner"
                      />
                    </div>

                    <div className="space-y-2">
                      <label className="text-[10px] font-black text-slate-400 dark:text-dark-muted uppercase tracking-widest flex items-center gap-2">
                        <MapPin className="w-3 h-3 text-indigo-500" /> Body Location
                      </label>
                      <input 
                        type="text" name="location" value={metadata.location} onChange={handleMetadataChange}
                        placeholder="e.g. Lower back"
                        className="w-full bg-slate-50 dark:bg-dark-bg border border-slate-100 dark:border-dark-border rounded-xl px-4 py-3 text-sm outline-none focus:ring-2 focus:ring-medical-500 dark:text-white transition-all shadow-inner"
                      />
                    </div>

                    <div className="space-y-2">
                      <label className="text-[10px] font-black text-slate-400 dark:text-dark-muted uppercase tracking-widest flex items-center gap-2">
                        <Clock className="w-3 h-3 text-indigo-500" /> Duration
                      </label>
                      <input 
                        type="text" name="duration" value={metadata.duration} onChange={handleMetadataChange}
                        placeholder="e.g. 2 months"
                        className="w-full bg-slate-50 dark:bg-dark-bg border border-slate-100 dark:border-dark-border rounded-xl px-4 py-3 text-sm outline-none focus:ring-2 focus:ring-medical-500 dark:text-white transition-all shadow-inner"
                      />
                    </div>

                    <div className="space-y-4 pt-2">
                      <label className="text-[10px] font-black text-slate-400 dark:text-dark-muted uppercase tracking-widest flex items-center gap-2">
                        <AlertTriangle className="w-3 h-3 text-rose-500" /> Symptom Red Flags
                      </label>
                      <div className="grid grid-cols-1 gap-3">
                        {[
                          { id: 'itching', label: 'Itching / Pruritus' },
                          { id: 'bleeding', label: 'Bleeding / Scabbing' },
                          { id: 'rapid_change', label: 'Rapid Change (Size/Color)' }
                        ].map(flag => (
                          <label key={flag.id} className="flex items-center gap-3 cursor-pointer group">
                            <div className="relative flex items-center">
                              <input 
                                type="checkbox" name={flag.id} checked={metadata[flag.id]} onChange={handleMetadataChange}
                                className="sr-only peer"
                              />
                              <div className="w-10 h-6 bg-slate-100 dark:bg-slate-800 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-medical-600"></div>
                            </div>
                            <span className="text-xs font-bold text-slate-600 dark:text-dark-muted group-hover:text-slate-900 dark:group-hover:text-white transition-colors">{flag.label}</span>
                          </label>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex flex-col items-center gap-6">
                  <button 
                    onClick={handleAnalyze}
                    disabled={isAnalyzing || !selectedImage}
                    aria-label={isAnalyzing ? "Analyzing image" : "Run diagnostic triage"}
                    className={`w-full md:w-auto min-w-[360px] py-5 px-12 bg-medical-600 hover:bg-medical-700 focus:ring-4 focus:ring-medical-500/30 outline-none text-white rounded-[2rem] font-black text-xl shadow-2xl shadow-medical-500/40 transition-all active:scale-[0.98] flex items-center justify-center gap-4 ${
                      isAnalyzing || !selectedImage ? 'opacity-50 cursor-not-allowed' : ''
                    }`}
                  >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="w-6 h-6 animate-spin" />
                    <span>{t('upload.multimodal_analysis')}</span>
                  </>
                ) : (
                  <>
                    Run Diagnostic Triage
                    <Search className="w-6 h-6" />
                  </>
                )}
              </button>

              <button 
                onClick={handleSampleImage}
                className="text-xs font-bold text-slate-400 dark:text-dark-muted hover:text-medical-600 transition-colors uppercase tracking-widest flex items-center gap-2"
              >
                <Microscope className="w-4 h-4" /> Load Clinical Sample for Demo
              </button>
            </div>
            
            <AnimatePresence>
              {error && (
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="mt-8 p-6 bg-rose-50 dark:bg-rose-900/20 border border-rose-100 dark:border-rose-900/30 rounded-3xl flex gap-4 items-center shadow-lg"
                >
                  <AlertCircle className="w-6 h-6 text-rose-600" />
                  <p className="text-sm text-rose-800 dark:text-rose-300 font-medium">{error}</p>
                </motion.div>
              )}
            </AnimatePresence>
          </ScrollReveal>
        </div>

        {/* ── Right Column: Context ───────────────────────────────────────── */}
        <aside className="w-full lg:w-[380px] space-y-8">
           <ScrollReveal direction="right">
              <div className="bg-slate-900 rounded-[2.5rem] p-10 text-white relative overflow-hidden shadow-2xl">
                <div className="relative z-10">
                  <Activity className="w-12 h-12 text-medical-400 mb-6" />
                  <h4 className="text-xl font-bold mb-4 tracking-tight">{t('upload.holistic')}</h4>
                  <p className="text-sm text-slate-400 leading-relaxed font-medium mb-6">
                    Our AI combines state-of-the-art vision models with patient clinical context. By providing symptoms, you activate our **Contextual Risk Engine** for a more accurate assessment.
                  </p>
                  <div className="flex items-center gap-2 px-4 py-2 bg-white/5 rounded-xl border border-white/10 text-[10px] font-bold text-emerald-400 uppercase tracking-widest">
                    <ShieldCheck className="w-4 h-4" /> Multimodal Engine Ready
                  </div>
                </div>
                {/* Decorative blob */}
                <div className="absolute top-0 right-0 w-32 h-32 bg-medical-500/10 rounded-full -mr-16 -mt-16 blur-3xl"></div>
              </div>
           </ScrollReveal>

           <ScrollReveal direction="right" delay={0.2}>
             <div className="bg-white dark:bg-dark-card rounded-[2.5rem] p-10 border border-slate-200 dark:border-dark-border shadow-xl shadow-slate-200/40 dark:shadow-none transition-colors">
               <h4 className="text-lg font-bold text-slate-900 dark:text-white mb-6 tracking-tight">{t('upload.safety_guidelines')}</h4>
               <ul className="space-y-5">
                 {[
                   { label: "Clear Focus", desc: "Keep the camera steady and in focus." },
                   { label: "Natural Light", desc: "Best results under indirect sunlight." },
                   { label: "Scale Reference", desc: "Include a ruler or coin if possible." }
                 ].map((item, i) => (
                   <li key={i} className="flex gap-4 group">
                     <div className="w-8 h-8 rounded-xl bg-slate-50 dark:bg-dark-bg flex items-center justify-center flex-shrink-0 group-hover:bg-medical-50 dark:group-hover:bg-indigo-900/30 transition-colors">
                       <CheckCircle2 className="w-4 h-4 text-medical-600 dark:text-medical-400" />
                     </div>
                     <div>
                       <div className="font-bold text-slate-900 dark:text-white text-sm">{item.label}</div>
                       <p className="text-[10px] text-slate-500 dark:text-dark-muted font-medium">{item.desc}</p>
                     </div>
                   </li>
                 ))}
               </ul>
             </div>
           </ScrollReveal>
        </aside>
      </div>
    </div>
  );
};

export default UploadPage;
