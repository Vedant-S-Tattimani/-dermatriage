import React from 'react';
import { Link } from 'react-router-dom';
import { 
  Shield, Zap, Activity, Microscope, 
  ArrowRight, CheckCircle2, Globe, HeartPulse,
  Smartphone, Award, Brain, Lock
} from 'lucide-react';
import { motion } from 'framer-motion';
import ScrollReveal from '../components/ScrollReveal';

const LandingPage = () => {
  return (
    <div className="bg-white dark:bg-dark-bg transition-colors duration-300 overflow-hidden">
      
      {/* ── Hero Section ─────────────────────────────────────────────────── */}
      <section className="relative pt-20 pb-32 md:pt-32 md:pb-48 bg-slate-50 dark:bg-dark-bg/50 overflow-hidden">
        {/* Background glow effects */}
        <div className="absolute top-0 left-0 w-full h-full overflow-hidden -z-10">
          <div className="absolute top-[-10%] right-[-5%] w-[50%] h-[50%] bg-medical-500/10 dark:bg-medical-500/5 rounded-full blur-[120px]"></div>
          <div className="absolute bottom-[-10%] left-[-5%] w-[40%] h-[40%] bg-indigo-500/10 dark:bg-indigo-500/5 rounded-full blur-[100px]"></div>
        </div>

        <div className="section-container relative">
          <div className="text-center max-w-5xl mx-auto">
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              className="inline-flex items-center gap-2 px-6 py-2.5 bg-white dark:bg-dark-card border border-slate-200 dark:border-dark-border rounded-full text-medical-600 dark:text-medical-400 font-black text-[10px] uppercase tracking-[0.4em] mb-10 shadow-xl transition-colors"
            >
              <Award className="w-4 h-4" />
              Advanced Multimodal Dermatology AI
            </motion.div>
            
            <motion.h1 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1, duration: 0.8 }}
              className="text-6xl md:text-8xl font-black text-slate-900 dark:text-white leading-[0.95] tracking-tighter mb-10"
            >
              The AI Standard for <br /><span className="text-gradient">Skin Triage.</span>
            </motion.h1>
            
            <motion.p 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="text-xl md:text-2xl text-slate-500 dark:text-dark-muted font-medium mb-16 max-w-3xl mx-auto leading-relaxed"
            >
              Harnessing ConvNeXt transformers and clinical metadata to categorize lesions across 7 clinical classes with audited precision.
            </motion.p>
            
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="flex flex-col sm:flex-row items-center justify-center gap-8"
            >
              <Link to="/upload" className="btn-primary text-xl px-14 py-6 rounded-[2.5rem] w-full sm:w-auto group shadow-2xl shadow-medical-500/40">
                Launch Diagnostic Scan
                <ArrowRight className="w-6 h-6 group-hover:translate-x-2 transition-transform duration-300" />
              </Link>
              <Link to="/how-it-works" className="btn-secondary text-xl px-14 py-6 rounded-[2.5rem] dark:bg-dark-card dark:border-dark-border dark:text-white w-full sm:w-auto hover:shadow-xl transition-all">
                Learn Methodology
              </Link>
            </motion.div>
          </div>

          {/* Dynamic Preview Display */}
          <motion.div 
            initial={{ opacity: 0, y: 80 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 1 }}
            className="mt-32 relative max-w-6xl mx-auto px-4"
          >
            <div className="aspect-video bg-white dark:bg-slate-900 rounded-[4rem] shadow-[0_50px_120px_rgba(0,0,0,0.15)] dark:shadow-none border-[16px] border-slate-900 dark:border-slate-800 relative overflow-hidden group">
               <div className="absolute inset-0 bg-gradient-to-br from-medical-50 to-white dark:from-dark-bg dark:to-slate-900 flex items-center justify-center transition-colors">
                  <div className="flex flex-col items-center gap-8">
                    <Brain className="w-28 h-28 text-medical-600 dark:text-medical-400 opacity-20 animate-pulse" />
                    <div className="h-2 w-72 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                      <motion.div 
                        initial={{ width: 0 }}
                        animate={{ width: "100%" }}
                        transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                        className="h-full bg-gradient-to-r from-medical-600 to-indigo-600"
                      />
                    </div>
                  </div>
               </div>
               {/* Reflection effect */}
               <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-white/5 to-transparent pointer-events-none group-hover:translate-x-full transition-transform duration-1000"></div>
            </div>
            
            {/* Floating feature badges */}
            <div className="absolute top-1/2 -left-8 -translate-y-1/2 hidden xl:block">
              <div className="bg-white/90 dark:bg-dark-card/90 backdrop-blur-xl p-8 rounded-[2.5rem] shadow-2xl border border-white/50 dark:border-dark-border space-y-3 animate-bounce-slow">
                <div className="text-[10px] font-black text-slate-400 dark:text-dark-muted uppercase tracking-[0.2em]">Verified Recall</div>
                <div className="text-4xl font-black text-medical-600 dark:text-medical-400 leading-none">90.0%</div>
                <p className="text-[10px] text-slate-500 font-bold">Actinic Keratosis</p>
              </div>
            </div>
            <div className="absolute top-1/4 -right-8 hidden xl:block">
              <div className="bg-white/90 dark:bg-dark-card/90 backdrop-blur-xl p-8 rounded-[2.5rem] shadow-2xl border border-white/50 dark:border-dark-border space-y-3 animate-bounce-delayed">
                <div className="text-[10px] font-black text-slate-400 dark:text-dark-muted uppercase tracking-[0.2em]">Processing</div>
                <div className="text-4xl font-black text-indigo-600 dark:text-indigo-400 leading-none">~350ms</div>
                <p className="text-[10px] text-slate-500 font-bold">Production Latency</p>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ── Capabilities ──────────────────────────────────────────────────── */}
      <section className="py-40 bg-white dark:bg-dark-bg transition-colors duration-300">
        <div className="section-container">
          <div className="text-center mb-24 space-y-4">
            <h2 className="text-4xl md:text-5xl font-black text-slate-900 dark:text-white tracking-tight">Clinical Grade Foundation</h2>
            <p className="text-slate-500 dark:text-dark-muted max-w-2xl mx-auto font-medium text-lg italic">Built for accuracy, audited for transparency, deployed for global impact.</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
            {[
              { 
                title: "Multimodal Engine", 
                desc: "Analyzes visual patterns alongside patient symptoms for superior risk profiling.",
                icon: <Activity className="w-10 h-10 text-medical-600" />
              },
              { 
                title: "XAI Explainability", 
                desc: "Supports multiple saliency methods including Grad-CAM++ and Eigen-CAM.",
                icon: <Zap className="w-10 h-10 text-amber-500" />
              },
              { 
                title: "Privacy Encrypted", 
                desc: "HIPAA-ready protocol ensures all image data is processed in temporary memory.",
                icon: <Lock className="w-10 h-10 text-emerald-500" />
              }
            ].map((feature, i) => (
              <ScrollReveal key={i} delay={i * 0.2}>
                <div className="group p-12 rounded-[3.5rem] bg-slate-50 dark:bg-dark-card border border-transparent hover:border-medical-200 dark:hover:border-dark-border hover:bg-white dark:hover:bg-slate-800 hover:shadow-2xl transition-all duration-500 h-full flex flex-col items-center text-center">
                  <div className="w-20 h-20 rounded-[1.8rem] bg-white dark:bg-dark-bg flex items-center justify-center mb-10 shadow-lg group-hover:scale-110 transition-transform">
                    {feature.icon}
                  </div>
                  <h3 className="text-2xl font-black text-slate-900 dark:text-white mb-6 tracking-tight">{feature.title}</h3>
                  <p className="text-slate-500 dark:text-dark-muted leading-relaxed font-medium">{feature.desc}</p>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </div>
      </section>

      {/* ── High Intensity CTA ────────────────────────────────────────────── */}
      <section className="py-40 bg-slate-900 dark:bg-black text-white relative overflow-hidden">
        <div className="absolute inset-0 -z-10 opacity-20">
          <Globe className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[1000px] h-[1000px] text-medical-500/30 animate-pulse" />
        </div>
        
        <div className="section-container text-center relative z-10">
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            className="space-y-12"
          >
            <h2 className="text-6xl md:text-8xl font-black tracking-tighter mb-8 leading-none">Global Precision <br />Triage.</h2>
            <p className="text-2xl text-slate-400 mb-16 max-w-3xl mx-auto font-medium leading-relaxed">
              Experience the convergence of medical expertise and deep learning. Optimized for clinicians and individuals alike.
            </p>
            <Link to="/upload" className="inline-flex items-center gap-4 px-16 py-7 bg-white text-slate-900 rounded-[3rem] font-black text-2xl hover:bg-medical-50 hover:scale-105 transition-all shadow-[0_20px_50px_rgba(255,255,255,0.1)] active:scale-95">
              Launch Diagnostic Console
              <Smartphone className="w-8 h-8 text-medical-600" />
            </Link>
          </motion.div>
        </div>
      </section>

      {/* ── Global Partners Footer ───────────────────────────────────────── */}
      <footer className="py-16 bg-white dark:bg-dark-bg border-t border-slate-100 dark:border-dark-border transition-colors">
        <div className="section-container">
           <div className="flex flex-wrap justify-center gap-12 md:gap-24 grayscale dark:invert opacity-40 mb-16">
             <div className="flex items-center gap-2 font-black text-2xl tracking-tighter italic">NEURAL<span className="text-medical-500">MED</span></div>
             <div className="flex items-center gap-2 font-black text-2xl tracking-tighter italic">VISTA<span className="text-indigo-500">LABS</span></div>
             <div className="flex items-center gap-2 font-black text-2xl tracking-tighter italic">SKIN<span className="text-emerald-500">AI</span></div>
             <div className="flex items-center gap-2 font-black text-2xl tracking-tighter italic">OPTIC<span className="text-medical-600 font-bold">CORE</span></div>
          </div>
          <div className="flex flex-col md:flex-row justify-between items-center gap-8 pt-12 border-t border-slate-50 dark:border-slate-800">
            <div className="flex items-center gap-3">
              <HeartPulse className="w-8 h-8 text-medical-600 shadow-xl" />
              <span className="text-2xl font-black text-slate-900 dark:text-white uppercase tracking-tighter">NeuralSkin AI</span>
            </div>
            <p className="text-xs text-slate-400 dark:text-dark-muted font-bold tracking-[0.2em] uppercase">© 2026 NeuralSkin AI · Built for impact.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
