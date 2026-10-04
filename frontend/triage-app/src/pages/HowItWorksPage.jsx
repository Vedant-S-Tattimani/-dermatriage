import React, { useRef } from 'react';
import { Cpu, Zap, Search, ShieldCheck, Microscope, Database, FileText, AlertTriangle, Scan, Activity, Thermometer } from 'lucide-react';
import { motion, useScroll, useTransform, useSpring } from 'framer-motion';
import CurvedLoop from '../components/CurvedLoop';

const ProcessStep = ({ icon: Icon, title, description, isLast }) => (
  <div className="flex flex-col items-center text-center relative group">
    <div className="w-20 h-20 bg-white dark:bg-dark-card shadow-xl rounded-2xl flex items-center justify-center mb-6 border border-slate-100 dark:border-dark-border z-10 transition-transform group-hover:scale-110 duration-500">
      <Icon className="w-10 h-10 text-medical-600 dark:text-medical-400" />
    </div>
    <h4 className="font-bold text-slate-900 dark:text-white mb-2">{title}</h4>
    <p className="text-xs text-slate-500 dark:text-dark-muted max-w-[150px] font-medium leading-relaxed">{description}</p>
    {!isLast && (
      <div className="hidden lg:block absolute top-10 left-[100%] w-full h-[1px] bg-gradient-to-r from-medical-200 dark:from-medical-900/50 to-transparent -translate-x-1/2 -z-0"></div>
    )}
  </div>
);

const HowItWorksPage = () => {
  const sectionRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ["start end", "end start"]
  });

  const smoothProgress = useSpring(scrollYProgress, {
    stiffness: 100,
    damping: 30,
    restDelta: 0.001
  });

  const modelY = useTransform(smoothProgress, [0, 1], [80, -80]);
  const modelScale = useTransform(smoothProgress, [0, 0.5, 1], [0.9, 1.05, 0.95]);
  const modelRotate = useTransform(smoothProgress, [0, 1], [-3, 3]);

  return (
    <div className="pb-32 overflow-hidden bg-white dark:bg-dark-bg transition-colors duration-300">
      
      {/* ── Hero / Pipeline Section ────────────────────────────────────────── */}
      <section className="bg-slate-900 dark:bg-black text-white py-32 relative overflow-hidden">
        <div className="absolute inset-0 opacity-10 pointer-events-none">
          <div className="absolute top-1/4 left-1/4 w-[400px] h-[400px] bg-medical-500 rounded-full blur-[150px]"></div>
        </div>

        <div className="section-container relative z-10">
          <div className="text-center mb-24 space-y-4">
            <motion.h1 
              initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
              className="text-5xl md:text-7xl font-black tracking-tighter"
            >
              The Multimodal <span className="text-medical-400">Pipeline.</span>
            </motion.h1>
            <p className="text-slate-400 text-lg max-w-2xl mx-auto font-medium">The journey from raw image capture to clinical-grade pattern analysis.</p>
          </div>
          
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-12 lg:gap-4 max-w-6xl mx-auto">
            <ProcessStep icon={Scan} title="Segmentation" description="AI isolates the lesion and crops background noise." />
            <ProcessStep icon={Search} title="Multimodal" description="Visual data combined with patient clinical metadata." />
            <ProcessStep icon={Cpu} title="Inference" description="ConvNeXt transformer extracts morphological features." />
            <ProcessStep icon={Zap} title="Calibration" description="Temperature scaling ensures realistic confidence." />
            <ProcessStep icon={FileText} title="Analytics" description="Advanced XAI saliency and risk profiling." isLast />
          </div>
        </div>
      </section>

      {/* Interactive Loop Divider */}
      <div className="py-16 bg-white dark:bg-dark-bg transition-colors">
        <CurvedLoop 
          marqueeText="CONVNEXT TRANSFORMERS ✦ DEEPLABV3 SEGMENTATION ✦ TEMPERATURE SCALING ✦ GRAD-CAM++ ✦"
          speed={2}
          curveAmount={100}
          direction="left"
          interactive={true}
        />
      </div>

      {/* ── Technical Deep Dive ────────────────────────────────────────────── */}
      <section ref={sectionRef} className="py-32 bg-white dark:bg-dark-bg relative transition-colors">
        <div className="section-container">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-24 items-center">
            
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="space-y-12"
            >
              <div>
                <span className="text-medical-600 dark:text-medical-400 font-black uppercase tracking-[0.3em] text-[10px] mb-4 block">Medical AI Stack</span>
                <h2 className="text-4xl md:text-5xl font-black text-slate-900 dark:text-white tracking-tighter leading-[1.05]">Next-Generation Architecture.</h2>
              </div>

              <div className="space-y-10">
                {[
                  { 
                    title: "ConvNeXt Tiny Backbone", 
                    icon: <Database className="w-6 h-6" />,
                    desc: "A pure convolutional network that competes with transformers. Fine-tuned on 10,000+ clinical skin samples for precise pattern recognition."
                  },
                  { 
                    title: "Advanced XAI Saliency", 
                    icon: <Zap className="w-6 h-6" />,
                    desc: "We utilize multiple Explainable AI methods like Grad-CAM++ and Eigen-CAM to provide verifiable visual evidence for every diagnostic assessment."
                  },
                  { 
                    title: "Confidence Calibration", 
                    icon: <Thermometer className="w-6 h-6" />,
                    desc: "Using Temperature Scaling to align model output probabilities with actual empirical accuracy, preventing overconfident clinical errors."
                  }
                ].map((item, i) => (
                  <div key={i} className="flex gap-6 group">
                    <div className="flex-shrink-0 w-14 h-14 bg-slate-50 dark:bg-dark-card rounded-2xl flex items-center justify-center border border-slate-100 dark:border-dark-border group-hover:bg-medical-600 transition-all duration-500">
                      <div className="text-medical-600 dark:text-medical-400 group-hover:text-white transition-colors">{item.icon}</div>
                    </div>
                    <div>
                      <h3 className="text-xl font-bold mb-2 text-slate-900 dark:text-white tracking-tight">{item.title}</h3>
                      <p className="text-slate-500 dark:text-dark-muted leading-relaxed text-sm font-medium">{item.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
            
            {/* 3D Visualizer */}
            <motion.div 
              style={{ y: modelY, scale: modelScale, rotateZ: modelRotate }}
              className="relative h-[600px] w-full flex items-center justify-center"
            >
              <div className="relative w-full h-full rounded-[4rem] overflow-hidden border border-slate-200 dark:border-dark-border bg-white dark:bg-dark-card shadow-2xl transition-colors">
                <iframe 
                  src='https://my.spline.design/untitled-eyh5rrPp1NEKbXXyy10Qltfl/' 
                  frameBorder='0' width='100%' height='100%'
                  className="w-full h-full pointer-events-none opacity-80 dark:opacity-40"
                  style={{ background: 'transparent' }}
                  title="Neural Network Map"
                ></iframe>

                <div className="absolute inset-0 bg-gradient-to-t from-white dark:from-dark-card via-transparent to-transparent pointer-events-none"></div>

                <div className="absolute bottom-10 left-1/2 -translate-x-1/2 z-20 flex items-center gap-4 bg-white/90 dark:bg-dark-bg/90 backdrop-blur-md px-6 py-3 rounded-2xl border border-white dark:border-dark-border shadow-2xl transition-colors">
                  <div className="w-8 h-8 bg-medical-600 rounded-xl flex items-center justify-center">
                    <Activity className="w-4 h-4 text-white" />
                  </div>
                  <span className="text-[10px] font-black text-slate-900 dark:text-white uppercase tracking-[0.2em]">Neural Engine Map</span>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* ── Ethics & Protocol ──────────────────────────────────────────────── */}
      <section className="py-32 bg-slate-50 dark:bg-dark-bg transition-colors duration-300">
        <div className="section-container">
          <div className="bg-white dark:bg-dark-card border border-slate-200 dark:border-dark-border rounded-[4rem] p-12 md:p-20 shadow-xl transition-colors">
            <div className="flex flex-col items-center text-center mb-20">
              <div className="w-20 h-20 bg-amber-50 dark:bg-amber-900/10 rounded-3xl flex items-center justify-center mb-8 border border-amber-100 dark:border-amber-900/30 shadow-lg">
                <AlertTriangle className="w-10 h-10 text-amber-600" />
              </div>
              <h2 className="text-4xl md:text-5xl font-black text-slate-900 dark:text-white tracking-tighter mb-6 uppercase">Safety Protocols</h2>
              <p className="text-slate-500 dark:text-dark-muted max-w-2xl text-lg font-medium">NeuralSkin AI is a triage prioritization system. It identifies high-risk morphological patterns to assist in early clinical evaluation.</p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-16">
              {[
                { label: "Pattern Detection", desc: "Our models identify statistical correlations in skin pathology. This is an assessment of probability, not a definitive medical diagnosis." },
                { label: "Zero Advice", desc: "The system does not recommend surgery, medication, or home remedies. It exclusively classifies risk into clinical urgency tiers." },
                { label: "Professional Vetting", desc: "Regardless of AI certainty, any growing, bleeding, or asymmetrical lesion requires evaluation by a board-certified specialist." }
              ].map((item, i) => (
                <div key={i} className="space-y-6">
                  <h4 className="font-black text-slate-900 dark:text-white text-xl tracking-tight uppercase underline decoration-medical-500 decoration-4 underline-offset-8">{item.label}</h4>
                  <p className="text-sm text-slate-500 dark:text-dark-muted leading-relaxed font-medium">{item.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default HowItWorksPage;
