import React, { useState, useEffect } from 'react';
import { 
  Activity, TrendingUp, AlertCircle, PieChart, 
  Clock, CheckCircle2, ArrowRight, BarChart3, 
  MessageSquare, Layers, Microscope
} from 'lucide-react';
import { motion } from 'framer-motion';
import ScrollReveal from '../components/ScrollReveal';

const MonitoringPage = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const backendUrl = "http://127.0.0.1:8001";

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const res = await fetch(`${backendUrl}/api/v1/analytics`);
        if (!res.ok) throw new Error("Failed to fetch production metrics.");
        const analytics = await res.json();
        setData(analytics);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchAnalytics();
  }, []);

  if (loading) return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 dark:bg-dark-bg">
      <div className="flex flex-col items-center gap-6">
        <div className="w-16 h-16 border-4 border-medical-600 border-t-transparent rounded-full animate-spin"></div>
        <p className="text-slate-400 font-bold uppercase tracking-widest text-xs">Initializing Analytics Engine...</p>
      </div>
    </div>
  );

  if (error) return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 dark:bg-dark-bg">
      <div className="text-center space-y-4">
        <AlertCircle className="w-12 h-12 text-rose-500 mx-auto" />
        <p className="text-slate-900 dark:text-white font-bold">{error}</p>
      </div>
    </div>
  );

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 md:py-20 bg-slate-50 dark:bg-dark-bg min-h-screen transition-colors duration-300">
      
      {/* ── Header ────────────────────────────────────────────────────────── */}
      <header className="mb-16">
        <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}>
          <div className="flex items-center gap-2 text-indigo-600 dark:text-indigo-400 font-black text-[10px] uppercase tracking-[0.3em] mb-3">
            <Activity className="w-4 h-4" />
            System Performance Dashboard
          </div>
          <h1 className="text-5xl font-black text-slate-900 dark:text-white tracking-tighter">Model Monitoring</h1>
        </motion.div>
      </header>

      {/* ── Key Metrics Grid ─────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
        {[
          { label: "Total Inferences", val: data.total_predictions, icon: <Layers className="w-5 h-5" />, color: "text-blue-600", bg: "bg-blue-50" },
          { label: "Avg Confidence", val: `${(data.avg_system_confidence * 100).toFixed(1)}%`, icon: <TrendingUp className="w-5 h-5" />, color: "text-emerald-600", bg: "bg-emerald-50" },
          { label: "Uncertain Cases", val: data.uncertain_count, icon: <AlertCircle className="w-5 h-5" />, color: "text-amber-600", bg: "bg-amber-50" },
          { label: "Feedback Loop", val: data.total_feedback, icon: <MessageSquare className="w-5 h-5" />, color: "text-indigo-600", bg: "bg-indigo-50" },
        ].map((stat, i) => (
          <motion.div 
            key={i}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className="bg-white dark:bg-dark-card p-8 rounded-[2rem] border border-slate-200 dark:border-dark-border shadow-sm flex flex-col justify-between"
          >
            <div className={`w-12 h-12 ${stat.bg} dark:bg-slate-800 rounded-2xl flex items-center justify-center ${stat.color} mb-6`}>
              {stat.icon}
            </div>
            <div>
              <div className="text-[10px] font-black text-slate-400 dark:text-dark-muted uppercase tracking-widest mb-1">{stat.label}</div>
              <div className="text-3xl font-black text-slate-900 dark:text-white tracking-tight">{stat.val}</div>
            </div>
          </motion.div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* ── Class Distribution (Chart Placeholder) ───────────────────────── */}
        <div className="lg:col-span-2">
          <ScrollReveal direction="left">
            <div className="bg-white dark:bg-dark-card rounded-[2.5rem] p-10 border border-slate-200 dark:border-dark-border shadow-xl min-h-[500px]">
              <div className="flex justify-between items-center mb-12">
                <h3 className="text-xl font-black text-slate-900 dark:text-white uppercase tracking-tighter flex items-center gap-3">
                  <BarChart3 className="w-6 h-6 text-medical-600" />
                  Prediction Drift & Accuracy
                </h3>
                <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Last 30 Days</span>
              </div>

              <div className="space-y-8">
                {data.class_distribution.map((cls, i) => (
                  <div key={i} className="space-y-3">
                    <div className="flex justify-between items-end">
                      <span className="text-sm font-black text-slate-700 dark:text-slate-300 uppercase tracking-tight">{cls.class_name}</span>
                      <span className="text-xs font-mono font-bold text-slate-400">{cls.count} hits</span>
                    </div>
                    <div className="h-4 bg-slate-100 dark:bg-dark-bg rounded-full overflow-hidden flex">
                      <motion.div 
                        initial={{ width: 0 }}
                        animate={{ width: `${(cls.count / data.total_predictions) * 100}%` }}
                        transition={{ duration: 1, delay: 0.5 + (i * 0.1) }}
                        className="h-full bg-medical-600 shadow-lg shadow-medical-500/20"
                      />
                    </div>
                    <div className="flex justify-between text-[10px] font-bold text-slate-400">
                      <span>Frequency</span>
                      <span className="text-emerald-500">{(cls.avg_confidence * 100).toFixed(1)}% Avg. Certainty</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </ScrollReveal>
        </div>

        {/* ── Right Column: Active Learning Queue ─────────────────────────── */}
        <div className="space-y-8">
          <ScrollReveal direction="right">
            <div className="bg-slate-900 rounded-[2.5rem] p-10 text-white relative overflow-hidden shadow-2xl">
              <div className="relative z-10">
                <Microscope className="w-12 h-12 text-medical-400 mb-6" />
                <h3 className="text-2xl font-black tracking-tighter uppercase mb-4 leading-none">Active Learning</h3>
                <p className="text-slate-400 text-sm font-medium leading-relaxed mb-10">
                  The system has isolated **{data.uncertain_count}** borderline cases for expert review. Correcting these will improve the next model weights (v1.2).
                </p>
                <button className="w-full py-4 bg-white text-slate-900 rounded-2xl font-black text-sm uppercase tracking-widest hover:bg-medical-50 transition-all flex items-center justify-center gap-3">
                  Label Difficult Cases
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
              <div className="absolute top-0 right-0 w-32 h-32 bg-medical-500/10 rounded-full -mr-16 -mt-16 blur-3xl"></div>
            </div>
          </ScrollReveal>

          <ScrollReveal direction="right" delay={0.2}>
             <div className="bg-white dark:bg-dark-card rounded-[2.5rem] p-10 border border-slate-200 dark:border-dark-border shadow-xl min-h-[300px]">
               <h3 className="text-lg font-black text-slate-900 dark:text-white uppercase tracking-tighter mb-8 flex items-center gap-3">
                 <CheckCircle2 className="w-5 h-5 text-emerald-500" />
                 Recent Feedback
               </h3>
               {data.recent_feedback.length === 0 ? (
                 <p className="text-xs text-slate-400 italic">No clinician feedback in queue.</p>
               ) : (
                 <ul className="space-y-6">
                   {data.recent_feedback.map((f, i) => (
                     <li key={i} className="flex gap-4 border-b border-slate-50 dark:border-dark-border pb-4 last:border-0 transition-colors">
                       <div className="w-8 h-8 rounded-xl bg-slate-50 dark:bg-dark-bg flex items-center justify-center text-slate-400 flex-shrink-0">
                         <Clock className="w-4 h-4" />
                       </div>
                       <div>
                         <div className="text-xs font-black text-slate-900 dark:text-white uppercase tracking-tight">Case {f.request_id.split('-')[0]}</div>
                         <p className="text-[10px] text-medical-600 dark:text-medical-400 font-bold">Correction: {f.correct_class}</p>
                       </div>
                     </li>
                   ))}
                 </ul>
               )}
             </div>
          </ScrollReveal>
        </div>
      </div>

    </div>
  );
};

export default MonitoringPage;
