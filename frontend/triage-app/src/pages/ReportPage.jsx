import React, { useRef } from 'react';
import { useLocation, Navigate, useNavigate } from 'react-router-dom';
import { 
  Printer, Download, ArrowLeft, Activity, 
  ShieldCheck, AlertTriangle, Info, Calendar, MapPin, CheckCircle2 
} from 'lucide-react';
import domtoimage from 'dom-to-image';
import { jsPDF } from 'jspdf';

const ReportPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const reportRef = useRef();

  const { result, originalImage, metadata } = location.state || {};

  if (!result) {
    return <Navigate to="/upload" replace />;
  }

  const backendUrl = "http://127.0.0.1:8001";

  // Overlay or Pure heatmap logic
  const getHeatmapUrl = () => {
    const path = result.xai_reports?.gradcampp || result.gradcam_url;
    return path ? `${backendUrl}${path}` : null;
  };

  const handleBack = () => {
    navigate('/results', { state: { result, originalImage, metadata } });
  };

  const handlePrint = () => {
    window.print();
  };

  const handleDownloadPdf = async () => {
    const element = reportRef.current;
    if (!element) return;
    
    try {
      // Temporarily override the styling during capture to prevent mx-auto from shifting the content in the PDF
      const dataUrl = await domtoimage.toPng(element, { 
        quality: 0.98, 
        bgcolor: '#ffffff',
        style: {
          margin: '0',
          left: '0',
          top: '0',
          position: 'relative'
        }
      });
      
      const pdf = new jsPDF('p', 'mm', 'a4');
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const pdfHeight = (element.offsetHeight * pdfWidth) / element.offsetWidth;
      
      let heightLeft = pdfHeight;
      let position = 0;

      // Add first page
      pdf.addImage(dataUrl, 'PNG', 0, position, pdfWidth, pdfHeight);
      heightLeft -= pageHeight;

      // Add subsequent pages if the content is taller than one A4 page
      while (heightLeft > 0) {
        position -= pageHeight;
        pdf.addPage();
        pdf.addImage(dataUrl, 'PNG', 0, position, pdfWidth, pdfHeight);
        heightLeft -= pageHeight;
      }

      pdf.save(`medical_triage_report_${Date.now()}.pdf`);
    } catch (error) {
      console.error('Error generating PDF:', error);
      alert('Failed to generate PDF. You can try the Print button instead.');
    }
  };

  const riskConfig = {
    "LOW": { 
      color: "text-emerald-700", 
      bg: "bg-emerald-50",
      icon: <ShieldCheck className="w-8 h-8 text-emerald-600" />
    },
    "MEDIUM": { 
      color: "text-amber-700", 
      bg: "bg-amber-50",
      icon: <Activity className="w-8 h-8 text-amber-600" />
    },
    "HIGH": { 
      color: "text-rose-700", 
      bg: "bg-rose-50",
      icon: <AlertTriangle className="w-8 h-8 text-rose-600" />
    },
    "UNCERTAIN": { 
      color: "text-slate-700", 
      bg: "bg-slate-50",
      icon: <Info className="w-8 h-8 text-slate-600" />
    }
  };

  const currentRisk = riskConfig[result.risk_level?.toUpperCase()] || riskConfig["UNCERTAIN"];

  return (
    <div className="min-h-screen bg-slate-100 p-8">
      {/* Non-printable action bar */}
      <div className="max-w-4xl mx-auto mb-6 flex justify-between items-center print:hidden">
        <button 
          onClick={handleBack}
          className="flex items-center gap-2 text-slate-600 hover:text-slate-900 transition-colors"
        >
          <ArrowLeft className="w-5 h-5" /> Back to Results
        </button>
        <div className="flex gap-4">
          <button 
            onClick={handlePrint}
            className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-200 text-slate-700 rounded-xl hover:bg-slate-50 shadow-sm transition-all font-medium"
          >
            <Printer className="w-4 h-4" /> Print
          </button>
          <button 
            onClick={handleDownloadPdf}
            className="flex items-center gap-2 px-4 py-2 bg-medical-600 text-white rounded-xl hover:bg-medical-700 shadow-sm transition-all font-medium"
          >
            <Download className="w-4 h-4" /> Download PDF
          </button>
        </div>
      </div>

      {/* Printable Report Container */}
      <div 
        ref={reportRef} 
        className="max-w-4xl mx-auto bg-white shadow-xl rounded-xl p-12 print:shadow-none print:p-0 print:m-0"
        style={{ minHeight: '297mm' }} // Roughly A4 height to ensure good layout
      >
        {/* Header */}
        <div className="border-b-2 border-slate-100 pb-6 mb-8 flex justify-between items-end">
          <div>
            <h1 className="text-3xl font-black text-slate-900 mb-1">Medical Skin Triage AI</h1>
            <p className="text-slate-500 font-medium tracking-wide">CONFIDENTIAL PATIENT REPORT</p>
          </div>
          <div className="text-right">
            <p className="text-slate-600 font-medium">Date: {new Date().toLocaleDateString()}</p>
            <p className="text-slate-400 text-sm">ID: {result.request_id || 'N/A'}</p>
          </div>
        </div>

        {/* Patient Info */}
        <div className="mb-10">
          <h2 className="text-lg font-bold text-slate-900 mb-4 uppercase tracking-wider border-l-4 border-medical-500 pl-3">Patient Information</h2>
          <div className="bg-slate-50 rounded-xl p-6 grid grid-cols-2 gap-y-4 gap-x-8">
            <div className="flex items-center gap-3">
              <Calendar className="w-5 h-5 text-slate-400" />
              <div>
                <p className="text-xs text-slate-500 uppercase font-bold">Age</p>
                <p className="text-slate-800 font-medium">{metadata?.age || 'Not provided'} years</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <MapPin className="w-5 h-5 text-slate-400" />
              <div>
                <p className="text-xs text-slate-500 uppercase font-bold">Location</p>
                <p className="text-slate-800 font-medium">{metadata?.location || 'Not provided'}</p>
              </div>
            </div>
            <div className="col-span-2 mt-2">
              <p className="text-xs text-slate-500 uppercase font-bold mb-1">Reported Symptoms</p>
              <div className="flex gap-2">
                {metadata?.rapid_change && (
                  <span className="px-3 py-1 bg-rose-100 text-rose-700 rounded-full text-xs font-bold">Rapidly Changing</span>
                )}
                {metadata?.bleeding && (
                  <span className="px-3 py-1 bg-rose-100 text-rose-700 rounded-full text-xs font-bold">Bleeding</span>
                )}
                {!metadata?.rapid_change && !metadata?.bleeding && (
                  <span className="text-slate-600 italic text-sm">No concerning symptoms reported.</span>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Visual Assessment */}
        <div className="mb-10 page-break-inside-avoid">
          <h2 className="text-lg font-bold text-slate-900 mb-4 uppercase tracking-wider border-l-4 border-medical-500 pl-3">Visual Assessment</h2>
          <div className="grid grid-cols-2 gap-6">
            <div>
              <p className="text-sm font-bold text-slate-500 mb-2 text-center">Original Image</p>
              <div className="rounded-xl overflow-hidden border border-slate-200 aspect-square bg-slate-100 flex items-center justify-center">
                {originalImage ? (
                  <img src={originalImage} alt="Original" className="w-full h-full object-cover" crossOrigin="anonymous" />
                ) : (
                  <span className="text-slate-400">No image available</span>
                )}
              </div>
            </div>
            <div>
              <p className="text-sm font-bold text-slate-500 mb-2 text-center">AI Heatmap (Attention)</p>
              <div className="rounded-xl overflow-hidden border border-slate-200 aspect-square bg-slate-100 flex items-center justify-center">
                {getHeatmapUrl() ? (
                  <img src={getHeatmapUrl()} alt="Heatmap" className="w-full h-full object-cover" crossOrigin="anonymous" />
                ) : (
                  <span className="text-slate-400">Heatmap not available</span>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* AI Analysis */}
        <div className="mb-10 page-break-inside-avoid">
          <h2 className="text-lg font-bold text-slate-900 mb-4 uppercase tracking-wider border-l-4 border-medical-500 pl-3">AI Analysis Results</h2>
          <div className={`rounded-2xl p-6 flex items-center gap-6 border ${currentRisk.bg} border-slate-100`}>
            <div className="bg-white p-4 rounded-xl shadow-sm">
              {currentRisk.icon}
            </div>
            <div className="flex-1">
              <p className="text-sm font-bold text-slate-500 uppercase mb-1">Primary Finding</p>
              <h3 className={`text-2xl font-black ${currentRisk.color}`}>{result.full_name}</h3>
              <p className="text-slate-600 font-medium mt-1">Risk Level: <span className="font-bold">{result.risk_level?.toUpperCase()}</span></p>
            </div>
            <div className="text-right border-l-2 border-slate-200 pl-6 py-2">
              <p className="text-sm font-bold text-slate-500 uppercase mb-1">AI Certainty</p>
              <p className="text-4xl font-black text-slate-800">{(result.confidence * 100).toFixed(0)}%</p>
            </div>
          </div>
        </div>

        {/* Clinical Notes & Recommendations */}
        <div className="mb-10 page-break-inside-avoid">
          <h2 className="text-lg font-bold text-slate-900 mb-4 uppercase tracking-wider border-l-4 border-medical-500 pl-3">Clinical Notes & Recommendations</h2>
          
          <div className="mb-6">
            <h4 className="font-bold text-slate-800 mb-2">Description</h4>
            <p className="text-slate-600 leading-relaxed text-justify">{result.clinical_description}</p>
          </div>

          <div className="bg-medical-50 border border-medical-100 rounded-xl p-6">
            <div className="flex items-start gap-4">
              <CheckCircle2 className="w-6 h-6 text-medical-600 flex-shrink-0 mt-1" />
              <div>
                <h4 className="font-bold text-medical-900 mb-2">Recommendation</h4>
                <p className="text-medical-800 leading-relaxed font-medium">{result.recommendation}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Disclaimer / Footer */}
        <div className="mt-16 pt-8 border-t-2 border-slate-100 text-sm text-slate-500 text-justify leading-relaxed page-break-inside-avoid">
          <p className="font-bold text-slate-700 mb-2 uppercase">Important Medical Disclaimer</p>
          <p>
            This report was generated by an Artificial Intelligence system and is intended for informational and triage purposes only. 
            <strong> It does not constitute a definitive medical diagnosis.</strong> The AI analyzes statistical patterns in images but cannot 
            replace the clinical judgment of a board-certified dermatologist. If you have any concerns about a skin lesion—especially if it is 
            growing, bleeding, changing color, or causing pain—you must consult a healthcare professional immediately regardless of the AI's 
            certainty score.
          </p>
          <div className="mt-12 flex justify-between items-end">
            <div>
              <div className="border-b border-slate-400 w-48 mb-2"></div>
              <p className="text-xs font-bold uppercase">Physician Signature</p>
            </div>
            <p className="text-xs">Generated by Medical Skin Triage API v1.1</p>
          </div>
        </div>

      </div>

      {/* Print styles injected directly */}
      <style>{`
        @media print {
          @page { margin: 0; }
          body { 
            background: white; 
            -webkit-print-color-adjust: exact; 
            print-color-adjust: exact; 
          }
          .page-break-inside-avoid {
            page-break-inside: avoid;
          }
        }
      `}</style>
    </div>
  );
};

export default ReportPage;
