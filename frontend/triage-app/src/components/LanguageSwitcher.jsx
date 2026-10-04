import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { motion, AnimatePresence } from 'framer-motion';
import { Globe, ChevronDown } from 'lucide-react';

const languages = [
  { code: 'en', name: 'English' },
  { code: 'hi', name: 'हिंदी' },
  { code: 'kn', name: 'ಕನ್ನಡ' },
  { code: 'mr', name: 'मराठी' },
  { code: 'ta', name: 'தமிழ்' },
  { code: 'te', name: 'తెలుగు' }
];

const LanguageSwitcher = () => {
  const { i18n } = useTranslation();
  const [isOpen, setIsOpen] = useState(false);

  const currentLang = languages.find(l => l.code === i18n.language) || languages[0];

  const changeLanguage = (code) => {
    i18n.changeLanguage(code);
    localStorage.setItem('i18nextLng', code);
    setIsOpen(false);
  };

  return (
    <div className="relative">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-2 bg-slate-100 dark:bg-slate-800 rounded-full hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
      >
        <Globe className="w-4 h-4 text-medical-600 dark:text-medical-400" />
        <span className="text-xs font-bold text-slate-700 dark:text-slate-300">{currentLang.name}</span>
        <ChevronDown className="w-3 h-3 text-slate-400" />
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div 
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="absolute right-0 mt-2 w-40 bg-white dark:bg-dark-card border border-slate-100 dark:border-dark-border rounded-2xl shadow-xl z-50 overflow-hidden"
          >
            {languages.map((lang) => (
              <button
                key={lang.code}
                onClick={() => changeLanguage(lang.code)}
                className={`w-full text-left px-4 py-3 text-sm font-medium hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors ${
                  i18n.language === lang.code 
                    ? 'text-medical-600 dark:text-medical-400 bg-medical-50 dark:bg-indigo-900/20' 
                    : 'text-slate-700 dark:text-slate-300'
                }`}
              >
                {lang.name}
              </button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default LanguageSwitcher;
