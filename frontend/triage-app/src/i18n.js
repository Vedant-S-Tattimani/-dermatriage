import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

import enTranslation from './locales/en.json';
import hiTranslation from './locales/hi.json';
import knTranslation from './locales/kn.json';
import mrTranslation from './locales/mr.json';
import taTranslation from './locales/ta.json';
import teTranslation from './locales/te.json';

const resources = {
  en: { translation: enTranslation },
  hi: { translation: hiTranslation },
  kn: { translation: knTranslation },
  mr: { translation: mrTranslation },
  ta: { translation: taTranslation },
  te: { translation: teTranslation }
};

const savedLang = localStorage.getItem('i18nextLng') || 'en';

i18n
  .use(initReactI18next)
  .init({
    resources,
    lng: savedLang,
    fallbackLng: 'en',
    interpolation: {
      escapeValue: false
    }
  });

export default i18n;
