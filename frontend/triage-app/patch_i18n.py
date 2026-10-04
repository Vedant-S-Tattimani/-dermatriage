import re
import os

upload_path = 'd:/medical assiastant/frontend/triage-app/src/pages/UploadPage.jsx'
with open(upload_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Add import if missing
if 'useTranslation' not in content:
    content = content.replace('import React', 'import React\nimport { useTranslation } from \'react-i18next\';')

if 'const { t } = useTranslation();' not in content:
    content = content.replace('const UploadPage = () => {', 'const UploadPage = () => {\n  const { t } = useTranslation();')

# Replacements mapping
replacements = {
    'Diagnostic Triage Intake': '{t(\'upload.intake\')}',
    'Multi-Stage': '{t(\'upload.title_multi\')}',
    'AI Assessment.': '{t(\'upload.title_ai\')}',
    'Combine visual AI pattern matching with clinical metadata for a holistic skin risk profile.': '{t(\'upload.subtitle\')}',
    'Step 1: Clinical Capture': '{t(\'upload.step1\')}',
    'Drop Scan Here': '{t(\'upload.drop_here\')}',
    'Supports JPEG, PNG, WebP (Max 10MB). Macro focus recommended.': '{t(\'upload.supports\')}',
    'Browse Files': '{t(\'upload.browse\')}',
    'Discard Image': '{t(\'upload.discard\')}',
    'Release to Upload': '{t(\'upload.release\')}',
    'Step 2: Clinical Context': '{t(\'upload.step2\')}',
    'Patient Age': '{t(\'upload.patient_age\')}',
    'Body Location': '{t(\'upload.body_location\')}',
    'Duration': '{t(\'upload.duration\')}',
    'Symptom Red Flags': '{t(\'upload.symptom_flags\')}',
    'Itching / Pruritus': '{t(\'upload.itching\')}',
    'Bleeding / Scabbing': '{t(\'upload.bleeding\')}',
    'Rapid Change (Size/Color)': '{t(\'upload.rapid_change\')}',
    'Run Diagnostic Triage': '{t(\'upload.run_triage\')}',
    'Multimodal Analysis...': '{t(\'upload.multimodal_analysis\')}',
    'Load Clinical Sample for Demo': '{t(\'upload.load_sample\')}',
    'Holistic Triage.': '{t(\'upload.holistic\')}',
    'Our AI combines state-of-the-art vision models with patient clinical context. By providing symptoms, you activate our Contextual Risk Engine for a more accurate assessment.': '{t(\'upload.holistic_desc\')}',
    'Multimodal Engine Ready': '{t(\'upload.engine_ready\')}',
    'Safety Guidelines': '{t(\'upload.safety_guidelines\')}',
    'Clear Focus': '{t(\'upload.clear_focus\')}',
    'Keep the camera steady and in focus.': '{t(\'upload.clear_focus_desc\')}',
    'Natural Light': '{t(\'upload.natural_light\')}',
    'Best results under indirect sunlight.': '{t(\'upload.natural_light_desc\')}',
    'Scale Reference': '{t(\'upload.scale_ref\')}',
    'Include a ruler or coin if possible.': '{t(\'upload.scale_ref_desc\')}',
}

for k, v in replacements.items():
    content = content.replace(f'> {k} <', f'> {v} <')
    content = content.replace(f'>{k}<', f'>{v}<')
    content = content.replace(f'{k} <br />', f'{v} <br />')
    content = content.replace(f'>\n              {k}\n            <', f'>\n              {v}\n            <')
    content = content.replace(f'>\n                {k}\n              <', f'>\n                {v}\n              <')
    content = content.replace(f'>\n                          {k}\n                        <', f'>\n                          {v}\n                        <')

with open(upload_path, 'w', encoding='utf-8') as f:
    f.write(content)
print('UploadPage updated.')

results_path = 'd:/medical assiastant/frontend/triage-app/src/pages/ResultsPage.jsx'
with open(results_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Add import if missing
if 'useTranslation' not in content:
    content = content.replace('import React', 'import React\nimport { useTranslation } from \'react-i18next\';')

if 'const { t } = useTranslation();' not in content:
    content = content.replace('const ResultsPage = () => {', 'const ResultsPage = () => {\n  const { t } = useTranslation();')

replacements = {
    'Verification Report · Clinical AI': '{t(\'results.report\')}',
    'Diagnostic Analytics': '{t(\'results.analytics\')}',
    'Vision Engine': '{t(\'results.engine\')}',
    'Restart Analysis': '{t(\'results.restart\')}',
    'Original': '{t(\'results.original\')}',
    'Focus': '{t(\'results.focus\')}',
    'Heatmap': '{t(\'results.heatmap\')}',
    'Mask': '{t(\'results.mask\')}',
    'Saliency Method': '{t(\'results.saliency_method\')}',
    'Heatmap Opacity': '{t(\'results.heatmap_opacity\')}',
    'Latency': '{t(\'results.latency\')}',
    'Status': '{t(\'results.status\')}',
    'Verified': '{t(\'results.verified\')}',
    'Risk Level': '{t(\'results.risk_level\')}',
    'Certainty': '{t(\'results.certainty\')}',
    'Recommendation': '{t(\'results.recommendation\')}',
    'Differential Diagnostics (Top 3)': '{t(\'results.differential\')}',
    'Verify Accuracy:': '{t(\'results.verify_accuracy\')}',
    'Feedback Recorded': '{t(\'results.feedback_recorded\')}',
    'Intake Profile': '{t(\'results.intake_profile\')}',
    'Symptom Markers': '{t(\'results.symptom_markers\')}',
    'No symptoms reported.': '{t(\'results.no_symptoms\')}',
    'Clinical AI Chat': '{t(\'results.clinical_chat\')}',
    'Consult our medically-trained assistant for pattern insights, condition papers, and clarified next steps.': '{t(\'results.consult_desc\')}',
    'Start Consultation': '{t(\'results.start_consultation\')}',
    'Medical AI Expert': '{t(\'results.expert\')}',
    'Connection Secure': '{t(\'results.secure\')}',
    'System Initialized': '{t(\'results.initialized\')}',
    'Cross-referencing medical database...': '{t(\'results.cross_ref\')}',
    'Protocol Disclaimer': '{t(\'results.disclaimer_title\')}',
    'This is an autonomous pattern recognition triage. Automated assessment cannot replace histological examination. If a lesion is bleeding, growing, or changing color rapidly, it must be clinically biopsied regardless of AI certainty scores.': '{t(\'results.disclaimer_desc\')}',
}

for k, v in replacements.items():
    content = content.replace(f'> {k} <', f'> {v} <')
    content = content.replace(f'>{k}<', f'>{v}<')
    content = content.replace(f'>\n              {k}\n            <', f'>\n              {v}\n            <')
    content = content.replace(f'>\n                {k}\n              <', f'>\n                {v}\n              <')
    content = content.replace(f'>\n                            {k}\n                          <', f'>\n                            {v}\n                          <')
    content = content.replace(f'>\n                          {k}\n                        <', f'>\n                          {v}\n                        <')

# Hardcoded input placeholder replacement
content = content.replace('placeholder="Describe your question in detail..."', 'placeholder={t(\'results.input_placeholder\')}')
content = content.replace('Verification Report · Clinical AI', '{t(\'results.report\')}')

with open(results_path, 'w', encoding='utf-8') as f:
    f.write(content)
print('ResultsPage updated.')
