import os
import sys
from pathlib import Path

# Ensure backend directory is in path
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

try:
    from app.core.classifier import predict
    from app.core.decision_layer import classify_risk
    from app.core.xai import xai_suite
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    sys.exit(1)

def test_pipeline():
    print("Starting Medical Triage Pipeline Integration Test...\n")
    
    # Get a sample image
    data_dir = base_dir / "data" / "processed"
    melanoma_dir = data_dir / "melanoma"
    benign_dir = data_dir / "benign"
    
    sample_image = None
    if melanoma_dir.exists() and list(melanoma_dir.glob("*.jpg")):
        sample_image = str(list(melanoma_dir.glob("*.jpg"))[0])
    elif benign_dir.exists() and list(benign_dir.glob("*.jpg")):
        sample_image = str(list(benign_dir.glob("*.jpg"))[0])
        
    if not sample_image:
        print("No sample images found for testing.")
        print("\nIntegration Test: FAIL")
        return
        
    print(f"Testing with image: {sample_image}\n")
    
    try:
        # Step 1: Run prediction
        print("--- Step 1: Classification ---")
        pred_result = predict(sample_image)
        if "error" in pred_result:
            print(f"Prediction Error: {pred_result['error']}")
            print("\nIntegration Test: FAIL")
            return
            
        prediction = pred_result.get("prediction", "Unknown")
        confidence = pred_result.get("confidence", 0.0)
        
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.2f}\n")
        
        # Step 2: Run decision layer
        print("--- Step 2: Decision Layer ---")
        risk_result = classify_risk(prediction, confidence)
        
        risk_level = risk_result.get("risk_level", "Unknown")
        message = risk_result.get("message", "No message provided.")
        
        print(f"Risk Level: {risk_level}")
        print(f"Message: {message}\n")
        
        # Step 3: Generate Grad-CAM
        print("--- Step 3: Grad-CAM Generation ---")
        gradcam_filename = xai_suite.generate(sample_image, method="gradcam")
        gradcam_output_path = xai_suite.output_dir / gradcam_filename
        print(f"Grad-CAM Output Path: {gradcam_output_path}\n")
        
        # Step 4: Validation
        print("--- Validation ---")
        if Path(gradcam_output_path).exists():
            print("Grad-CAM image successfully generated and verified on disk.")
            print("\nIntegration Test: PASS")
        else:
            print("Error: Grad-CAM output file does not exist on disk.")
            print("\nIntegration Test: FAIL")
            
    except Exception as e:
        print(f"An unexpected error occurred during the pipeline test: {e}")
        print("\nIntegration Test: FAIL")

if __name__ == "__main__":
    test_pipeline()
