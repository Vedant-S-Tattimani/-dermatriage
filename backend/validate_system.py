import requests
import os
from pathlib import Path

BASE_URL = "http://127.0.0.1:8005"
UPLOADS_DIR = Path("uploads")

def test_triage():
    print("--- Testing Triage Endpoint ---")
    # Find a valid image in uploads
    images = [f for f in os.listdir(UPLOADS_DIR) if f.endswith(".jpg") or f.endswith(".png")]
    if not images:
        print("No images found in uploads to test with.")
        return
    
    test_image = UPLOADS_DIR / images[0]
    print(f"Testing with image: {test_image}")
    
    with open(test_image, "rb") as f:
        files = {"file": (test_image.name, f, "image/jpeg")}
        response = requests.post(f"{BASE_URL}/api/v1/triage", files=files)
    
    if response.status_code in [200, 201]:
        data = response.json()
        print("SUCCESS!")
        print(f"Prediction (class_name): {data.get('class_name')}")
        print(f"Confidence: {data.get('confidence')}")
        print(f"Risk Level: {data.get('risk_level')}")
        print(f"Grad-CAM URL: {data.get('gradcam_url')}")
        
        # Check if heatmap exists
        if data.get('gradcam_url'):
            filename = data.get('gradcam_url').split('/')[-1]
            heatmap_full_path = UPLOADS_DIR / filename
            if heatmap_full_path.exists():
                print(f"Heatmap file exists at: {heatmap_full_path}")
            else:
                print(f"ERROR: Heatmap file NOT FOUND at: {heatmap_full_path}")
    else:
        print(f"FAILED! Status code: {response.status_code}")
        # Safe printing for Windows command prompt
        try:
            print(response.text)
        except UnicodeEncodeError:
            print(response.text.encode('ascii', errors='replace').decode('ascii'))

if __name__ == "__main__":
    test_triage()
