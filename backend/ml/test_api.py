import os
import sys
import json
import requests
from pathlib import Path

def test_triage_api():
    print("Starting API Test for /triage endpoint...\n")
    
    base_dir = Path(__file__).resolve().parent.parent
    melanoma_dir = base_dir / "data" / "processed" / "melanoma"
    
    if not melanoma_dir.exists() or not list(melanoma_dir.glob("*.jpg")):
        print(f"No sample images found in {melanoma_dir}")
        print("\nFINAL RESULT: FAIL")
        return
        
    sample_image = str(list(melanoma_dir.glob("*.jpg"))[0])
    print(f"Using sample image: {sample_image}")
    
    url = "http://127.0.0.1:8001/triage"
    
    try:
        with open(sample_image, "rb") as f:
            files = {"file": (Path(sample_image).name, f, "image/jpeg")}
            print(f"Sending POST request to {url}...")
            response = requests.post(url, files=files)
            
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Response text: {response.text}")
            print("\nFINAL RESULT: FAIL")
            return
            
        try:
            data = response.json()
        except ValueError:
            print("Response is not valid JSON.")
            print(f"Response text: {response.text}")
            print("\nFINAL RESULT: FAIL")
            return
            
        print("\nJSON Response:")
        print(json.dumps(data, indent=2))
        
        expected_keys = ["prediction", "confidence", "risk_level", "message", "heatmap_path"]
        missing_keys = [key for key in expected_keys if key not in data]
        
        if missing_keys:
            print(f"\nError: Missing keys in JSON response: {missing_keys}")
            print("\nFINAL RESULT: FAIL")
            return
            
        heatmap_path = data.get("heatmap_path")
        if not heatmap_path or not Path(heatmap_path).exists():
            print(f"\nError: Heatmap file does not exist at {heatmap_path}")
            print("\nFINAL RESULT: FAIL")
            return
            
        print("\nValidation passed: All keys present and heatmap file exists on disk.")
        print("\nFINAL RESULT: PASS")
        
    except requests.exceptions.RequestException as e:
        print(f"\nRequest failed: {e}")
        print("\nFINAL RESULT: FAIL")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("\nFINAL RESULT: FAIL")

if __name__ == "__main__":
    test_triage_api()
