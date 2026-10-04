import os
import sys
import random
from pathlib import Path

# Ensure backend directory is in path
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

try:
    from app.core.classifier import predict
except ImportError as e:
    print(f"Failed to import classifier module: {e}")
    sys.exit(1)

def get_images_for_class(folder_name, expected_label):
    data_dir = base_dir / "data" / "processed" / folder_name
    if not data_dir.exists() or not data_dir.is_dir():
        print(f"Data directory not found: {data_dir}")
        return []
    
    images = list(data_dir.glob("*.jpg"))
    return [(img, expected_label) for img in images]

def main():
    print("Starting inference test...\n")
    
    melanoma_images = get_images_for_class("melanoma", "melanoma")
    benign_images = get_images_for_class("benign", "benign")
    
    # Ensure mix of both classes, max 25 each to reach 50 max
    num_melanoma = min(len(melanoma_images), 25)
    num_benign = min(len(benign_images), 25)
    
    # Randomly sample without replacement to avoid selecting same images repeatedly
    test_set = []
    if num_melanoma > 0:
        test_set.extend(random.sample(melanoma_images, num_melanoma))
    if num_benign > 0:
        test_set.extend(random.sample(benign_images, num_benign))
        
    # Shuffle the combined test set
    random.shuffle(test_set)
    
    if not test_set:
        print("No images found to test.")
        return

    correct = 0
    total = len(test_set)
    total_confidence = 0.0
    all_near_one = True
    
    for image_path, actual_label in test_set:
        print(f"Testing image: {image_path.name}")
        
        result = predict(str(image_path))
        if "error" in result:
            print(f"Error during prediction: {result['error']}")
            print("Result: FAIL\n")
            continue
            
        predicted = result.get("prediction", "")
        confidence = result.get("confidence", 0.0)
        
        print(f"Actual: {actual_label}")
        print(f"Predicted: {predicted}")
        print(f"Confidence: {confidence:.2f}")
        
        total_confidence += confidence
        if confidence < 0.99:
            all_near_one = False
            
        if predicted == actual_label:
            print("Result: PASS\n")
            correct += 1
        else:
            print("Result: FAIL\n")

    print("--- Summary ---")
    print(f"Total images tested: {total}")
    print(f"Correct predictions: {correct}")
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"Overall accuracy: {accuracy:.2f}%")
    
    if total > 0:
        avg_confidence = total_confidence / total
        print(f"\n--- Confidence Analysis ---")
        print(f"Average confidence: {avg_confidence:.2f}")
        if all_near_one:
            print("Flag: Confidence is always near 1.0 (Suspiciously high)")
            
    if accuracy > 95.0 and total < 50:
        print("\nWarning: Possible overfitting or data leakage")

if __name__ == "__main__":
    main()
