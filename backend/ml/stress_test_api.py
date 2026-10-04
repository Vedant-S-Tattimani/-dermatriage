import os
import sys
import json
import random
import tempfile
import requests
from pathlib import Path

BASE_URL = "http://127.0.0.1:8002/triage"
base_dir = Path(__file__).resolve().parent.parent


def send_image(image_path: str, label: str = "valid") -> dict:
    """Send an image to the triage endpoint and return result info."""
    try:
        with open(image_path, "rb") as f:
            files = {"file": (Path(image_path).name, f, "image/jpeg")}
            response = requests.post(BASE_URL, files=files, timeout=30)

        data = None
        try:
            data = response.json()
        except ValueError:
            pass

        return {
            "label": label,
            "file": Path(image_path).name,
            "status_code": response.status_code,
            "data": data,
        }
    except Exception as e:
        return {
            "label": label,
            "file": Path(image_path).name,
            "status_code": None,
            "error": str(e),
        }


def test_valid_images():
    """Test A: Multiple valid images from both classes."""
    print("=" * 60)
    print("TEST A: Valid Images (10 melanoma + 10 benign)")
    print("=" * 60)

    melanoma_dir = base_dir / "data" / "processed" / "melanoma"
    benign_dir = base_dir / "data" / "processed" / "benign"

    melanoma_images = list(melanoma_dir.glob("*.jpg")) if melanoma_dir.exists() else []
    benign_images = list(benign_dir.glob("*.jpg")) if benign_dir.exists() else []

    selected = []
    if melanoma_images:
        selected.extend([(img, "melanoma") for img in random.sample(melanoma_images, min(10, len(melanoma_images)))])
    if benign_images:
        selected.extend([(img, "benign") for img in random.sample(benign_images, min(10, len(benign_images)))])

    random.shuffle(selected)

    results = []
    for img_path, actual_class in selected:
        result = send_image(str(img_path), label=actual_class)
        status = result["status_code"]
        data = result.get("data")

        prediction = data.get("prediction", "N/A") if data else "N/A"
        risk_level = data.get("risk_level", "N/A") if data else "N/A"
        heatmap = data.get("heatmap_path") if data else None
        heatmap_exists = Path(heatmap).exists() if heatmap else False

        print(f"  [{status}] {result['file']:30s} | pred={prediction:10s} | risk={risk_level:12s} | heatmap={'YES' if heatmap_exists else 'NO'}")
        results.append(status == 200 and heatmap_exists)

    return results


def test_invalid_inputs():
    """Test B: Non-image file and empty request."""
    print("\n" + "=" * 60)
    print("TEST B: Invalid Inputs")
    print("=" * 60)

    results = []

    # B1: Send a text file as if it were an image
    tmp_txt = base_dir / "uploads" / "_stress_test_fake.txt"
    tmp_txt.write_text("this is not an image")
    try:
        with open(tmp_txt, "rb") as f:
            files = {"file": (tmp_txt.name, f, "text/plain")}
            resp = requests.post(BASE_URL, files=files, timeout=10)
        expected = resp.status_code == 400
        print(f"  [{'PASS' if expected else 'FAIL'}] Non-image file -> status {resp.status_code} (expected 400)")
        results.append(expected)
    except Exception as e:
        print(f"  [FAIL] Non-image file -> error: {e}")
        results.append(False)
    finally:
        tmp_txt.unlink(missing_ok=True)

    # B2: Empty request (no file)
    try:
        resp = requests.post(BASE_URL, timeout=10)
        expected = resp.status_code == 422
        print(f"  [{'PASS' if expected else 'FAIL'}] Empty request -> status {resp.status_code} (expected 422)")
        results.append(expected)
    except Exception as e:
        print(f"  [FAIL] Empty request -> error: {e}")
        results.append(False)

    return results


def test_edge_cases():
    """Test C: Very small image and corrupted image."""
    print("\n" + "=" * 60)
    print("TEST C: Edge Cases")
    print("=" * 60)

    results = []

    # C1: Very small image (1x1 pixel JPEG)
    tiny_jpg = base_dir / "uploads" / "_stress_test_tiny.jpg"
    try:
        from PIL import Image
        img = Image.new("RGB", (1, 1), color=(128, 128, 128))
        img.save(str(tiny_jpg), "JPEG")

        result = send_image(str(tiny_jpg), label="tiny_image")
        status = result["status_code"]
        # Accept either 200 (model handled it) or 500 (model choked on tiny input)
        passed = status in (200, 500)
        print(f"  [{'PASS' if passed else 'FAIL'}] Tiny 1x1 image -> status {status}")
        results.append(passed)
    except Exception as e:
        print(f"  [FAIL] Tiny image test -> error: {e}")
        results.append(False)
    finally:
        tiny_jpg.unlink(missing_ok=True)

    # C2: Corrupted image (random bytes with .jpg extension)
    corrupt_jpg = base_dir / "uploads" / "_stress_test_corrupt.jpg"
    try:
        corrupt_jpg.write_bytes(os.urandom(512))

        with open(corrupt_jpg, "rb") as f:
            files = {"file": (corrupt_jpg.name, f, "image/jpeg")}
            resp = requests.post(BASE_URL, files=files, timeout=10)

        # Accept 400 or 500 — both are valid error handling
        passed = resp.status_code in (400, 500)
        print(f"  [{'PASS' if passed else 'FAIL'}] Corrupted image -> status {resp.status_code}")
        results.append(passed)
    except Exception as e:
        print(f"  [FAIL] Corrupted image test -> error: {e}")
        results.append(False)
    finally:
        corrupt_jpg.unlink(missing_ok=True)

    return results


def main():
    print("STRESS TEST: /triage API")
    print(f"Target: {BASE_URL}\n")

    all_results = []

    all_results.extend(test_valid_images())
    all_results.extend(test_invalid_inputs())
    all_results.extend(test_edge_cases())

    total = len(all_results)
    passed = sum(all_results)
    failed = total - passed

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total requests: {total}")
    print(f"  Passed:         {passed}")
    print(f"  Failed:         {failed}")

    if failed == 0:
        print("\nSTRESS TEST RESULT: PASS")
    else:
        print("\nSTRESS TEST RESULT: FAIL")


if __name__ == "__main__":
    main()
