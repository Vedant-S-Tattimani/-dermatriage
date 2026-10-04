import os
import shutil
from pathlib import Path

def setup_external_val():
    """Sets up the directory structure for external validation (Phase 4)."""
    BASE = Path("C:/Users/Lenovo/medical-assistant-working/backend/data/external_val")
    
    classes = [
        "actinic", "bcc", "bkl", "df", "melanoma", "nevus", "vascular"
    ]
    
    for cls in classes:
        (BASE / cls).mkdir(parents=True, exist_ok=True)
        
    print(f"External validation structure created at {BASE}")
    print("Please place your external (Google/Smartphone) images into the respective class folders.")

if __name__ == "__main__":
    setup_external_val()
