import urllib.request
import json
import os

url = "https://stitch.googleapis.com/mcp"
headers = {
    "Content-Type": "application/json",
    "X-Goog-Api-Key": os.environ.get("STITCH_API_KEY", "")
}

project_id = "2442653139091666131"
screens = [
    {"name": "Design System", "id": "asset-stub-assets-1d8ee7a3c75e414c9fb22e711a3f347c-1777970895898"},
    {"name": "Patient Image Upload", "id": "0ab0e9fcd36a4d48a3b87d86629abd0c"},
    {"name": "Analyzing Image...", "id": "fc6fcad7ab23470b9e759f2a3864f8cd"},
    {"name": "Analysis Results", "id": "78aa91b4618d4c3f8e0194e2e02ba877"},
    {"name": "Triage History", "id": "e5adcaa149014e869d5ead185ea393a7"}
]

out_dir = "frontend/stitch_screens"
os.makedirs(out_dir, exist_ok=True)

for i, screen in enumerate(screens):
    print(f"\nFetching {screen['name']}...")
    data = {
        "jsonrpc": "2.0",
        "id": i + 1,
        "method": "tools/call",
        "params": {
            "name": "get_screen",
            "arguments": {
                "name": f"projects/{project_id}/screens/{screen['id']}",
                "projectId": project_id,
                "screenId": screen["id"]
            }
        }
    }
    
    req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            if "error" in result:
                print(f"Error fetching {screen['name']}: {result['error']}")
                continue
            
            content = result.get("result", {}).get("content", [])
            for item in content:
                if item["type"] == "text":
                    try:
                        # Try to parse it as JSON to extract downloadUrl or htmlCode
                        parsed = json.loads(item["text"])
                        
                        html_code = parsed.get("htmlCode", {})
                        download_url = html_code.get("downloadUrl")
                        
                        if download_url:
                            print(f"Downloading HTML from {download_url}...")
                            html_req = urllib.request.Request(download_url)
                            with urllib.request.urlopen(html_req) as html_resp:
                                html_content = html_resp.read().decode('utf-8')
                                
                            file_path = os.path.join(out_dir, f"{screen['id']}.html")
                            with open(file_path, "w", encoding="utf-8") as f:
                                f.write(html_content)
                            print(f"Saved {file_path}")
                        else:
                            print("No HTML downloadUrl found. Saving full JSON response.")
                            file_path = os.path.join(out_dir, f"{screen['id']}.json")
                            with open(file_path, "w", encoding="utf-8") as f:
                                f.write(json.dumps(parsed, indent=2))
                            print(f"Saved {file_path}")
                    except json.JSONDecodeError:
                        file_path = os.path.join(out_dir, f"{screen['id']}.txt")
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(item["text"])
                        print(f"Saved plain text to {file_path}")
    except Exception as e:
        print(f"Failed: {e}")
