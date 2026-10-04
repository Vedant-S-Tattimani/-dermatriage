import urllib.request
import json
import os

url = "https://stitch.googleapis.com/mcp"
headers = {
    "Content-Type": "application/json",
    "X-Goog-Api-Key": os.environ.get("STITCH_API_KEY", "")
}
data = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list"
}

req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers)
try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
        tools = result.get("result", {}).get("tools", [])
        for tool in tools:
            if tool.get("name") == "get_screen":
                print(json.dumps(tool, indent=2))
except Exception as e:
    print(f"Error: {e}")
