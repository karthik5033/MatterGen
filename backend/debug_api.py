import requests
import json

url = "http://localhost:8000/api/v1/generate"
payload = {
    "prompt": "Test material",
    "weights": {
        "density": 0.5,
        "stability": 0.5,
        "band_gap": 0.5
    },
    "n_candidates": 3
}

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 422:
        print("Validation Error Details:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(response.text)
except Exception as e:
    print(f"Request failed: {e}")
