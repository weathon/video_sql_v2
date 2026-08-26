"""
single image+text request against the served endpoint, printing the real answer
so we can eyeball correctness (no hallucination / no infinite-loop / not empty)
before trusting throughput numbers for a config.
"""
import argparse, base64, io, json, requests
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
ap.add_argument("--model", required=True)
args = ap.parse_args()

img = Image.new("RGB", (256, 256), (30, 60, 120))
d = ImageDraw.Draw(img)
d.rectangle([40, 40, 200, 200], fill=(220, 60, 60))
d.text((50, 100), "RED BOX", fill=(255, 255, 255))
buf = io.BytesIO(); img.save(buf, format="JPEG")
url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

messages = [
    {"role": "user", "content": [
        {"type": "text", "text": "what is the dominant shape and color in this image? one short sentence."},
        {"type": "image_url", "image_url": {"url": url}},
    ]},
]
resp = requests.post(args.url, json={"model": args.model, "messages": messages,
                                     "temperature": 0.0}, timeout=300)
print("status", resp.status_code)
obj = resp.json()
print(json.dumps(obj.get("usage", {}), indent=2))
print("ANSWER:", obj["choices"][0]["message"]["content"])
