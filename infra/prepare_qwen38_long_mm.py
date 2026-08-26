import json
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor


root = Path(__file__).resolve().parents[1] / "runs" / "qwen38_bench"
images_dir = root / "images"
images_dir.mkdir()

image_specs = [(448, 256)] * 946 + [(928, 128)] * 78
for index, size in enumerate(image_specs):
    image = Image.new(
        "RGB",
        size,
        (index % 256, (index // 256) % 256, (index * 29) % 256),
    )
    image.putpixel(
        (index % size[0], index % size[1]),
        ((index * 31) % 256, (index * 47) % 256, (index * 61) % 256),
    )
    image.save(images_dir / f"image_{index:04d}.png")

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.8-27B")
instruction = (
    "\n你现在是一名持续自由联想的多模态智能体。观察前面的1024张合成图像，"
    "不断编造故事、观察、推测和细节，不要调用工具，不要自行结束回答，"
    "不要输出结束标记，一直说到服务端强制停止。"
)
content = [
    {
        "type": "image",
        "url": f"file:///bench/images/image_{index:04d}.png",
    }
    for index in range(1024)
]
content.append({"type": "text", "text": instruction})
messages = [{"role": "user", "content": content}]
rendered = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
)
stub_tokens = len(processor.tokenizer.encode(rendered, add_special_tokens=False))
filler_tokens = 116024 - stub_tokens
content[-1]["text"] = "的" * filler_tokens + instruction
rendered = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
)
assert len(processor.tokenizer.encode(rendered, add_special_tokens=False)) == 116024

images = [
    Image.open(images_dir / f"image_{index:04d}.png").convert("RGB")
    for index in range(1024)
]
processor_content = [
    {"type": "image", "image": image}
    for image in images
]
processor_content.append({"type": "text", "text": content[-1]["text"]})
processed = processor.apply_chat_template(
    [{"role": "user", "content": processor_content}],
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
prompt_tokens = processed["input_ids"].shape[-1]
visual_tokens = (processed["input_ids"] == 248056).sum().item()
grid_visual_tokens = (
    processed["image_grid_thw"][:, 0]
    * processed["image_grid_thw"][:, 1]
    * processed["image_grid_thw"][:, 2]
).sum().item() // 4
assert prompt_tokens == 230000
assert visual_tokens == 115000
assert grid_visual_tokens == 115000
assert processed["image_grid_thw"].shape[0] == 1024
assert processed["pixel_values"].shape == (460000, 1536)

api_content = [
    {
        "type": "image_url",
        "image_url": {
            "url": f"file:///bench/images/image_{index:04d}.png",
        },
    }
    for index in range(1024)
]
api_content.append({"type": "text", "text": content[-1]["text"]})
with (root / "request.json").open("x") as file:
    json.dump(
        {"messages": [{"role": "user", "content": api_content}]},
        file,
        ensure_ascii=False,
    )
with (root / "request_summary.json").open("x") as file:
    json.dump(
        {
            "model": "Qwen/Qwen3.8-27B",
            "image_count": 1024,
            "image_sizes": {
                "448x256": 946,
                "928x128": 78,
            },
            "prompt_tokens": prompt_tokens,
            "visual_tokens": visual_tokens,
            "text_and_template_tokens": prompt_tokens - visual_tokens,
            "pixel_values_shape": list(processed["pixel_values"].shape),
            "max_model_len": 262144,
            "requested_completion_tokens": 32144,
        },
        file,
        ensure_ascii=False,
        indent=2,
    )
print(
    json.dumps(
        {
            "prompt_tokens": prompt_tokens,
            "visual_tokens": visual_tokens,
            "text_and_template_tokens": prompt_tokens - visual_tokens,
            "pixel_values_shape": list(processed["pixel_values"].shape),
        },
        ensure_ascii=False,
    )
)
