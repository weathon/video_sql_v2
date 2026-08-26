"""
generate synthetic long multi-turn image+text conversations for infra benchmarking.
targets a per-request prompt token budget (default ~100k) using the served model's
own tokenizer, and embeds a few generated images so the multimodal path is exercised.
each conversation is a multi-turn dialogue: alternating user/assistant text turns
(filler "memory log" style content) with images attached to some user turns.
"""
import argparse, base64, io, json, random
from transformers import AutoTokenizer
from PIL import Image, ImageDraw

# deterministic corpus so token counting is stable; varied enough to avoid trivial cache hits per request
WORDS = ("kitchen breakfast alex maria coffee garage bicycle repair evening walk park "
         "dog running meeting laptop presentation slide budget quarter revenue garden "
         "tomato harvest rain umbrella subway platform delayed train ticket museum "
         "painting sculpture guitar practice chord dinner pasta sauce basil wine glass "
         "notebook sketch drawing bridge river sunset photograph camera lens tripod").split()


def make_image(seed):
    r = random.Random(seed)
    img = Image.new("RGB", (512, 512), (r.randint(0, 255), r.randint(0, 255), r.randint(0, 255)))
    d = ImageDraw.Draw(img)
    for _ in range(30):
        x0, y0 = r.randint(0, 511), r.randint(0, 511)
        x1, y1 = r.randint(0, 511), r.randint(0, 511)
        col = (r.randint(0, 255), r.randint(0, 255), r.randint(0, 255))
        if r.random() < 0.5:
            d.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=col)
        else:
            d.ellipse([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=col)
    d.text((10, 10), f"frame {seed}", fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def filler(rng, n_words):
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def build_conversation(conv_seed, target_tokens, tok, n_images, turns):
    rng = random.Random(conv_seed)
    images = [make_image(conv_seed * 1000 + i) for i in range(n_images)]
    # rough token budget per text turn: reserve ~1200 tokens per image (qwen vl @512px), rest split across turns
    text_budget = max(2000, target_tokens - n_images * 1200)
    per_turn_words = max(20, int((text_budget / turns) * 0.75))

    messages = [{"role": "system", "content": "you are a lifelong memory assistant reviewing a long activity log."}]
    img_idx = 0
    for t in range(turns):
        if t % 2 == 0:  # user turn
            content = [{"type": "text",
                        "text": f"log segment {t}: " + filler(rng, per_turn_words) +
                                " question: what happened around this segment?"}]
            # attach an image to a few early user turns
            if img_idx < n_images and t < n_images * 2:
                content.append({"type": "image_url", "image_url": {"url": images[img_idx]}})
                img_idx += 1
            messages.append({"role": "user", "content": content})
        else:  # assistant turn (part of the recorded history)
            messages.append({"role": "assistant",
                              "content": "noted. " + filler(rng, per_turn_words)})
    # final live user question that the server must actually answer
    messages.append({"role": "user",
                     "content": "based on the entire log above, summarize the three most important events. be brief."})

    # measure real text token count directly (the local chat template does not
    # render the openai-style multimodal content list, so count the raw text;
    # image tokens are added by the server on top of this)
    parts = []
    for m in messages:
        if isinstance(m["content"], str):
            parts.append(m["content"])
        else:
            for p in m["content"]:
                if p["type"] == "text":
                    parts.append(p["text"])
    n_tok = len(tok(" ".join(parts)).input_ids)
    return messages, n_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target-tokens", type=int, default=100000)
    ap.add_argument("--n-images", type=int, default=4)
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--n-convs", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    with open(args.out, "w") as f:
        for i in range(args.n_convs):
            msgs, n = build_conversation(i, args.target_tokens, tok, args.n_images, args.turns)
            f.write(json.dumps({"conv_id": i, "prompt_tokens_est": n, "messages": msgs}) + "\n")
            print(f"conv {i}: est_prompt_tokens={n} n_images={args.n_images}", flush=True)
    print(f"wrote {args.n_convs} convs to {args.out}", flush=True)


if __name__ == "__main__":
    main()
