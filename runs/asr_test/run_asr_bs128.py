import subprocess, time
from pathlib import Path
import duckdb, soundfile as sf, torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

OUT = Path("runs/asr_test/bs128")
OUT.mkdir(parents=True, exist_ok=True)
MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"

con = duckdb.connect("data/pilot.duckdb", read_only=True)
video_path = con.execute("SELECT video_path FROM videos WHERE video_id='4'").fetchone()[0]
rows = con.execute("""
  SELECT start_sec, end_sec FROM asr_segments
  WHERE video_id='4' AND (end_sec-start_sec) BETWEEN 5 AND 15
  ORDER BY start_sec LIMIT 128
""").fetchall()
con.close()

wavs = []
for start, end in rows:
    wav = OUT / f"{start:.2f}_{end:.2f}.wav"
    if not wav.exists():
        subprocess.run([
            "ffmpeg", "-y", "-v", "error",
            "-ss", str(start), "-to", str(end), "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000", str(wav),
        ], check=True)
    wavs.append(str(wav))

audio_s = sum(sf.info(w).duration for w in wavs)
print("n_wavs", len(wavs), "audio_seconds", round(audio_s, 2))

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda").eval()
inputs = processor.apply_transcription_request(audio=wavs, language=[None] * len(wavs)).to(model.device, model.dtype)

with torch.inference_mode():
    _ = model.generate(**inputs)  # warmup
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
with torch.inference_mode():
    output_ids = model.generate(**inputs)  # warm timed
torch.cuda.synchronize()
warm = time.time() - t0

generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
trans = processor.decode(generated_ids, return_format="transcription_only")
empty = sum(1 for t in trans if len(t.strip()) == 0)

print("warm_gen_s", round(warm, 2))
print("warm_realtime_x", round(audio_s / warm, 1))
print("warm_proc_min_per_audio_min", round(warm / audio_s, 4))
print("peak_mem_GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))
print("empty_transcriptions", empty, "of", len(trans))
