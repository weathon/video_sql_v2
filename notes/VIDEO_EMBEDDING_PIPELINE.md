# Video embedding pipeline

## Final approach

- Text embeddings remain batched at 512 rows per request.
- Video embeddings use one cached video clip per HTTP request.
- Exactly 64 video requests may be in flight. Requests are launched without an artificial delay and the semaphore limits active requests.
- Each completed video embedding is immediately written to its own stable `video-%06d.npz` shard. A resumed run verifies the shard row ID before skipping it.
- Completed shards are consolidated into one normalized `.vectors.npy` matrix and one `.row_ids.npy` array per searchable table. DuckDB stores metadata, not embedding vectors.
- Video content is sent as a local `file://` URL. vLLM is started with `--allowed-local-media-path /raid/wg25r/video_sql_v2/data/video_clips`; video bytes are not serialized into the request JSON.
- vLLM decodes through TorchCodec using the Qwen video loader at 2 fps, with all frames selected and four FFmpeg threads.
- OpenCV is not used anywhere in clip caching, harness sampling, evidence screenshots, or video embedding.

## Clip cache

- Source videos are split once with FFmpeg stream copy into `data/video_clips/<video_id>/%06d.mp4`.
- The segment target is 30 seconds, but boundaries follow existing keyframes. A segment may be somewhat shorter or longer than 30 seconds; it is not re-encoded to force an exact duration.
- DuckDB `video_clips` boundaries are updated to the measured cumulative durations of the cached clips. Harness sampling and evidence extraction read these discrete files directly instead of seeking through the source video.
- Cache generation resumes from existing verified files.
- The final Pilot cache contains 9,167 database rows and 9,167 files. Every video's final cached boundary matches its measured video-stream duration within 0.05 seconds.
- Video 23 had 751 nominal 30-second rows but FFmpeg produced 750 keyframe-aligned clips whose summed duration was exactly the full 22,503.7 seconds. The final generated clip already covered the tail, so the redundant trailing row `23:video_clips:750` was removed and the remaining 750 boundaries were updated.

## Verified service configuration

```text
vllm serve Qwen/Qwen3-VL-Embedding-8B
  --served-model-name qwen3-vl-embedding-8b
  --runner pooling
  --host 127.0.0.1
  --port 8012
  --trust-remote-code
  --allowed-local-media-path /raid/wg25r/video_sql_v2/data/video_clips
  --mm-processor-cache-gb 0
  --renderer-num-workers 4
  --max-num-batched-tokens 32768
  --middleware src.vllm_local_error_middleware.expose_runtime_error
  --media-io-kwargs {"video":{"video_backend":"qwen3_vl","backend":"torchcodec","fps":2,"num_frames":-1,"seek_mode":"approximate","num_ffmpeg_threads":4}}
```

- The Pilot service was launched in tmux session `stage1_embedding_server_torchcodec` on physical GPU 4 before Codex commands were changed to use foreground harness sessions.
- The Pilot embedding run used tmux session `stage1_pilot_video_embeddings`; new Codex-run jobs use a foreground harness session instead of tmux.
- Run log: `logs/stage1_pilot_video_embeddings_batch2_128_1s.log`.
- Server log: `logs/stage1_embedding_server_no_mm_cache_all_raid.log`.
- Runtime caches, configuration, and temporary compilation files are directed to `/raid/wg25r/video_sql_v2/runs/`; the Hugging Face cache is `/raid/huggingface`.

## Verification performed

- A real local clip request returned HTTP 200 with one finite 4,096-dimensional unit-normalized vector.
- The real loader reported TorchCodec and decoded 60 frames from a 30-second clip at 2 fps.
- The complete test suite passed: 13 tests.
- Database count, disk file count, and per-video duration coverage were checked after cache generation.
- The final Video 23 contact sheet at 22,495, 22,499, and 22,503 seconds and its 22,502-second evidence screenshot were opened and visually checked.
- The completed Pilot run produced 813 text shards containing 416,029 rows and 9,164 video shards. Three video rows were skipped after the exact decoder error `Requested next frame while there are no more frames left to decode.`: `14:video_clips:557`, `17:video_clips:557`, and `21:video_clips:557`.
- A real 25-request concurrent search against the largest NumPy matrix completed with all requests returning ranked rows.

## Approaches replaced

- Runtime source-video seeking and cutting were replaced by the reusable discrete clip cache.
- OpenCV decoding was replaced by FFmpeg stream-copy caching and TorchCodec reading.
- Text uses batches of 512 rows and video requests use one clip per request with 64 maximum in flight.
- Ten-second and one-second request spacing were replaced by no artificial delay; the semaphore supplies the concurrency limit.
- The available observations did not establish a Python HTTP-client deadlock, so no such diagnosis is recorded as a cause.
