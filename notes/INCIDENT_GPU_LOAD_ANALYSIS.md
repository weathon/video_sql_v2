# Incident: used GPU utilization as a proxy for vLLM service load

## Date
2026-08-28

## Summary
While planning EgoLife Stage 3.2 alongside the running Stage 3.1 index build, I used the instantaneous GPU utilization reported by `nvidia-smi` to characterize the shared `Inferact/Qwen3.8-Flash-Next-NVFP4` vLLM service as highly loaded. I then said that adding Stage 3.2 requests would compete materially with Stage 3.1 and proposed waiting or using separate GPUs. That conclusion was wrong because GPU utilization is not a measure of vLLM request load or remaining serving capacity.

## What went wrong
1. I treated GPU 5 at 96% utilization and GPU 6 at 82% utilization as evidence that the vLLM service was heavily loaded.
2. I ignored the service-level measurements that actually described the workload: 13 running requests and 0 waiting requests.
3. I failed to interpret 13 running requests relative to B200 serving capacity. For B200, 13 running requests is extremely low load; these systems can generally handle 256 to 512 concurrent requests.
4. I incorrectly warned that directly reusing the existing TP2 service would interfere with Stage 3.1, even though the scheduler had no waiting requests and substantial request capacity remained.

## Evidence available at the time
- The service exposed `Inferact/Qwen3.8-Flash-Next-NVFP4` on port 8000 with tensor parallel size 2 across GPU 5 and GPU 6.
- The vLLM metrics reported 13 running requests and 0 waiting requests.
- The server was configured with `--max-num-seqs 200`.
- The GPU utilization percentages only showed that active kernels were occupying the devices during the sampling interval; they did not show scheduler saturation, request backlog, throughput headroom, or request latency.

## Impact
No service was stopped and no experiment was launched from this incorrect conclusion. The error affected the planning discussion: I recommended unnecessary resource separation and gave the user a false description of the existing service load.

## Root cause
I substituted a hardware activity percentage for the serving system's workload measurements instead of evaluating the vLLM scheduler state and the number of running requests relative to the capacity of the B200 service.

## Corrective actions
1. Never use GPU utilization as the primary measure of vLLM service load.
2. Determine serving load from `num_requests_running`, `num_requests_waiting`, configured concurrency, request latency, token throughput, preemption, and scheduler capacity.
3. Interpret running-request counts relative to the hardware and server configuration. Treat 13 running requests on this B200 service as extremely low load, not high load.
4. When the queue has no waiting requests, do not claim that the service is saturated based on GPU utilization alone.
5. For EgoLife Stage 3.2, directly reuse the existing port 8000 Flash Next TP2 service as the user instructed.

## Files touched
`notes/INCIDENT_GPU_LOAD_ANALYSIS.md`
