FROM vllm/vllm-openai:qwen38

RUN grep -q 'filename, device=current_platform.device_type' /usr/local/lib/python3.12/dist-packages/vllm/distributed/ec_transfer/ec_connector/example_connector.py \
    && sed -i 's/filename, device=current_platform.device_type/filename, device=torch.cuda.current_device()/' /usr/local/lib/python3.12/dist-packages/vllm/distributed/ec_transfer/ec_connector/example_connector.py
