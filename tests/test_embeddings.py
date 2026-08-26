import base64
from io import BytesIO

import cv2
import httpx
import numpy as np
from PIL import Image

from src.embeddings import clip_data_url, request_embeddings


def create_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 4, (160, 120))
    if not writer.isOpened():
        raise RuntimeError("test video writer did not open")
    for index in range(8):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:, :, index % 3] = 25 * index
        writer.write(frame)
    writer.release()


def test_clip_data_url_covers_video_at_two_fps(tmp_path):
    video_path = tmp_path / "video.avi"
    create_video(video_path)
    data_url, media_io_kwargs = clip_data_url(str(video_path), 0.0, 2.0)
    encoded_frames = data_url.removeprefix("data:video/jpeg;base64,").split(",")
    assert len(encoded_frames) == 4
    for frame in encoded_frames:
        with Image.open(BytesIO(base64.b64decode(frame))) as image:
            assert image.size == (160, 120)
    assert media_io_kwargs == {"video": {"num_frames": -1, "fps": 4.0, "frames_indices": [0, 2, 4, 6], "total_num_frames": 8, "duration": 2.0, "do_sample_frames": False}}


async def test_embedding_batch_request_uses_qwen_chat_protocol():
    captured = {}

    def respond(request):
        captured.update(__import__("json").loads(request.content))
        return httpx.Response(200, json={"data": [{"index": 1, "embedding": [0.25, -0.25]}, {"index": 0, "embedding": [0.5, -0.5]}]})

    messages = [{"role": "user", "content": [{"type": "text", "text": "query"}]}]
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        results = await request_embeddings(client, "http://embedding/v1/embeddings", "qwen", [messages, messages], ["row-1", "row-2"], None)
    assert results == [("row-1", [0.5, -0.5]), ("row-2", [0.25, -0.25])]
    assert captured == {"model": "qwen", "messages": [messages, messages], "encoding_format": "float", "continue_final_message": True, "add_special_tokens": True}
    assert "input" not in captured
