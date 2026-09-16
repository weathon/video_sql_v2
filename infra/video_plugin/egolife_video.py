import json

from vllm.multimodal.video import Qwen3VLVideoBackend, VIDEO_LOADER_REGISTRY


class EgoLifeUniform(Qwen3VLVideoBackend):
    @classmethod
    def load_bytes(cls, data, **kwargs):
        path = kwargs.pop('trace_path')
        frames, metadata = super().load_bytes(data, **kwargs)
        metadata['do_sample_frames'] = False
        print('UNIFORM_DECODE ' + json.dumps({'path': path, 'indices': metadata['frames_indices'], 'fps': metadata['fps'], 'source_frames': metadata['total_num_frames'], 'returned_frames': len(frames)}), flush=True)
        return frames, metadata


def register():
    VIDEO_LOADER_REGISTRY.register('qwen3_vl_uniform')(EgoLifeUniform)
