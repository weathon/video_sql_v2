from vllm.multimodal.video import Qwen3VLVideoBackend, VIDEO_LOADER_REGISTRY


class EgoLifeIFrames(Qwen3VLVideoBackend):
    @classmethod
    def compute_frames_index_to_sample(cls, source, target, **kwargs):
        return kwargs['iframe_indices']

    @classmethod
    def load_bytes(cls, data, **kwargs):
        requested = kwargs['iframe_indices']
        if not requested or requested != sorted(set(requested)):
            raise ValueError(f'Invalid I-frame indices: {requested}')
        frames, metadata = super().load_bytes(data, **kwargs)
        if metadata['frames_indices'] != requested:
            raise ValueError(f'I-frame decode mismatch: requested={requested} decoded={metadata["frames_indices"]}')
        metadata['do_sample_frames'] = False
        print(f'IFRAME_DECODE indices={requested} fps={metadata["fps"]} source_frames={metadata["total_num_frames"]} returned_frames={len(frames)}', flush=True)
        return frames, metadata


def register():
    VIDEO_LOADER_REGISTRY.register('qwen3_vl_iframes')(EgoLifeIFrames)
