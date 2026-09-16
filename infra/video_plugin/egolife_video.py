import json

from vllm.multimodal.video import Qwen3VLVideoBackend, VIDEO_LOADER_REGISTRY


class EgoLifeUniform(Qwen3VLVideoBackend):
    @classmethod
    def compute_frames_index_to_sample(cls, source, target, **kwargs):
        if 'start_frame' not in kwargs:
            return super().compute_frames_index_to_sample(source, target, **kwargs)
        first, stop = kwargs['start_frame'], kwargs['stop_frame']
        if not 0 <= first < stop <= source.total_frames_num:
            raise ValueError(f'Invalid source interval: {first}:{stop} of {source.total_frames_num}')
        fps = kwargs['source_fps']
        interval = source._replace(total_frames_num=stop-first, original_fps=fps, duration=(stop-first)/fps)
        return [first + index for index in super().compute_frames_index_to_sample(interval, target, **kwargs)]

    @classmethod
    def load_bytes(cls, data, **kwargs):
        path = kwargs.pop('trace_path')
        frames, metadata = super().load_bytes(data, **kwargs)
        source_indices = metadata['frames_indices']
        source_frames = metadata['total_num_frames']
        if 'start_frame' in kwargs:
            first, stop = kwargs['start_frame'], kwargs['stop_frame']
            metadata.update({'frames_indices': [index-first for index in source_indices], 'total_num_frames': stop-first, 'fps': kwargs['source_fps'], 'duration': (stop-first)/kwargs['source_fps']})
        metadata['do_sample_frames'] = False
        print('UNIFORM_DECODE ' + json.dumps({'path': path, 'indices': metadata['frames_indices'], 'source_indices': source_indices, 'fps': metadata['fps'], 'source_frames': source_frames, 'returned_frames': len(frames)}), flush=True)
        return frames, metadata


def register():
    VIDEO_LOADER_REGISTRY.register('qwen3_vl_uniform')(EgoLifeUniform)
