"""Backport only the NVFP4 gate/up padding fix from vLLM e126687a9a82."""

from pathlib import Path


path = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/flashinfer_utils.py")
source = path.read_text()
start = source.index("def align_fp4_moe_weights_for_fi(")
end = source.index("\ndef ", start + 1)
function = source[start:end]
replacements = {
    "    padded_w13[:, : w13.shape[1], :] = w13": """    if is_act_and_mul:
        padded_w13[:, :intermediate, :] = w13[:, :intermediate, :]
        padded_w13[:, padded_intermediate : padded_intermediate + intermediate, :] = (
            w13[:, intermediate:, :]
        )
    else:
        padded_w13[:, : w13.shape[1], :] = w13""",
    "    padded_w13_scale[:, : w13_scale.shape[1], :] = w13_scale": """    if is_act_and_mul:
        padded_w13_scale[:, :intermediate, :] = w13_scale[:, :intermediate, :]
        padded_w13_scale[
            :, padded_intermediate : padded_intermediate + intermediate, :
        ] = w13_scale[:, intermediate:, :]
    else:
        padded_w13_scale[:, : w13_scale.shape[1], :] = w13_scale""",
}
for old, new in replacements.items():
    if function.count(old) != 1:
        raise RuntimeError(f"Unexpected NVFP4 padding source: {old}")
    function = function.replace(old, new)
path.write_text(source[:start] + function + source[end:])
print(f"Applied NVFP4 gate/up padding fix to {path}")
