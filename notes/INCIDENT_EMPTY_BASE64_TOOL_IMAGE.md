# Incident: empty base64 tool images

## Symptom

Twenty-eight Pilot `clip_qa` calls failed with an OpenAI 400 error stating that a prior `function_call_output` image data URL contained empty base64 bytes. The final image files on disk were non-empty, so checking only the completed media directory hid the failure window.

## Verified root cause

`sample_video` originally named each contact sheet using only video ID, start time, end time, and page number. The clip agent can issue parallel calls for the same interval with different fps or crop coordinates. A scan of the saved Pilot traces found 143 parallel call groups that wrote the same output filename; no group had completely identical sampling arguments. One thread could therefore read the file for base64 encoding while another thread had truncated the same filename and had not finished writing it.

## Fix

- Contact-sheet filenames now include fps, all four crop coordinates, the caption flag, and a unique call ID in addition to video ID, interval, and page number.
- Function tools use `failure_error_function=None`, so SDK and tool failures propagate as their real exceptions instead of becoming generic model-visible tool output.
- Nested clip runs stream and record their responses, usage, and partial transcript before propagating a run failure.

## Verification

A real Pilot clip was sampled concurrently over the same interval with two different fps/crop configurations. The calls produced three distinct JPEG paths; all three files were non-empty and decoded as 1920×900 images.
