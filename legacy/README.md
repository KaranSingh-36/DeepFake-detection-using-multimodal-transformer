# Legacy experiments (kept for reference)

These are the scripts and models from the first phase of the project: a whole-clip
classifier (8 frames + first 4 s of audio, fully fine-tuned ViT + Wav2Vec2 with
cross-attention). They produced the numbers in `results/error_analysis*`,
`results/shortcut_check` and the `multimodal_25000_2epoch_epoch_2.pth` checkpoint.

They are **not** part of the current pipeline. The shortcut check in
`evaluation/shortcut_check.py` showed why: swapping the audio of a fake clip
with the real source audio did not change the prediction at all, and fakes whose
manipulated segment was never shown to the model were still flagged. The model
had learned a per-file cue rather than the manipulation.

The current pipeline lives in `fusiontrace/` and `scripts/` (see the top-level README).

To run anything in here, put this folder on the Python path, because the scripts
import `models.*` and `dataset.*`:

```powershell
$env:PYTHONPATH = "legacy"
python legacy\evaluation\shortcut_check.py
```
