# DeepGuard dashboard

This is a local-only dashboard for the trained multimodal checkpoint. It is
intentionally isolated from the training and evaluation files.

## Start

From the project root:

```powershell
.\.venv\Scripts\python.exe .\dashboard\app.py
```

Then open <http://127.0.0.1:8501>.

The dashboard uses the existing checkpoint
`results/checkpoints/multimodal_25000_2epoch_epoch_2.pth`, samples 8 video
frames, extracts the first 4 seconds of mono 16 kHz audio, and runs the same
multimodal model used by the evaluation script. FFmpeg must be available on
`PATH`. These fixed inputs match the checkpoint's training and evaluation
configuration.
