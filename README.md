# STIT-SD

Minimal PyTorch implementation of:

**STIT-SD: A Spatiotemporal Interleaved Transformer with Self-Distillation for Unilateral Motor Imagery Decoding**

`STIT_SD.py` contains the STIT model, self-distillation loss, EMA teacher update, and a runnable example using random EEG data.

## Requirements

- Python 3
- PyTorch

## Run

```bash
python STIT_SD.py
```

The example uses an input tensor with shape `(batch, channels, samples)` and performs one student update followed by one EMA teacher update.
