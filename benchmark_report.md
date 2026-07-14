# Benchmark & E2E Test Report

- **Repository**: SG_integration_step1
- **Date**: 2026-07-14 22:41:07

## 1. E2E Testing Summary
❌ **Status**: FAILED

### Test Logs (Snippet)
```text
============================= test session starts ==============================
platform darwin -- Python 3.13.9, pytest-9.0.3, pluggy-1.5.0
rootdir: /Users/hyunchanan/Documents/GitHub/SG_integration_step1
configfile: pyproject.toml
plugins: anyio-4.12.1, cov-7.1.0, hypothesis-6.155.7, hydra-core-1.3.2, respx-0.23.1
collected 23 items / 1 error

==================================== ERRORS ====================================
________________ ERROR collecting SG_proj_007/tests/test_api.py ________________
import file mismatch:
imported module 'test_api' has this __file__ attribute:
  /Users/hyunchanan/Documents/GitHub/SG_integration_step1/SG_proj_002/tests/test_api.py
which is not the same as the test file we want to collect:
  /Users/hyunchanan/Documents/GitHub/SG_integration_step1/SG_proj_007/tests/test_api.py
HINT: remove __pycache__ / .pyc files and/or use a unique basename for your test file modules
=========================== short test summary info ============================
ERROR SG_proj_007/tests/test_api.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
============================== 1 error in 13.73s ===============================

```

## 2. Models Detected
- `checkpoints/sam2_hiera_large.pt` (856.35 MB)
- `checkpoints/mobile_sam.pt` (38.84 MB)
- `exported_models/weights/torch/model.pt` (125.12 MB)
- `SG_proj_003/checkpoints/mobile_sam.pt` (38.84 MB)
- `SG_proj_002/weights/mobile_sam.pt` (38.84 MB)
- `checkpoints/v_sams_model.pth` (98.01 MB)
- `models/depth_anything_v2/depth_anything_v2_vits.pth` (94.62 MB)
- `SG_proj_003/checkpoints/v_sams_model.pth` (98.01 MB)
- `SG_proj_003/vsams/data/visual_library.pth` (4.18 MB)
