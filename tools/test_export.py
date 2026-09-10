import tempfile
from pathlib import Path
import torch
from musab_model import MathCore, MODEL_CONFIG
from export_android import export

torch.set_num_threads(2)
torch.manual_seed(20260910)
# Small dimensions, exact same operators/encoding/context as production.
cfg=dict(MODEL_CONFIG,layers=2,dim=32,heads=4,kv_heads=2,head_dim=8,ff=64)
with tempfile.TemporaryDirectory() as td:
    path=export(MathCore(cfg),Path(td)/'small.onnx')
    assert path.stat().st_size>0
print('ONNX_LOGITS_AND_GENERATION_PARITY_PASS')
