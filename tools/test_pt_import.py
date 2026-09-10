"""Generate real torch.save fixtures; Java imports them and matches reference logits."""
from pathlib import Path
import torch
import numpy as np
from build_pt_template import build
from musab_model import MathCore, MODEL_CONFIG
from musab_data import encode_prompt
import zipfile

torch.set_num_threads(2);torch.manual_seed(20260910)
root=Path('build/pt-fixtures');root.mkdir(parents=True,exist_ok=True)
cfg=dict(MODEL_CONFIG,layers=2,dim=32,heads=4,kv_heads=2,head_dim=8,ff=64)
model=MathCore(cfg).eval();build(model,root)
for dtype in ['fp32','fp16']:
    state={k:(v.half() if dtype=='fp16' else v.clone()) for k,v in model.state_dict().items()}
    ref=MathCore(cfg).eval();ref.load_state_dict(state)
    for protocol in [2,4]:
        name=f'{dtype}-p{protocol}'
        torch.save(dict(model=state,metrics={'loss':0.5,'complete':True,'modules':168}),root/(name+'.pt'),pickle_protocol=protocol)
    with torch.inference_mode():
        for length in [23,37,191,255]:
            ids=torch.tensor([[1]+[3+(i*17)%256 for i in range(length-1)]])
            values=ref(ids)[0,-1].numpy()
            (root/f'{dtype}-{length}.txt').write_text('\n'.join(map(str,values.tolist())))
# OrderedDict uses BUILD metadata, unlike trainer's plain-dict cpu_weights.
torch.save({'model':model.state_dict(),'metrics':None},root/'ordered.pt')
# Negative fixtures: malformed storage, wrong tensor shape, unsupported GLOBAL descriptor.
state=dict(model.state_dict());state['embed.weight']=torch.zeros(2,2)
torch.save({'model':state},root/'wrong-shape.pt')
state=dict(model.state_dict());state['norm.weight']=torch.full_like(state['norm.weight'],float('nan'))
torch.save({'model':state},root/'nan.pt')
with zipfile.ZipFile(root/'fp32-p2.pt') as z, zipfile.ZipFile(root/'truncated.pt','w') as out:
    for info in z.infolist():
        data=z.read(info.filename)
        if '/data/0' in info.filename:data=data[:-4]
        out.writestr(info.filename,data)
with zipfile.ZipFile(root/'unknown-type.pt','w') as z:
    z.writestr('archive/data.pkl',b'\x80\x02cunsupported\nUnknownType\n.')
print('PT_FIXTURES_READY')

# Production-size manifest + direct FP32 checkpoint, no training or user weights.
full=root/'full';full.mkdir(exist_ok=True)
torch.manual_seed(4512)
production=MathCore().eval();build(production,full)
torch.save({'model':{k:v.clone() for k,v in production.state_dict().items()},'metrics':None},full/'best.pt')
ids=torch.tensor([[1]+[3+(i*17)%256 for i in range(22)]])
with torch.inference_mode():values=production(ids)[0,-1].numpy()
(full/'reference.txt').write_text('\n'.join(map(str,values.tolist())))
print('PRODUCTION_PT_FIXTURE_READY')
