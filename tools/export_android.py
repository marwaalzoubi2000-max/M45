"""Convert a COPY of laptop_fp16.pt to standalone Android ONNX. Never trains."""
import argparse
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
import onnx
import onnxruntime as ort
from musab_model import MathCore, MODEL_CONFIG
from musab_data import encode_prompt

class LastLogits(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model=model
    def forward(self, input_ids):
        return self.model(input_ids)[:, -1, :]

def export(model, destination):
    model=model.float().cpu().eval()
    wrapper=LastLogits(model).eval()
    destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as td:
        temp=Path(td)/'model.onnx'
        with torch.inference_mode():
            torch.onnx.export(wrapper, (torch.tensor([encode_prompt('1+1')]),), str(temp),
                input_names=['input_ids'], output_names=['logits'], opset_version=17,
                dynamic_axes={'input_ids': {1:'sequence'}}, dynamo=False,
                external_data=False)
        graph=onnx.load(str(temp))
        onnx.helper.set_model_props(graph, {'musab_format':'musab-v12-full-v1',
            'tokenizer':'utf8-byte-offset-3', 'context':'256', 'answer_budget':'64'})
        onnx.checker.check_model(graph)
        onnx.save(graph,str(temp))
        session=ort.InferenceSession(str(temp),providers=['CPUExecutionProvider'])
        # Compare logits at multiple sequence lengths, including the context boundary.
        for length in (20,37,191,255):
            ids=torch.tensor([[1]+[3+(i*17)%256 for i in range(length-1)]])
            with torch.inference_mode(): expected=wrapper(ids).numpy()
            actual=session.run(None,{'input_ids':ids.numpy()})[0]
            np.testing.assert_allclose(actual,expected,rtol=2e-3,atol=2e-4)
        # Compare several consecutive autoregressive decisions against PyTorch.
        ids=torch.tensor([encode_prompt('What is 12 + 7?')])
        for _ in range(8):
            with torch.inference_mode(): a=wrapper(ids).numpy()
            b=session.run(None,{'input_ids':ids.numpy()})[0]
            a[:,:2]=-np.inf;b[:,:2]=-np.inf
            if int(a.argmax())!=int(b.argmax()): raise ValueError('Export changed greedy token decisions')
            token=int(a.argmax())
            if token==2:break
            ids=torch.cat([ids,torch.tensor([[token]])],dim=1)
        del session
        temp.replace(destination)
    return destination

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--weights',required=True,help='laptop_fp16.pt exported by V12.2')
    parser.add_argument('--output',default='musab_android.onnx')
    args=parser.parse_args()
    if Path(args.weights).resolve()==Path(args.output).resolve():raise ValueError('Input and output must differ')
    torch.set_num_threads(2)
    ck=torch.load(args.weights,map_location='cpu',weights_only=True)
    if ck.get('format')!='MUSAB_V12_INFERENCE' or ck.get('model_config')!=MODEL_CONFIG:
        raise ValueError('Use V12.2 laptop_fp16.pt; resume.pt is training state, not an inference export')
    model=MathCore();model.load_state_dict(ck['model'],strict=True);del ck
    print('EXPORT_VERIFIED:', export(model,args.output))
if __name__=='__main__':main()
