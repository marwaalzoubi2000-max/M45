"""Build architecture-only ONNX assets. User weights are graph inputs, never bundled."""
import argparse
from pathlib import Path
import numpy as np
import torch
import onnx
from onnx import numpy_helper
from musab_model import MathCore, MODEL_CONFIG
from export_android import LastLogits
from musab_data import encode_prompt

def build(model, out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    wrapper=LastLogits(model.float().eval())
    path=out/'pt_graph.onnx'
    # No parameter folding: preserve named original weights for direct .pt binding.
    torch.onnx.export(wrapper,(torch.tensor([encode_prompt('1+1')]),),str(path),
        export_params=False,do_constant_folding=False,dynamo=False,opset_version=17,
        input_names=['input_ids'],output_names=['logits'],dynamic_axes={'input_ids':{1:'sequence'}})
    graph=onnx.load(str(path))
    params=dict(wrapper.named_parameters());buffers=dict(wrapper.named_buffers())
    bindings=[];keep=[]
    for inp in graph.graph.input:
        name=inp.name
        if name=='input_ids':keep.append(inp)
        elif name in buffers:
            graph.graph.initializer.append(numpy_helper.from_array(buffers[name].detach().numpy(),name))
        elif name in params:
            assert name.startswith('model.')
            bindings.append((name,name[len('model.'):],list(params[name].shape)))
            keep.append(inp)
        else:raise ValueError('Unmapped graph input: '+name)
    if set(k for _,k,_ in bindings)!=set(model.state_dict()):
        raise ValueError('Template did not expose every trainable weight')
    del graph.graph.input[:];graph.graph.input.extend(keep)
    onnx.helper.set_model_props(graph,{'musab_format':'musab-v12-pt-v1'})
    onnx.checker.check_model(graph);onnx.save(graph,str(path))
    (out/'pt_weights.tsv').write_text(''.join(n+'\t'+k+'\t'+','.join(map(str,s))+'\n' for n,k,s in bindings))
    print('PT_TEMPLATE_READY',len(bindings),path.stat().st_size)
    return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='app/src/main/assets');a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(45)
    build(MathCore(),a.output)
