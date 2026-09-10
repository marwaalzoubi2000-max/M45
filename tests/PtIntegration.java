package com.musab.math;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
public final class PtIntegration {
    static MathEngine open(Path root,String file)throws Exception {
        return new MathEngine(root.resolve(file).toFile(),Files.readAllBytes(root.resolve("pt_graph.onnx")),Files.newInputStream(root.resolve("pt_weights.tsv")),(a,b)->{});
    }
    public static void main(String[] args)throws Exception {
        Path root=Paths.get(args[0]);int checks=0;
        for(String dtype:new String[]{"fp32","fp16"})for(int protocol:new int[]{2,4}) {
            try(MathEngine engine=open(root,dtype+"-p"+protocol+".pt")) {
                for(int length:new int[]{23,37,191,255}) {
                    long[] ids=new long[length];ids[0]=1;for(int i=1;i<length;i++)ids[i]=3+((i-1)*17)%256;
                    float[] actual=engine.logits(ids);List<String> ref=Files.readAllLines(root.resolve(dtype+"-"+length+".txt"));
                    if(actual.length!=ref.size())throw new AssertionError("shape");
                    for(int i=0;i<actual.length;i++){float expected=Float.parseFloat(ref.get(i));if(Math.abs(actual[i]-expected)>2e-4+2e-3*Math.abs(expected))throw new AssertionError("Logits mismatch "+dtype+" "+length+" "+i);}
                    checks++;
                }
                AtomicBoolean cancel=new AtomicBoolean(true);boolean stopped=false;
                try{engine.solve("1+1",cancel);}catch(InterruptedIOException expected){stopped=true;}
                if(!stopped)throw new AssertionError("cancellation");checks++;
            }
        }
        try(MathEngine engine=open(root,"ordered.pt")){if(engine.logits(Tokenizer.prompt("1+1")).length!=259)throw new AssertionError();checks++;}
        for(String name:new String[]{"wrong-shape.pt","nan.pt","truncated.pt","unknown-type.pt"}) {
            boolean rejected=false;try(MathEngine unused=open(root,name)){}catch(IOException expected){rejected=true;}
            if(!rejected)throw new AssertionError("must reject "+name);checks++;
        }
        if(PtCheckpoint.half(1)!=Math.scalb(1f,-24)||PtCheckpoint.half(0x3c00)!=1f||PtCheckpoint.half(0xc000)!=-2f)throw new AssertionError("half conversion");
        try(MathEngine full=open(root.resolve("full"),"best.pt")) {
            long[] ids=new long[23];ids[0]=1;for(int i=1;i<23;i++)ids[i]=3+((i-1)*17)%256;
            float[] actual=full.logits(ids);List<String> ref=Files.readAllLines(root.resolve("full/reference.txt"));
            for(int i=0;i<actual.length;i++){float expected=Float.parseFloat(ref.get(i));if(Math.abs(actual[i]-expected)>2e-4+2e-3*Math.abs(expected))throw new AssertionError("Production logits mismatch "+i);}
            checks++;
        }
        System.out.println("DIRECT_PT_IMPORT_PARITY_PASS checks="+(checks+1));
    }
}
