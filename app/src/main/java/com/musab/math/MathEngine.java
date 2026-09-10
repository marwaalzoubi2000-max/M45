package com.musab.math;

import ai.onnxruntime.*;
import java.io.*;
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

/** Owned exclusively by the Activity's serial worker; no native handles cross threads. */
public final class MathEngine implements AutoCloseable {
    private final OrtEnvironment env=OrtEnvironment.getEnvironment();
    private OrtSession session;
    private final Map<String,OnnxTensor> parameters=new HashMap<>();
    public interface Progress {void update(int done,int total);}
    public interface TokenProgress {void update(String answer,int tokens);}
    private String description="ONNX";
    public String description(){return description;}
    public MathEngine(File checkpoint, byte[] template, InputStream manifest, Progress progress) throws Exception {
        try (InputStream manifestInput=manifest) {
            try(OrtSession.SessionOptions opts=new OrtSession.SessionOptions()) {
                opts.setIntraOpNumThreads(2);opts.setInterOpNumThreads(1);
                session=env.createSession(template,opts);
            }
            Map<String,String[]> entries=new LinkedHashMap<>();
            try(BufferedReader reader=new BufferedReader(new InputStreamReader(manifestInput,StandardCharsets.UTF_8))) {
                String line;while((line=reader.readLine())!=null){String[] parts=line.split("\\t");if(parts.length!=3)throw new IOException("قالب أوزان غير صالح");entries.put(parts[1],parts);}
            }
            try(PtCheckpoint pt=new PtCheckpoint(checkpoint)) {
                if(!pt.keys().equals(entries.keySet()))throw new IOException("هذه الأوزان لا تطابق معمارية MUSAB V12.2");
                int done=0;
                for(String[] entry:entries.values()) {
                    String[] dims=entry[2].split(",");long[] shape=new long[dims.length];for(int i=0;i<dims.length;i++)shape[i]=Long.parseLong(dims[i]);
                    parameters.put(entry[0],OnnxTensor.createTensor(env,pt.tensor(entry[1],shape),shape));
                    progress.update(++done,entries.size());
                }
                description="PyTorch • "+pt.precision+" • V12.2";
            }
            Set<String> expected=new HashSet<>(parameters.keySet());expected.add("input_ids");
            if(!session.getInputNames().equals(expected))throw new IOException("مدخلات القالب غير متوافقة");
            next(Tokenizer.prompt("1+1"));
        }catch(Exception|OutOfMemoryError e){try{close();}catch(Exception ignored){}throw e;}
    }
    public MathEngine(File model) throws Exception {
        OrtSession candidate;
        try (OrtSession.SessionOptions opts=new OrtSession.SessionOptions()) {
            opts.setIntraOpNumThreads(2); opts.setInterOpNumThreads(1);
            candidate=env.createSession(model.getAbsolutePath(),opts);
        }
        session=candidate;
        try {
            Map<String,String> meta=session.getMetadata().getCustomMetadata();
            if (!"musab-v12-full-v1".equals(meta.get("musab_format")))
                throw new IOException("هذا الملف ليس تصدير MUSAB V12.2 المتوافق");
            if (!session.getInputNames().equals(Collections.singleton("input_ids")) ||
                !session.getOutputNames().equals(Collections.singleton("logits")))
                throw new IOException("واجهة النموذج غير متوافقة");
            next(Tokenizer.prompt("1+1")); // Real shape/type/finite-value check before installation.
        } catch (Exception e) { session.close(); throw e; }
    }
    float[] logits(long[] ids) throws Exception {
        try(OnnxTensor input=OnnxTensor.createTensor(env,LongBuffer.wrap(ids),new long[]{1,ids.length})) {
            Map<String,OnnxTensor> feeds=new HashMap<>(parameters);feeds.put("input_ids",input);
            try(OrtSession.Result result=session.run(feeds)) {
                Object value=result.get(0).getValue();
                if(!(value instanceof float[][])||((float[][])value).length!=1)throw new IOException("نوع خرج النموذج غير متوافق");
                return ((float[][])value)[0];
            }
        }
    }
    private int next(long[] ids) throws Exception { return Tokenizer.choose(logits(ids)); }
    public String solve(String question, AtomicBoolean cancel) throws Exception {return solve(question,cancel,(answer,tokens)->{});}
    public String solve(String question, AtomicBoolean cancel, TokenProgress progress) throws Exception {
        long[] ids=Tokenizer.prompt(question);
        ByteArrayOutputStream bytes=new ByteArrayOutputStream(); boolean ended=false;
        for (int i=0;i<Tokenizer.ANSWER && ids.length<Tokenizer.CONTEXT;i++) {
            if (cancel.get()) throw new InterruptedIOException("تم إيقاف الحل");
            int token=next(ids);
            if (cancel.get()) throw new InterruptedIOException("تم إيقاف الحل");
            if (token==Tokenizer.EOS) {ended=true;break;}
            bytes.write(token-Tokenizer.OFFSET);
            progress.update(Tokenizer.decode(bytes),i+1);
            ids=Arrays.copyOf(ids,ids.length+1);ids[ids.length-1]=token;
        }
        String answer=Tokenizer.decode(bytes);
        if (answer.isEmpty()) return "لم ينتج النموذج جوابًا. جرّب صياغة السؤال بالإنجليزية.";
        return answer + (ended ? "" : "\n\n[بلغ الجواب حد الطول وقد يكون غير مكتمل]");
    }
    @Override public void close() throws OrtException {
        try{if(session!=null){session.close();session=null;}}finally{for(OnnxTensor t:parameters.values())t.close();parameters.clear();}
    }
}
