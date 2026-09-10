package com.musab.math;

import ai.onnxruntime.*;
import java.io.*;
import java.nio.LongBuffer;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

/** Owned exclusively by the Activity's serial worker; no native handles cross threads. */
public final class MathEngine implements AutoCloseable {
    private final OrtEnvironment env=OrtEnvironment.getEnvironment();
    private final OrtSession session;
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
    private int next(long[] ids) throws Exception {
        try (OnnxTensor input=OnnxTensor.createTensor(env,LongBuffer.wrap(ids),new long[]{1,ids.length});
             OrtSession.Result result=session.run(Collections.singletonMap("input_ids",input))) {
            Object value=result.get(0).getValue();
            if (!(value instanceof float[][])) throw new IOException("نوع خرج النموذج غير متوافق");
            float[][] logits=(float[][])value;
            if (logits.length!=1) throw new IOException("أبعاد الخرج غير متوافقة");
            return Tokenizer.choose(logits[0]);
        }
    }
    public String solve(String question, AtomicBoolean cancel) throws Exception {
        long[] ids=Tokenizer.prompt(question);
        ByteArrayOutputStream bytes=new ByteArrayOutputStream(); boolean ended=false;
        for (int i=0;i<Tokenizer.ANSWER && ids.length<Tokenizer.CONTEXT;i++) {
            if (cancel.get()) throw new InterruptedIOException("تم إيقاف الحل");
            int token=next(ids);
            if (cancel.get()) throw new InterruptedIOException("تم إيقاف الحل");
            if (token==Tokenizer.EOS) {ended=true;break;}
            bytes.write(token-Tokenizer.OFFSET);
            ids=Arrays.copyOf(ids,ids.length+1);ids[ids.length-1]=token;
        }
        String answer=Tokenizer.decode(bytes);
        if (answer.isEmpty()) return "لم ينتج النموذج جوابًا. جرّب صياغة السؤال بالإنجليزية.";
        return answer + (ended ? "" : "\n\n[بلغ الجواب حد الطول وقد يكون غير مكتمل]");
    }
    @Override public void close() throws OrtException {session.close();}
}
