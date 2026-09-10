package com.musab.math;

import java.nio.charset.StandardCharsets;
import java.io.ByteArrayOutputStream;

public final class Tokenizer {
    public static final int PAD=0, BOS=1, EOS=2, OFFSET=3, VOCAB=259, CONTEXT=256, ANSWER=64;
    private Tokenizer() {}
    public static long[] prompt(String question) {
        if (question.trim().isEmpty()) throw new IllegalArgumentException("اكتب المسألة أولًا");
        byte[] bytes = ("Question: " + question + "\nAnswer: ").getBytes(StandardCharsets.UTF_8);
        if (bytes.length + 1 > 192) throw new IllegalArgumentException("المسألة أطول من سعة النموذج؛ اختصرها");
        long[] ids = new long[bytes.length + 1]; ids[0]=BOS;
        for (int i=0;i<bytes.length;i++) ids[i+1]=(bytes[i]&255)+OFFSET;
        return ids;
    }
    public static int choose(float[] logits) {
        if (logits.length != VOCAB) throw new IllegalArgumentException("أبعاد النموذج غير متوافقة");
        int best=EOS;
        for (int i=EOS;i<logits.length;i++) {
            if (!Float.isFinite(logits[i])) throw new IllegalArgumentException("النموذج أعاد قيمًا غير صالحة");
            if (logits[i]>logits[best]) best=i;
        }
        return best;
    }
    public static String decode(ByteArrayOutputStream bytes) {return new String(bytes.toByteArray(),StandardCharsets.UTF_8).trim();}
}
