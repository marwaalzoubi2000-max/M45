package com.musab.math;

import java.io.*;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.zip.*;

/** Reads the tensor-only subset of torch.save ZIP files. Never executes Python/pickle globals. */
public final class PtCheckpoint implements AutoCloseable {
    public static final long MAX_FILE=300L*1024*1024;
    private final ZipFile zip;
    private final String prefix;
    private final Map<?,?> weights;
    public final String precision;
    public PtCheckpoint(File file) throws IOException {
        if(file.length()>MAX_FILE) throw bad("حجم الملف يتجاوز 300 MB");
        zip=new ZipFile(file);
        try {
            String root=null; Set<String> names=new HashSet<>(); int entries=0;
            Enumeration<? extends ZipEntry> it=zip.entries();
            while(it.hasMoreElements()) {
                ZipEntry z=it.nextElement();String name=z.getName();
                if(++entries>2048 || !names.add(name) || name.startsWith("/") || Arrays.asList(name.split("/")).contains("..")) throw bad("بنية ZIP غير صالحة");
                if(name.endsWith("/data.pkl")) {if(root!=null)throw bad("أكثر من سجل أوزان");root=name.substring(0,name.length()-8);}
            }
            if(root==null)throw bad("الملف ليس أوزان PyTorch ZIP مدعومة");prefix=root;
            ZipEntry order=zip.getEntry(prefix+"byteorder");
            if(order!=null && !new String(read(order,16),StandardCharsets.US_ASCII).trim().equals("little"))throw bad("ترتيب البايت غير مدعوم");
            Object obj=new Reader(read(zip.getEntry(prefix+"data.pkl"),4*1024*1024)).parse();
            if(!(obj instanceof Map))throw bad("لا يوجد قاموس أوزان");
            Map<?,?> top=(Map<?,?>)obj;
            if(top.containsKey("optimizer")||top.containsKey("rng"))throw bad("اختر best.pt أو laptop_fp16.pt بدل resume.pt");
            Object model=top.get("model");
            if(!(model instanceof Map))throw bad("الملف لا يحتوي model الخاص بـ V12.2");
            weights=(Map<?,?>)model;
            if(weights.isEmpty()||weights.size()>512)throw bad("عدد مصفوفات الأوزان غير صالح");
            Set<String> types=new HashSet<>();
            for(Object v:weights.values()){if(!(v instanceof Tensor))throw bad("عنصر غير مصفوفة داخل model");types.add(((Tensor)v).storage.dtype);}
            precision=types.size()==1?(types.contains("torch HalfStorage")?"FP16":"FP32"):"FP16 / FP32";
        }catch(IOException|RuntimeException e){zip.close();if(e instanceof IOException)throw (IOException)e;throw bad("بيانات الأوزان غير صالحة");}
    }
    public Set<String> keys() throws IOException {Set<String> result=new HashSet<>();for(Object k:weights.keySet()){if(!(k instanceof String))throw bad("اسم وزن غير صالح");result.add((String)k);}return result;}
    public FloatBuffer tensor(String key,long[] expected) throws IOException {
        Object obj=weights.get(key);if(!(obj instanceof Tensor))throw bad("وزن مفقود: "+key);
        Tensor t=(Tensor)obj;
        if(!Arrays.equals(expected,t.shape))throw bad("معمارية الأوزان لا تطابق V12.2: "+key);
        long count=1,stride=1;
        for(int i=t.shape.length-1;i>=0;i--){if(t.shape[i]<=0||t.shape[i]>10_000_000||t.stride[i]!=stride)throw bad("ترتيب مصفوفة غير مدعوم");stride=Math.multiplyExact(stride,t.shape[i]);}
        count=stride;
        if(count>10_000_000 || t.offset<0 || t.offset>t.storage.count-count)throw bad("حدود مصفوفة غير صالحة");
        int unit=t.storage.dtype.equals("torch HalfStorage")?2:4;
        ZipEntry entry=zip.getEntry(prefix+"data/"+t.storage.key);
        if(entry==null||entry.getSize()!=t.storage.count*unit)throw bad("ملف وزن ناقص: "+key);
        FloatBuffer buffer=ByteBuffer.allocateDirect((int)count*4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        try(InputStream raw=new BufferedInputStream(zip.getInputStream(entry))) {
            long skip=t.offset*unit;while(skip>0){long n=raw.skip(skip);if(n==0){if(raw.read()<0)throw bad("وزن مبتور");n=1;}skip-=n;}
            byte[] bytes=new byte[65536];long left=count;
            while(left>0){int n=(int)Math.min(left,bytes.length/unit);int need=n*unit,got=0;
                while(got<need){int k=raw.read(bytes,got,need-got);if(k<0)throw bad("وزن مبتور");got+=k;}
                ByteBuffer block=ByteBuffer.wrap(bytes,0,need).order(ByteOrder.LITTLE_ENDIAN);
                for(int i=0;i<n;i++){float v=unit==4?block.getFloat():half(block.getShort()&65535);if(!Float.isFinite(v))throw bad("الأوزان تحتوي NaN أو Infinity");buffer.put(v);}left-=n;
            }
        }
        buffer.flip();return buffer;
    }
    static float half(int h) {
        int sign=(h&0x8000)<<16,exp=(h>>>10)&31,mant=h&1023;
        if(exp==0){if(mant==0)return Float.intBitsToFloat(sign);return (sign==0?1f:-1f)*Math.scalb((float)mant,-24);}
        return Float.intBitsToFloat(sign | (exp==31?0x7f800000:(exp+112)<<23) | (mant<<13));
    }
    private byte[] read(ZipEntry e,int max)throws IOException {
        if(e==null||e.getSize()<0||e.getSize()>max)throw bad("ملف بيانات وصفية مفقود أو كبير");
        try(InputStream in=zip.getInputStream(e);ByteArrayOutputStream out=new ByteArrayOutputStream()) {
            byte[] b=new byte[8192];int n;while((n=in.read(b))!=-1){if(out.size()+n>max)throw bad("تجاوز حد البيانات");out.write(b,0,n);}return out.toByteArray();
        }
    }
    @Override public void close()throws IOException{zip.close();}
    private static IOException bad(String s){return new IOException(s);}
    static final class Global {final String name;Global(String s){name=s;}}
    static final class Storage {final String dtype,key;final long count;Storage(String d,String k,long n){dtype=d;key=k;count=n;}}
    static final class Tensor {final Storage storage;final long offset;final long[] shape,stride;Tensor(Storage s,long o,long[] d,long[] st){storage=s;offset=o;shape=d;stride=st;}}
    static final class Reader {
        final ByteBuffer b; final ArrayList<Object> stack=new ArrayList<>(); final Map<Integer,Object> memo=new HashMap<>();static final Object MARK=new Object();
        Reader(byte[] bytes){b=ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);}
        Object parse()throws IOException {
            int ops=0;
            try {while(b.hasRemaining()) {
                if(++ops>500000||stack.size()>4096||memo.size()>50000)throw bad("تجاوز حدود القارئ");
                int op=u();switch(op){
                    case 0x80: int protocol=u();if(protocol!=2&&protocol!=4)throw bad("بروتوكول الحفظ غير مدعوم");break;
                    case 0x95: long frame=b.getLong();if(frame<0||frame>b.remaining())throw bad("إطار مبتور");break;
                    case '.': if(stack.size()!=1||b.hasRemaining())throw bad("نهاية بيانات غير صالحة");return pop();
                    case '}': push(new LinkedHashMap<>());break;
                    case ']': push(new ArrayList<>());break;
                    case ')': push(new Object[0]);break;
                    case '(': push(MARK);break;
                    case 'N': push(null);break;
                    case 0x88: push(true);break;case 0x89:push(false);break;
                    case 'K':push((long)u());break;
                    case 'M':push((long)(b.getShort()&65535));break;
                    case 'J':push((long)b.getInt());break;
                    case 0x8a: int len=u();if(len>8)throw bad("عدد أكبر من المدعوم");long value=0;int last=0;for(int i=0;i<len;i++){last=u();value|=(long)last<<(i*8);}if(len>0&&len<8&&(last&128)!=0)value|=(-1L)<<(len*8);push(value);break;
                    case 'G':push(Double.longBitsToDouble(Long.reverseBytes(b.getLong())));break;
                    case 'X':push(string(b.getInt()));break;
                    case 0x8c:push(string(u()));break;
                    case 'q':memo.put(u(),peek());break;
                    case 'r':int mi=b.getInt();if(mi<0||mi>50000)throw bad("مرجع غير صالح");memo.put(mi,peek());break;
                    case 0x94:memo.put(memo.size(),peek());break;
                    case 'h':push(ref(u()));break;case 'j':push(ref(b.getInt()));break;
                    case 'c':push(global(line()+" "+line()));break;
                    case 0x93:Object name=pop(),module=pop();push(global(module+" "+name));break;
                    case 't':push(marked());break;
                    case 0x85:push(new Object[]{pop()});break;
                    case 0x86:Object two=pop(),one=pop();push(new Object[]{one,two});break;
                    case 0x87:Object three=pop(),second=pop(),first=pop();push(new Object[]{first,second,three});break;
                    case 'u':Object[] pairs=marked();if(pairs.length%2!=0)throw bad("قاموس غير صالح");for(int i=0;i<pairs.length;i+=2)set(peek(),pairs[i],pairs[i+1]);break;
                    case 's':Object v=pop(),k=pop();set(peek(),k,v);break;
                    case 'e':Object[] items=marked();for(Object item:items)list(peek()).add(item);break;
                    case 'a':Object item=pop();list(peek()).add(item);break;
                    case 'Q':push(storage(tuple(pop())));break;
                    case 'R':Object[] args=tuple(pop());Object callable=pop();if(!(callable instanceof Global))throw bad("استدعاء مرفوض");push(reduce(((Global)callable).name,args));break;
                    case 'b':Object state=pop();if(!(peek() instanceof Map)||!(state instanceof Map))throw bad("حالة كائن غير مدعومة");break; // OrderedDict _metadata only; no code execution.
                    default:throw bad("عملية pickle غير مدعومة: "+op);
                }
            }}catch(BufferUnderflowException|IndexOutOfBoundsException|ClassCastException e){throw bad("بيانات pickle مبتورة أو غير صالحة");}
            throw bad("لا توجد نهاية لسجل الأوزان");
        }
        Global global(String name)throws IOException {
            if(!Arrays.asList("collections OrderedDict","torch._utils _rebuild_tensor_v2","torch._utils _rebuild_tensor","torch FloatStorage","torch HalfStorage").contains(name))throw bad("نوع غير مدعوم داخل الملف: "+name);
            return new Global(name);
        }
        Object reduce(String name,Object[] a)throws IOException {
            if(name.equals("collections OrderedDict")){if(a.length!=0)throw bad("قاموس غير مدعوم");return new LinkedHashMap<>();}
            if((name.equals("torch._utils _rebuild_tensor_v2")||name.equals("torch._utils _rebuild_tensor"))&&a.length>=4&&a[0] instanceof Storage){
                long[] dims=numbers(a[2]),strides=numbers(a[3]);if(dims.length<1||dims.length>4||dims.length!=strides.length)throw bad("شكل غير صالح");
                return new Tensor((Storage)a[0],number(a[1]),dims,strides);
            }
            throw bad("استدعاء غير مسموح");
        }
        Storage storage(Object[] a)throws IOException {
            if(a.length!=5||!"storage".equals(a[0])||!(a[1] instanceof Global)||!(a[2] instanceof String))throw bad("تخزين غير صالح");
            String dtype=((Global)a[1]).name,key=(String)a[2];long n=number(a[4]);
            if(!(dtype.equals("torch FloatStorage")||dtype.equals("torch HalfStorage"))||!key.matches("[0-9]{1,12}")||n<=0||n>40_000_000)throw bad("تخزين غير مدعوم");
            return new Storage(dtype,key,n);
        }
        long number(Object o)throws IOException{if(!(o instanceof Long))throw bad("عدد غير صالح");return (Long)o;}
        long[] numbers(Object o)throws IOException{Object[] a=tuple(o);long[] n=new long[a.length];for(int i=0;i<n.length;i++)n[i]=number(a[i]);return n;}
        Object[] tuple(Object o)throws IOException{if(!(o instanceof Object[]))throw bad("تسلسل غير صالح");return (Object[])o;}
        @SuppressWarnings("unchecked") void set(Object target,Object key,Object value)throws IOException {if(!(target instanceof Map)||!(key instanceof String||key instanceof Long))throw bad("قاموس غير صالح");Map<Object,Object> m=(Map<Object,Object>)target;if(m.size()>20000)throw bad("قاموس كبير");if(m.containsKey(key))throw bad("مفتاح مكرر");m.put(key,value);}
        @SuppressWarnings("unchecked") List<Object> list(Object o)throws IOException{if(!(o instanceof List))throw bad("قائمة غير صالحة");return (List<Object>)o;}
        Object ref(int i)throws IOException{if(!memo.containsKey(i))throw bad("مرجع مفقود");return memo.get(i);}
        Object[] marked()throws IOException{int i=stack.lastIndexOf(MARK);if(i<0)throw bad("بداية تسلسل مفقودة");Object[] a=stack.subList(i+1,stack.size()).toArray();stack.subList(i,stack.size()).clear();return a;}
        String line()throws IOException{int start=b.position();while(b.hasRemaining()){if(u()==10){int end=b.position();b.position(start);String s=string(end-start-1);b.get();return s;}}throw bad("نص مبتور");}
        String string(int n)throws IOException{if(n<0||n>1024*1024||n>b.remaining())throw bad("طول نص غير صالح");byte[] data=new byte[n];b.get(data);return new String(data,StandardCharsets.UTF_8);}
        int u(){return b.get()&255;}void push(Object o){stack.add(o);}Object pop(){return stack.remove(stack.size()-1);}Object peek(){return stack.get(stack.size()-1);}
    }
}
