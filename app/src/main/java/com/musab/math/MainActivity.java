package com.musab.math;

import android.app.*;
import android.os.*;
import android.content.*;
import android.net.Uri;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.view.*;
import android.widget.*;
import java.io.*;
import java.nio.file.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;

public final class MainActivity extends Activity {
    private final ExecutorService worker=Executors.newSingleThreadExecutor();
    private final AtomicBoolean cancel=new AtomicBoolean();
    private MathEngine engine; // worker only
    private TextView status,answer; private EditText question;
    private Button load,solve,clear,reset;
    private volatile boolean dead; private boolean busy,ready;
    private File modelFile;
    @Override public void onCreate(Bundle state) {
        super.onCreate(state); modelFile=new File(getFilesDir(),"model.onnx");
        ScrollView scroll=new ScrollView(this);
        LinearLayout body=new LinearLayout(this);body.setOrientation(LinearLayout.VERTICAL);
        body.setPadding(dp(24),dp(24),dp(24),dp(32));body.setLayoutDirection(View.LAYOUT_DIRECTION_RTL);
        body.setBackgroundColor(Color.rgb(244,247,245));scroll.addView(body);setContentView(scroll);
        body.setOnApplyWindowInsetsListener((v,insets)->{
            v.setPadding(dp(24),dp(24)+insets.getSystemWindowInsetTop(),dp(24),dp(32)+insets.getSystemWindowInsetBottom());
            return insets;
        });
        TextView title=text("مصعب • رياضيات",28);title.setTypeface(null,Typeface.BOLD);body.addView(title);
        body.addView(text("نموذجك، على جهازك",16));
        load=button("تحميل الأوزان",body);status=text("حمّل ملف النموذج لبدء الحل",14);body.addView(status);
        body.addView(text("أدخل المسألة",18));
        question=new EditText(this);question.setHint("What is 12 + 7?");question.setMinLines(3);
        question.setGravity(Gravity.TOP|Gravity.START);question.setTextDirection(View.TEXT_DIRECTION_LTR);
        question.setInputType(android.text.InputType.TYPE_CLASS_TEXT|android.text.InputType.TYPE_TEXT_FLAG_MULTI_LINE);
        question.setBackground(card());question.setPadding(dp(16),dp(16),dp(16),dp(16));body.addView(question);
        body.addView(text("استخدم الإنجليزية أو الرموز؛ بيانات التدريب باللغة الإنجليزية.",13));
        solve=button("حل",body);clear=button("مسح",body);
        body.addView(text("جواب النموذج",18));answer=text("سيظهر الجواب هنا",22);answer.setMinHeight(dp(160));
        answer.setTextIsSelectable(true);answer.setTextDirection(View.TEXT_DIRECTION_FIRST_STRONG);
        answer.setPadding(dp(16),dp(16),dp(16),dp(16));answer.setBackground(card());body.addView(answer);
        body.addView(text("قد يخطئ النموذج؛ تحقّق من النتيجة.",13));reset=button("إعادة تعيين",body);
        load.setOnClickListener(v->{Intent pick=new Intent(Intent.ACTION_OPEN_DOCUMENT);pick.addCategory(Intent.CATEGORY_OPENABLE);pick.setType("*/*");startActivityForResult(pick,7);});
        clear.setOnClickListener(v->{question.setText("");answer.setText("سيظهر الجواب هنا");});
        solve.setOnClickListener(v->{if(busy){cancel.set(true);status.setText("جارٍ إيقاف الحل…");return;}startSolve();});
        reset.setOnClickListener(v->new AlertDialog.Builder(this).setTitle("إعادة تعيين؟")
            .setMessage("سيُحذف النموذج المستورد من التطبيق فقط، وتُمسح المسألة والجواب.")
            .setNegativeButton("إلغاء",null).setPositiveButton("إعادة تعيين",(d,w)->resetModel()).show());
        if(state!=null){question.setText(state.getString("question",""));answer.setText(state.getString("answer","سيظهر الجواب هنا"));}
        controls(); if(modelFile.isFile()) openSaved();
    }
    private int dp(int n){return (int)(n*getResources().getDisplayMetrics().density+0.5f);}
    private GradientDrawable card(){GradientDrawable d=new GradientDrawable();d.setColor(Color.WHITE);d.setCornerRadius(dp(16));d.setStroke(dp(1),0xFFD5E2DA);return d;}
    private TextView text(String value,int size){TextView t=new TextView(this);t.setText(value);t.setTextSize(size);t.setTextColor(0xFF17382E);t.setPadding(0,dp(10),0,dp(10));return t;}
    private Button button(String value,LinearLayout body){Button b=new Button(this);b.setText(value);b.setTextSize(17);b.setAllCaps(false);body.addView(b,new LinearLayout.LayoutParams(-1,dp(56)));return b;}
    private void ui(Runnable action){runOnUiThread(()->{if(!dead)action.run();});}
    private void controls(){load.setEnabled(!busy);reset.setEnabled(!busy);clear.setEnabled(!busy);question.setEnabled(!busy);solve.setEnabled(ready);solve.setText(busy?"إيقاف":"حل");}
    private void openSaved(){busy=true;controls();status.setText("جارٍ تحميل النموذج المحفوظ…");worker.execute(()->{
        try{engine=new MathEngine(modelFile);ui(()->{ready=true;busy=false;status.setText("النموذج جاهز • يعمل دون إنترنت");controls();});}
        catch(Exception e){failure(e);}
    });}
    private void failure(Exception e){ui(()->{busy=false;status.setText("تعذّر التنفيذ: "+e.getMessage());controls();});}
    @Override protected void onActivityResult(int request,int result,Intent data){super.onActivityResult(request,result,data);
        if(request!=7||result!=RESULT_OK||data==null||data.getData()==null)return;
        Uri uri=data.getData();busy=true;controls();status.setText("جارٍ فحص وتحميل النموذج…");
        worker.execute(()->{
            File temp=new File(getFilesDir(),"incoming.onnx");MathEngine candidate=null;
            try{
                try(InputStream input=getContentResolver().openInputStream(uri);FileOutputStream out=new FileOutputStream(temp)){
                    if(input==null)throw new IOException("تعذّر فتح الملف");byte[] buffer=new byte[65536];long total=0;int count;
                    while((count=input.read(buffer))!=-1){if(dead)throw new InterruptedIOException("تم الإلغاء");total+=count;
                        if(total>300L*1024*1024)throw new IOException("الملف أكبر من الحد المسموح 300 MB");out.write(buffer,0,count);}
                    out.getFD().sync();
                }
                candidate=new MathEngine(temp);
                Files.move(temp.toPath(),modelFile.toPath(),StandardCopyOption.ATOMIC_MOVE,StandardCopyOption.REPLACE_EXISTING);
                MathEngine previous=engine;engine=candidate;candidate=null;
                if(previous!=null)try{previous.close();}catch(Exception ignored){}
                ui(()->{ready=true;busy=false;answer.setText("سيظهر الجواب هنا");status.setText("تم تحميل النموذج • جاهز للحل");controls();});
            }catch(Exception e){if(candidate!=null)try{candidate.close();}catch(Exception ignored){}failure(e);}
            finally{temp.delete();}
        });
    }
    private void startSolve(){String q=question.getText().toString();try{Tokenizer.prompt(q);}catch(Exception e){status.setText(e.getMessage());return;}
        cancel.set(false);busy=true;controls();status.setText("جارٍ الحل على الجهاز…");
        worker.execute(()->{try{String result=engine.solve(q,cancel);ui(()->{answer.setText(result);status.setText("اكتمل الحل");busy=false;controls();});}catch(Exception e){failure(e);}});
    }
    private void resetModel(){busy=true;controls();worker.execute(()->{try{if(engine!=null){engine.close();engine=null;}
        Files.deleteIfExists(modelFile.toPath());ui(()->{busy=false;ready=false;question.setText("");answer.setText("سيظهر الجواب هنا");status.setText("حمّل ملف النموذج لبدء الحل");controls();});
    }catch(Exception e){ui(()->ready=false);failure(e);}});}
    @Override protected void onSaveInstanceState(Bundle out){super.onSaveInstanceState(out);out.putString("question",question.getText().toString());out.putString("answer",answer.getText().toString());}
    @Override protected void onDestroy(){dead=true;cancel.set(true);worker.execute(()->{if(engine!=null)try{engine.close();}catch(Exception ignored){}});worker.shutdown();super.onDestroy();}
}
