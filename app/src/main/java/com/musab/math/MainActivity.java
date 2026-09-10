package com.musab.math;

import android.app.*;
import android.os.*;
import android.content.*;
import android.net.Uri;
import org.json.*;
import android.provider.OpenableColumns;
import android.database.Cursor;
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
    private Button load,solve,clear,reset,copy,share,historyButton;
    private ProgressBar progress;
    private boolean solving;
    private String selectedName="";
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
        body.addView(text("نموذجك، على جهازك • الإصدار 2",16));
        load=button("تحميل best.pt من الهاتف",body);status=text("حمّل ملف النموذج لبدء الحل",14);body.addView(status);
        progress=new ProgressBar(this,null,android.R.attr.progressBarStyleHorizontal);progress.setMax(100);progress.setVisibility(View.GONE);body.addView(progress);
        body.addView(text("يدعم best.pt وlaptop_fp16.pt من V12.2 مباشرةً",13));
        body.addView(text("أدخل المسألة",18));
        question=new EditText(this);question.setHint("What is 12 + 7?");question.setMinLines(3);
        question.setGravity(Gravity.TOP|Gravity.START);question.setTextDirection(View.TEXT_DIRECTION_LTR);
        question.setInputType(android.text.InputType.TYPE_CLASS_TEXT|android.text.InputType.TYPE_TEXT_FLAG_MULTI_LINE);
        question.setBackground(card());question.setPadding(dp(16),dp(16),dp(16),dp(16));body.addView(question);
        body.addView(text("استخدم الإنجليزية أو الرموز؛ بيانات التدريب باللغة الإنجليزية.",13));
        LinearLayout actions=new LinearLayout(this);body.addView(actions);
        solve=button("حل",actions);clear=button("مسح",actions);
        body.addView(text("جواب النموذج",18));answer=text("سيظهر الجواب هنا",22);answer.setMinHeight(dp(160));
        answer.setTextIsSelectable(true);answer.setTextDirection(View.TEXT_DIRECTION_FIRST_STRONG);
        answer.setPadding(dp(16),dp(16),dp(16),dp(16));answer.setBackground(card());body.addView(answer);
        LinearLayout answerActions=new LinearLayout(this);body.addView(answerActions);
        copy=button("نسخ",answerActions);share=button("مشاركة",answerActions);
        historyButton=button("سجل المسائل",body);
        body.addView(text("قد يخطئ النموذج؛ تحقّق من النتيجة.",13));reset=button("إعادة تعيين",body);
        load.setOnClickListener(v->{Intent pick=new Intent(Intent.ACTION_OPEN_DOCUMENT);pick.addCategory(Intent.CATEGORY_OPENABLE);pick.setType("*/*");startActivityForResult(pick,7);});
        copy.setOnClickListener(v->{((android.content.ClipboardManager)getSystemService(CLIPBOARD_SERVICE)).setPrimaryClip(ClipData.newPlainText("جواب مصعب",answer.getText()));Toast.makeText(this,"تم نسخ الجواب",Toast.LENGTH_SHORT).show();});
        share.setOnClickListener(v->{Intent send=new Intent(Intent.ACTION_SEND);send.setType("text/plain");send.putExtra(Intent.EXTRA_TEXT,question.getText()+"\n\n"+answer.getText());startActivity(Intent.createChooser(send,"مشاركة الجواب"));});
        historyButton.setOnClickListener(v->showHistory());
        clear.setOnClickListener(v->{question.setText("");answer.setText("سيظهر الجواب هنا");});
        solve.setOnClickListener(v->{if(busy){cancel.set(true);status.setText("جارٍ إيقاف الحل…");return;}startSolve();});
        reset.setOnClickListener(v->new AlertDialog.Builder(this).setTitle("إعادة تعيين؟")
            .setMessage("سيُحذف النموذج المستورد من التطبيق فقط، وتُمسح المسألة والجواب وسجل المسائل.")
            .setNegativeButton("إلغاء",null).setPositiveButton("إعادة تعيين",(d,w)->resetModel()).show());
        if(state!=null){question.setText(state.getString("question",""));answer.setText(state.getString("answer","سيظهر الجواب هنا"));}
        controls(); if(modelFile.isFile()) openSaved();
    }
    private int dp(int n){return (int)(n*getResources().getDisplayMetrics().density+0.5f);}
    private GradientDrawable card(){GradientDrawable d=new GradientDrawable();d.setColor(Color.WHITE);d.setCornerRadius(dp(16));d.setStroke(dp(1),0xFFD5E2DA);return d;}
    private TextView text(String value,int size){TextView t=new TextView(this);t.setText(value);t.setTextSize(size);t.setTextColor(0xFF17382E);t.setPadding(0,dp(10),0,dp(10));return t;}
    private Button button(String value,LinearLayout body){
        Button b=new Button(this);b.setText(value);b.setTextSize(17);b.setAllCaps(false);b.setTextColor(Color.WHITE);
        GradientDrawable bg=new GradientDrawable();bg.setColor(0xFF176B5B);bg.setCornerRadius(dp(14));b.setBackground(bg);
        LinearLayout.LayoutParams lp=body.getOrientation()==LinearLayout.HORIZONTAL?new LinearLayout.LayoutParams(0,dp(54),1):new LinearLayout.LayoutParams(-1,dp(54));
        lp.setMargins(dp(3),dp(7),dp(3),dp(7));body.addView(b,lp);return b;
    }
    private void ui(Runnable action){runOnUiThread(()->{if(!dead)action.run();});}
    private void controls(){load.setEnabled(!busy);reset.setEnabled(!busy);clear.setEnabled(!busy);question.setEnabled(!busy);solve.setEnabled(ready&&(!busy||solving));solve.setText(solving?"إيقاف":"حل");historyButton.setEnabled(!busy);copy.setEnabled(!solving);share.setEnabled(!solving);progress.setVisibility(busy?View.VISIBLE:View.GONE);}
    private void openSaved(){busy=true;controls();status.setText("جارٍ تحميل النموذج المحفوظ…");worker.execute(()->{
        try{engine=openModel(modelFile);ui(()->{ready=true;busy=false;status.setText(getPreferences(0).getString("modelName","النموذج")+" • جاهز دون إنترنت");controls();});}
        catch(Exception e){failure(e);}
    });}
    private void failure(Exception e){ui(()->{busy=false;solving=false;status.setText("تعذّر التنفيذ: "+e.getMessage());controls();});}
    @Override protected void onActivityResult(int request,int result,Intent data){super.onActivityResult(request,result,data);
        if(request!=7||result!=RESULT_OK||data==null||data.getData()==null)return;
        Uri uri=data.getData();selectedName=displayName(uri);busy=true;progress.setProgress(0);controls();status.setText("جارٍ فحص وتحميل النموذج…");
        worker.execute(()->{
            File temp=new File(getFilesDir(),"incoming.onnx");MathEngine candidate=null;
            try{
                try(InputStream input=getContentResolver().openInputStream(uri);FileOutputStream out=new FileOutputStream(temp)){
                    if(input==null)throw new IOException("تعذّر فتح الملف");byte[] buffer=new byte[65536];long total=0;int count;
                    while((count=input.read(buffer))!=-1){if(dead)throw new InterruptedIOException("تم الإلغاء");total+=count;
                        if(total>300L*1024*1024)throw new IOException("الملف أكبر من الحد المسموح 300 MB");out.write(buffer,0,count);}
                    out.getFD().sync();
                }
                candidate=openModel(temp);
                Files.move(temp.toPath(),modelFile.toPath(),StandardCopyOption.ATOMIC_MOVE,StandardCopyOption.REPLACE_EXISTING);
                MathEngine previous=engine;engine=candidate;candidate=null;
                if(previous!=null)try{previous.close();}catch(Exception ignored){}
                ui(()->{ready=true;busy=false;answer.setText("سيظهر الجواب هنا");getPreferences(0).edit().putString("modelName",selectedName).apply();status.setText(selectedName+" • جاهز للحل");controls();});
            }catch(Exception e){if(candidate!=null)try{candidate.close();}catch(Exception ignored){}failure(e);}
            finally{temp.delete();}
        });
    }
    private void startSolve(){String q=question.getText().toString();try{Tokenizer.prompt(q);}catch(Exception e){status.setText(e.getMessage());return;}
        cancel.set(false);busy=true;solving=true;progress.setProgress(0);controls();status.setText("جارٍ الحل على الجهاز…");long started=SystemClock.elapsedRealtime();
        worker.execute(()->{try{String result=engine.solve(q,cancel,(partial,tokens)->ui(()->{answer.setText(partial);progress.setProgress(tokens*100/64);}));ui(()->{answer.setText(result);status.setText("اكتمل الحل خلال "+((SystemClock.elapsedRealtime()-started)/1000.0)+" ثانية");busy=false;solving=false;saveHistory(q,result);controls();});}catch(Exception e){failure(e);}});
    }
    private void resetModel(){busy=true;controls();worker.execute(()->{try{if(engine!=null){engine.close();engine=null;}
        Files.deleteIfExists(modelFile.toPath());ui(()->{busy=false;ready=false;getPreferences(0).edit().clear().apply();question.setText("");answer.setText("سيظهر الجواب هنا");status.setText("حمّل ملف النموذج لبدء الحل");controls();});
    }catch(Exception e){ui(()->ready=false);failure(e);}});}
    private MathEngine openModel(File file) throws Exception {
        boolean pt;try(FileInputStream in=new FileInputStream(file)){pt=in.read()==80&&in.read()==75;}
        if(!pt)return new MathEngine(file);
        byte[] template;
        try(InputStream in=getAssets().open("pt_graph.onnx");ByteArrayOutputStream out=new ByteArrayOutputStream()){
            byte[] b=new byte[8192];int n;while((n=in.read(b))!=-1)out.write(b,0,n);template=out.toByteArray();
        }
        return new MathEngine(file,template,getAssets().open("pt_weights.tsv"),(done,total)->{
            if(dead)throw new java.util.concurrent.CancellationException("تم إغلاق التطبيق");
            ui(()->{progress.setProgress(done*100/total);status.setText("تحميل الأوزان: "+done*100/total+"٪");});
        });
    }
    private String displayName(Uri uri){
        try(Cursor c=getContentResolver().query(uri,new String[]{OpenableColumns.DISPLAY_NAME},null,null,null)){
            if(c!=null&&c.moveToFirst())return c.getString(0);
        }catch(Exception ignored){}return "النموذج المستورد";
    }
    private JSONArray history(){try{return new JSONArray(getPreferences(0).getString("history","[]"));}catch(JSONException e){return new JSONArray();}}
    private void saveHistory(String q,String a){
        try{JSONArray old=history(),next=new JSONArray();next.put(new JSONObject().put("q",q).put("a",a));for(int i=0;i<Math.min(old.length(),19);i++)next.put(old.get(i));getPreferences(0).edit().putString("history",next.toString()).apply();}catch(JSONException ignored){}
    }
    private void showHistory(){
        JSONArray list=history();if(list.length()==0){Toast.makeText(this,"لا توجد مسائل محفوظة بعد",Toast.LENGTH_SHORT).show();return;}
        String[] titles=new String[list.length()];for(int i=0;i<titles.length;i++)titles[i]=list.optJSONObject(i).optString("q");
        new AlertDialog.Builder(this).setTitle("آخر 20 مسألة").setItems(titles,(d,which)->{
            JSONObject item=list.optJSONObject(which);question.setText(item.optString("q"));answer.setText(item.optString("a"));
        }).setNegativeButton("إغلاق",null).setNeutralButton("مسح السجل",(d,w)->getPreferences(0).edit().remove("history").apply()).show();
    }
    @Override protected void onSaveInstanceState(Bundle out){super.onSaveInstanceState(out);out.putString("question",question.getText().toString());out.putString("answer",answer.getText().toString());}
    @Override protected void onDestroy(){dead=true;cancel.set(true);worker.execute(()->{if(engine!=null)try{engine.close();}catch(Exception ignored){}});worker.shutdown();super.onDestroy();}
}
