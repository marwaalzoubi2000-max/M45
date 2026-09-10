package com.musab.math;
import org.junit.Test;
import static org.junit.Assert.*;
public class TokenizerTest {
 @Test public void exactTrainingPrompt(){long[] x=Tokenizer.prompt("x=2");assertEquals(1,x[0]);assertEquals('Q'+3,x[1]);assertEquals(' '+3,x[x.length-1]);assertEquals(23,x.length);}
 @Test public void utf8UnsignedBytes(){long[] x=Tokenizer.prompt("é");assertEquals(198,x[11]);assertEquals(172,x[12]);}
 @Test(expected=IllegalArgumentException.class) public void rejectEmpty(){Tokenizer.prompt("  ");}
 @Test(expected=IllegalArgumentException.class) public void rejectLong(){Tokenizer.prompt(new String(new char[193]).replace('\0','a'));}
 @Test public void suppressSpecialTokens(){float[] x=new float[259];x[0]=100;x[1]=99;x[2]=2;assertEquals(2,Tokenizer.choose(x));}
 @Test(expected=IllegalArgumentException.class) public void rejectNaN(){float[] x=new float[259];x[100]=Float.NaN;Tokenizer.choose(x);}
}
