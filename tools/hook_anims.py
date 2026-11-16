import numpy as np, subprocess, math, cv2, sys
from PIL import Image, ImageDraw, ImageFont
W,H=1080,1920; RW,RH=540,960; FPS=60000/1001; CY=860; SC=0.65
F=ImageFont.truetype('anton.ttf',int(130*SC))
def hanger(sz,col=(255,255,255,255)):
    S=sz*4; im=Image.new('RGBA',(S,S),(0,0,0,0)); d=ImageDraw.Draw(im); w_=int(S*0.08); cx=S/2; top=S*0.12
    d.arc([cx-S*0.11,top,cx+S*0.11,top+S*0.22],180,450,fill=col,width=w_)
    d.line([(cx,top+S*0.22),(cx,top+S*0.32)],fill=col,width=w_)
    d.line([(cx,top+S*0.32),(S*0.06,S*0.78)],fill=col,width=w_); d.line([(cx,top+S*0.32),(S*0.94,S*0.78)],fill=col,width=w_)
    d.line([(S*0.06,S*0.78),(S*0.94,S*0.78)],fill=col,width=w_); return im.resize((sz,sz),Image.LANCZOS)
PAD=int(28*SC); RHt=int(154*SC); HS=int(104*SC); GAP=int(22*SC)
words=["RATE","THE","FIT"]; sp=F.getlength(" ")
ww=[F.getlength(w) for w in words]; TW=sum(ww)+sp*2
BW=int(PAD*2+TW+GAP+HS); BH=RHt
HANG=hanger(HS)
def ease_out_back(x,c1=1.70158*1.4): c3=c1+1; return 1+c3*(x-1)**3+c1*(x-1)**2
def box(words_on=3, hang_rot=0.0, hang_dy=0, hang_on=True, inv=0.0, wipe=1.0, rgb_split=0):
    L=Image.new('RGBA',(BW,BH),(0,0,0,0)); d=ImageDraw.Draw(L)
    bg=tuple(int(0+(255-0)*inv) for _ in range(3)); fg=tuple(int(255-255*inv) for _ in range(3))
    d.rectangle([0,0,int(BW*wipe),BH],fill=bg+(255,))
    x=PAD
    for i,w in enumerate(words):
        if i<words_on:
            if rgb_split:
                d.text((x-rgb_split,BH/2+2),w,font=F,fill=(255,0,60,200),anchor='lm'); d.text((x+rgb_split,BH/2+2),w,font=F,fill=(0,220,255,200),anchor='lm')
            d.text((x,BH/2+2),w,font=F,fill=fg+(255,),anchor='lm')
        x+=ww[i]+sp
    if hang_on:
        hg=HANG if fg==(255,255,255) else hanger(HS,fg+(255,))
        r=hg.rotate(hang_rot,resample=Image.BICUBIC,center=(HS/2,HS*0.15))
        L.alpha_composite(r,(int(PAD+TW+GAP),int((BH-HS)/2+hang_dy)))
    if wipe<1: L=L.crop((0,0,max(1,int(BW*wipe)),BH)); L2=Image.new('RGBA',(BW,BH),(0,0,0,0)); L2.alpha_composite(L,(0,0)); L=L2
    return L
def place(ov,im,sx=1.0,sy=1.0,dx=0,dy=0,alpha=1.0):
    if sx<=0.01 or sy<=0.01: return
    im=im.resize((max(1,int(im.width*sx)),max(1,int(im.height*sy))),Image.LANCZOS)
    if alpha<1: im.putalpha(im.getchannel('A').point(lambda v:int(v*alpha)))
    ov.alpha_composite(im,(int(W/2-im.width/2+dx),int(CY-im.height/2+dy)))
def swing(t): return 18*math.exp(-t*5)*math.cos(t*14)
T_OUT=1.8
def A(t):  # POP
    ov=Image.new('RGBA',(W,H),(0,0,0,0))
    if t<0.28: s=max(0,ease_out_back(t/0.28))
    elif t<T_OUT: s=1
    elif t<T_OUT+0.06: s=1+0.06*(t-T_OUT)/0.06
    elif t<T_OUT+0.18: s=1.06*(1-(t-T_OUT-0.06)/0.12)
    else: return ov
    place(ov,box(hang_rot=swing(max(0,t-0.2))),s,s); return ov
def B(t):  # SLAM
    ov=Image.new('RGBA',(W,H),(0,0,0,0))
    if t>=T_OUT: return ov
    if t<0.14: x=t/0.14; s=2.0-1.0*x*x; a=min(1,x*2); dx=dy=0
    else:
        s=1; a=1; k=math.exp(-(t-0.14)*14); dx=10*k*math.sin(t*90); dy=8*k*math.cos(t*77)
    place(ov,box(hang_rot=swing(max(0,t-0.14))),s,s,dx,dy,a); return ov
def C(t):  # STAGGER
    ov=Image.new('RGBA',(W,H),(0,0,0,0))
    if t<0.18: wipe=1-(1-t/0.18)**3; place(ov,box(words_on=0,hang_on=False,wipe=wipe)); return ov
    if t<T_OUT:
        n=min(3,1+int((t-0.18)/0.09)); hd=-80*max(0,1-(t-0.45)/0.15) if t<0.6 else 0
        place(ov,box(words_on=n,hang_on=t>=0.45,hang_dy=hd,hang_rot=swing(max(0,t-0.6)))); return ov
    if t<T_OUT+0.15: x=(t-T_OUT)/0.15; place(ov,box(wipe=1-x*x)); return ov
    return ov
def D(t):  # FLASH
    ov=Image.new('RGBA',(W,H),(0,0,0,0))
    if t<0.05: place(ov,box(inv=1.0)); return ov
    if t<0.18: place(ov,box(rgb_split=int(8*(1-(t-0.05)/0.13)), hang_rot=swing(t)),1.04,1.04); return ov
    if t<T_OUT: place(ov,box(hang_rot=swing(t))); return ov
    if t<T_OUT+0.12: place(ov,box(),1.0,1-(t-T_OUT)/0.12); return ov
    return ov
def E(t):  # SLAM + double invert flash, shorter
    ov=Image.new('RGBA',(W,H),(0,0,0,0)); T2=0.72
    if t>=T2+0.08: return ov
    if t>=T2: place(ov,box(),1.0,1-(t-T2)/0.08); return ov
    if t<0.14: x=t/0.14; s=2.0-1.0*x*x; a=min(1,x*2); dx=dy=0
    else:
        s=1; a=1; k=math.exp(-(t-0.14)*14); dx=10*k*math.sin(t*90); dy=8*k*math.cos(t*77)
    inv=1.0 if (0.26<=t<0.30 or 0.35<=t<0.39) else 0.0
    place(ov,box(inv=inv,hang_rot=swing(max(0,t-0.14))),s,s,dx,dy,a); return ov
V={'E_slamflash':E,'A_pop':A,'B_slam':B,'C_stagger':C,'D_flash':D}
name=sys.argv[1]; fn=V[name]
dec=subprocess.Popen(['ffmpeg','-v','error','-i','hook_bg.mp4','-t','3.0','-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{RW}x{RH}','-r','60000/1001','-i','-','-i','hook_bg.mp4','-map','0:v','-map','1:a','-t','3.0',
 '-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart',f'HOOK_{name}_PREVIEW.mp4'],stdin=subprocess.PIPE)
i=0
while True:
    b=dec.stdout.read(RW*RH*3)
    if len(b)<RW*RH*3: break
    fr=np.frombuffer(b,np.uint8).reshape(RH,RW,3)
    ov=np.asarray(fn(i/FPS)); ov=cv2.resize(ov,(RW,RH),interpolation=cv2.INTER_AREA); a=ov[...,3:4]/255.
    enc.stdin.write((fr*(1-a)+ov[...,:3]*a).astype(np.uint8).tobytes()); i+=1
enc.stdin.close(); enc.wait(); print(name,i)
