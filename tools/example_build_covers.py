import sys; sys.path.insert(0,'/home/claude/scripts')
import numpy as np, cv2
from PIL import Image, ImageEnhance, ImageFilter
from layout import *
F='anton.ttf'
cardi=Image.open('cut_cardi_clean.png'); travis=Image.open('cut_travis.png')
bgc=Image.open('bg_vicecity.jpg').convert('RGB'); poster=Image.open('rs/festival_poster.jpg').convert('RGB')
logo=np.asarray(poster.crop((285,165,805,515))).astype(np.float32); lm=np.clip((logo.mean(2)-110)/100,0,1)
H,W=lm.shape; t=np.linspace(0,1,H)[:,None]
grad=np.dstack([255-0*t, 120+60*(1-t)-90*t, 60+150*t])  # orange (top) -> hot pink -> magenta
grad=np.broadcast_to(grad,(H,W,3)).clip(0,255).astype(np.uint8)
lr=Image.fromarray(np.dstack([grad,(lm*255).astype(np.uint8)]))
gl=Image.new('RGBA',lr.size,(255,70,160,255)); gl.putalpha(lr.getchannel('A').filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(10)))
sq=Image.new('RGBA',(560,560)); yy=np.linspace(0,1,560)[:,None]
bgarr=np.dstack([np.broadcast_to(20+40*yy,(560,560)),np.broadcast_to(10+10*yy,(560,560)),np.broadcast_to(45+40*yy,(560,560))]).astype(np.uint8)
sq=Image.fromarray(bgarr).convert('RGBA'); lr2=lr.resize((470,int(470*H/W))); gl2=gl.resize(lr2.size)
pos=(45,(560-lr2.size[1])//2); sq.alpha_composite(gl2,pos); sq.alpha_composite(lr2,pos)
fest_badge=badge(sq.convert('RGB'),(0,0,560,560),size=243)
bgv=bgc; shift=300
b=Image.new('RGB',(1080,1350)); b.paste(bgv.crop((0,shift,1080,1350)),(0,0)); b.paste(bgv.crop((0,1349,1080,1350)).resize((1080,shift)),(0,1350-shift))
b=ImageEnhance.Color(ImageEnhance.Contrast(b).enhance(1.1)).enhance(1.2).convert('RGBA')
art=Image.open('prof/svn_pegasus_art.jpg').convert('RGB')
G=lambda g,c=1.15,s=1.05,sh=70: dict(gamma=g,contrast=c,sat=s,sharpen=sh)
jobs={
 's1':dict(bg=b,people=[dict(name='cardi',cut=cardi,x=800,grade=G(0.67,1.14,1.02,90)),dict(name='travis',cut=travis,x=300,z=1,grade=G(0.9))],lines=["TRAVIS & CARDI","HEADLINE GTA VI'S","OWN FESTIVAL"],accent=[2],badge_img=badge(Image.open('justice/rs_art_clean.jpg').convert('RGB'),(30,240,1010,1220),size=243)),
 's2':dict(bg=bg_from('bgsrc_sots_20.png',blur=14,bright=0.6,sat=1.1),people=[dict(name='bex',cut=Image.open('cut_bex2.png'),x=520,grade=G(0.8))],lines=["PRODUCER BEXBLU","JUST HIT","1 MILLION LISTENERS"],accent=[2]),
 's3':dict(bg=bg_from('bgsrc_ray_20.png',blur=16,bright=0.55,sat=1.15),people=[dict(name='gotit',cut=Image.open('cut_gotit.png'),x=790,grade=G(0.9,1.12,1.0,60)),dict(name='ray',cut=Image.open('cut_ray.png'),x=290,z=1,grade=G(0.5,1.3))],lines=["RAYMATIX LINKS","LIL GOTIT ON","\"DONE DIS TING B4\""],accent=[1]),
 's5':dict(bg=bg_from('prof/svn_pegasus_art.jpg',blur=6,bright=0.7,sat=1.1),people=[dict(cut=Image.open('cut_svn.png'),x=520,grade=G(0.85,1.12))],lines=["SVN4VR IS BACK","WITH \"PEGASUS\""],accent=[1],badge_img=badge(Image.open('svnpp/svn_pp.png').convert('RGB'),(205,95,505,395),size=243)),
}
for k,j in jobs.items():
    _,issues,p,spot=build(j['bg'],j['people'],j['lines'],j['accent'],badge_img=j.get('badge_img'),out=f'final_{k}_cover.jpg',debug=f'dbg_{k}.jpg')
    print(k,[(pl['how'],round(pl['target_h']),round(pl['scale'],2)) for pl in p],'badge:',spot,'ISSUES:',issues or 'none')
m=Image.new('RGB',(432*4,540))
for i,k in enumerate(jobs): m.paste(Image.open(f'dbg_{k}.jpg'),(i*432,0))
m.save('dbg_all.jpg')
m=Image.new('RGB',(432*4,540))
for i,k in enumerate(jobs): m.paste(Image.open(f'final_{k}_cover.jpg').resize((432,540)),(i*432,0))
m.save('final_all.jpg')
