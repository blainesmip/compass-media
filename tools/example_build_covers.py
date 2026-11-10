import sys; sys.path.insert(0,'/home/claude/scripts')
import numpy as np, cv2
from PIL import Image, ImageEnhance, ImageFilter
from layout import *
F='anton.ttf'
cardi=Image.open('cut_cardi_ready.png'); travis=Image.open('cut_travis.png')
bgc=Image.open('bg_vicecity.jpg').convert('RGB'); poster=Image.open('rs/festival_poster.jpg').convert('RGB')
sun=ImageEnhance.Color(bgc.crop((180,830,900,1190)).resize((560,560))).enhance(1.3)
logo=np.asarray(poster.crop((295,175,795,505))).astype(np.float32); lm=np.clip((logo.mean(2)-120)/100,0,1)
lr=Image.fromarray(np.dstack([np.full(lm.shape+(3,),255,np.uint8),(lm*255).astype(np.uint8)]))
gl=Image.new('RGBA',lr.size,(255,60,170,255)); gl.putalpha(lr.getchannel('A').filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(8)))
sq=sun.convert('RGBA'); sq.alpha_composite(Image.new('RGBA',sq.size,(10,6,30,120))); sq.alpha_composite(gl,(30,115)); sq.alpha_composite(gl,(30,115)); sq.alpha_composite(lr,(30,115))
fest_badge=badge(sq.convert('RGB'),(0,0,560,560),size=225)
bgv=bgc; shift=300
b=Image.new('RGB',(1080,1350)); b.paste(bgv.crop((0,shift,1080,1350)),(0,0)); b.paste(bgv.crop((0,1349,1080,1350)).resize((1080,shift)),(0,1350-shift))
b=ImageEnhance.Color(ImageEnhance.Contrast(b).enhance(1.1)).enhance(1.2).convert('RGBA')
art=Image.open('prof/svn_pegasus_art.jpg').convert('RGB')
G=lambda g,c=1.15,s=1.05,sh=70: dict(gamma=g,contrast=c,sat=s,sharpen=sh)
jobs={
 's1':dict(bg=b,people=[dict(cut=cardi,x=800,grade=G(0.67,1.14,1.02,90)),dict(cut=travis,x=300,z=1,grade=G(0.9))],lines=["TRAVIS & CARDI","HEADLINE GTA VI'S","OWN FESTIVAL"],accent=[2],badge_img=fest_badge),
 's2':dict(bg=bg_from('bgsrc_sots_20.png',blur=14,bright=0.6,sat=1.1),people=[dict(cut=Image.open('cut_bex2.png'),x=520,grade=G(0.8))],lines=["PRODUCER BEXBLU","JUST HIT","1 MILLION LISTENERS"],accent=[2],badge_img=stat_badge('1M','MONTHLY LISTENERS',F)),
 's3':dict(bg=bg_from('bgsrc_ray_20.png',blur=16,bright=0.55,sat=1.15),people=[dict(cut=Image.open('cut_aux.png'),x=820,grade=G(0.62,1.2,1.05,80)),dict(cut=Image.open('cut_ray.png'),x=290,z=1,grade=G(0.58,1.3))],lines=["RAYMATIX GETS THE","AUXLUA CO-SIGN"],accent=[1]),
 's5':dict(bg=bg_from('prof/svn_pegasus_art.jpg',blur=6,bright=0.7,sat=1.1),people=[dict(cut=Image.open('cut_svn.png'),x=520,grade=G(0.85,1.12))],lines=["SVN4VR IS BACK","WITH \"PEGASUS\""],accent=[1],badge_img=badge(art,(1450,0,2379,929),size=225)),
}
for k,j in jobs.items():
    _,issues,p=build(j['bg'],j['people'],j['lines'],j['accent'],badge_img=j.get('badge_img'),out=f'final_{k}_cover.jpg',debug=f'dbg_{k}.jpg')
    print(k,[pl['how'] for pl in p],'ISSUES:',issues or 'none')
m=Image.new('RGB',(432*4,540))
for i,k in enumerate(jobs): m.paste(Image.open(f'dbg_{k}.jpg'),(i*432,0))
m.save('dbg_all.jpg')
m=Image.new('RGB',(432*4,540))
for i,k in enumerate(jobs): m.paste(Image.open(f'final_{k}_cover.jpg').resize((432,540)),(i*432,0))
m.save('final_all.jpg')
