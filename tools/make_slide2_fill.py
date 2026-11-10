import argparse, subprocess
from PIL import Image
p=argparse.ArgumentParser(); p.add_argument("video"); p.add_argument("out")
p.add_argument("--crop", default=None); p.add_argument("--logo", default="compass_white.png")
p.add_argument("--overlay", default="s2fill_ov.png")
a=p.parse_args(); W,H=1080,1350
lg=Image.open(a.logo).convert("RGBA"); lw=120; lh=int(lg.height*lw/lg.width)
lg=lg.resize((lw,lh),Image.LANCZOS); lg.putalpha(lg.getchannel("A").point(lambda v:int(v*0.5)))
ov=Image.new("RGBA",(W,H),(0,0,0,0)); ov.alpha_composite(lg,((W-lw)//2,H-70-lh)); ov.save(a.overlay)
pre=f"crop={a.crop}," if a.crop else ""
fc=f"[0:v]{pre}scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1[b];[b][1:v]overlay=0:0[v]"
subprocess.run(["ffmpeg","-y","-i",a.video,"-i",a.overlay,"-filter_complex",fc,"-map","[v]","-map","0:a?","-c:v","libx264","-preset","veryfast","-pix_fmt","yuv420p","-crf","19","-r","30","-c:a","aac","-b:a","192k","-shortest","-movflags","+faststart",a.out],check=True,capture_output=True)
print("ok",a.out)
