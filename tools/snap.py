import json, wave, numpy as np
def load(wav):
    w=wave.open(wav); sr=w.getframerate(); a=np.frombuffer(w.readframes(w.getnframes()),np.int16).astype(np.float32)/32768
    if w.getnchannels()>1: a=a.reshape(-1,w.getnchannels()).mean(1)
    hop=int(sr*0.01); n=len(a)//hop
    db=20*np.log10(np.sqrt((a[:n*hop].reshape(n,hop)**2).mean(1))+1e-6); return db
def snap(words, db, i, j, pre=0.12, post=0.18, maxext=0.45):
    s=words[i][0]-pre; e=words[j][1]+post
    if i>0: s=max(s, words[i-1][1]+0.03)
    nxt = words[j+1][0]-0.04 if j+1<len(words) else e+maxext
    # extend end while still loud (word tail), relative to local floor
    k0=int(words[j][1]*100); seg=db[max(0,k0-200):k0+100]; floor=np.percentile(seg,15)
    k=k0
    while k<len(db) and k<k0+int(maxext*100) and db[k]>floor+8: k+=1
    e=max(e, k/100+0.10); e=min(e, nxt)
    # start: walk back to quiet
    k=int(words[i][0]*100); lim=k-25
    while k>lim and db[k-1]>floor+8: k-=1
    s=min(s, k/100-0.05)
    if i>0: s=max(s, words[i-1][1]+0.03)
    return round(s,2), round(e,2)
