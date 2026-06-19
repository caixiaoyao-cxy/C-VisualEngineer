#!/usr/bin/env python3
import json, math, os, random, re, sys, io, urllib
from pathlib import Path
import numpy as np, requests, cv2
from PIL import Image, ImageDraw, ImageFont

OUTPUT = Path("output_emoji_video"); OUTPUT.mkdir(exist_ok=True)
TEMP_ICONS = OUTPUT / "temp_icons"; TEMP_ICONS.mkdir(exist_ok=True)
VIDEO_FPS = 24; VIDEO_DURATION = 5.0; ANIM_DURATION = 3.0; W, H = 800, 600

def get_map_shape(place):
    mask_path = str(OUTPUT / "mask.png")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; EmojiMap/1.0)"}
    url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(place)}&format=json&limit=1"
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200 and r.json(): lat, lon = float(r.json()[0]["lat"]), float(r.json()[0]["lon"])
        else: raise ValueError
    except:
        print("  Nominatim fail, use fallback ellipse")
        img=Image.new("L",(W,H),0); ImageDraw.Draw(img).ellipse([W//4,H//6,W*3//4,H*5//6],fill=255); img.save(mask_path); return mask_path
    zoom=12; size=600; n=2.0**zoom
    cx=int((lon+180)/360*n); cy=int((1-math.log(math.tan(math.radians(lat))+1/math.cos(math.radians(lat)))/math.pi)/2*n)
    ts=256; tl=int(math.ceil(size/ts))
    img=Image.new("RGB",(size,size),(255,255,255)); loaded=0
    for dx in range(-tl//2,tl//2+1):
        for dy in range(-tl//2,tl//2+1):
            try:
                r=requests.get(f"https://tile.openstreetmap.org/{zoom}/{cx+dx}/{cy+dy}.png",headers=headers,timeout=10)
                if r.status_code==200: img.paste(Image.open(io.BytesIO(r.content)),((dx+tl//2)*ts,(dy+tl//2)*ts)); loaded+=1
            except: pass
    if loaded==0:
        print("  no tiles, use fallback")
        img=Image.new("L",(W,H),0); ImageDraw.Draw(img).ellipse([W//4,H//6,W*3//4,H*5//6],fill=255); img.save(mask_path); return mask_path
    gray=cv2.cvtColor(np.array(img),cv2.COLOR_RGB2GRAY)
    _,thresh=cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)
    contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    ma=np.zeros((size,size),np.uint8)
    if contours: cv2.drawContours(ma,[max(contours,key=cv2.contourArea)],-1,255,thickness=cv2.FILLED)
    ys,xs=np.where(ma>0)
    if len(xs)>0:
        cxo,cyo=int(xs.mean()),int(ys.mean()); bw=int(xs.max()-xs.min())+20; bh=int(ys.max()-ys.min())+20
        s=min(W*0.6/bw,H*0.6/bh); out=np.zeros((H,W),np.uint8)
        for y,x in zip(ys,xs):
            nx=int((x-(cxo-bw//2))*s)+(W-int(bw*s))//2; ny=int((y-(cyo-bh//2))*s)+(H-int(bh*s))//2
            if 0<=nx<W and 0<=ny<H: out[ny,nx]=255
        Image.fromarray(out).save(mask_path)
    else: Image.fromarray(ma).resize((W,H)).save(mask_path)
    print(f"  mask saved ({loaded} tiles)"); return mask_path

def get_local_symbols(place):
    headers = {"User-Agent": "Mozilla/5.0"}
    queries = [f"{place}\u6587\u5316", f"{place}\u666f\u70b9", f"{place}\u7279\u8272", f"{place}\u53e4\u8ff9", place]
    stopwords = {"\u81f3\u4eca","\u5171\u63a5","\u8fdb\u57ce","\u52a1\u5de5","\u5b50\u5973","\u4e49\u52a1","\u6559\u80b2","\u9636\u6bb5","\u4f4d\u4e8e","\u5c5e\u4e8e","\u5305\u62ec","\u4e3b\u8981","\u8457\u540d","\u91cd\u8981","\u5408\u8ba1","\u653f\u7b56","\u65b9\u6848","\u5b9e\u65bd","\u63d0\u4f9b","\u8fdb\u884c","\u5efa\u7acb"}
    seen=set(); result=[]
    for q in queries:
        if len(result)>=6: break
        url=f"https://zh.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(q)}&format=json&srlimit=20&utf8=1"
        try:
            r=requests.get(url,headers=headers,timeout=15).json()
            for p in r.get("query",{}).get("search",[]):
                title=p.get("title","")
                for w in re.findall(r"[\u4e00-\u9fff]{2,6}",title):
                    if len(w)>=2 and w[:2] not in seen and not any(s in w for s in stopwords):
                        seen.add(w[:2]); result.append(w)
                    if len(result)>=6: break
                if len(result)>=6: break
        except: pass
    if result: print(f"  culture: {result}"); return result
    for q in [f"{place} culture", f"{place} tourism", place]:
        if len(result)>=3: break
        url=f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(q)}&format=json&srlimit=10&utf8=1"
        try:
            r=requests.get(url,headers=headers,timeout=15).json()
            for p in r.get("query",{}).get("search",[]):
                title=p.get("title","")
                for w in re.findall(r"[A-Z][a-z]+",title):
                    if len(w)>=4 and w.lower() not in seen:
                        seen.add(w.lower()); result.append(w)
                    if len(result)>=3: break
                if len(result)>=3: break
        except: pass
    if result: print(f"  culture(en): {result}"); return result
    print(f"  fallback"); return ["\U0001f30d", "\u2b50", "\U0001f332"]

EMOJI_MAP = {"\u8336":"\U0001f375","\u9152":"\U0001f377","\u9c7c":"\U0001f41f","\u5854":"\U0001f5fc","\u6865":"\U0001f309","\u57ce":"\U0001f3f0","\u5c71":"\u26f0\ufe0f","\u6e56":"\U0001f3de\ufe0f","\u6d77":"\U0001f30a","\u9f99":"\U0001f409","\u82b1":"\U0001f338","\u706f":"\U0001f3ee","\u8239":"\U0001f6a2","\u706b":"\U0001f525","\u6c34":"\U0001f4a7","\u7af9":"\U0001f38b","\u9a6c":"\U0001f434","\u9e1f":"\U0001f426","\u5e08":"\U0001f981","\u8c61":"\U0001f418","\u6843":"\U0001f351","\u6885":"\U0001f338","\u84b8":"\U0001f373","\u997a":"\U0001f95f","\u9762":"\U0001f35c","\u996d":"\U0001f35a","\u7c73":"\U0001f35a","\u5706":"\U0001f30d","\u5b9d":"\U0001f4e6","\u5251":"\U0001f5e1\ufe0f","\u6cc9":"\U0001f4a7","\u5ef6":"\U0001f3db\ufe0f"}
COLORS = [(255,100,100),(100,200,255),(255,200,80),(120,220,120),(200,150,255),(255,160,200)]

def find_font(pathes, size):
    for p in pathes:
        try: return ImageFont.truetype(p, size)
        except: pass
    return ImageFont.load_default()

def gen_icons(symbols):
    paths=[]
    for idx,sym in enumerate(symbols[:8]):
        emoji=next((v for k,v in EMOJI_MAP.items() if k in sym or sym in k), None)
        sz=random.randint(50,80); img=Image.new("RGBA",(sz+20,sz+20),(0,0,0,0)); d=ImageDraw.Draw(img)
        ef=find_font(["NotoEmoji-Regular.ttf","seguiemj.ttf"],sz)
        if emoji and ef:
            bb=d.textbbox((0,0),emoji,font=ef)
            d.text(((img.width-(bb[2]-bb[0]))//2-bb[0],(img.height-(bb[3]-bb[1]))//2-bb[1]),emoji,font=ef,fill=(255,255,255,255))
        else:
            c=COLORS[idx%len(COLORS)]; d.rounded_rectangle([4,4,img.width-4,img.height-4],radius=10,fill=c+(200,))
            tf=find_font(["NotoSansCJK-Regular.ttc","msyh.ttc","simhei.ttf"],32)
            bb=d.textbbox((0,0),sym[:4],font=tf)
            d.text(((img.width-(bb[2]-bb[0]))//2-bb[0],(img.height-(bb[3]-bb[1]))//2-bb[1]),sym[:4],font=tf,fill=(255,255,255,255))
        p=TEMP_ICONS/f"icon_{idx:02d}.png"; img.save(p); paths.append(str(p)); print(f"  icon: {sym}")
    return paths

def compose_static(icon_paths,mask_path):
    mask=np.array(Image.open(mask_path).convert("L")); canvas=Image.new("RGBA",(W,H),(255,255,255,255))
    ys,xs=np.where(mask>0)
    if len(xs)==0: return canvas
    pts=[(int(xs[j]),int(ys[j])) for j in np.linspace(0,len(xs)-1,max(40,len(icon_paths)*8),endpoint=False,dtype=int)]
    for i,(px,py) in enumerate(pts):
        icon=Image.open(icon_paths[i%len(icon_paths)]).convert("RGBA"); sz=random.randint(25,55)
        r=icon.resize((sz,sz),Image.LANCZOS).rotate(random.randint(0,360),expand=True,fillcolor=(0,0,0,0))
        canvas.paste(r,(px-r.width//2+random.randint(-6,6),py-r.height//2+random.randint(-6,6)),r)
    return canvas

def gen_video(icon_paths,mask_path):
    mask=np.array(Image.open(mask_path).convert("L")); ys,xs=np.where(mask>0)
    pts=[(int(xs[j]),int(ys[j])) for j in np.linspace(0,len(xs)-1,max(40,len(icon_paths)*8),endpoint=False,dtype=int)] if len(xs)>0 else [(W//2,H//2)]*40
    dirs=[(-W//2,-H//2),(W,-H//2),(W+W//2,-H//2),(-W//2,H+H//2),(W+W//2,H+H//2)]
    data=[]
    for i,(ex,ey) in enumerate(pts):
        icon=Image.open(icon_paths[i%len(icon_paths)]).convert("RGBA"); sz=random.randint(25,55)
        r=icon.resize((sz,sz),Image.LANCZOS).rotate(random.randint(0,360),expand=True,fillcolor=(0,0,0,0))
        dx,dy=random.choice(dirs); data.append((r,dx+random.randint(-80,80),dy+random.randint(-80,80),ex,ey))
    nf=int(VIDEO_DURATION*VIDEO_FPS); af=int(ANIM_DURATION*VIDEO_FPS)
    fourcc=cv2.VideoWriter_fourcc(*"mp4v"); w=cv2.VideoWriter(str(OUTPUT/"output_video.mp4"),fourcc,VIDEO_FPS,(W,H))
    for fi in range(nf):
        canvas=Image.new("RGBA",(W,H),(255,255,255,255))
        if fi<af:
            t=1-(1-fi/af)**2
            for img,sx,sy,ex,ey in data:
                cx=int(sx+(ex-sx)*t); cy=int(sy+(ey-sy)*t)
                r=img.rotate(int(360*(1-t)),expand=True,fillcolor=(0,0,0,0))
                if cx+r.width>=0 and cx<W and cy+r.height>=0 and cy<H: canvas.paste(r,(cx-r.width//2,cy-r.height//2),r)
        else:
            bt=(fi-af)/max(nf-af,1); oy=int(4*math.sin(bt*4*math.pi))
            for img,*_,ex,ey in data:
                if ex+img.width>=0 and ex<W and ey+img.height+oy>=0 and ey+oy<H: canvas.paste(img,(ex-img.width//2,ey-img.height//2+oy),img)
        w.write(cv2.cvtColor(np.array(canvas.convert("RGB")),cv2.COLOR_RGB2BGR))
    w.release(); print(f"  Done: {OUTPUT/'output_video.mp4'}")

if __name__=="__main__":
    place=sys.argv[1] if len(sys.argv)>1 else "\u676d\u5dde"
    print(f"1. Map {place}"); mp=get_map_shape(place)
    print(f"2. Culture"); syms=get_local_symbols(place)
    print(f"3. Icons"); icons=gen_icons(syms)
    print(f"4. Static"); s=compose_static(icons,mp); s.save(OUTPUT/"final_static.png")
    print(f"5. Video"); gen_video(icons,mp)
