"""Lay out actual RGBA outputs. Alpha is preserved; no generated backgrounds or cutout retouching."""
from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1] / 'assets'
INK, PAPER, MUTED = '#173f3c', '#f7f4eb', '#6c7d75'
SANS = '/usr/share/fonts/truetype/lato/Lato-Regular.ttf'


def face(size, italic=False, weight=650):
    font = ImageFont.truetype(str(ROOT / 'fonts' / ('Fraunces-Italic.ttf' if italic else 'Fraunces.ttf')), size)
    font.set_variation_by_axes([144, weight, 60, 1])
    return font


def text(canvas, xy, value, size=26, color=INK, display=False, italic=False, weight=650):
    font = face(size, italic, weight) if display else ImageFont.truetype(SANS, size)
    ImageDraw.Draw(canvas).text(xy, value, font=font, fill=color, anchor='lt')


def star(canvas, x, y, radius, color=INK):
    points=[]
    for i in range(8):
        angle=math.pi*i/4
        r=radius if i%2==0 else radius*.2
        points.append((x+math.cos(angle)*r,y+math.sin(angle)*r))
    ImageDraw.Draw(canvas).polygon(points,fill=color)


def asset(canvas, name, box):
    image=Image.open(ROOT/'samples'/f'{name}.png').convert('RGBA')
    # Crop only fully transparent margins; keep the actual soft alpha edges.
    crop=image.getchannel('A').getbbox()
    image=image.crop(crop)
    x,y,w,h=box
    image.thumbnail((w,h),Image.Resampling.LANCZOS)
    canvas.alpha_composite(image,(int(x+(w-image.width)/2),int(y+(h-image.height)/2)))


def checker(canvas, box, step=32):
    x,y,w,h=box;draw=ImageDraw.Draw(canvas)
    draw.rectangle((x,y,x+w,y+h),fill='#f1eee7')
    for yy in range(y,y+h,step):
        for xx in range(x,x+w,step):
            if ((xx-x)//step+(yy-y)//step)%2==0:
                draw.rectangle((xx,yy,min(xx+step,x+w),min(yy+step,y+h)),fill='#e7e6df')


def corners(canvas, box, color='#92aaa0'):
    x,y,w,h=box;d=ImageDraw.Draw(canvas);n=30
    for xx,yy,dx,dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
        d.line([(xx+dx*n,yy),(xx,yy),(xx,yy+dy*n)],fill=color,width=2)
        d.rectangle((xx-4,yy-4,xx+4,yy+4),fill=PAPER,outline=color,width=2)


def banner():
    im=Image.new('RGBA',(2000,900),PAPER);d=ImageDraw.Draw(im)
    d.ellipse((1020,-240,2140,910),fill='#e3eade')
    d.ellipse((1005,55,1435,485),fill='#ecccbb')
    text(im,(82,70),'CKMOSS  /  TRANSPARENT IMAGE GENERATION',23,MUTED)
    text(im,(78,172),'CKMoss',143,display=True,italic=True,weight=650)
    text(im,(72,338),'RGBA',245,display=True,weight=700)
    star(im,865,360,39,'#ca6f4f')
    text(im,(88,639),'Let your ideas',61,display=True,italic=True,weight=500)
    text(im,(88,713),'live anywhere.',61,display=True,italic=True,weight=500)
    d.line((92,828,770,828),fill='#bec7bd',width=2)
    text(im,(90,846),'CHARACTERS. OBJECTS. EVERY LAYER OF YOUR WORLD.',18,MUTED)
    checker(im,(1460,145,390,600),30)
    corners(im,(1440,125,430,640))
    asset(im,'navigator',(1300,28,680,840))
    asset(im,'fox',(968,80,470,420))
    asset(im,'teapot',(955,561,430,270))
    star(im,1220,515,18,'#ca6f4f');star(im,1908,796,31)
    im.convert('RGB').save(ROOT/'banner.png',optimize=True)


def poster():
    im=Image.new('RGBA',(2000,2520),PAPER);d=ImageDraw.Draw(im)
    text(im,(95,65),'CKMOSS  /  AN OPEN CANVAS FOR YOUR IDEAS',25,MUTED)
    text(im,(86,140),'Made to',154,display=True,italic=True,weight=550)
    text(im,(86,307),'stand out.',162,display=True,weight=680)
    star(im,924,204,40,'#ca6f4f')
    text(im,(103,530),'Characters, little worlds, and objects to make your own.',28,MUTED)
    d.ellipse((1190,55,1920,785),fill='#dfe8dc')
    corners(im,(1310,190,500,690))
    asset(im,'navigator',(1230,120,680,850))
    asset(im,'fox',(115,655,600,490))
    d.ellipse((708,625,1118,1035),fill='#e9ded0')
    asset(im,'teapot',(670,665,490,430))
    text(im,(104,1190),'A world of possibilities.',72,display=True,italic=True,weight=550)
    text(im,(109,1290),'One prompt. A real alpha channel. Ready for your composition.',27,MUTED)
    d.line((105,1360,1895,1360),fill='#bec7bd',width=2)
    panels=[(105,1430,410,780),(565,1430,410,780),(1025,1430,410,780),(1485,1430,410,780)]
    colors=['#e5e9e0','#e9e4ee','#f0e1d1','#e3e7e5']
    for panel,color in zip(panels,colors):
        x,y,w,h=panel;d.rounded_rectangle((x,y,x+w,y+h),radius=205,fill=color)
    asset(im,'botanist',(110,1460,400,710))
    asset(im,'witch',(540,1450,460,730))
    asset(im,'boy',(1020,1460,420,720))
    asset(im,'knight',(1460,1450,460,730))
    for x,label in [(105,'01 / CHARACTERS'),(565,'02 / FANTASY'),(1025,'03 / PERSONALITY'),(1485,'04 / ADVENTURE')]:
        text(im,(x+15,2250),label,22,MUTED)
    d.line((105,2330,1895,2330),fill='#bec7bd',width=2)
    text(im,(105,2387),'CKMoss RGBA',51,display=True,italic=True,weight=600)
    text(im,(1080,2394),'Actual model outputs · composited for display',25,MUTED)
    star(im,1870,2410,22,'#ca6f4f')
    im.convert('RGB').save(ROOT/'showcase.png',optimize=True)


if __name__=='__main__':
    banner();poster()
