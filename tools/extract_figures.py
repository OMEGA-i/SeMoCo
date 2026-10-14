import subprocess, os
from PIL import Image, ImageChops

PDF = "/Users/a26044/projects/Semoco_Webpage/2608.24334v2.pdf"
OUT = "/tmp/semoco_figs"
os.makedirs(OUT, exist_ok=True)

def render(page, dpi=500):
    p = f"{OUT}/full_{page}"
    subprocess.run(["pdftoppm", "-r", str(dpi), "-f", str(page), "-l", str(page),
                    "-png", PDF, p], check=True)
    # pdftoppm appends -NN
    for f in os.listdir(OUT):
        if f.startswith(f"full_{page}-") and f.endswith(".png"):
            return os.path.join(OUT, f)
    raise RuntimeError("no render")

def trim(im, bg=(255, 255, 255), pad=24):
    bgimg = Image.new(im.mode, im.size, bg)
    diff = ImageChops.difference(im, bgimg)
    diff = ImageChops.add(diff, diff, 2.0, -30)
    bbox = diff.getbbox()
    if not bbox:
        return im
    l, t, r, b = bbox
    l = max(0, l - pad); t = max(0, t - pad)
    r = min(im.width, r + pad); b = min(im.height, b + pad)
    return im.crop((l, t, r, b))

jobs = [
    # page, (y0, y1) normalized band holding the figure only — stop short of
    # the printed caption, the page reproduces it in HTML instead.
    (4, 0.035, 0.322, "tokenizer"),
    (6, 0.070, 0.480, "generator"),
    (7, 0.060, 0.318, "dataset"),
    (8, 0.620, 0.806, "qualitative"),
]

for page, y0, y1, name in jobs:
    src = render(page)
    im = Image.open(src).convert("RGB")
    W, H = im.size
    band = im.crop((0, int(H * y0), W, int(H * y1)))
    fig = trim(band)
    out = f"{OUT}/{name}.png"
    fig.save(out)
    print(name, "page", page, "page_size", (W, H), "->", fig.size, out)
