#!/usr/bin/env python3
"""Transcode the raw capture folders into web assets under assets/.

Source folders (g1/, soma_probe/) are left untouched; everything written here
is a derived, compressed copy that is safe to commit and serve.
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ASSETS = os.path.join(ROOT, "assets")

G1_SRC = os.path.join(ROOT, "g1")
PROBE_SRC = os.path.join(ROOT, "soma_probe")

DIR_G1 = os.path.join(ASSETS, "video", "g1")
DIR_PROBE = os.path.join(ASSETS, "video", "probe")
DIR_LOOP = os.path.join(ASSETS, "video", "loop")
DIR_POSTER = os.path.join(ASSETS, "poster")
DIR_FIG = os.path.join(ASSETS, "figures")

# Ordered so the gallery alternates between the two camera setups.
G1_ORDER = [
    "飞书20260920-105318.qt",
    "20260920-105059.mp4",
    "飞书20260920-105328.qt",
    "飞书20260920-105215.mp4",
    "飞书20260920-105334.qt",
    "飞书20260920-105240.mp4",
    "飞书20260920-105343.qt",
    "飞书20260920-105258.mp4",
    "飞书20260920-105348.qt",
    "飞书20260920-105303.mp4",
    "飞书20260920-105354.qt",
    "飞书20260920-105313.mp4",
    "飞书20260920-105358.qt",
    "飞书20260920-105245.mp4",
    "飞书20260920-105323.qt",
    "飞书20260920-105253.mp4",
    "飞书20260920-105338.qt",
    "20260920-105206.mp4",
]

PROBE_CROP_W, PROBE_CROP_H = 1488, 1036

PROBE_META = [
    ("rec_01KWEQSEMNBN24M3FV3XPA6JYF", "vogue"),
    ("rec_01KWG2PCC3JZFD7C1KHJHCJDE3", "walk-sway"),
    ("rec_01KWEKJYD75TA4QN88GESQTX5X", "crouch-forward"),
    ("rec_01KWECSJSEE0KWJ0MT9AT0B8C4", "jog-backward"),
    ("rec_01KWEGCJPXG8JAVJP4FECBM5VG", "side-steps"),
]

FFMPEG = ["ffmpeg", "-v", "error", "-y"]
X264 = ["-c:v", "libx264", "-preset", "slow", "-pix_fmt", "yuv420p",
        "-profile:v", "high", "-level", "4.0", "-an", "-movflags", "+faststart"]


def run(cmd):
    subprocess.run(cmd, check=True)


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-show_entries", "format=duration",
         "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    return (int(j["streams"][0]["width"]), int(j["streams"][0]["height"]),
            float(j["format"]["duration"]))


def poster(src, dst, at, width=900, grade=""):
    run(FFMPEG + ["-ss", f"{at:.2f}", "-i", src, "-frames:v", "1",
                  "-vf", f"{grade}scale={width}:-2:flags=lanczos",
                  "-q:v", "4", dst])


def build_g1():
    os.makedirs(DIR_G1, exist_ok=True)
    items = []
    for i, name in enumerate(G1_ORDER, 1):
        src = os.path.join(G1_SRC, name)
        if not os.path.exists(src):
            print("  ! missing", name)
            continue
        w, h, dur = probe(src)
        slug = f"g1-{i:02d}"
        dst = os.path.join(DIR_G1, slug + ".mp4")
        # lift the grey lab footage so it sits on the paper ground
        grade = ("eq=saturation=0.94:contrast=1.02:gamma=1.12:brightness=0.03,")
        run(FFMPEG + ["-i", src,
                      "-vf", grade + "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                      "-crf", "27"] + X264 + [dst])
        poster(src, os.path.join(DIR_POSTER, slug + ".jpg"), dur * 0.45,
               grade=grade)
        items.append({"id": slug, "src": f"assets/video/g1/{slug}.mp4",
                      "poster": f"assets/poster/{slug}.jpg",
                      "w": w, "h": h, "dur": round(dur, 2)})
        print(f"  {slug}  {w}x{h}  {dur:5.1f}s  "
              f"{os.path.getsize(dst)/1e6:5.2f} MB")
    return items


def build_probe():
    os.makedirs(DIR_PROBE, exist_ok=True)
    items = []
    for i, (rec, slug_tail) in enumerate(PROBE_META, 1):
        src = os.path.join(PROBE_SRC, rec + ".mp4")
        txt = os.path.join(PROBE_SRC, rec + ".txt")
        if not os.path.exists(src):
            print("  ! missing", rec)
            continue
        w, h, dur = probe(src)
        lines = []
        if os.path.exists(txt):
            lines = [l.strip() for l in open(txt).read().splitlines() if l.strip()]
        slug = f"probe-{i:02d}-{slug_tail}"
        dst = os.path.join(DIR_PROBE, slug + ".mp4")
        # the renders carry a grey viewport band along the bottom; character
        # pixels never reach below row 1022, so 1036 clips it away safely
        vf = f"crop={PROBE_CROP_W}:{PROBE_CROP_H}:0:0,scale=1116:-2:flags=lanczos"
        run(FFMPEG + ["-i", src, "-vf", vf, "-crf", "26"] + X264 + [dst])
        run(FFMPEG + ["-ss", f"{dur * 0.5:.2f}", "-i", src, "-frames:v", "1",
                      "-vf", f"crop={PROBE_CROP_W}:{PROBE_CROP_H}:0:0,"
                             "scale=900:-2:flags=lanczos",
                      "-q:v", "4", os.path.join(DIR_POSTER, slug + ".jpg")])
        items.append({
            "id": slug,
            "src": f"assets/video/probe/{slug}.mp4",
            "poster": f"assets/poster/{slug}.jpg",
            "caption": lines[0] if lines else "",
            "rmse": next((l.split(":", 1)[1].strip()
                          for l in lines if l.lower().startswith("smpl-x fit")), ""),
            "w": w, "h": h, "dur": round(dur, 2),
        })
        print(f"  {slug}  {w}x{h}  {dur:5.1f}s  "
              f"{os.path.getsize(dst)/1e6:5.2f} MB")
    return items


def build_loops(g1_items):
    """Tiny silent loops for the hero film strip."""
    os.makedirs(DIR_LOOP, exist_ok=True)
    picks = g1_items[:10]
    out = []
    for it in picks:
        src = os.path.join(ROOT, it["src"])
        start = max(0.0, it["dur"] * 0.35)
        dst = os.path.join(DIR_LOOP, it["id"] + ".mp4")
        run(FFMPEG + ["-ss", f"{start:.2f}", "-t", "4.5", "-i", src,
                      "-vf", "scale=512:-2:flags=lanczos", "-r", "24",
                      "-crf", "31"] + X264 + [dst])
        out.append({"id": it["id"], "src": f"assets/video/loop/{it['id']}.mp4",
                    "poster": it["poster"]})
        print(f"  loop {it['id']}  {os.path.getsize(dst)/1e3:6.0f} KB")
    return out


def build_figures():
    os.makedirs(DIR_FIG, exist_ok=True)
    src_dir = "/tmp/semoco_figs"
    for name in ("tokenizer", "generator", "dataset", "qualitative"):
        s = os.path.join(src_dir, name + ".png")
        if not os.path.exists(s):
            print("  ! missing figure", name, "- run tools/extract_figures.py")
            continue
        from PIL import Image
        im = Image.open(s).convert("RGB")
        if im.width > 2400:
            im = im.resize((2400, round(im.height * 2400 / im.width)),
                           Image.LANCZOS)
        d = os.path.join(DIR_FIG, name + ".png")
        im.save(d, optimize=True)
        im.save(os.path.join(DIR_FIG, name + ".webp"), quality=92, method=6)
        print(f"  {name}  {im.size}  {os.path.getsize(d)/1e6:.2f} MB")


if __name__ == "__main__":
    only = sys.argv[1:] or ["figures", "g1", "probe", "loops"]
    for d in (DIR_POSTER, DIR_FIG, ASSETS):
        os.makedirs(d, exist_ok=True)

    manifest_path = os.path.join(ASSETS, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))

    if "figures" in only:
        print("figures:"); build_figures()
    if "g1" in only:
        print("g1:"); manifest["g1"] = build_g1()
    if "probe" in only:
        print("probe:"); manifest["probe"] = build_probe()
    if "loops" in only and manifest.get("g1"):
        print("loops:"); manifest["loop"] = build_loops(manifest["g1"])

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("\nmanifest ->", manifest_path)
