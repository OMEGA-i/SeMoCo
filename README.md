# SeMoCo — project page

Static project page for *SeMoCo: A Semantic-First Motion Codec for Motion Language Modeling*
(arXiv:2608.24334). No build step, no dependencies — plain HTML, CSS and JavaScript.

## Run locally

```bash
python3 -m http.server 4321
# http://127.0.0.1:4321
```

## Layout

```
index.html          markup and copy
styles.css          theme, layout, components
main.js             lazy video playback, lightbox, scroll reveal, hero canvas
archive/v1-dark/    the earlier dark-theme build, kept runnable
assets/
  fonts/            self-hosted woff2 + fonts.css
  figures/          Figures 1–4 extracted from the PDF (png + webp)
  video/g1/         18 Unitree G1 takes
  video/probe/      5 stylized-avatar probe clips
  video/loop/       short low-bitrate loops for the hero film strip
  poster/           first-paint stills for every clip
  paper/SeMoCo.pdf  copy of the manuscript
  manifest.json     what the media build produced
tools/
  extract_figures.py  crops the figures out of the PDF
  build_media.py      transcodes g1/ and soma_probe/ into assets/
```

## Design

A blueprint-blue ground carries the page, with a violet band on the codec and
avatar sections, one bright band on the corpus, and an inverted navy section for
the hardware footage. `.section--ink` re-points the same custom properties rather
than restating any component, so every card, table and badge inverts on its own.

Sections are not butted colour blocks. One four-column rule system runs from the
hero to the footer at the same x in every section, showing through the whitespace
and hidden behind the cards, and each section opens on a full-bleed rule plus a
running head carrying its index, its name and the packet mark. The transition is
the design element; the change of ground is secondary.

Type is Fraunces (display), Archivo (text) and IBM Plex Mono (labels and data),
all served from `assets/fonts/` — the page makes no external requests at all,
which matters on networks where the font CDNs are slow or blocked. Fraunces and
Archivo are variable; `fonts.css` ships one file per subset and axis range.

Not every clip is on the page. The G1 grid shows three takes with distinct
silhouettes and the rail under the player reaches all eighteen; the probe grid
shows four of the five renders.

## Regenerating media

The raw capture folders (`g1/`, `soma_probe/`) and the source PDF are gitignored.
With them present, and `ffmpeg` plus `pdftoppm` on PATH:

```bash
python3 tools/extract_figures.py     # writes /tmp/semoco_figs
python3 tools/build_media.py         # writes assets/
python3 tools/build_media.py g1      # or a single group: figures|g1|probe|loops|paper
```

`build_media.py` strips audio, lifts the grey lab footage so it sits on the light
ground, and crops the grey viewport band out of the probe renders. Total shipped
media is ~30 MB.

## Archive

`archive/v1-dark/` holds the first, dark-theme version. It shares the same media
folder (paths rewritten to `../../assets/`), so it stays viewable at
`/archive/v1-dark/` without duplicating any video. It renders the current, lighter
colour grade of the G1 clips rather than the darker one it originally shipped with.

## Deploying to GitHub Pages

Push to a repository, then set **Settings → Pages → Source** to the default branch,
root folder. The page is entirely relative-path based, so it also works from a
project subpath such as `user.github.io/semoco/`.

## Notes

- Videos use `preload="none"` with poster stills and only play while on screen.
- Author links in the hero are placeholders (`href="#"`); fill them in as needed.
- The avatar section is labelled work in progress, matching the current state of
  that experiment.
