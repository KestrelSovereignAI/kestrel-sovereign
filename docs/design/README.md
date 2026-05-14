# docs/design/

Canonical published brand assets — the refreshed kestrel-on-rook mark
adopted 2026-05-05. These ship in the public repo so READMEs, the website,
and OSS contributors can reference them directly.

## What's here

| Path | Asset |
|---|---|
| `kestrel_icon.svg` | Mark only — master SVG |
| `kestrel_logo.svg` | Full lockup (mark + outlined wordmark) — master SVG |
| `kestrel_bird_only.svg` | Bird silhouette without the castle |
| `KESTREL_LOGO.png` | Master raster (1254×1254) |
| `kestrel_avatar_400x400.png` | Square avatar |
| `kestrel_avatar_400x400_circle.png` | Round-cropped avatar |
| `launch/mark/` | Published mark bundle — `.min.svg`, `.ico`, PNG at 16/32/48/64/128/180/192/256/400/512/1024 (transparent + white-bg) |
| `launch/lockup/` | Published lockup bundle — `.min.svg`, PNG at 256/512/1024/1280/2048 widths (transparent + white-bg) |

The browser favicon and in-app nav mark in [`../../kestrel_sovereign/static/`](../../kestrel_sovereign/static/) (`favicon.svg`, `favicon.ico`, `kestrel_logo.svg`) are bit-identical copies of files under `launch/`.

## What's *not* here (and where it went)

Working brand toolkit — concept exploration, raster source files, fonts,
build pipeline, design spec, pre-refresh archive — lives in the private
[`kestrel-internal`](https://github.com/KestrelSovereignAI/kestrel-internal)
repo under `brand/`. That includes:

- Concept history (V1/V2 exploration, ecosystem variants, AI-generated grids)
- Pre-May-5 snapshot (the old painterly bird and its derivatives)
- Master raster sources the SVGs were traced from
- Variable Montserrat font (the lockup wordmark is committed as outlined paths, so no font is required at render time)
- Asset-pipeline scripts (`build_logo_assets.py`, `trace_logo.py`, `smooth_svg_paths.py`, `generate_branding_suite.py`)
- Brand guide, design spec, ASCII concept, original designer PDF
- Kestrel reference photography

The build script that regenerates the bundle in this directory now lives in
`kestrel-internal/brand/scripts/build_logo_assets.py`. Output paths still
target `kestrel-sovereign/docs/design/launch/`.

## Trademarks

The product code is Apache-2.0. The Kestrel mark and wordmark are trademarks
of the Kestrel Sovereign AI entity and are not covered by the OSS license.
