# MERL README artwork

The cover and trust-decision diagram extend the project website's SLAI violet
(`#92278F`), magenta (`#C82E70`) and pale editorial palette. Grounded experience is
blue; imagination is orange. The SVGs are static vector artwork with outlined
Manrope glyphs, so they retain their appearance without external fonts, scripts,
SVG animation or embedded HTML.

The two GIFs use the original MERL-Full videos prepared for the
[project website](https://li-wenzhan.github.io/MERL-project/#robots). They retain
the complete source chronology and the visible **2× physical-time playback**
mark. Conversion uses 440-pixel width, 8 fps and a 128-color palette. Only spatial
resolution, frame sampling and color quantization change; the clips are not
retimed. Evaluation trial counts describe the study, not a count of clips shown.

To regenerate, check out the public `MERL-project` repository beside this one,
install fontTools with WOFF2 support, and make ffmpeg available:

```sh
python assets/readme/build_assets.py \
  --font ../MERL-project/assets/fonts/manrope.woff2 \
  --video-dir ../MERL-project/assets/videos
```

Manrope is used under its SIL Open Font License; the original font and license
remain in the homepage repository. Research video and figure rights remain with
their respective authors. This directory does not grant a blanket license over
the repository or upstream research assets.
