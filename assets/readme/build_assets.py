"""Build MERL's README artwork from the project font and original robot videos.

Example, with the public homepage repository checked out beside this repository:
  python assets/readme/build_assets.py \
      --font ../MERL-project/assets/fonts/manrope.woff2 \
      --video-dir ../MERL-project/assets/videos

Requires fontTools (with WOFF2 support) and ffmpeg. No model output is synthesized.
"""
import argparse
from pathlib import Path
import subprocess

from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--font', type=Path, required=True)
parser.add_argument('--video-dir', type=Path, required=True)
args = parser.parse_args()
output = Path(__file__).resolve().parent
base = TTFont(args.font)
fonts = {weight: instantiateVariableFont(base, {'wght': weight}, inplace=False)
         if 'fvar' in base else base for weight in (500, 600, 800)}


def text(label, x, y, size, color='#291c32', weight=600, anchor='start'):
    """Outline the supplied Manrope glyphs, keeping SVGs font-independent."""
    font = fonts[weight]
    glyphs, cmap = font.getGlyphSet(), font.getBestCmap()
    scale = size / font['head'].unitsPerEm
    width = sum(font['hmtx'][cmap[ord(c)]][0] for c in label) * scale
    if anchor == 'middle':
        x -= width / 2
    elif anchor == 'end':
        x -= width
    pen = SVGPathPen(glyphs, ntos=lambda n: f'{n:.3f}'.rstrip('0').rstrip('.') or '0')
    for char in label:
        glyph = cmap[ord(char)]
        glyphs[glyph].draw(TransformPen(pen, (scale, 0, 0, -scale, x, y)))
        x += font['hmtx'][glyph][0] * scale
    return f'<path fill="{color}" d="{pen.getCommands()}"/>'


def rect(x, y, w, h, fill, stroke='none', radius=12):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>'


def save(name, width, height, title, content, defs=''):
    (output / name).write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{title}"><title>{title}</title><defs>{defs}</defs>'
        + ''.join(content) + '</svg>\n', encoding='utf-8')


banner = [rect(1, 1, 1598, 418, 'url(#paper)', '#e3d4e8', 26)]
for y in range(38, 420, 29):
    for x in range(845, 1600, 29):
        banner.append(f'<circle cx="{x}" cy="{y}" r="1" fill="#c5a8ce" opacity=".23"/>')
for angle in (-30, 30):
    banner.append(f'<ellipse cx="1255" cy="215" rx="252" ry="130" transform="rotate({angle} 1255 215)" fill="none" stroke="#c390c2" stroke-width="1.5" opacity=".38"/>')
period_x = 58 + sum(fonts[800]['hmtx'][fonts[800].getBestCmap()[ord(c)]][0] for c in 'MERL') * 151 / fonts[800]['head'].unitsPerEm - 3
banner += [text('MODEL-EVOLVING REINFORCEMENT LEARNING', 65, 66, 17, '#986296'),
           text('MERL', 58, 224, 151, '#291c32', 800),
           text('.', period_x, 224, 151, '#c82e70', 800),
           text('Learning to trust imagination.', 65, 290, 39, '#92278f'),
           text('World models  /  VLA post-training  /  Two-level trust', 65, 365, 19, '#977aa1', 500),
           rect(1390, 37, 153, 39, '#f0deef', '#e5cbe3', 8),
           text('CoRL 2026', 1466, 63, 19, '#92278f', anchor='middle'),
           rect(994, 151, 480, 125, '#ffffff', '#dec7e3', 19),
           text('Evolve. Calibrate. Refine.', 1234, 206, 29, '#7c2f7b', anchor='middle'),
           text('Learning influence follows reliability.', 1234, 240, 18, '#a77dad', 500, 'middle')]
for x, y, color, label in [(1063, 102, '#d58a49', 'IMAGINATION'), (1007, 328, '#5483b9', 'GROUNDED'), (1467, 323, '#c82e70', 'TRUST')]:
    banner += [f'<circle cx="{x}" cy="{y}" r="7" fill="{color}"/>',
               f'<circle cx="{x}" cy="{y}" r="14" fill="none" stroke="{color}" opacity=".22"/>',
               text(label, x, y + 36, 13, color, anchor='middle')]
save('banner.svg', 1600, 420, 'MERL: learning to trust imagination', banner,
     '<linearGradient id="paper" x2="1" y2=".3"><stop stop-color="#fcfafe"/><stop offset="1" stop-color="#f0e3f2"/></linearGradient>')

compass = [rect(1, 1, 1198, 558, '#faf7fc', '#e8dcec', 22),
           text('Four decisions. One reliability interface.', 36, 53, 33, '#502556'),
           text('Global scheduling across a stage. Local control within imagined experience.', 36, 84, 18, '#9a7aa4', 500)]
cards = [(26, 112, 'STAGE LEVEL', 'How much to learn?', 'Loss mixing', 'rho', '#92278f'),
         (612, 112, 'STAGE LEVEL', 'How far to imagine?', 'Rollout horizon', 'h', '#92278f'),
         (26, 330, 'CHUNK LEVEL', 'Which chunks?', 'Replay probability', 'p', '#c82e70'),
         (612, 330, 'CHUNK LEVEL', 'How much influence?', 'Optimization weight', 'w', '#c82e70')]
card_fragments = []
for i, (x, y, level, question, mechanism, symbol, color) in enumerate(cards):
    start = len(compass)
    compass += [rect(x, y, 562, 202, '#ffffff', '#e3d3e9', 15),
                rect(x + 20, y + 20, 4, 18, color, radius=2),
                text(level, x + 34, y + 34, 14, '#ac82ad'),
                text(question, x + 22, y + 82, 28, '#4b2d54'),
                text(mechanism, x + 22, y + 113, 18, '#9a7ba4', 500),
                rect(x + 432, y + 52, 105, 78, '#f6ebf6', '#eeddee', 13),
                text(symbol, x + 484, y + 103, 36 if i else 29, color, anchor='middle')]
    if i == 0:
        compass += [rect(x + 22, y + 152, 218, 20, '#5483b9', radius=4),
                    rect(x + 243, y + 152, 135, 20, '#d58a49', radius=4)]
    elif i == 1:
        for j in range(12):
            compass.append(rect(x + 22 + j * 29, y + 151, 23, 23, '#aa64a2' if j < 8 else '#ede3f1', radius=4))
    else:
        for j, width in enumerate((156, 107, 62)):
            compass.append(rect(x + 22 + j * 122, y + 158 - j * 5, min(width, 100), 8 + j * 4, color if j == 0 else '#d8b6d4', radius=3))
    card_fragments.append(''.join(compass[start:]))
save('trust-compass.svg', 1200, 560, 'MERL trust decisions: loss mixture, rollout horizon, replay probability and optimization weight', compass)
mobile = [rect(1, 1, 588, 1038, '#faf7fc', '#e8dcec', 22),
          text('Four decisions.', 28, 53, 38, '#502556'),
          text('One reliability interface.', 28, 94, 30, '#92278f'),
          text('Schedule globally. Control locally.', 28, 128, 19, '#9a7aa4', 500)]
for index, ((x, y, *_), fragment) in enumerate(zip(cards, card_fragments)):
    mobile.append(f'<g transform="translate({14-x} {154+index*218-y})">{fragment}</g>')
save('trust-compass-mobile.svg', 590, 1040, 'The four MERL trust decisions in a mobile layout', mobile)

for slug in ('block', 'lid'):
    source = args.video_dir / f'{slug}-merl.mp4'
    target = output / f'{slug}-merl.gif'
    filters = '[0:v]fps=8,scale=440:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source),
                    '-filter_complex', filters, '-loop', '0', str(target)], check=True)
for path in output.iterdir():
    if path.suffix in ('.svg', '.gif'):
        print(f'{path.name}: {path.stat().st_size:,} bytes')
