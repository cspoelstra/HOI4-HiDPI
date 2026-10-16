#!/usr/bin/env python3
"""Builds the "HiDPI Text" HOI4 mod: 2x font atlases plus a text shader that samples them.

Run against the installed game; see README.md.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from hidpi.bmfont import load_rgba, read_fnt, save_dds_rgba, write_fnt
from hidpi.fontgen import FontSpec, build_atlas
from hidpi.imageops import luminance

MOD_NAME = 'HiDPI Text (2x)'
MOD_DIR = 'hidpi_text'
STEAM = Path.home() / '.local/share/Steam'
DEFAULT_GAME = STEAM / 'steamapps/common/Hearts of Iron IV'
PROTON_DOCS = STEAM / 'steamapps/compatdata/394360/pfx/drive_c/users/steamuser/Documents/Paradox Interactive/Hearts of Iron IV'
PROTON_DOCS_WIN = 'C:/users/steamuser/Documents/Paradox Interactive/Hearts of Iron IV'

# Drawn in world space by the map shaders, not text.shader.
SKIP = {'hoi_mapfont4', 'hoi_arrow_font'}

# Outline fonts and em sizes (1x pixels) that best reproduce each vanilla bitmap font, found by
# maximising the correlation between rendered and original glyphs. Fonts not listed here are traced
# from their bitmaps instead (see hidpi/trace.py).
UBUNTU_B = 'Ubuntu:style=Bold'
UBUNTU_M = 'Ubuntu:style=Medium'
GOTHIC_DEMI = 'URW Gothic:style=Demi'
GOTHIC_BOOK = 'URW Gothic:style=Book'
SANS = 'Liberation Sans:style=Regular'
SERIF = 'Liberation Serif:style=Regular'
OUTLINES = {
    'hoi_16mbs': (UBUNTU_B, 12.59),
    'hoi_16mbs_cryllic': (UBUNTU_B, 12.59),
    'hoi_18mbs': (UBUNTU_B, 14.09),
    'hoi_18mbs_cryllic': (UBUNTU_B, 14.09),
    'hoi_22tech': (UBUNTU_B, 16.84),
    'hoi_18': (UBUNTU_M, 13.83),
    'hoi_18_cryllic': (UBUNTU_M, 13.32),
    'hoi_20b': (UBUNTU_M, 15.10),
    'hoi_20b_cryllic': (UBUNTU_M, 14.62),
    'hoi_20bs': (UBUNTU_M, 15.31),
    'hoi_20bs_cryllic': (UBUNTU_M, 15.31),
    'hoi_26mbs': (UBUNTU_M, 20.35),
    'hoi_26mbs_cryllic': (UBUNTU_M, 20.35),
    'hoi_30': (UBUNTU_M, 29.85),
    'hoi_33': (UBUNTU_M, 25.83),
    'cg_16b': (GOTHIC_DEMI, 13.31),
    'hoi_16tooltip3': (GOTHIC_BOOK, 12.97, 1.0),
    'Arial12': (SANS, 11.59),
    'hoi_16typewriter': (SERIF, 14.58),
    'hoi_16typewriter_cryllic': (SERIF, 14.58),
    'hoi_16typewriter_inverted': (SERIF, 14.58),
    'hoi_16typewriter_cryllic_inverted': (SERIF, 14.58),
    'hoi_22typewriter': (SERIF, 19.82),
    'hoi_22typewriter_cryllic': (SERIF, 19.82),
    'hoi_22typewriter_inverted': (SERIF, 19.82),
    'hoi_22typewriter_cryllic_inverted': (SERIF, 19.82),
}

DX9_ALIGN = '''#ifdef PDX_DIRECTX_9
			// At UI scale 2 the DX9 GUI is half a screen pixel off D3D9's texel/pixel alignment, so 1:1
			// glyphs get averaged with their neighbours and 2x magnified art comes out unevenly jagged.
			// Shift sampling back onto texel centres.
			vHdUV += 0.5 * float2( ddx( vHdUV ).x, ddy( vHdUV ).y );
#endif'''


def rcas(sampler):
    """HLSL that sets vHdSharp to OutColor.rgb sharpened with AMD FSR 1 RCAS (on premultiplied colour)."""
    return f'''float2 vHdDX = float2( ddx( vHdUV ).x, 0.0 );
			float2 vHdDY = float2( 0.0, ddy( vHdUV ).y );
			float4 vHdN = tex2D( {sampler}, vHdUV - vHdDY );
			float4 vHdS = tex2D( {sampler}, vHdUV + vHdDY );
			float4 vHdW = tex2D( {sampler}, vHdUV - vHdDX );
			float4 vHdE = tex2D( {sampler}, vHdUV + vHdDX );
			float3 vHdB = vHdN.rgb * vHdN.a;
			float3 vHdH = vHdS.rgb * vHdS.a;
			float3 vHdD = vHdW.rgb * vHdW.a;
			float3 vHdF = vHdE.rgb * vHdE.a;
			float3 vHdC = OutColor.rgb * OutColor.a;
			float3 vHdMin = min( min( vHdB, vHdD ), min( vHdF, vHdH ) );
			float3 vHdMax = max( max( vHdB, vHdD ), max( vHdF, vHdH ) );
			float3 vHdLobe3 = max( -vHdMin / ( 4.0 * vHdMax + 0.0001 ), ( 1.0 - vHdMax ) / ( 4.0 * vHdMin - 4.0001 ) );
			float vHdLobe = max( -0.1875, min( max( vHdLobe3.r, max( vHdLobe3.g, vHdLobe3.b ) ), 0.0 ) );
			float3 vHdSharp = saturate( ( vHdLobe * ( vHdB + vHdD + vHdF + vHdH ) + vHdC ) / ( 4.0 * vHdLobe + 1.0 ) );
			vHdSharp = saturate( vHdSharp / max( OutColor.a, 1.0 / 255.0 ) );'''


def tooltip_text(sampler):
    '''HLSL that updates vHdRGB for texels of a tooltip render target.

    Tooltips are rendered at 1x into an offscreen texture that is then magnified 2x, so their text
    can't use the 2x atlases. Their background (gfx/interface/tiles/tiles_dialog.dds) has alpha
    253/255, which identifies them. There the anti-aliased 1x text is treated as an implicit outline:
    interpolate bicubically, then redraw each pixel on the dark or bright side of the 50% contour
    between the darkest and brightest of the 4x4 surrounding texels, with a ~1px edge. Low-contrast
    areas are left as they are.
    '''
    taps = []
    for j in range(-1, 3):
        for i in range(-1, 3):
            taps += [
                f'vHdTap = tex2Dlod0( {sampler}, vHdUV00 + float2( {i}.0, {j}.0 ) * vHdTexel ).rgb;',
                f'vHdCubic += vHdWY.{"xyzw"[j + 1]} * vHdWX.{"xyzw"[i + 1]} * vHdTap;',
                'fHdTapL = dot( vHdTap, vHdLuma );',
                'if ( fHdTapL < fHdLo ) { fHdLo = fHdTapL; vHdLo = vHdTap; }',
                'if ( fHdTapL > fHdHi ) { fHdHi = fHdTapL; vHdHi = vHdTap; }',
            ]
    taps = '\n\t\t\t\t'.join(taps)
    return f'''float2 vHdTexel = 2.0 * float2( ddx( vHdUV ).x, ddy( vHdUV ).y );
			float2 vHdPos = vHdUV / vHdTexel - 0.5;
			float2 vHdFrac = vHdPos - floor( vHdPos );
			float2 vHdUV00 = ( floor( vHdPos ) + 0.5 ) * vHdTexel;
			float4 vHdT00 = tex2Dlod0( {sampler}, vHdUV00 );
			float4 vHdT10 = tex2Dlod0( {sampler}, vHdUV00 + float2( vHdTexel.x, 0.0 ) );
			float4 vHdT01 = tex2Dlod0( {sampler}, vHdUV00 + float2( 0.0, vHdTexel.y ) );
			float4 vHdT11 = tex2Dlod0( {sampler}, vHdUV00 + vHdTexel );
			float4 vHdTag4 = abs( float4( vHdT00.a, vHdT10.a, vHdT01.a, vHdT11.a ) * 255.0 - 253.0 );
			if ( min( min( vHdTag4.x, vHdTag4.y ), min( vHdTag4.z, vHdTag4.w ) ) < 0.5 )
			{{
				float2 vHdT = vHdFrac;
				float2 vHdT2 = vHdT * vHdT;
				float2 vHdT3 = vHdT2 * vHdT;
				float4 vHdWX = float4( -0.5 * vHdT3.x + vHdT2.x - 0.5 * vHdT.x, 1.5 * vHdT3.x - 2.5 * vHdT2.x + 1.0, -1.5 * vHdT3.x + 2.0 * vHdT2.x + 0.5 * vHdT.x, 0.5 * vHdT3.x - 0.5 * vHdT2.x );
				float4 vHdWY = float4( -0.5 * vHdT3.y + vHdT2.y - 0.5 * vHdT.y, 1.5 * vHdT3.y - 2.5 * vHdT2.y + 1.0, -1.5 * vHdT3.y + 2.0 * vHdT2.y + 0.5 * vHdT.y, 0.5 * vHdT3.y - 0.5 * vHdT2.y );
				float3 vHdLuma = float3( 0.299, 0.587, 0.114 );
				float3 vHdCubic = float3( 0.0, 0.0, 0.0 );
				float3 vHdTap = float3( 0.0, 0.0, 0.0 );
				float3 vHdLo = float3( 1.0, 1.0, 1.0 );
				float3 vHdHi = float3( 0.0, 0.0, 0.0 );
				float fHdLo = 2.0;
				float fHdHi = -1.0;
				float fHdTapL = 0.0;
				{taps}
				vHdCubic = saturate( vHdCubic );
				float fHdEdge = saturate( ( ( dot( vHdCubic, vHdLuma ) - fHdLo ) / max( fHdHi - fHdLo, 0.0001 ) - 0.5 ) / 0.4 + 0.5 );
				fHdEdge = fHdEdge * fHdEdge * ( 3.0 - 2.0 * fHdEdge );
				float fHdContrast = saturate( ( fHdHi - fHdLo - 0.08 ) / 0.15 );
				vHdRGB = lerp( vHdCubic, lerp( vHdLo, vHdHi, fHdEdge ), fHdContrast );
			}}'''


TEXT_ORIG = 'float4 OutColor = tex2D( SimpleTexture, v.vTexCoord );'
TEXT_PATCH = f'''// Font atlases from the HiDPI Text mod are stored at twice the resolution their .fnt describes
			// and carry magenta, alpha 1/255 corner texels. The engine derives glyph UVs from the real
			// texture size, so for those textures the UVs come out halved.
			float4 vHdTag = tex2Dlod0( SimpleTexture, float2( 0.0, 0.0 ) );
			bool bHdFont = vHdTag.r > 0.99 && vHdTag.b > 0.99 && vHdTag.a > 0.5 / 255.0 && vHdTag.a < 1.5 / 255.0;
			float2 vHdUV = v.vTexCoord * ( bHdFont ? 2.0 : 1.0 );
#ifdef PDX_DIRECTX_9
			// At UI scale 2 the DX9 GUI is half a screen pixel off D3D9's texel/pixel alignment, so 1:1
			// glyphs get averaged with their neighbours and 2x magnified art comes out unevenly jagged.
			// Shift sampling back onto texel centres. Glyphs drawn into the 1x tooltip render target
			// (two atlas texels per pixel, atlas width from the marker's green channel) are aligned
			// already and must stay that way so each pixel averages exactly a 2x2 texel block.
			float fHdTexelsPerPixel = bHdFont ? ddx( vHdUV ).x * exp2( floor( vHdTag.g * 32.0 + 0.5 ) ) : 1.0;
			vHdUV += saturate( 2.0 - fHdTexelsPerPixel ) * 0.5 * float2( ddx( vHdUV ).x, ddy( vHdUV ).y );
#endif
		    float4 OutColor = tex2D( SimpleTexture, vHdUV );
			// Other textures through here are magnified UI art, e.g. the tooltip render target on DX11.
			{rcas('SimpleTexture')}
			float3 vHdRGB = vHdSharp;
			{tooltip_text('SimpleTexture')}
			OutColor.rgb = bHdFont ? OutColor.rgb : vHdRGB;'''

SPRITE_ORIG = 'float4 OutColor = tex2D( MapTexture, v.vTexCoord );'
SPRITE_PATCH = f'''float2 vHdUV = v.vTexCoord;
{DX9_ALIGN}
			float4 OutColor = tex2D( MapTexture, vHdUV );
			// UI art, and on DX9 the tooltip render target, is magnified 2x: sharpen it.
			{rcas('MapTexture')}
			float3 vHdRGB = vHdSharp;
			{tooltip_text('MapTexture')}
			OutColor.rgb = vHdRGB;'''

SHADER_PATCHES = {
    'text.shader': (TEXT_ORIG, TEXT_PATCH),
    **{name: (SPRITE_ORIG, SPRITE_PATCH) for name in (
        'buttonstate.shader', 'buttonstate_nodowneffect.shader', 'buttonstate_onlydisable.shader',
        'buttonstate_nodownordisableeffect.shader', 'buttonstate_linear.shader',
        'buttonstate_nontransparent.shader', 'buttonstate_fade_frames_to_black.shader')},
}


def referenced_fonts(game):
    text = ''.join(p.read_text(encoding='utf-8', errors='replace') for p in (game / 'interface').glob('*.gfx'))
    text = re.sub(r'#[^\n]*', '', text)
    return sorted(set(re.findall(r'"gfx/fonts/([A-Za-z0-9_]+)"', text)) - SKIP)


def font_file(pattern):
    try:
        path = subprocess.run(['fc-match', '-f', '%{file}', pattern], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return None
    family = pattern.split(':')[0].lower().replace(' ', '')
    return path if path and family in Path(path).stem.lower().replace(' ', '').replace('-', '') else None


def texture_path(fonts_dir, name):
    for ext in ('.dds', '.tga'):
        if (fonts_dir / (name + ext)).exists():
            return fonts_dir / (name + ext)
    raise FileNotFoundError(name)


def has_effects(tex):
    a = tex[..., 3]
    dark = (a * (luminance(tex) < 0.35)).sum() / max(a.sum(), 1e-6)
    return 0.15 < dark < 0.9


def convert(args):
    name, fonts_dir, out_dir, outline = args
    fnt = read_fnt(fonts_dir / (name + '.fnt'))
    tex = load_rgba(texture_path(fonts_dir, name))
    spec = FontSpec(effects=has_effects(tex))
    if outline:
        spec.ttf, spec.em = outline[0], outline[1]
        spec.embolden = outline[2] if len(outline) > 2 else 0.0
    new_fnt, atlas, counts, (w, h) = build_atlas(fnt, tex, spec)
    write_fnt(out_dir / (name + '.fnt'), new_fnt, name + '.dds', w, h)
    save_dds_rgba(out_dir / (name + '.dds'), atlas)
    if (fonts_dir / (name + '.tga')).exists():
        px = np.clip(np.rint(atlas * 255), 0, 255).astype(np.uint8)
        Image.fromarray(px, 'RGBA').save(out_dir / (name + '.tga'), orientation=1)
    return name, counts, spec


def write_shaders(game, out):
    out.mkdir(parents=True, exist_ok=True)
    for name, (orig, patch) in SHADER_PATCHES.items():
        src = (game / 'gfx/FX' / name).read_text()
        if orig not in src:
            sys.exit(f'gfx/FX/{name} has changed in this game version; update SHADER_PATCHES')
        (out / name).write_text(src.replace(orig, patch))


def write_descriptors(mod_root, version, path_value):
    body = (f'version="1.0"\nname="{MOD_NAME}"\ntags={{\n\t"Graphics"\n\t"Utilities"\n}}\n'
            f'supported_version="{version}"\n')
    (mod_root / 'descriptor.mod').write_text(body)
    return body + f'path="{path_value}"\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--game', type=Path, default=DEFAULT_GAME, help='HOI4 install directory')
    ap.add_argument('--out', type=Path, default=Path(__file__).parent / 'dist',
                    help='where to build the mod (default: ./dist)')
    ap.add_argument('--install', type=Path, nargs='?', const=PROTON_DOCS, default=None,
                    help='copy the built mod into this HOI4 user directory (default: the Proton prefix)')
    ap.add_argument('--jobs', type=int, default=os.cpu_count())
    args = ap.parse_args()

    fonts_dir = args.game / 'gfx/fonts'
    mod_root = args.out / MOD_DIR
    out_fonts = mod_root / 'gfx/fonts'
    if mod_root.exists():
        shutil.rmtree(mod_root)
    out_fonts.mkdir(parents=True)

    outlines = {}
    for name, (pattern, *rest) in OUTLINES.items():
        path = font_file(pattern)
        if path:
            outlines[name] = (path, *rest)
        else:
            print(f'warning: {pattern} not installed, {name} will be traced from its bitmap')

    names = referenced_fonts(args.game)
    jobs = [(n, fonts_dir, out_fonts, outlines.get(n)) for n in names]
    with ProcessPoolExecutor(args.jobs) as pool:
        for name, counts, spec in pool.map(convert, jobs):
            src = Path(spec.ttf).name if spec.ttf else 'bitmap'
            fx = ' +effects' if spec.effects else ''
            print(f'{name:36s} {src:28s}{fx:9s} ' + ' '.join(f'{k}={v}' for k, v in sorted(counts.items())))

    write_shaders(args.game, mod_root / 'gfx/FX')
    raw = re.search(r'"rawVersion":\s*"(\d+\.\d+)', (args.game / 'launcher-settings.json').read_text())
    version = f'{raw.group(1)}.*' if raw else '1.*'
    write_descriptors(mod_root, version, '')
    print(f'built {mod_root}')

    if args.install:
        user_dir = args.install
        dest = user_dir / 'mod' / MOD_DIR
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(mod_root, dest)
        path_value = f'{PROTON_DOCS_WIN}/mod/{MOD_DIR}' if user_dir == PROTON_DOCS else str(dest)
        (user_dir / 'mod' / f'{MOD_DIR}.mod').write_text(write_descriptors(dest, version, path_value))
        print(f'installed {dest}')


if __name__ == '__main__':
    main()
