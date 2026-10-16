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
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
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
REPO = Path(__file__).parent
FONTS = REPO / 'fonts'
LICENSES = REPO / 'licenses'
REPO_URL = 'https://github.com/cspoelstra/HOI4-HiDPI'

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
# The hoi4_typewriter fonts (AFL Font nonmetric) have no freely licensed outlines; their traced bitmaps
# come out lumpy. They are drawn from Podkova, the best-fitting openly licensed slab serif, at weights
# that match the vanilla strokes. Podkova is bundled in fonts/ (SIL OFL) as a variable font, given here
# as (file, weight).
PODKOVA_SEMIBOLD = ('Podkova[wght].ttf', 600)
PODKOVA_MEDIUM = ('Podkova[wght].ttf', 500)
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
    'hoi4_typewriter16': (PODKOVA_SEMIBOLD, 15.0),
    'hoi4_typewriter16_inverted': (PODKOVA_SEMIBOLD, 15.0),
    'hoi4_typewriter22': (PODKOVA_MEDIUM, 19.94),
    'hoi4_typewriter22_inverted': (PODKOVA_MEDIUM, 19.94),
}

# Licence texts in licenses/ for each family of outline fonts the atlases are drawn from; they ship with
# the mod, listed in its CREDITS.txt.
FONT_LICENSES = {
    'Ubuntu': ['Ubuntu-Font-Licence-1.0.txt'],
    'URW Gothic': ['URW-base35-LICENSE.txt', 'URW-base35-COPYING-AGPL-3.0.txt'],
    'Liberation Sans': ['Liberation-OFL.txt'],
    'Liberation Serif': ['Liberation-OFL.txt'],
    'Podkova': ['Podkova-OFL.txt'],
}

# Tooltip fonts (cg_16b and its fallback file) are drawn straight from their outline font with a
# synthesised drop shadow, so text.shader knows the fill grey and can carry them into tooltips at 2x.
# Fill grey and drop shadow (dx, dy, sigma, strength in pixels) were measured from the vanilla bitmaps.
REDRAWN = {
    'cg_16b': dict(fill=0.77, shadow=(0.0, 1.0, 0.75, 1.2)),
    'hoi_16tooltip3': dict(fill=0.9, shadow=(0.5, 1.0, 2.0, 1.5)),
}

DX9_ALIGN = '''#ifdef PDX_DIRECTX_9
			// At UI scale 2 the DX9 GUI is half a screen pixel off D3D9's texel/pixel alignment, so 1:1
			// glyphs get averaged with their neighbours and 2x magnified art comes out unevenly jagged.
			// Shift sampling back onto texel centres.
			vHdUV += 0.5 * float2( ddx( vHdUV ).x, ddy( vHdUV ).y );
#endif'''

# Tooltips and buttons with baked-in labels (textSpriteType) are drawn by text.shader into an offscreen
# texture at one pixel per UI unit, which is then magnified to the UI scale like any other sprite.
# text.shader stores text drawn there as 2x2 sub-pixel coverage plus colour in texels with alpha 255
# (everything else it draws there stays below 255), and marks texel (0,0) with this colour so the
# shaders that draw the texture know to decode it. Texel (0,1) holds the colour at the centre of the
# background art, for text texels with no background next to them.
BEACON = '( float4( 90.0, 60.0, 165.0, 255.0 ) / 255.0 )'


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


def target_text(sampler):
    '''HLSL that redraws a texture written by text.shader as a 1x render target (see BEACON).

    Texels with alpha 255 hold text: each screen pixel shows its sub-pixel's coverage of the text
    colour over the average of the surrounding non-text texels, or over the reference background if
    there are none. Next to text, the texture is magnified without blending in those texels. Updates
    vHdRGB and OutColor.a.
    '''
    names = {(i, j): f'vHdT{"mzp"[i + 1]}{"mzp"[j + 1]}' for j in (-1, 0, 1) for i in (-1, 0, 1)}
    # Text texels and the two marker texels are never blended into their neighbours.
    taps = '\n\t\t\t\t'.join(
        f'float4 {n} = tex2Dlod0( {sampler}, ( vHdCell + float2( {i + 0.5}, {j + 0.5} ) ) * vHdUnit );\n'
        f'\t\t\t\tfloat fHdP{n[4:]} = max( step( 254.5 / 255.0, {n}.a ), vHdCell.x == {-i}.0 && abs( vHdCell.y + {j - 0.5} ) < 1.0 ? 1.0 : 0.0 );'
        for (i, j), n in names.items())
    ring = [n for (i, j), n in names.items() if (i, j) != (0, 0)]
    bg_sum = ' + '.join(f'( 1.0 - fHdP{n[4:]} ) * {n}' for n in ring)
    bg_n = ' + '.join(f'fHdP{n[4:]}' for n in ring)
    return f'''// Texel size from the texture's UV change per UI unit, snapped to a whole number of texels so the
			// samples below hit texel centres exactly.
			float2 vHdUnit = abs( float2( ddx( vHdUV ).x, ddy( vHdUV ).y ) ) / max( abs( float2( ddx( v.vHdUI.x ), ddy( v.vHdUI.y ) ) ), 0.00001 );
			vHdUnit = 1.0 / max( floor( 1.0 / vHdUnit + 0.5 ), 1.0 );
			if ( all( abs( tex2Dlod0( {sampler}, 0.5 * vHdUnit ) - {BEACON} ) < 0.5 / 255.0 ) )
			{{
				float2 vHdPos = vHdUV / vHdUnit;
				float2 vHdCell = floor( vHdPos );
				float2 vHdSub = step( 0.5, vHdPos - vHdCell );
				{taps}
				float fHdText = fHdPzz + {bg_n};
				// The marker texels show their right-hand neighbour.
				float fHdCorner = vHdCell.x == 0.0 && abs( vHdCell.y - 0.5 ) < 1.0 ? 1.0 : 0.0;
				vHdTzz = lerp( vHdTzz, vHdTpz, fHdCorner );
				fHdPzz = lerp( fHdPzz, fHdPpz, fHdCorner );
				if ( fHdPzz > 0.5 )
				{{
					float fHdBgN = 8.0 - ( {bg_n} );
					float4 vHdBg = fHdBgN > 0.5 ? ( {bg_sum} ) / fHdBgN : tex2Dlod0( {sampler}, float2( 0.5, 1.5 ) * vHdUnit );
					float3 vHdByte = floor( vHdTzz.rgb * 255.0 + 0.5 );
					float3 vHdHi = floor( vHdByte / 16.0 );
					float3 vHdLo = vHdByte - 16.0 * vHdHi;
					float3 vHdQ = floor( vHdLo / 2.0 );
					float4 vHdCov = float4( vHdQ, dot( vHdLo - 2.0 * vHdQ, float3( 1.0, 2.0, 4.0 ) ) ) / 7.0;
					float2 vHdRow = lerp( vHdCov.xz, vHdCov.yw, vHdSub.x );
					float fHdCov = lerp( vHdRow.x, vHdRow.y, vHdSub.y );
					vHdRGB = lerp( vHdBg.rgb, vHdHi / 15.0, fHdCov );
					OutColor.a = lerp( vHdBg.a, 1.0, fHdCov );
				}}
				else if ( fHdText > 0.5 )
				{{
					float2 vHdW = abs( vHdPos - vHdCell - 0.5 );
					bool bHdR = vHdSub.x > 0.5;
					bool bHdD = vHdSub.y > 0.5;
					float4 vHdX = lerp( bHdR ? vHdTpz : vHdTmz, vHdTzz, bHdR ? fHdPpz : fHdPmz );
					float4 vHdY = lerp( bHdD ? vHdTzp : vHdTzm, vHdTzz, bHdD ? fHdPzp : fHdPzm );
					float4 vHdXY = bHdR ? ( bHdD ? vHdTpp : vHdTpm ) : ( bHdD ? vHdTmp : vHdTmm );
					vHdXY = lerp( vHdXY, vHdTzz, bHdR ? ( bHdD ? fHdPpp : fHdPpm ) : ( bHdD ? fHdPmp : fHdPmm ) );
					float4 vHdLin = lerp( lerp( vHdTzz, vHdX, vHdW.x ), lerp( vHdY, vHdXY, vHdW.x ), vHdW.y );
					vHdRGB = vHdLin.rgb;
					OutColor.a = vHdLin.a;
				}}
			}}'''


TEXT_VS_STRUCT = ('\tfloat4  vColor\t  : TEXCOORD1;\n};',
                  '\tfloat4  vColor\t  : TEXCOORD1;\n\tfloat4  vHdUI\t  : TEXCOORD2;\n\tfloat2  vHdClip  : TEXCOORD3;\n};')
TEXT_VS = '\t[[\n\t\tVS_OUTPUT main(const VS_INPUT v )\n\t\t{\n\t\t    VS_OUTPUT Out;'
TEXT_VS_3D = ('MainCode VertexShaderText3D\n' + TEXT_VS,
              'MainCode VertexShaderText3D\n' + TEXT_VS + '''
			Out.vHdUI = float4( 0.0, 0.0, 0.0, 0.0 );
			Out.vHdClip = float2( 0.0, 0.0 );''')
TEXT_VS_2D = ('MainCode VertexShaderText\n' + TEXT_VS,
              'MainCode VertexShaderText\n' + TEXT_VS + '''
			// UI coordinates, and the width of what is drawn into as 2 / that.
			Out.vHdUI = float4( v.vPosition.xy, abs( Matrix[0][0] ), 0.0 );
			Out.vHdClip = mul( Matrix, v.vPosition ).xy;''')

TEXT_ORIG = '''float4 OutColor = tex2D( SimpleTexture, v.vTexCoord );
			OutColor = OutColor * v.vColor;
		    return OutColor;'''
TEXT_PATCH = f'''// Font atlases from the HiDPI Text mod are stored at twice the resolution their .fnt describes
			// and carry red, alpha 1/255 corner texels (hidpi/fontgen.py). The engine derives glyph UVs from
			// the real texture size, so for those textures the UVs come out halved.
			float4 vHdTag = tex2Dlod0( SimpleTexture, float2( 0.0, 0.0 ) );
			bool bHdFont = vHdTag.r > 0.99 && vHdTag.a > 0.5 / 255.0 && vHdTag.a < 1.5 / 255.0;
			float2 vHdUV = v.vTexCoord * ( bHdFont ? 2.0 : 1.0 );
			// Drawing into a 1x render target (see BEACON in build_mod.py): one UI unit per pixel, and
			// narrower than any screen.
			bool bHdTarget = abs( ddx( v.vHdUI.x ) ) > 0.75 && v.vHdUI.z > 2.0 / 1100.0;
			float2 vHdEdge = float2( v.vHdClip.x + 1.0, 1.0 - v.vHdClip.y ) / abs( float2( ddx( v.vHdClip.x ), ddy( v.vHdClip.y ) ) );
			float2 vHdQuarter = 0.25 * float2( ddx( vHdUV ).x, ddy( vHdUV ).y );
#ifdef PDX_DIRECTX_9
			// At UI scale 2 the DX9 GUI is half a screen pixel off D3D9's texel/pixel alignment, so 1:1
			// glyphs get averaged with their neighbours and 2x magnified art comes out unevenly jagged.
			// Shift sampling back onto texel centres. Render targets are aligned already.
			float fHdTexelsPerPixel = bHdFont ? ddx( vHdUV ).x * exp2( floor( vHdTag.g * 32.0 + 0.5 ) ) : 1.0;
			vHdUV += bHdTarget ? 0.0 : saturate( 2.0 - fHdTexelsPerPixel ) * 0.5 * float2( ddx( vHdUV ).x, ddy( vHdUV ).y );
#endif
		    float4 OutColor = tex2D( SimpleTexture, vHdUV );
			float4 vHdPlain = OutColor * v.vColor;
			// Other textures through here are magnified UI art, and on DX11 the render targets.
			{rcas('SimpleTexture')}
			float3 vHdRGB = vHdSharp;
			{target_text('SimpleTexture')}
			OutColor.rgb = bHdFont ? OutColor.rgb : vHdRGB;
			OutColor = OutColor * v.vColor;
			if ( bHdTarget )
			{{
				OutColor = vHdPlain;
				OutColor.a = min( OutColor.a, 254.0 / 255.0 );
				// Dark text is the engine's drop shadow pass, drawn before the text itself; it stays a plain
				// 1x shadow.
				float3 vHdLuma = float3( 0.299, 0.587, 0.114 );
				if ( bHdFont && vHdTag.b > 0.0 && dot( v.vColor.rgb, vHdLuma ) > 0.1 )
				{{
					// Coverage of the glyph fill (no shadow) in the four atlas texels under this pixel, 3 bits
					// each, plus the colour of the fill at 4 bits per channel.
					float4 vHd00 = tex2Dlod0( SimpleTexture, vHdUV - vHdQuarter );
					float4 vHd10 = tex2Dlod0( SimpleTexture, vHdUV + float2( vHdQuarter.x, -vHdQuarter.y ) );
					float4 vHd01 = tex2Dlod0( SimpleTexture, vHdUV + float2( -vHdQuarter.x, vHdQuarter.y ) );
					float4 vHd11 = tex2Dlod0( SimpleTexture, vHdUV + vHdQuarter );
					float4 vHdFill = float4( dot( vHd00.rgb, vHdLuma ) * vHd00.a, dot( vHd10.rgb, vHdLuma ) * vHd10.a,
					                         dot( vHd01.rgb, vHdLuma ) * vHd01.a, dot( vHd11.rgb, vHdLuma ) * vHd11.a );
					float4 vHdQ = floor( saturate( vHdFill * v.vColor.a / vHdTag.b ) * 7.0 + 0.5 );
					float3 vHdCol = floor( saturate( v.vColor.rgb * vHdTag.b ) * 15.0 + 0.5 );
					float3 vHdBits = float3( fmod( vHdQ.w, 2.0 ), fmod( floor( vHdQ.w / 2.0 ), 2.0 ), floor( vHdQ.w / 4.0 ) );
					// Fully transparent where there is no fill, which leaves the target untouched.
					OutColor = max( max( vHdQ.x, vHdQ.y ), max( vHdQ.z, vHdQ.w ) ) > 0.5
						? float4( ( vHdCol * 16.0 + vHdQ.xyz * 2.0 + vHdBits ) / 255.0, 1.0 ) : float4( 0.0, 0.0, 0.0, 0.0 );
				}}
				if ( !bHdFont && vHdEdge.x < 0.75 && vHdEdge.y >= 0.75 && vHdEdge.y < 1.75 )
					OutColor = float4( ( tex2Dlod0( SimpleTexture, float2( 0.5, 0.5 ) ) * v.vColor ).rgb, 254.0 / 255.0 );
				if ( vHdEdge.x < 0.75 && vHdEdge.y < 0.75 )
					OutColor = {BEACON};
			}}
		    return OutColor;'''

SPRITE_VS_STRUCT = ('\tfloat2  vTexCoord : TEXCOORD0;', '\tfloat2  vTexCoord : TEXCOORD0;\n\tfloat2  vHdUI : TEXCOORD3;')
SPRITE_VS = ('Out.vPosition  = mul( WorldViewProjectionMatrix, float4( v.vPosition.xyz, 1 ) );',
             'Out.vPosition  = mul( WorldViewProjectionMatrix, float4( v.vPosition.xyz, 1 ) );\n\t\t\tOut.vHdUI = v.vPosition.xy;')
SPRITE_ORIG = 'float4 OutColor = tex2D( MapTexture, v.vTexCoord );'
SPRITE_PATCH = f'''float2 vHdUV = v.vTexCoord;
{DX9_ALIGN}
			float4 OutColor = tex2D( MapTexture, vHdUV );
			// UI art, and on DX9 the render targets, is magnified 2x: sharpen it.
			{rcas('MapTexture')}
			float3 vHdRGB = vHdSharp;
			{target_text('MapTexture')}
			OutColor.rgb = vHdRGB;'''

# (original, replacement[, count]) in order; count defaults to the first occurrence, -1 means all.
SHADER_PATCHES = {
    'text.shader': [TEXT_VS_STRUCT, TEXT_VS_3D, TEXT_VS_2D, (TEXT_ORIG, TEXT_PATCH)],
    **{name: [SPRITE_VS_STRUCT, SPRITE_VS, (SPRITE_ORIG, SPRITE_PATCH, -1)] for name in (
        'buttonstate.shader', 'buttonstate_nodowneffect.shader', 'buttonstate_onlydisable.shader',
        'buttonstate_nodownordisableeffect.shader', 'buttonstate_linear.shader',
        'buttonstate_nontransparent.shader', 'buttonstate_fade_frames_to_black.shader')},
}


def referenced_fonts(game):
    text = ''.join(p.read_text(encoding='utf-8', errors='replace') for p in (game / 'interface').glob('*.gfx'))
    text = re.sub(r'#[^\n]*', '', text)
    return sorted(set(re.findall(r'"gfx/fonts/([A-Za-z0-9_]+)"', text)) - SKIP)


def font_file(pattern, cache):
    """Outline font file for an OUTLINES entry: a fontconfig pattern, or a bundled variable font and
    weight, which is instanced into `cache`."""
    if isinstance(pattern, tuple):
        file, weight = pattern
        path = cache / f'{file.split("[")[0]}-{weight}.ttf'
        if not path.exists():
            cache.mkdir(parents=True, exist_ok=True)
            instantiateVariableFont(TTFont(FONTS / file), {'wght': weight}).save(path)
        return str(path)
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
        spec.substitute = isinstance(OUTLINES[name][0], tuple)
        if name in REDRAWN:
            spec.redraw = True
            spec.fill = REDRAWN[name]['fill']
            spec.shadow = REDRAWN[name]['shadow']
    new_fnt, atlas, counts, (w, h) = build_atlas(fnt, tex, spec)
    write_fnt(out_dir / (name + '.fnt'), new_fnt, name + '.dds', w, h)
    save_dds_rgba(out_dir / (name + '.dds'), atlas)
    if (fonts_dir / (name + '.tga')).exists():
        px = np.clip(np.rint(atlas * 255), 0, 255).astype(np.uint8)
        Image.fromarray(px, 'RGBA').save(out_dir / (name + '.tga'), orientation=1)
    return name, counts, spec


def write_patched(game, mod_root):
    patches = {f'gfx/FX/{name}': patch for name, patch in SHADER_PATCHES.items()}
    for rel, pairs in patches.items():
        src = (game / rel).read_text(encoding='utf-8', errors='surrogateescape')
        for orig, patch, *count in pairs:
            if orig not in src:
                sys.exit(f'{rel} has changed in this game version; update SHADER_PATCHES')
            src = src.replace(orig, patch, *(count or [1]))
        dest = mod_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src, encoding='utf-8', errors='surrogateescape')


def write_credits(mod_root, used):
    """CREDITS.txt and licence texts for the outline fonts in `used` ({font file: [game font names]})."""
    lines = [
        f'{MOD_NAME} for Hearts of Iron IV',
        REPO_URL,
        '',
        'The font textures (gfx/fonts) are redrawings of Hearts of Iron IV\'s fonts at twice the resolution,',
        'and the shaders (gfx/FX) are modified copies of the game\'s own. The game\'s assets remain the',
        'property of Paradox Interactive. The build script that generates this mod is MIT licensed.',
        '',
        'Font textures drawn from the outline fonts below are distributed under the licence of the font',
        'they were drawn from; the licence texts are in the licenses folder. Their source is that font plus',
        'the build script in the repository above. The other font textures are traced from the game\'s own',
        'bitmaps.',
    ]
    (mod_root / 'licenses').mkdir()
    for path, names in sorted(used.items(), key=lambda kv: kv[0]):
        info = TTFont(path, lazy=True)['name']
        family = info.getDebugName(16) or info.getDebugName(1)
        if family not in FONT_LICENSES:
            sys.exit(f'{family} ({path}) has no entry in FONT_LICENSES')
        for licence in FONT_LICENSES[family]:
            shutil.copy(LICENSES / licence, mod_root / 'licenses' / licence)
        lines += ['', f'{family} ({Path(path).name})', '  ' + ' '.join(info.getDebugName(0).split()),
                  '  Licence: ' + ', '.join(FONT_LICENSES[family]), '  Used for: ' + ', '.join(sorted(names))]
    (mod_root / 'CREDITS.txt').write_text('\n'.join(lines) + '\n')


def write_descriptors(mod_root, version, path_value, remote_file_id=None):
    body = (f'version="1.0"\nname="{MOD_NAME}"\ntags={{\n\t"Graphics"\n\t"Utilities"\n}}\n'
            f'picture="thumbnail.png"\nsupported_version="{version}"\n')
    if remote_file_id:
        body += f'remote_file_id="{remote_file_id}"\n'
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
        path = font_file(pattern, args.out / 'fonts')
        if path:
            outlines[name] = (path, *rest)
        else:
            print(f'warning: {pattern} not installed, {name} will be traced from its bitmap')

    names = referenced_fonts(args.game)
    jobs = [(n, fonts_dir, out_fonts, outlines.get(n)) for n in names]
    used = {}
    with ProcessPoolExecutor(args.jobs) as pool:
        for name, counts, spec in pool.map(convert, jobs):
            src = Path(spec.ttf).name if spec.ttf else 'bitmap'
            fx = ' +effects' if spec.effects else ''
            print(f'{name:36s} {src:28s}{fx:9s} ' + ' '.join(f'{k}={v}' for k, v in sorted(counts.items())))
            if counts.get('ttf'):
                used.setdefault(spec.ttf, []).append(name)

    write_patched(args.game, mod_root)
    write_credits(mod_root, used)
    shutil.copy(REPO / 'workshop/thumbnail.png', mod_root / 'thumbnail.png')
    raw = re.search(r'"rawVersion":\s*"(\d+\.\d+)', (args.game / 'launcher-settings.json').read_text())
    version = f'{raw.group(1)}.*' if raw else '1.*'
    write_descriptors(mod_root, version, '')
    print(f'built {mod_root}')

    if args.install:
        user_dir = args.install
        dest = user_dir / 'mod' / MOD_DIR
        # Keep the Workshop item id the Paradox launcher adds on upload, so later uploads update it.
        mod_file = user_dir / 'mod' / f'{MOD_DIR}.mod'
        remote = re.search(r'remote_file_id="(\d+)"', mod_file.read_text()) if mod_file.exists() else None
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(mod_root, dest)
        path_value = f'{PROTON_DOCS_WIN}/mod/{MOD_DIR}' if user_dir == PROTON_DOCS else str(dest)
        mod_file.write_text(write_descriptors(dest, version, path_value, remote and remote.group(1)))
        print(f'installed {dest}')


if __name__ == '__main__':
    main()
