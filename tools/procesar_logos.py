#!/usr/bin/env python3
"""Quita el fondo blanco, pasa el texto negro/gris a blanco (conserva el dorado)
y recorta los margenes. Salida PNG con transparencia, lista para fondo oscuro."""
import sys
import numpy as np
from PIL import Image


def procesar(src, dst, max_w=None):
    im = Image.open(src).convert("RGB")
    arr = im.numpy() if hasattr(im, "numpy") else np.asarray(im).astype(float)
    arr = arr.astype(float)
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    chroma = mx - mn                      # 0 = gris/negro/blanco, alto = color (dorado)

    # --- Alfa: fondo blanco -> transparente, con borde suave (anti-aliasing) ---
    ramp_lo, ramp_hi = 198.0, 238.0       # mn>=hi: transparente ; mn<=lo: opaco
    alpha = np.clip((ramp_hi - mn) / (ramp_hi - ramp_lo) * 255.0, 0, 255)

    # --- Color: dorado (chroma alto) se conserva; gris/negro -> blanco ---
    es_color = chroma > 40
    out = np.zeros_like(arr)
    # pixeles de color: mantener su RGB original
    out[es_color] = arr[es_color]
    # pixeles acromaticos (texto negro, bordes grises): blanco
    out[~es_color] = 255.0

    rgba = np.dstack([out, alpha]).astype(np.uint8)
    res = Image.fromarray(rgba, "RGBA")

    # --- Recorte a la caja del contenido visible (alfa > 12) ---
    a = np.asarray(res)[..., 3]
    ys, xs = np.where(a > 12)
    if len(xs):
        pad = 6
        x0, x1 = max(0, xs.min() - pad), min(res.width, xs.max() + 1 + pad)
        y0, y1 = max(0, ys.min() - pad), min(res.height, ys.max() + 1 + pad)
        res = res.crop((x0, y0, x1, y1))

    if max_w and res.width > max_w:
        h = round(res.height * max_w / res.width)
        res = res.resize((max_w, h), Image.LANCZOS)

    res.save(dst)
    print("%s -> %s  (%dx%d)" % (src, dst, res.width, res.height))


if __name__ == "__main__":
    base = sys.argv[1]
    procesar(base + "/Customer Hub - S-fondo.jpg", base + "/fradar/logo_customerhub.png", max_w=900)
    procesar(base + "/Big Logo Niage white.png", base + "/fradar/logo_niage.png", max_w=900)
