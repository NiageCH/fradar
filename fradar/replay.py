#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replay.py - validacion y afinado OFFLINE de FRadar (sin lidar conectado)
========================================================================
Dos modos:

  python replay.py --test
      Genera escenarios SINTETICOS (dos personas que se cruzan + una
      oclusion) y comprueba que el tracker mantiene el id. Imprime cuantos
      "saltos de id" hay (objetivo: 0).

  python replay.py --file grabacion.jsonl
      Reproduce una grabacion real (cuando exista) por el pipeline
      clustering+tracking y reporta tracks/saltos. (la grabacion la crea el
      panel cuando el radar este conectado).

No necesita el lidar: prueba la logica de seguimiento de forma repetible.
"""

import sys
import json
import argparse

import numpy as np
from sklearn.cluster import DBSCAN

from tracker import Tracker

DT = 1.0 / 8.0     # 8 fps, como el panel


# --------------------------------------------------------------------------
# Clustering (mismo criterio que el panel)
# --------------------------------------------------------------------------
def fusionar_centroides(cen, d):
    """Une centroides a < d (m): una persona partida en 2 clusters -> 1 deteccion.
    (misma logica que el panel; d<=0 lo desactiva)."""
    if d <= 0 or len(cen) < 2:
        return cen
    usados = [False] * len(cen); out = []
    for i in range(len(cen)):
        if usados[i]:
            continue
        gx = [cen[i][0]]; gy = [cen[i][1]]; usados[i] = True
        cambiado = True
        while cambiado:
            cambiado = False
            cx = sum(gx) / len(gx); cy = sum(gy) / len(gy)
            for j in range(len(cen)):
                if usados[j]:
                    continue
                if np.hypot(cx - cen[j][0], cy - cen[j][1]) <= d:
                    gx.append(cen[j][0]); gy.append(cen[j][1]); usados[j] = True
                    cambiado = True
        out.append((sum(gx) / len(gx), sum(gy) / len(gy)))
    return out


def clusterizar(pts, eps=0.35, min_samples=4, max_size=0.9, merge_dist=0.0):
    if len(pts) < min_samples:
        return []
    pts = np.asarray(pts, dtype=float)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    out = []
    for lab in set(labels):
        if lab == -1:
            continue
        m = pts[labels == lab]
        if max(np.ptp(m[:, 0]), np.ptp(m[:, 1])) > max_size:
            continue
        out.append((float(m[:, 0].mean()), float(m[:, 1].mean())))
    return fusionar_centroides(out, merge_dist)


# --------------------------------------------------------------------------
# Escenarios sinteticos (detecciones a nivel de centroide, con ruido)
# --------------------------------------------------------------------------
def _ruido(p, s=0.03):
    return (p[0] + np.random.normal(0, s), p[1] + np.random.normal(0, s))


def escenario_cruce(n=60):
    """Dos personas que caminan en sentidos opuestos y se cruzan en el centro.
    Devuelve lista de frames; cada frame = lista de (xy, gt_label)."""
    frames = []
    for k in range(n):
        f = []
        x = -2.0 + 4.0 * k / (n - 1)        # A: de izquierda a derecha
        f.append((_ruido((x, 0.3)), "A"))
        x2 = 2.0 - 4.0 * k / (n - 1)        # B: de derecha a izquierda
        f.append((_ruido((x2, -0.3)), "B"))
        frames.append(f)
    return frames


def escenario_oclusion(n=60, gap=(25, 40)):
    """Una persona cruza pero desaparece entre los frames gap[0]..gap[1]
    (oclusion tras un expositor). Debe recuperar el MISMO id al reaparecer."""
    frames = []
    for k in range(n):
        f = []
        if not (gap[0] <= k < gap[1]):
            x = -2.5 + 5.0 * k / (n - 1)
            f.append((_ruido((x, 0.5)), "A"))
        frames.append(f)
    return frames


def escenario_giro(n=44, s=0.12):
    """Una persona entra rapido, se da la VUELTA (180 grados) dentro del alcance
    y se aleja. Con ruido de centroide alto (como saltos pierna/torso reales).
    Es el caso que rompia el id con velocidad constante: debe MANTENERLO."""
    frames = []
    for k in range(n):
        t = k / (n - 1)
        tri = 2 * t if t < 0.5 else 2 * (1 - t)   # 0 -> 1 -> 0 (pico = el giro)
        y = 0.4 + 2.8 * tri                        # ~0.4m .. 3.2m .. 0.4m, rapido
        frames.append([(_ruido((0.5, y), s), "A")])
    return frames


def evaluar(nombre, frames, tracker, merge_dist=0.0):
    """Pasa los frames por el tracker y cuenta saltos de id por persona GT.
    merge_dist>0 fusiona los centroides antes de trackear (como el panel)."""
    ultimo_id = {}      # gt_label -> id del tracker
    saltos = 0
    ids_por_gt = {}
    for f in frames:
        dets = fusionar_centroides([xy for xy, _ in f], merge_dist)
        gts = [g for _, g in f]
        salida = tracker.update(dets, DT)
        # asociar cada track de salida al gt mas cercano de este frame
        for tr in salida:
            if not f:
                continue
            dmin, gbest = 1e9, None
            for (xy, g) in f:
                d = np.hypot(tr["x"] - xy[0], tr["y"] - xy[1])
                if d < dmin:
                    dmin, gbest = d, g
            if gbest is None or dmin > 0.6:
                continue
            ids_por_gt.setdefault(gbest, set()).add(tr["id"])
            if gbest in ultimo_id and ultimo_id[gbest] != tr["id"]:
                saltos += 1
            ultimo_id[gbest] = tr["id"]
    print("[%s]" % nombre)
    for g, ids in sorted(ids_por_gt.items()):
        print("   persona GT %s -> ids asignados: %s" % (g, sorted(ids)))
    print("   SALTOS de id: %d   (ideal 0)" % saltos)
    return saltos


def escenario_fragmentacion(n=60, sep=0.45, s=0.05):
    """UNA persona que el lidar parte en DOS clusters (torso + piernas) a ~sep m.
    Sin fusion nacen 2 ids para la misma persona; con la fusion de clusters activa
    debe quedar UN solo id y 0 saltos."""
    frames = []
    for k in range(n):
        x = -2.0 + 4.0 * k / (n - 1)
        frames.append([(_ruido((x, 0.6), s), "A"),
                       (_ruido((x + sep, 0.6), s), "A")])
    return frames


# Parametros afinados (los mismos que usa el panel por defecto): gating robusto
# + velocidad amortiguada -> aguanta cambios de direccion sin cambiar de id.
TUNED = dict(gate_dist=1.0, n_confirm=3, max_misses=8, reid_frames=60,
             reid_dist=1.4, q=0.12, r=0.05, vel_damp=0.88,
             dedup_dist=0.45, dedup_frames=12)


def run_test():
    np.random.seed(1)
    print("=== Prueba sintetica del tracker (objetivo: 0 saltos) ===\n")
    print("-- Continuidad de id --")
    s1 = evaluar("cruce de 2 personas", escenario_cruce(), Tracker(**TUNED))
    print()
    s2 = evaluar("oclusion ~2s (re-id)", escenario_oclusion(), Tracker(**TUNED))
    print()
    s3 = evaluar("giro 180 dentro del alcance (cambio de direccion)",
                 escenario_giro(), Tracker(**TUNED))
    print("\n-- Robustez ante id duplicado (persona fragmentada) --")
    # sin fusion: nacen 2 ids (informativo, se espera que sea peor)
    sin = evaluar("fragmentacion SIN fusion", escenario_fragmentacion(),
                  Tracker(**TUNED), merge_dist=0.0)
    print()
    # con fusion de clusters (como se configuraria en sitio): 1 id, 0 saltos
    con = evaluar("fragmentacion CON fusion", escenario_fragmentacion(),
                  Tracker(**TUNED), merge_dist=0.60)
    cont = s1 + s2 + s3
    print("\n=== Continuidad: %s ===" %
          ("OK (0 saltos)" if cont == 0 else "%d saltos" % cont))
    print("=== Duplicados: fusion %s (sin=%d saltos -> con=%d saltos) ===" %
          ("OK" if con == 0 and con < sin else "REVISAR", sin, con))
    ok = (cont == 0 and con == 0 and con < sin)
    print("\n=== RESULTADO GLOBAL: %s ===" % ("OK" if ok else "FALLO"))
    return 0 if ok else 1


def run_file(path):
    tracker = Tracker()
    nframes = 0
    with open(path) as fh:
        for linea in fh:
            try:
                scan = json.loads(linea)
            except ValueError:
                continue
            pts = [(r * np.cos(np.radians(a)), r * np.sin(np.radians(a)))
                   for a, r in scan.get("pts", []) if r > 0]
            dets = clusterizar(pts)
            tracker.update(dets, DT)
            nframes += 1
    print("Reproducidos %d frames de %s" % (nframes, path))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="escenarios sinteticos")
    ap.add_argument("--file", help="reproducir una grabacion .jsonl")
    a = ap.parse_args()
    if a.file:
        sys.exit(run_file(a.file))
    sys.exit(run_test())
