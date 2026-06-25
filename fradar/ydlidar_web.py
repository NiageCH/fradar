#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Panel web del YDLidar X2L  (con pagina de ajustes)
==================================================
- Lee el lidar en un hilo de fondo y, EN ESE MISMO HILO, renderiza la vista
  cenital a PNG (el binding SWIG de ydlidar no libera el GIL en doProcessSimple,
  asi que el render se hace junto al lidar y el web solo sirve el ultimo PNG).
- Puntos coloreados por distancia (barra de color cerca<->lejos).
- Pagina /settings para cambiar ROI, visualizacion y parametros del radar.
  Los cambios de ROI/visual son instantaneos; los de lidar (frecuencia, sample
  rate) reinician el sensor de forma segura.

    source venv/bin/activate
    python ydlidar_web.py        # http://10.10.10.22:8080  (o localhost:8080)
"""

import io
import os
import json
import math
import time
import threading
from collections import defaultdict, deque

import numpy as np
from sklearn.cluster import DBSCAN

from tracker import Tracker     # tracker robusto (Kalman + Hungarian + re-ID)

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize, to_rgb
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

from flask import Flask, Response, request, redirect

import ydlidar

PUERTO_WEB  = 8080
_DIR        = os.path.dirname(os.path.abspath(__file__))
CFG_PATH    = os.path.join(_DIR, "config.json")
BG_PATH     = os.path.join(_DIR, "bg_reference.json")
CSV_PATH    = os.path.join(_DIR, "eventos.csv")
N_BINS      = 360            # un valor de fondo por grado

# Parametros de lidar que, al cambiar, obligan a reiniciar el sensor.
LIDAR_KEYS = ("scan_frequency", "sample_rate")

DEFAULTS = {
    # --- ROI delante del stand ---
    "roi_angle_min": -45.0,
    "roi_angle_max":  45.0,
    "roi_dist_min":   0.20,
    "roi_dist_max":   3.00,
    "min_points":     8,
    # --- Zona excluida (p.ej. donde esta el empleado): se ignora del todo ---
    "excl_enabled":   False,
    "excl_angle_min": 120.0,
    "excl_angle_max": 180.0,
    "excl_dist_min":  0.0,
    "excl_dist_max":  2.5,
    # --- Filtrado de fondo (resta muebles/paredes fijos) ---
    "bg_filter_enabled": False,
    "bg_tolerance":      0.20,   # margen (m): mas alto = filtra mas agresivo
    "show_background":   True,    # dibujar el fondo en gris tenue
    # --- Conteo de personas (clustering + tracking) ---
    "track_enabled":       False,
    "cluster_eps":         0.35,  # radio DBSCAN (m): puntos mas cerca = mismo objeto
    "cluster_min_samples": 4,     # puntos minimos para formar un cluster
    "cluster_max_size":    0.90,  # tamano max de un cluster (m); mayor = no es persona
    "track_max_dist":      0.75,  # distancia max (m) para asociar persona entre frames
    "track_max_missing":   40,    # frames que un id sigue recuperable (re-ID) tras perderse
    "track_confirm":       3,     # barridos seguidos para confirmar un id (anti-espurios)
    "track_max_misses":    5,     # fallos seguidos antes de mandar el id al limbo
    "track_reid_dist":     1.0,   # distancia max (m) para re-identificar un id perdido
    "near_dist":           1.0,   # "cerca" = a menos de esta distancia (m)
    "stay_near_s":        10.0,   # >= este tiempo a <near_dist => "se queda cerca"
    "log_csv":             True,  # registrar eventos en eventos.csv
    # --- Visualizacion ---
    "view_max":       4.0,     # alcance mostrado en el grafico (m)
    "point_size":     7,       # tamano puntos del entorno
    "roi_point_size": 22,      # tamano puntos dentro de la ROI
    "show_trails":    True,     # dibujar la trayectoria (estela) de cada persona
    "trail_len":      40,       # nº de posiciones (suavizadas por Kalman) de la estela
    "flip_x":         False,    # espejo horizontal (si la vista sale derecha<->izquierda)
    "cmap":           "turbo", # mapa de color por distancia
    # --- Lidar (requieren reinicio del sensor) ---
    "scan_frequency": 7.0,
    "sample_rate":    3,
}

# Estado compartido --------------------------------------------------------
_lock = threading.Lock()
_cfg = dict(DEFAULTS)
_cfg_version = 0          # sube en cada guardado -> el render reconstruye figura
_need_restart = False     # pedir reinicio del lidar
_frame_png = None
_lidar_ok = False
_bg_ref = None            # referencia de fondo: lista de N_BINS (dist o None)
_bg_capture = False       # peticion de capturar fondo (stand vacio)
_bg_status = "sin fondo"  # texto de estado del fondo
_rec_request = None       # segundos de grabacion pedidos (None = nada)
_rec_status = "sin grabacion"


def cargar_cfg():
    global _cfg
    try:
        with open(CFG_PATH) as f:
            disco = json.load(f)
        merged = dict(DEFAULTS)
        merged.update({k: disco[k] for k in disco if k in DEFAULTS})
        _cfg = merged
        print("Config cargada de", CFG_PATH)
    except FileNotFoundError:
        guardar_cfg(_cfg)
    except Exception as e:
        print("Aviso: config.json ilegible (%s), uso defaults" % e)


def guardar_cfg(cfg):
    try:
        with open(CFG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print("No se pudo guardar config:", e)


def guardar_bg(ref):
    try:
        with open(BG_PATH, "w") as f:
            json.dump(ref, f)
    except Exception as e:
        print("No se pudo guardar el fondo:", e)


def get_cfg():
    with _lock:
        return dict(_cfg)


def en_roi(cfg, ang_deg, dist):
    return (cfg["roi_angle_min"] <= ang_deg <= cfg["roi_angle_max"] and
            cfg["roi_dist_min"] <= dist <= cfg["roi_dist_max"])


def en_excluida(cfg, ang_deg, dist):
    if not cfg.get("excl_enabled"):
        return False
    return (cfg["excl_angle_min"] <= ang_deg <= cfg["excl_angle_max"] and
            cfg["excl_dist_min"] <= dist <= cfg["excl_dist_max"])


def cargar_bg():
    global _bg_ref, _bg_status
    try:
        with open(BG_PATH) as f:
            ref = json.load(f)
        if isinstance(ref, list) and len(ref) == N_BINS:
            _bg_ref = ref
            _bg_status = "fondo cargado de disco"
    except FileNotFoundError:
        pass
    except Exception as e:
        print("Aviso: bg_reference ilegible (%s)" % e)


def es_fondo(ref, tol, ang_deg, dist):
    """True si el punto coincide con un obstaculo fijo conocido (fondo).
    Compara contra el bin y sus vecinos para absorber el jitter angular."""
    if ref is None:
        return False
    b = int(round(ang_deg)) % N_BINS
    for bb in (b - 1, b, b + 1):
        r = ref[bb % N_BINS]
        if r is not None and abs(dist - r) <= tol:
            return True
    return False


def log_evento(track_id, dwell, tipo):
    """Anade una fila a eventos.csv (la crea con cabecera si no existe)."""
    nuevo = not os.path.exists(CSV_PATH)
    try:
        with open(CSV_PATH, "a") as f:
            if nuevo:
                f.write("fecha_hora,track_id,dwell_s,tipo\n")
            f.write("%s,%d,%.1f,%s\n" %
                    (time.strftime("%Y-%m-%d %H:%M:%S"), track_id, dwell, tipo))
    except Exception as e:
        print("No se pudo escribir CSV:", e)


def clusterizar(fgx, fgy, cfg):
    """Agrupa puntos de primer plano en personas. Devuelve lista de centroides
    (cx, cy) de clusters cuyo tamano es compatible con una persona."""
    if len(fgx) < int(cfg["cluster_min_samples"]):
        return []
    pts = np.column_stack([fgx, fgy])
    labels = DBSCAN(eps=float(cfg["cluster_eps"]),
                    min_samples=int(cfg["cluster_min_samples"])).fit_predict(pts)
    centroides = []
    for lab in set(labels):
        if lab == -1:                      # ruido
            continue
        m = pts[labels == lab]
        span = max(np.ptp(m[:, 0]), np.ptp(m[:, 1]))   # tamano del cluster
        if span > float(cfg["cluster_max_size"]):  # demasiado grande -> no es persona
            continue
        centroides.append((float(m[:, 0].mean()), float(m[:, 1].mean())))
    return centroides


# --- Figura ----------------------------------------------------------------
def crear_figura(cfg):
    # Paleta del dashboard Niage (oscuro + acento dorado)
    BG_PAGE = "#0a0a0a"; BG_CARD = "#1d1d1d"; BORDE = "#3f3f3f"
    GRID = "#2f2f2f"; APAGADO = "#a8a8a8"; TEXTO = "#e3e3e3"
    ORO = "#fad51b"; ROJO = "#ef4444"
    lim = float(cfg["view_max"])
    fig = Figure(figsize=(7.6, 7), facecolor=BG_PAGE)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.08, 0.06, 0.78, 0.88])   # deja hueco a la derecha p/ barra
    ax.set_facecolor(BG_CARD)
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    if cfg.get("flip_x"):      # espejo horizontal: corrige vista derecha<->izquierda
        ax.invert_xaxis()      # afecta a TODO (puntos, ROI, personas, estelas, ejes)
    ax.tick_params(colors=APAGADO)
    for s in ax.spines.values():
        s.set_color(BORDE)
    for r in range(1, int(lim) + 1):
        ax.add_patch(patches.Circle((0, 0), r, fill=False, ec=GRID, lw=0.8))
        ax.text(0, r, "%dm" % r, color=APAGADO, fontsize=7, ha="center", va="bottom")
    # Radios y etiquetas de grados (referencia para ajustar las zonas).
    for ang in (0, 45, 90, 135, 180, -135, -90, -45):
        rad = math.radians(ang)
        cx, cy = math.cos(rad), math.sin(rad)
        es_cero = (ang == 0)
        ax.plot([0, lim * cx], [0, lim * cy], "-",
                color=ORO if es_cero else GRID,
                lw=1.4 if es_cero else 0.6, alpha=0.95 if es_cero else 0.6, zorder=1)
        ax.text(0.90 * lim * cx, 0.90 * lim * cy, "%d°" % ang,
                color=ORO if es_cero else APAGADO, fontsize=9,
                ha="center", va="center",
                weight="bold" if es_cero else "normal",
                bbox=dict(boxstyle="round,pad=0.1", fc=BG_CARD, ec="none", alpha=0.7))
    roi_w = cfg["roi_dist_max"] - cfg["roi_dist_min"]
    ax.add_patch(patches.Wedge((0, 0), cfg["roi_dist_max"], cfg["roi_angle_min"],
                               cfg["roi_angle_max"], width=roi_w,
                               facecolor=ORO, edgecolor="none", alpha=0.08))
    ax.add_patch(patches.Wedge((0, 0), cfg["roi_dist_max"], cfg["roi_angle_min"],
                               cfg["roi_angle_max"], width=roi_w,
                               facecolor="none", edgecolor=ORO, lw=1.3, alpha=0.55))
    if cfg.get("excl_enabled"):    # zona excluida (empleado): roja tenue, rayada
        ax.add_patch(patches.Wedge((0, 0), cfg["excl_dist_max"], cfg["excl_angle_min"],
                                   cfg["excl_angle_max"],
                                   width=cfg["excl_dist_max"] - cfg["excl_dist_min"],
                                   alpha=0.18, color=ROJO, hatch="///"))
    if cfg.get("track_enabled"):   # circulo de "cerca" (<near_dist)
        nd = float(cfg["near_dist"])
        ax.add_patch(patches.Circle((0, 0), nd, fill=False, ec=ROJO,
                                    lw=1.3, ls="--"))
        ax.text(0, -nd, "cerca <%.1gm" % nd, color=ROJO, fontsize=7,
                ha="center", va="top")
    ax.plot(0, 0, "^", color=ORO, markersize=13, markeredgecolor="#0a0a0a",
            markeredgewidth=0.8)

    cmap = matplotlib.colormaps[cfg.get("cmap", "turbo")]
    norm = Normalize(0, lim)
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cax = fig.add_axes([0.88, 0.10, 0.03, 0.80])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("distancia (m)", color=APAGADO)
    cb.outline.set_edgecolor(BORDE)
    cb.ax.yaxis.set_tick_params(color=BORDE)
    for t in cb.ax.get_yticklabels():
        t.set_color(APAGADO)

    # HUD anclado a coordenadas de ejes -> siempre arriba-izquierda, aunque flip_x invierta X
    txt = ax.text(0.02, 0.975, "", transform=ax.transAxes, va="top", ha="left",
                  color=TEXTO, fontsize=10.5, weight="bold")
    return {"fig": fig, "ax": ax, "txt": txt, "cmap": cmap, "norm": norm,
            "scatters": [], "overlays": []}


def render_frame(F, cfg, xs, ys, ds, rxs, rys, rds, exs, eys, bxs, bys, info):
    ax = F["ax"]
    for c in F["scatters"]:
        c.remove()
    F["scatters"] = []
    for o in F["overlays"]:
        o.remove()
    F["overlays"] = []
    if bxs and cfg.get("show_background"):   # fondo (muebles/paredes): gris muy tenue
        F["scatters"].append(ax.scatter(bxs, bys, c="#3a3a3a", s=2, linewidths=0))
    if exs:   # puntos en zona excluida: gris tenue, no cuentan
        F["scatters"].append(ax.scatter(exs, eys, c="#555555",
                                         s=max(3, cfg["point_size"] - 3), linewidths=0))
    if xs:
        F["scatters"].append(ax.scatter(xs, ys, c=ds, cmap=F["cmap"], norm=F["norm"],
                                         s=cfg["point_size"], linewidths=0))
    if rxs:   # puntos en ROI: mismo color por distancia pero grandes y con borde
        F["scatters"].append(ax.scatter(rxs, rys, c=rds, cmap=F["cmap"], norm=F["norm"],
                                         s=cfg["roi_point_size"], edgecolors="white",
                                         linewidths=0.6))
    # personas seguidas: trayectoria (estela) + circulo + ID + tiempo
    COL = {"queda": "#ef4444", "cerca": "#fad51b", "roi": "#ffffff", "fuera": "#a8a8a8"}
    mostrar_estela = cfg.get("show_trails", True)
    for tid, x, y, estado, secs, trail in info.get("tracks", []):
        col = COL.get(estado, "#ff9a00")
        # estela: linea que une las posiciones suavizadas por el Kalman, con las
        # mas antiguas mas tenues (asi se ve de donde viene y hacia donde va).
        if mostrar_estela and trail and len(trail) >= 2:
            pts = np.asarray(trail, dtype=float).reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            n = len(segs)
            base = to_rgb(col)
            colores = [(base[0], base[1], base[2], 0.10 + 0.85 * (k + 1) / n)
                       for k in range(n)]
            lc = LineCollection(segs, colors=colores, linewidths=2.2, capstyle="round")
            F["overlays"].append(ax.add_collection(lc))
        F["overlays"].append(ax.add_patch(patches.Circle((x, y), 0.30, fill=False,
                                                          ec=col, lw=2.0)))
        if estado == "queda":
            etq = "#%d QUEDA %.0fs" % (tid, secs)
        elif estado == "cerca":
            etq = "#%d %.0fs" % (tid, secs)        # cuenta hacia los 10s
        else:
            etq = "#%d" % tid
        F["overlays"].append(ax.text(x + 0.18, y + 0.18, etq, color=col,
                                     fontsize=9, family="monospace", weight="bold"))

    filtro = "FILTRO FONDO ON" if (cfg.get("bg_filter_enabled") and _bg_ref is not None) else "filtro fondo off"
    if cfg.get("track_enabled"):
        F["txt"].set_text("PERSONAS  total:%d  en zona ahora:%d\n"
                          "PASAN:%d   SE QUEDAN (<%.1gm,>%gs):%d   %s" %
                          (info["total"], info["en_roi_ahora"], info["pasan"],
                           float(cfg["near_dist"]), float(cfg["stay_near_s"]),
                           info["quedan"], filtro))
    else:
        estado = ("PRESENCIA  %.1fs" % info["dwell"]) if info["presente"] else "vacio"
        F["txt"].set_text("en ROI: %3d   %s\neventos: %d   %s" %
                          (info["n_roi"], estado, info["eventos"], filtro))
    F["ax"].set_title("FRadar  ·  %s" % time.strftime("%H:%M:%S"),
                      color="#e3e3e3", weight="bold")
    buf = io.BytesIO()
    F["fig"].savefig(buf, format="png", facecolor=F["fig"].get_facecolor(), dpi=80)
    return buf.getvalue()


# --- Hilo del lidar --------------------------------------------------------
def capturar_fondo(laser, n_scans=25):
    """Promedia varios barridos del stand VACIO -> distancia fija por angulo."""
    global _bg_ref, _bg_capture, _bg_status
    with _lock:
        _bg_status = "capturando fondo..."
    samples = defaultdict(list)
    scan = ydlidar.LaserScan()
    n = 0
    while n < n_scans and ydlidar.os_isOk():
        if not laser.doProcessSimple(scan):
            time.sleep(0.02); continue
        for p in scan.points:
            if p.range > 0:
                b = int(round(p.angle * 180.0 / math.pi)) % N_BINS
                samples[b].append(p.range)
        n += 1
    ref = [None] * N_BINS
    for b, vals in samples.items():
        vals.sort()
        ref[b] = vals[len(vals) // 2]      # mediana = distancia fija de ese angulo
    ocup = sum(1 for v in ref if v is not None)
    with _lock:
        _bg_ref = ref
        _bg_capture = False
        _bg_status = "fondo OK (%d/%d angulos)" % (ocup, N_BINS)
    guardar_bg(ref)
    print("Fondo capturado:", ocup, "angulos")


def build_laser(cfg):
    laser = ydlidar.CYdLidar()
    laser.setlidaropt(ydlidar.LidarPropSerialPort, "/dev/ttyUSB0")
    laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 115200)
    laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
    laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
    laser.setlidaropt(ydlidar.LidarPropScanFrequency, float(cfg["scan_frequency"]))
    laser.setlidaropt(ydlidar.LidarPropSampleRate, int(cfg["sample_rate"]))
    laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)
    laser.setlidaropt(ydlidar.LidarPropIntenstiy, False)
    return laser


def hilo_lidar():
    global _frame_png, _need_restart, _lidar_ok, _rec_request, _rec_status
    ydlidar.os_init()
    laser = None
    F = None
    cfg = get_cfg()
    cfg_ver_local = -1
    presente = False; t_entrada = None; eventos = 0
    intervalo = 1.0 / 8.0; ultimo_render = 0.0

    # --- Estado de tracking de personas ---
    tracker_obj = None     # instancia de Tracker (se crea/recrea segun config)
    tracker_ver = -1
    personas = {}          # id -> estado de ROI/dwell/conteo (web), keyed por id del tracker
    frame_no = 0
    t_prev = None          # tiempo del barrido anterior (para dt del Kalman)
    cnt_total = 0          # personas distintas que han entrado en la ROI
    cnt_pasan = 0          # han pasado (no se quedaron cerca)
    cnt_quedan = 0         # se han quedado cerca >umbral
    # --- Grabador de barridos ---
    rec_fh = None
    rec_until = 0.0

    while ydlidar.os_isOk():
        # ¿Hay que (re)arrancar el lidar?
        with _lock:
            restart = _need_restart or laser is None
            _need_restart = False
        if restart:
            cfg = get_cfg()
            if laser is not None:
                laser.turnOff(); laser.disconnecting()
            laser = build_laser(cfg)
            ok = laser.initialize() and laser.turnOn()
            with _lock:
                _lidar_ok = ok
            if not ok:
                print("ERROR: no se pudo arrancar el lidar; reintento en 3s")
                laser = None; time.sleep(3); continue
            print("Lidar arrancado (freq=%.1f, sample=%s)" %
                  (cfg["scan_frequency"], cfg["sample_rate"]))
            F = None  # forzar reconstruccion de figura

        # ¿Capturar fondo? (stand vacio) -> bloquea ~3s
        with _lock:
            do_capture = _bg_capture
        if do_capture:
            capturar_fondo(laser)
            continue

        # ¿Ha cambiado la config visual? -> reconstruir figura
        with _lock:
            ver = _cfg_version
        if F is None or ver != cfg_ver_local:
            cfg = get_cfg()
            F = crear_figura(cfg)
            cfg_ver_local = ver

        scan = ydlidar.LaserScan()
        if not laser.doProcessSimple(scan):
            time.sleep(0.02); continue

        with _lock:
            ref = _bg_ref
        usar_bg = cfg["bg_filter_enabled"] and ref is not None
        tol = cfg["bg_tolerance"]

        xs, ys, ds, rxs, rys, rds, exs, eys, bxs, bys = [], [], [], [], [], [], [], [], [], []
        fgx, fgy = [], []     # primer plano (todo lo movil) para clustering
        raw = []              # puntos crudos (ang_deg, range) para el grabador
        n_roi = 0
        for p in scan.points:
            if p.range <= 0:
                continue
            ang_deg = p.angle * 180.0 / math.pi
            raw.append((round(ang_deg, 2), round(p.range, 3)))
            x = p.range * math.cos(p.angle); y = p.range * math.sin(p.angle)
            if en_excluida(cfg, ang_deg, p.range):     # ignorada (empleado)
                exs.append(x); eys.append(y)
            elif usar_bg and es_fondo(ref, tol, ang_deg, p.range):  # mueble/pared fijo
                bxs.append(x); bys.append(y)
            elif en_roi(cfg, ang_deg, p.range):
                n_roi += 1; rxs.append(x); rys.append(y); rds.append(p.range)
                fgx.append(x); fgy.append(y)
            else:
                xs.append(x); ys.append(y); ds.append(p.range)
                fgx.append(x); fgy.append(y)

        ahora = time.monotonic()

        # --- Grabador de barridos (para afinar offline con replay.py) ---
        with _lock:
            req = _rec_request; _rec_request = None
        if req:
            try:
                nombre = time.strftime("grabacion_%Y%m%d_%H%M%S.jsonl")
                rec_fh = open(os.path.join(_DIR, nombre), "w")
                rec_until = ahora + float(req)
                with _lock:
                    _rec_status = "grabando %ss -> %s" % (req, nombre)
                print("[FRadar] grabando %ss -> %s" % (req, nombre))
            except Exception as e:
                print("error grabador:", e); rec_fh = None
        if rec_fh is not None:
            if ahora < rec_until:
                rec_fh.write(json.dumps({"pts": raw}) + "\n")
            else:
                rec_fh.close(); rec_fh = None
                with _lock:
                    _rec_status = "grabacion terminada"
                print("[FRadar] grabacion terminada")

        # --- Presencia agregada (simple, sin tracking) ---
        hay_algo = n_roi >= cfg["min_points"]
        if hay_algo and not presente:
            presente = True; t_entrada = ahora; eventos += 1
        elif not hay_algo and presente:
            presente = False
        dwell = (ahora - t_entrada) if (presente and t_entrada) else 0.0

        # --- Clustering + tracking de personas (Kalman + Hungarian + re-ID) ---
        tracks_vis = []
        if cfg["track_enabled"]:
            frame_no += 1
            dt = (ahora - t_prev) if t_prev else (1.0 / 7.0)
            dt = min(0.5, max(0.01, dt))         # clamp por seguridad
            # (re)crear el tracker si cambio la configuracion
            if tracker_obj is None or tracker_ver != cfg_ver_local:
                tracker_obj = Tracker(gate_dist=float(cfg["track_max_dist"]),
                                      n_confirm=int(cfg["track_confirm"]),
                                      max_misses=int(cfg["track_max_misses"]),
                                      reid_frames=int(cfg["track_max_missing"]),
                                      reid_dist=float(cfg["track_reid_dist"]))
                tracker_ver = cfg_ver_local
                personas.clear()

            centros = clusterizar(fgx, fgy, cfg)
            salida = tracker_obj.update(centros, dt)     # ids ESTABLES

            near_d = float(cfg["near_dist"]); stay_s = float(cfg["stay_near_s"])
            vistos = set()
            for tr in salida:
                tid = tr["id"]; tx = tr["x"]; ty = tr["y"]
                vistos.add(tid)
                st = personas.get(tid)
                if st is None:
                    st = {"entered_roi": False, "in_roi": False, "t_enter": None,
                          "in_near": False, "t_near": None, "near_dwell": 0.0,
                          "stayed": False, "pasa_done": False, "last": frame_no,
                          "trail": deque(maxlen=max(2, int(cfg["trail_len"])))}
                    personas[tid] = st
                st["last"] = frame_no
                st["trail"].append((tx, ty))   # historial de la posicion Kalman
                ang = math.degrees(math.atan2(ty, tx)); dist = math.hypot(tx, ty)
                en_zona = en_roi(cfg, ang, dist) and not en_excluida(cfg, ang, dist)
                en_cerca = en_zona and dist < near_d

                if en_zona and not st["in_roi"]:
                    st["in_roi"] = True; st["t_enter"] = ahora
                    if not st["entered_roi"]:
                        st["entered_roi"] = True; cnt_total += 1
                elif not en_zona and st["in_roi"]:
                    st["in_roi"] = False; st["t_enter"] = None

                if en_cerca and not st["in_near"]:
                    st["in_near"] = True; st["t_near"] = ahora
                elif not en_cerca and st["in_near"]:
                    st["in_near"] = False; st["t_near"] = None
                st["near_dwell"] = (ahora - st["t_near"]) if st["in_near"] else 0.0

                if st["in_near"] and st["near_dwell"] >= stay_s and not st["stayed"]:
                    st["stayed"] = True; cnt_quedan += 1
                    if cfg["log_csv"]:
                        log_evento(tid, st["near_dwell"], "se_queda_cerca")
                    print("[FRadar] persona #%d SE QUEDA CERCA (%.1fs)" % (tid, st["near_dwell"]))

                if st["stayed"]:
                    estado = "queda"; secs = st["near_dwell"]
                elif en_cerca:
                    estado = "cerca"; secs = st["near_dwell"]
                elif st["in_roi"]:
                    estado = "roi"; secs = (ahora - st["t_enter"]) if st["t_enter"] else 0.0
                else:
                    estado = "fuera"; secs = 0.0
                tracks_vis.append((tid, tx, ty, estado, secs, list(st["trail"])))

            # personas no vistas un rato -> las que entraron y no se quedaron = "pasan"
            for tid in list(personas):
                if frame_no - personas[tid]["last"] > cfg["track_max_missing"]:
                    st = personas[tid]
                    if st["entered_roi"] and not st["stayed"] and not st["pasa_done"]:
                        cnt_pasan += 1; st["pasa_done"] = True
                        if cfg["log_csv"]:
                            log_evento(tid, 0.0, "pasa")
                        print("[FRadar] persona #%d PASA" % tid)
                    del personas[tid]
        t_prev = ahora

        if ahora - ultimo_render < intervalo:
            continue
        ultimo_render = ahora
        en_roi_ahora = sum(1 for _, _, _, st, _, _ in tracks_vis if st in ("roi", "cerca", "queda"))
        info = {"n_roi": n_roi, "presente": presente, "dwell": dwell, "eventos": eventos,
                "tracks": tracks_vis, "total": cnt_total, "pasan": cnt_pasan,
                "quedan": cnt_quedan, "en_roi_ahora": en_roi_ahora}
        png = render_frame(F, cfg, xs, ys, ds, rxs, rys, rds, exs, eys, bxs, bys, info)
        with _lock:
            _frame_png = png


# --- Web -------------------------------------------------------------------
app = Flask(__name__)

PAGINA = """<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FRadar</title>
<style>
 :root{
   --bg:#0a0a0a; --card:#1d1d1d; --elev:#2a2a2a; --borde:#3f3f3f;
   --texto:#e3e3e3; --apag:#a8a8a8; --oro:#fad51b; --rojo:#ef4444; --verde:#10b981;
   --r-md:.375rem; --r-lg:.5rem; --r-xl:.75rem;
   --sans:Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 }
 *{box-sizing:border-box}
 body{background:var(--bg);color:var(--texto);font-family:var(--sans);
      margin:0;padding:0;-webkit-font-smoothing:antialiased}
 .topbar{display:flex;align-items:center;justify-content:space-between;
   gap:16px;padding:14px 22px;border-bottom:1px solid var(--borde);
   background:#0d0d0d;position:sticky;top:0;z-index:10}
 .brand{display:flex;align-items:center;gap:11px}
 .brand .dot{width:12px;height:12px;border-radius:50%;background:var(--oro);
   box-shadow:0 0 14px #fad51b80}
 .brand b{font-size:18px;font-weight:700;letter-spacing:.5px}
 .brand span{color:var(--apag);font-size:12px;font-weight:500}
 .nav{display:flex;gap:10px}
 .btn{display:inline-flex;align-items:center;gap:7px;padding:8px 15px;
   border:1px solid var(--borde);color:var(--texto);text-decoration:none;
   border-radius:var(--r-md);font-size:13px;font-weight:500;background:var(--card);
   transition:.15s}
 .btn:hover{border-color:var(--oro);color:var(--oro)}
 .btn.primary{background:var(--oro);color:#1a1a1a;border-color:var(--oro);font-weight:600}
 .btn.primary:hover{background:#e8c515;color:#1a1a1a}
 .wrap{max-width:920px;margin:22px auto;padding:0 18px;display:flex;
   flex-direction:column;gap:18px}
 .card{background:var(--card);border:1px solid var(--borde);
   border-radius:var(--r-xl);box-shadow:0 10px 15px -3px #0000001a,0 4px 6px -2px #0000000d}
 .card .head{display:flex;align-items:center;justify-content:space-between;
   padding:13px 18px;border-bottom:1px solid var(--borde)}
 .card .head h2{margin:0;font-size:14px;font-weight:600;color:var(--texto)}
 .card .head .live{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--apag)}
 .card .head .live i{width:8px;height:8px;border-radius:50%;background:var(--verde);
   box-shadow:0 0 8px #10b981aa;animation:pulse 1.6s infinite}
 @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
 .radar-box{padding:14px;text-align:center}
 #radar{max-width:100%;height:auto;border-radius:var(--r-lg);display:block;margin:0 auto}
 .leyenda{padding:16px 18px;display:flex;flex-wrap:wrap;gap:9px 14px;font-size:12.5px}
 .chip{display:inline-flex;align-items:center;gap:7px;padding:5px 11px;
   background:var(--elev);border:1px solid var(--borde);border-radius:9999px;color:var(--apag)}
 .chip .sw{width:11px;height:11px;border-radius:50%;flex:0 0 auto}
 .leyenda .note{flex-basis:100%;color:var(--apag);font-size:12px;margin-top:4px;line-height:1.5}
 .leyenda .note b{color:var(--texto)}
</style></head><body>
 <header class="topbar">
   <div class="brand"><span class="dot"></span>
     <div><b>FRadar</b> <span>· conteo y permanencia LiDAR</span></div>
   </div>
   <nav class="nav">
     <a class="btn" href="/stats">&#128202; Estadísticas</a>
     <a class="btn primary" href="/settings">&#9881; Ajustes</a>
   </nav>
 </header>
 <main class="wrap">
   <section class="card">
     <div class="head"><h2>Vista del radar</h2>
       <div class="live"><i></i> en vivo</div></div>
     <div class="radar-box"><img id="radar" src="/frame.png" alt="radar"></div>
   </section>
   <section class="card leyenda">
     <span class="chip"><span class="sw" style="background:#fad51b"></span> LiDAR / eje 0°</span>
     <span class="chip"><span class="sw" style="background:#ffffff;border:1px solid #777"></span> borde blanco = ROI</span>
     <span class="chip"><span class="sw" style="background:#a8a8a8"></span> fuera</span>
     <span class="chip"><span class="sw" style="background:#ffffff"></span> en zona</span>
     <span class="chip"><span class="sw" style="background:#fad51b"></span> cerca (&lt;1m)</span>
     <span class="chip"><span class="sw" style="background:#ef4444"></span> se queda (&gt;10s)</span>
     <span class="chip"><span class="sw" style="background:#3a3a3a"></span> gris = fondo fijo</span>
     <span class="note">El color de los puntos = distancia (barra derecha). La
       <b>estela</b> de cada persona es su trayectoria suavizada por filtro de Kalman
       (lo más tenue = más antiguo). Ejes: 0° dorado, giro antihorario.</span>
   </section>
 </main>
<script>
 var img=document.getElementById('radar');
 function sig(){ img.src='/frame.png?t='+Date.now(); }
 img.onload=function(){ setTimeout(sig,120); };
 img.onerror=function(){ setTimeout(sig,500); };
</script>
</body></html>"""


def campo(label, name, cfg, paso="any", nota=""):
    return ("<tr><td class='lbl'>%s</td>"
            "<td><input class='inp' name='%s' value='%s' step='%s' type='number'></td>"
            "<td class='nota'>%s</td></tr>"
            % (label, name, cfg[name], paso, nota))


@app.route("/settings", methods=["GET", "POST"])
def settings():
    global _cfg, _cfg_version, _need_restart
    msg = ""
    if request.method == "POST":
        nuevo = get_cfg()
        reinicio = False
        for k in DEFAULTS:
            if isinstance(DEFAULTS[k], bool):
                nuevo[k] = (k in request.form)   # checkbox: presente = marcado
                continue
            if k in request.form and request.form[k] != "":
                try:
                    val = type(DEFAULTS[k])(request.form[k]) if not isinstance(DEFAULTS[k], str) else request.form[k]
                except ValueError:
                    continue
                if k in LIDAR_KEYS and val != nuevo[k]:
                    reinicio = True
                nuevo[k] = val
        with _lock:
            _cfg = nuevo
            _cfg_version += 1
            if reinicio:
                _need_restart = True
        guardar_cfg(nuevo)
        return redirect("/settings?ok=1" + ("&r=1" if reinicio else ""))

    cfg = get_cfg()
    if request.args.get("ok"):
        msg = "Guardado." + (" Reiniciando el lidar..." if request.args.get("r") else "")
    filas_roi = (campo("Angulo min (deg)", "roi_angle_min", cfg) +
                 campo("Angulo max (deg)", "roi_angle_max", cfg) +
                 campo("Distancia min (m)", "roi_dist_min", cfg) +
                 campo("Distancia max (m)", "roi_dist_max", cfg) +
                 campo("Min. puntos presencia", "min_points", cfg))
    chk = "checked" if cfg["excl_enabled"] else ""
    filas_excl = ("<tr><td class='lbl'>Activar exclusion</td>"
                  "<td><input type='checkbox' name='excl_enabled' %s></td>"
                  "<td class='nota'>"
                  "ignora por completo lo que haya en esta zona</td></tr>" % chk +
                  campo("Angulo min (deg)", "excl_angle_min", cfg) +
                  campo("Angulo max (deg)", "excl_angle_max", cfg) +
                  campo("Distancia min (m)", "excl_dist_min", cfg) +
                  campo("Distancia max (m)", "excl_dist_max", cfg))
    chk_trail = "checked" if cfg["show_trails"] else ""
    chk_flip = "checked" if cfg["flip_x"] else ""
    filas_vis = (campo("Alcance vista (m)", "view_max", cfg) +
                 campo("Tamano punto", "point_size", cfg) +
                 campo("Tamano punto ROI", "roi_point_size", cfg) +
                 "<tr><td class='lbl'>Espejo horizontal</td>"
                 "<td><input type='checkbox' name='flip_x' %s></td>"
                 "<td class='nota'>"
                 "actívalo si la vista sale invertida (derecha&harr;izquierda)</td></tr>" % chk_flip +
                 "<tr><td class='lbl'>Mostrar trayectorias</td>"
                 "<td><input type='checkbox' name='show_trails' %s></td>"
                 "<td class='nota'>"
                 "dibuja la estela (recorrido) de cada persona</td></tr>" % chk_trail +
                 campo("Longitud estela (puntos)", "trail_len", cfg,
                       nota="cuantas posiciones recientes se dibujan") +
                 "<tr><td class='lbl'>Mapa de color</td>"
                 "<td><select class='inp' name='cmap'>" +
                 "".join("<option%s>%s</option>" % (" selected" if c == cfg["cmap"] else "", c)
                         for c in ("turbo", "jet", "plasma", "viridis", "cool", "hot")) +
                 "</select></td><td></td></tr>")
    filas_lidar = (campo("Frecuencia giro (Hz)", "scan_frequency", cfg, nota="X2L ~7 Hz") +
                   campo("Sample rate (kHz)", "sample_rate", cfg, nota="X2L = 3"))
    chk_bg = "checked" if cfg["bg_filter_enabled"] else ""
    chk_show = "checked" if cfg["show_background"] else ""
    filas_bg = ("<tr><td class='lbl'>Activar filtrado</td>"
                "<td><input type='checkbox' name='bg_filter_enabled' %s></td>"
                "<td class='nota'>"
                "oculta lo fijo, deja solo lo que se mueve (personas)</td></tr>" % chk_bg +
                campo("Tolerancia (m)", "bg_tolerance", cfg,
                      nota="margen; mas alto filtra mas") +
                "<tr><td class='lbl'>Mostrar fondo (gris)</td>"
                "<td><input type='checkbox' name='show_background' %s></td>"
                "<td class='nota'>"
                "dibujar lo fijo en gris tenue (o ocultarlo)</td></tr>" % chk_show)
    chk_tr = "checked" if cfg["track_enabled"] else ""
    chk_csv = "checked" if cfg["log_csv"] else ""
    filas_track = ("<tr><td class='lbl'>Activar conteo</td>"
                   "<td><input type='checkbox' name='track_enabled' %s></td>"
                   "<td class='nota'>"
                   "agrupa puntos en personas y las sigue con un ID</td></tr>" % chk_tr +
                   campo("Radio cluster eps (m)", "cluster_eps", cfg,
                         nota="puntos mas cerca de esto = misma persona") +
                   campo("Min. puntos cluster", "cluster_min_samples", cfg) +
                   campo("Tamano max persona (m)", "cluster_max_size", cfg,
                         nota="clusters mayores se descartan (paredes)") +
                   campo("Dist. asociacion (m)", "track_max_dist", cfg,
                         nota="cuanto puede moverse entre frames") +
                   campo("Confirmar id (barridos)", "track_confirm", cfg,
                         nota="visto N veces seguidas para crear id (anti-espurios)") +
                   campo("Fallos antes de limbo", "track_max_misses", cfg) +
                   campo("Frames recuperable (re-ID)", "track_max_missing", cfg,
                         nota="cuanto tiempo se recupera el MISMO id tras perderse") +
                   campo("Dist. re-ID (m)", "track_reid_dist", cfg) +
                   campo("Distancia 'cerca' (m)", "near_dist", cfg,
                         nota="se quedan si llegan a menos de esto") +
                   campo("Tiempo 'se queda' (s)", "stay_near_s", cfg,
                         nota="segundos a <cerca para contar como 'se queda'") +
                   "<tr><td class='lbl'>Registrar CSV</td>"
                   "<td><input type='checkbox' name='log_csv' %s></td>"
                   "<td class='nota'>"
                   "guarda cada evento en eventos.csv</td></tr>" % chk_csv)
    with _lock:
        bg_status = _bg_status
        rec_status = _rec_status
    return """<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>FRadar · Ajustes</title>
<style>
 :root{
   --bg:#0a0a0a; --card:#1d1d1d; --elev:#2a2a2a; --borde:#3f3f3f;
   --texto:#e3e3e3; --apag:#a8a8a8; --oro:#fad51b; --rojo:#ef4444; --verde:#10b981;
   --r-md:.375rem; --r-lg:.5rem; --r-xl:.75rem;
   --sans:Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 }
 *{box-sizing:border-box}
 body{background:var(--bg);color:var(--texto);font-family:var(--sans);
      margin:0;padding:0;-webkit-font-smoothing:antialiased;font-size:14px}
 .topbar{display:flex;align-items:center;justify-content:space-between;gap:16px;
   padding:14px 22px;border-bottom:1px solid var(--borde);background:#0d0d0d;
   position:sticky;top:0;z-index:10}
 .brand{display:flex;align-items:center;gap:11px}
 .brand .dot{width:12px;height:12px;border-radius:9999px;background:var(--oro);box-shadow:0 0 14px #fad51b80}
 .brand b{font-size:18px;font-weight:700;letter-spacing:.5px}
 .brand span{color:var(--apag);font-size:12px;font-weight:500}
 .nav{display:flex;gap:10px}
 .btn{display:inline-flex;align-items:center;gap:7px;padding:8px 15px;border:1px solid var(--borde);
   color:var(--texto);text-decoration:none;border-radius:var(--r-md);font-size:13px;
   font-weight:500;background:var(--card);cursor:pointer;font-family:var(--sans);transition:.15s}
 .btn:hover{border-color:var(--oro);color:var(--oro)}
 .btn.primary{background:var(--oro);color:#1a1a1a;border-color:var(--oro);font-weight:600}
 .btn.primary:hover{background:#e8c515}
 .wrap{max-width:760px;margin:22px auto;padding:0 18px;display:flex;flex-direction:column;gap:18px}
 .card{background:var(--card);border:1px solid var(--borde);border-radius:var(--r-xl);
   box-shadow:0 10px 15px -3px #0000001a,0 4px 6px -2px #0000000d;overflow:hidden}
 .card>.head{display:flex;align-items:center;gap:10px;padding:13px 18px;border-bottom:1px solid var(--borde)}
 .card>.head:before{content:"";width:4px;height:16px;border-radius:2px;background:var(--oro)}
 .card>.head h2{margin:0;font-size:14px;font-weight:600;border:0}
 .card>.body{padding:16px 18px}
 table{width:100%%;border-collapse:collapse}
 td{padding:5px 4px;vertical-align:middle}
 td.lbl{text-align:right;color:var(--apag);width:42%%;padding-right:12px}
 td.nota{color:#8a8a8a;font-size:12px;padding-left:10px;line-height:1.4}
 .inp{background:#141414;color:var(--texto);border:1px solid var(--borde);
   border-radius:var(--r-md);padding:6px 9px;width:130px;font-family:var(--sans);font-size:13px;outline:none}
 .inp:focus{border-color:var(--oro);box-shadow:0 0 0 3px #fad51b33}
 select.inp{width:140px}
 input[type=checkbox]{width:16px;height:16px;accent-color:var(--oro);cursor:pointer}
 .ok{background:#10b98122;border:1px solid #10b98155;color:#6ee7b7;
   padding:10px 14px;border-radius:var(--r-md);font-size:13px}
 .ok:empty{display:none}
 .tip{color:var(--apag);font-size:12.5px;line-height:1.5;margin:0 0 12px}
 .tip.warn{color:#fbe36c}.tip.danger{color:#fca5a5}
 .tip a,.body a{color:var(--oro)}
 .save{position:sticky;bottom:0;background:linear-gradient(transparent,#0a0a0a 40%%);
   padding:14px 0 4px;text-align:right}
 @media(max-width:560px){td.lbl{width:auto}.wrap{padding:0 12px}}
</style></head><body>
 <header class="topbar">
   <div class="brand"><span class="dot"></span>
     <div><b>FRadar</b> <span>· ajustes</span></div></div>
   <nav class="nav">
     <a class="btn" href="/stats">&#128202; Estadísticas</a>
     <a class="btn primary" href="/">&larr; Radar</a></nav>
 </header>
 <main class="wrap">
  <div class="ok">%s</div>

  <section class="card"><div class="head"><h2>Filtrado de fondo</h2></div><div class="body">
    <p class="tip">Con el stand <b>VACÍO</b>, pulsa el botón: el sistema aprende
     muebles/paredes y luego los resta. Estado: <b>%s</b></p>
    <form method="POST" action="/capturar_fondo">
      <button class="btn primary" type="submit">&#128247; Capturar fondo (stand vacío)</button>
    </form>
  </div></section>

  <section class="card"><div class="head"><h2>Grabador (afinado offline)</h2></div><div class="body">
    <p class="tip">Graba unos segundos de barridos reales para reproducirlos y ajustar el
     seguimiento sin estar delante del radar. Estado: <b>%s</b>
     &nbsp;<a href="/grabaciones">ver grabaciones</a></p>
    <form method="POST" action="/grabar">
      <input class="inp" style="width:80px" name="seg" value="30" type="number" min="1" max="600"> segundos
      &nbsp;<button class="btn" type="submit">&#9210; Grabar</button>
    </form>
  </div></section>

  <form method="POST">
   <section class="card"><div class="head"><h2>Filtrado de fondo · ajustes</h2></div>
     <div class="body"><table>%s</table></div></section>

   <section class="card"><div class="head"><h2>Conteo de personas</h2></div><div class="body">
     <p class="tip warn">Funciona mejor con el filtrado de fondo activado. Cuenta personas
      distintas, distingue "pasa" de "se queda" y mide el tiempo.
      &nbsp;<a href="/eventos.csv">&#11015; descargar eventos.csv</a></p>
     <table>%s</table></div></section>

   <section class="card"><div class="head"><h2>Zona de interés (ROI)</h2></div>
     <div class="body"><table>%s</table></div></section>

   <section class="card"><div class="head"><h2>Zona excluida (empleado)</h2></div><div class="body">
     <p class="tip danger">Lo que caiga aquí NO cuenta ni dispara presencia. Útil para tapar
      al empleado / mostrador.</p>
     <table>%s</table></div></section>

   <section class="card"><div class="head"><h2>Visualización</h2></div>
     <div class="body"><table>%s</table></div></section>

   <section class="card"><div class="head"><h2>LiDAR (avanzado)</h2></div><div class="body">
     <p class="tip danger">&#9888; Cambiar estos valores reinicia el sensor y puede romper la
      lectura. Para el X2L lo normal es 7 Hz y sample rate 3.</p>
     <table>%s</table></div></section>

   <div class="save"><button class="btn primary" type="submit">Guardar cambios</button></div>
  </form>
 </main>
</body></html>""" % (msg, bg_status, rec_status, filas_bg, filas_track, filas_roi, filas_excl, filas_vis, filas_lidar)


@app.route("/capturar_fondo", methods=["POST"])
def capturar_fondo_web():
    global _bg_capture
    with _lock:
        _bg_capture = True
    return redirect("/settings?ok=1")


@app.route("/grabar", methods=["POST"])
def grabar_web():
    global _rec_request
    try:
        seg = max(1, min(600, int(float(request.form.get("seg", "30")))))
    except ValueError:
        seg = 30
    with _lock:
        _rec_request = seg
    return redirect("/settings?ok=1")


@app.route("/grabaciones")
def grabaciones():
    files = sorted([f for f in os.listdir(_DIR)
                    if f.startswith("grabacion_") and f.endswith(".jsonl")], reverse=True)
    pedido = request.args.get("f")
    if pedido:
        nombre = os.path.basename(pedido)
        ruta = os.path.join(_DIR, nombre)
        if nombre in files and os.path.exists(ruta):
            with open(ruta) as fh:
                return Response(fh.read(), mimetype="application/json",
                                headers={"Content-Disposition": "attachment; filename=" + nombre})
        return Response("no existe", status=404)
    items = "".join("<li style='margin:7px 0'><a href='/grabaciones?f=%s' "
                    "style='color:#fad51b'>%s</a> "
                    "<span style='color:#a8a8a8;font-size:12px'>&mdash; %d KB</span></li>" %
                    (f, f, os.path.getsize(os.path.join(_DIR, f)) // 1024) for f in files)
    return ("<!doctype html><html lang=es><head><meta charset=utf-8>"
            "<meta name=viewport content='width=device-width, initial-scale=1'>"
            "<title>FRadar · Grabaciones</title><style>"
            "*{box-sizing:border-box}"
            "body{background:#0a0a0a;color:#e3e3e3;margin:0;padding:0;"
            "font-family:Inter,system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;"
            "-webkit-font-smoothing:antialiased}"
            ".topbar{display:flex;align-items:center;justify-content:space-between;"
            "padding:14px 22px;border-bottom:1px solid #3f3f3f;background:#0d0d0d}"
            ".brand{display:flex;align-items:center;gap:11px}"
            ".dot{width:12px;height:12px;border-radius:9999px;background:#fad51b;box-shadow:0 0 14px #fad51b80}"
            ".brand b{font-size:18px;font-weight:700}.brand span{color:#a8a8a8;font-size:12px}"
            ".btn{padding:8px 15px;border:1px solid #3f3f3f;color:#e3e3e3;text-decoration:none;"
            "border-radius:.375rem;font-size:13px;font-weight:500;background:#1d1d1d}"
            ".wrap{max-width:680px;margin:22px auto;padding:0 18px}"
            ".panel{background:#1d1d1d;border:1px solid #3f3f3f;border-radius:.75rem;padding:8px 22px}"
            "h2{font-size:14px;font-weight:600;margin:0 0 12px;display:flex;align-items:center;gap:9px}"
            "h2:before{content:'';width:4px;height:16px;border-radius:2px;background:#fad51b}"
            "ul{list-style:none;padding:0;margin:0}</style></head><body>"
            "<header class=topbar><div class=brand><span class=dot></span>"
            "<div><b>FRadar</b> <span>· grabaciones</span></div></div>"
            "<a class=btn href='/settings'>&larr; Ajustes</a></header>"
            "<main class=wrap><h2>Grabaciones</h2><div class=panel><ul>%s</ul></div></main>"
            "</body></html>" % (items or "<li style='color:#a8a8a8'>(ninguna todavía)</li>"))


@app.route("/stats")
def stats():
    pasan = quedan = 0
    dwells = []
    por_hora = defaultdict(int)       # hora del dia -> nº personas
    hoy = time.strftime("%Y-%m-%d")
    hoy_pasan = hoy_quedan = 0
    try:
        with open(CSV_PATH) as f:
            next(f, None)             # cabecera
            for linea in f:
                p = linea.strip().split(",")
                if len(p) < 4:
                    continue
                fecha_hora, _id, dwell, tipo = p[0], p[1], p[2], p[3]
                es_queda = tipo.startswith("se_queda")
                if tipo == "pasa":
                    pasan += 1
                elif es_queda:
                    quedan += 1
                    try:
                        dwells.append(float(dwell))
                    except ValueError:
                        pass
                else:
                    continue
                if len(fecha_hora) >= 13:
                    por_hora[fecha_hora[11:13]] += 1
                if fecha_hora[:10] == hoy:
                    if tipo == "pasa":
                        hoy_pasan += 1
                    elif es_queda:
                        hoy_quedan += 1
    except FileNotFoundError:
        pass

    total = pasan + quedan
    pct = (100.0 * quedan / total) if total else 0.0
    dwell_medio = (sum(dwells) / len(dwells)) if dwells else 0.0
    dwell_max = max(dwells) if dwells else 0.0

    # barras por hora
    maxh = max(por_hora.values()) if por_hora else 1
    filas = []
    for h in range(24):
        c = por_hora.get("%02d" % h, 0)
        w = int(100 * c / maxh) if maxh else 0
        filas.append(
            "<div style='display:flex;align-items:center;margin:2px 0'>"
            "<span style='width:46px;text-align:right;color:#a8a8a8;font-size:12px'>%02d:00</span>"
            "<div style='flex:1;margin:0 10px;background:#141414;border-radius:9999px;overflow:hidden'>"
            "<div style='height:12px;width:%d%%;background:#fad51b;border-radius:9999px'></div></div>"
            "<span style='width:40px;color:#a8a8a8;font-size:12px'>%d</span></div>" % (h, w, c))
    barras = "".join(filas)

    return """<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FRadar · Estadísticas</title>
<meta http-equiv="refresh" content="15">
<style>
 :root{
   --bg:#0a0a0a; --card:#1d1d1d; --elev:#2a2a2a; --borde:#3f3f3f;
   --texto:#e3e3e3; --apag:#a8a8a8; --oro:#fad51b;
   --r-md:.375rem; --r-xl:.75rem;
   --sans:Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 }
 *{box-sizing:border-box}
 body{background:var(--bg);color:var(--texto);font-family:var(--sans);
      margin:0;padding:0;-webkit-font-smoothing:antialiased}
 .topbar{display:flex;align-items:center;justify-content:space-between;gap:16px;
   padding:14px 22px;border-bottom:1px solid var(--borde);background:#0d0d0d}
 .brand{display:flex;align-items:center;gap:11px}
 .brand .dot{width:12px;height:12px;border-radius:9999px;background:var(--oro);box-shadow:0 0 14px #fad51b80}
 .brand b{font-size:18px;font-weight:700;letter-spacing:.5px}
 .brand span{color:var(--apag);font-size:12px;font-weight:500}
 .nav{display:flex;gap:10px}
 .btn{display:inline-flex;align-items:center;gap:7px;padding:8px 15px;border:1px solid var(--borde);
   color:var(--texto);text-decoration:none;border-radius:var(--r-md);font-size:13px;font-weight:500;
   background:var(--card);transition:.15s}
 .btn:hover{border-color:var(--oro);color:var(--oro)}
 .btn.primary{background:var(--oro);color:#1a1a1a;border-color:var(--oro);font-weight:600}
 .wrap{max-width:880px;margin:22px auto;padding:0 18px}
 h2{font-size:14px;font-weight:600;color:var(--texto);margin:26px 0 10px;
    display:flex;align-items:center;gap:9px}
 h2:before{content:"";width:4px;height:16px;border-radius:2px;background:var(--oro)}
 .sub{color:var(--apag);font-size:12px;margin:4px 0 0}
 .cards{display:flex;flex-wrap:wrap;gap:14px;margin:12px 0}
 .card{background:var(--card);border:1px solid var(--borde);border-radius:var(--r-xl);
   padding:16px 20px;min-width:140px;flex:1;
   box-shadow:0 10px 15px -3px #0000001a,0 4px 6px -2px #0000000d}
 .num{font-size:30px;font-weight:700;color:var(--oro);line-height:1.1}
 .lbl{font-size:12px;color:var(--apag);margin-top:4px}
 .panel{background:var(--card);border:1px solid var(--borde);border-radius:var(--r-xl);padding:16px 20px}
</style></head><body>
 <header class="topbar">
   <div class="brand"><span class="dot"></span>
     <div><b>FRadar</b> <span>· estadísticas</span></div></div>
   <nav class="nav">
     <a class="btn" href="/eventos.csv">&#11015; CSV</a>
     <a class="btn" href="/settings">&#9881; Ajustes</a>
     <a class="btn primary" href="/">&larr; Radar</a></nav>
 </header>
 <main class="wrap">
 <p class="sub">Se actualiza cada 15 s</p>
 <h2>Acumulado histórico</h2>
 <div class="cards">
  <div class="card"><div class="num">%d</div><div class="lbl">total personas</div></div>
  <div class="card"><div class="num">%d</div><div class="lbl">PASAN</div></div>
  <div class="card"><div class="num">%d</div><div class="lbl">SE QUEDAN (&lt;1m,&gt;10s)</div></div>
  <div class="card"><div class="num">%.0f%%</div><div class="lbl">%% que se queda</div></div>
  <div class="card"><div class="num">%.0fs</div><div class="lbl">permanencia media</div></div>
  <div class="card"><div class="num">%.0fs</div><div class="lbl">permanencia maxima</div></div>
 </div>
 <h2>Hoy (%s)</h2>
 <div class="cards">
  <div class="card"><div class="num">%d</div><div class="lbl">pasan hoy</div></div>
  <div class="card"><div class="num">%d</div><div class="lbl">se quedan hoy</div></div>
 </div>
 <h2>Afluencia por hora del día</h2>
 <div class="panel">%s</div>
 </main>
</body></html>""" % (total, pasan, quedan, pct, dwell_medio, dwell_max,
                     hoy, hoy_pasan, hoy_quedan, barras)


@app.route("/eventos.csv")
def eventos_csv():
    try:
        with open(CSV_PATH) as f:
            data = f.read()
    except FileNotFoundError:
        data = "fecha_hora,track_id,dwell_s,tipo\n"
    return Response(data, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=eventos.csv"})


@app.route("/")
def index():
    return PAGINA


@app.route("/frame.png")
def frame():
    with _lock:
        png = _frame_png
    if png is None:
        return Response("esperando lidar...", status=503)
    return Response(png, mimetype="image/png", headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    cargar_cfg()
    cargar_bg()
    threading.Thread(target=hilo_lidar, daemon=True).start()
    time.sleep(2)
    app.run(host="0.0.0.0", port=PUERTO_WEB, threaded=True)
