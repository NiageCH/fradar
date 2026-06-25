#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tracker.py - Seguimiento de personas robusto para FRadar
========================================================
Tracker tipo SORT mejorado, pensado para mantener el MISMO id por persona:

  B1. Filtro de Kalman por persona (modelo velocidad constante) -> prediccion
      suave y tolerante al temblor del centroide.
  B2. Asignacion optima (Hungarian / linear_sum_assignment) -> no intercambia
      ids cuando dos personas se cruzan.
  B3. Confirmacion M-de-N -> un id "oficial" solo nace tras verse en varios
      barridos seguidos (mata los ids espurios de un solo frame).
  B4. Re-identificacion -> los tracks perdidos se guardan un tiempo y, si algo
      reaparece cerca de su posicion prevista, se recupera el MISMO id.

No depende del lidar: se alimenta con listas de detecciones (x, y) en metros,
asi se puede probar offline con datos grabados o sinteticos (ver replay.py).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanCV:
    """Kalman 2D, estado [x, y, vx, vy], modelo de velocidad constante."""

    def __init__(self, xy, q=0.04, r=0.05):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0], dtype=float)
        self.P = np.diag([0.1, 0.1, 1.0, 1.0])
        self.q = q          # ruido de proceso (cuanto puede acelerar)
        self.r = r          # ruido de medida (cuanto se fia del centroide)

    def predict(self, dt):
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=float)
        # ruido de proceso (aceleracion aleatoria)
        G = np.array([0.5 * dt * dt, 0.5 * dt * dt, dt, dt])
        Q = np.outer(G, G) * self.q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        R = np.eye(2) * self.r
        y = np.array(z, dtype=float) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    @property
    def pos(self):
        return self.x[0], self.x[1]

    @property
    def vel(self):
        return self.x[2], self.x[3]


class Track:
    __slots__ = ("id", "kf", "hits", "misses", "lost", "confirmed", "age")

    def __init__(self, tid, xy, q, r):
        self.id = tid
        self.kf = KalmanCV(xy, q, r)
        self.hits = 1
        self.misses = 0
        self.lost = 0          # frames en el limbo (perdido pero recuperable)
        self.confirmed = False
        self.age = 0


class Tracker:
    """Mantiene los tracks frame a frame.

    Parametros clave (todos ajustables, p.ej. desde la config del panel):
      gate_dist   distancia max (m) para asociar deteccion<->track
      n_confirm   barridos seguidos para confirmar un id (B3)
      max_misses  fallos seguidos antes de mandar el track al limbo
      reid_frames frames que un track perdido sigue siendo recuperable (B4)
      reid_dist   distancia max (m) para re-identificar (B4)
    """

    def __init__(self, gate_dist=0.8, n_confirm=3, max_misses=5,
                 reid_frames=40, reid_dist=1.0, q=0.04, r=0.05):
        self.gate_dist = gate_dist
        self.n_confirm = n_confirm
        self.max_misses = max_misses
        self.reid_frames = reid_frames
        self.reid_dist = reid_dist
        self.q = q
        self.r = r
        self.activos = []      # tracks vivos
        self.limbo = []        # tracks perdidos, candidatos a re-id
        self._next = 1

    def _nuevo(self, xy):
        t = Track(self._next, xy, self.q, self.r)
        self._next += 1
        return t

    @staticmethod
    def _asignar(tracks, dets, gate):
        """Hungarian con puerta. Devuelve (parejas, tracks_libres, dets_libres)."""
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        C = np.zeros((len(tracks), len(dets)))
        for i, t in enumerate(tracks):
            px, py = t.kf.pos
            for j, (dx, dy) in enumerate(dets):
                C[i, j] = np.hypot(px - dx, py - dy)
        ri, cj = linear_sum_assignment(C)
        parejas, ti_used, dj_used = [], set(), set()
        for i, j in zip(ri, cj):
            if C[i, j] <= gate:
                parejas.append((i, j)); ti_used.add(i); dj_used.add(j)
        t_libres = [i for i in range(len(tracks)) if i not in ti_used]
        d_libres = [j for j in range(len(dets)) if j not in dj_used]
        return parejas, t_libres, d_libres

    def update(self, detecciones, dt):
        """detecciones: lista de (x, y) en metros. dt: segundos desde el frame
        anterior. Devuelve la lista de tracks CONFIRMADOS y visibles ahora."""
        # 1) predecir activos
        for t in self.activos:
            t.kf.predict(dt); t.age += 1

        # 2) asociacion optima activos<->detecciones (B2)
        parejas, t_libres, d_libres = self._asignar(self.activos, detecciones, self.gate_dist)
        for i, j in parejas:
            t = self.activos[i]
            t.kf.update(detecciones[j])
            t.hits += 1; t.misses = 0
            if not t.confirmed and t.hits >= self.n_confirm:
                t.confirmed = True                         # B3

        # 3) activos no emparejados -> suben fallos; si pasan, al limbo
        sobreviven = []
        for idx, t in enumerate(self.activos):
            if idx in [i for i, _ in parejas]:
                sobreviven.append(t); continue
            t.misses += 1
            if t.misses > self.max_misses:
                t.lost = 0
                if t.confirmed:
                    self.limbo.append(t)                   # solo re-id de ids reales
            else:
                sobreviven.append(t)
        self.activos = sobreviven

        # 4) detecciones libres: re-id contra el limbo (B4) o track nuevo
        dets_pendientes = [detecciones[j] for j in d_libres]
        for t in self.limbo:
            t.kf.predict(dt); t.lost += 1
        re_parejas, _, rl_libres = self._asignar(self.limbo, dets_pendientes, self.reid_dist)
        reusadas = set()
        for i, j in re_parejas:
            t = self.limbo[i]
            t.kf.update(dets_pendientes[j])
            t.misses = 0; t.lost = 0
            self.activos.append(t); reusadas.add(i)        # recupera el MISMO id
        self.limbo = [t for k, t in enumerate(self.limbo)
                      if k not in reusadas and t.lost <= self.reid_frames]
        for j in rl_libres:                                # nadie reidentificado -> nuevo
            self.activos.append(self._nuevo(dets_pendientes[j]))

        # 5) salida: solo confirmados vistos en este frame
        salida = []
        for t in self.activos:
            if t.confirmed and t.misses == 0:
                x, y = t.kf.pos; vx, vy = t.kf.vel
                salida.append({"id": t.id, "x": x, "y": y, "vx": vx, "vy": vy})
        return salida
