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

    def __init__(self, xy, q=0.04, r=0.05, vel_damp=1.0):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0], dtype=float)
        self.P = np.diag([0.1, 0.1, 1.0, 1.0])
        self.q = q          # ruido de proceso (cuanto puede acelerar)
        self.r = r          # ruido de medida (cuanto se fia del centroide)
        self.vel_damp = vel_damp   # 1.0 = velocidad constante; <1 amortigua la
                                   # velocidad para no "pasarse de largo" al girar

    def predict(self, dt):
        d = self.vel_damp
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, d, 0],
                      [0, 0, 0, d]], dtype=float)
        # ruido de proceso (aceleracion aleatoria)
        G = np.array([0.5 * dt * dt, 0.5 * dt * dt, dt, dt])
        Q = np.outer(G, G) * self.q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def freeze(self):
        """Track perdido (limbo): anula la velocidad e infla la incertidumbre de
        posicion. Asi la re-identificacion se basa en la ULTIMA posicion vista y
        no en una prediccion que sigue avanzando en la direccion antigua."""
        self.x[2] = 0.0
        self.x[3] = 0.0
        self.P[0, 0] += 0.5
        self.P[1, 1] += 0.5

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
    __slots__ = ("id", "kf", "hits", "misses", "lost", "confirmed", "age",
                 "last_xy", "dup", "sep", "muted")

    def __init__(self, tid, xy, q, r, vel_damp=1.0):
        self.id = tid
        self.kf = KalmanCV(xy, q, r, vel_damp)
        self.hits = 1
        self.misses = 0
        self.lost = 0          # frames en el limbo (perdido pero recuperable)
        self.confirmed = False
        self.age = 0
        self.last_xy = (xy[0], xy[1])   # ultima deteccion asociada (gating robusto)
        self.dup = 0           # frames solapado sobre un track mas antiguo (dedup)
        self.sep = 0           # frames separado de el (para reactivar si se separan)
        self.muted = False     # duplicado persistente: sigue vivo pero NO se emite


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
                 reid_frames=40, reid_dist=1.0, q=0.04, r=0.05, vel_damp=1.0,
                 dedup_dist=0.45, dedup_frames=10):
        self.gate_dist = gate_dist
        self.n_confirm = n_confirm
        self.max_misses = max_misses
        self.reid_frames = reid_frames
        self.reid_dist = reid_dist
        self.q = q
        self.r = r
        self.vel_damp = vel_damp
        # dedup: dos tracks que se solapan de forma persistente = misma persona
        # (reflexion/pierna fragmentada). El mas nuevo se silencia (dedup_dist<=0
        # lo desactiva). dedup_frames = solape sostenido antes de silenciar.
        self.dedup_dist = dedup_dist
        self.dedup_frames = dedup_frames
        self.activos = []      # tracks vivos
        self.limbo = []        # tracks perdidos, candidatos a re-id
        self._next = 1

    def _nuevo(self, xy):
        t = Track(self._next, xy, self.q, self.r, self.vel_damp)
        self._next += 1
        return t

    def _match(self, tracks, dets, idx, dt, gate_base):
        """Hungarian con puerta sobre un SUBCONJUNTO de detecciones (idx = indices
        aun libres). Devuelve (parejas[(track, j)], tracks_libres, idx_libres).

        La puerta es ADAPTATIVA al movimiento: crece con la velocidad del track y
        con dt, asi los frames perdidos y los caminantes rapidos no rompen el id
        (tope: como mucho el doble del radio base)."""
        if not tracks or not idx:
            return [], list(tracks), list(idx)
        C = np.zeros((len(tracks), len(idx)))
        gates = np.empty(len(tracks))
        for i, t in enumerate(tracks):
            px, py = t.kf.pos                      # posicion prevista (Kalman)
            lx, ly = t.last_xy                     # ultima deteccion vista
            vx, vy = t.kf.vel
            gates[i] = gate_base + min(np.hypot(vx, vy) * dt, gate_base)
            for jj, j in enumerate(idx):
                dx, dy = dets[j]
                # coste = la MENOR de las dos distancias: asi un giro brusco (la
                # prediccion se pasa de largo) no rompe la asociacion -> mismo id
                C[i, jj] = min(np.hypot(px - dx, py - dy),
                               np.hypot(lx - dx, ly - dy))
        ri, cj = linear_sum_assignment(C)
        parejas, ti_used, dj_used = [], set(), set()
        for i, jj in zip(ri, cj):
            if C[i, jj] <= gates[i]:
                parejas.append((tracks[i], idx[jj])); ti_used.add(i); dj_used.add(jj)
        t_libres = [tracks[i] for i in range(len(tracks)) if i not in ti_used]
        i_libres = [idx[jj] for jj in range(len(idx)) if jj not in dj_used]
        return parejas, t_libres, i_libres

    def _dedup(self):
        """Silencia (no borra) el track mas NUEVO cuando lleva pegado a otro mas
        antiguo un tiempo sostenido: es el MISMO objeto (reflexion / persona
        fragmentada en dos clusters). Se conserva el id viejo. El track silenciado
        sigue vivo y absorbiendo su deteccion, asi el duplicado no renace en bucle.
        Si de verdad se separan (dos personas que iban juntas) se reactiva."""
        if self.dedup_dist <= 0:
            return
        conf = sorted([t for t in self.activos if t.confirmed], key=lambda t: t.id)
        for b in range(len(conf)):
            joven = conf[b]
            solapado = False
            for a in range(b):                    # solo contra tracks MAS antiguos
                viejo = conf[a]
                if viejo.muted:
                    continue
                d = np.hypot(joven.kf.pos[0] - viejo.kf.pos[0],
                             joven.kf.pos[1] - viejo.kf.pos[1])
                if d <= self.dedup_dist:
                    solapado = True
                    break
            if solapado:
                joven.dup += 1; joven.sep = 0
                if joven.dup >= self.dedup_frames:
                    joven.muted = True
            else:
                joven.sep += 1
                if joven.sep >= self.dedup_frames:   # separados de verdad -> reactivar
                    joven.muted = False; joven.dup = 0

    def update(self, detecciones, dt):
        """detecciones: lista de (x, y) en metros. dt: segundos desde el frame
        anterior. Devuelve la lista de tracks CONFIRMADOS y visibles ahora."""
        # 1) predecir activos
        for t in self.activos:
            t.kf.predict(dt); t.age += 1

        libres = list(range(len(detecciones)))

        # 2) asociacion en DOS FASES: primero los tracks CONFIRMADOS (protege los
        #    ids establecidos: una deteccion espuria ya no le roba la deteccion a
        #    un id real), y solo con lo que sobra, los tentativos (B2).
        confirmados = [t for t in self.activos if t.confirmed]
        tentativos = [t for t in self.activos if not t.confirmed]
        m1, _, libres = self._match(confirmados, detecciones, libres, dt, self.gate_dist)
        m2, _, libres = self._match(tentativos, detecciones, libres, dt, self.gate_dist)

        emparejados = set()
        for t, j in (m1 + m2):
            t.kf.update(detecciones[j])
            t.last_xy = detecciones[j]                     # ancla para gating robusto
            t.hits += 1; t.misses = 0
            if not t.confirmed and t.hits >= self.n_confirm:
                t.confirmed = True                         # B3
            emparejados.add(id(t))

        # 3) activos no emparejados -> suben fallos; si pasan, al limbo
        sobreviven = [t for t in self.activos if id(t) in emparejados]
        for t in self.activos:
            if id(t) in emparejados:
                continue
            t.misses += 1
            if t.misses > self.max_misses:
                t.lost = 0
                if t.confirmed:
                    t.kf.freeze()                          # ancla re-id a la ult. posicion
                    self.limbo.append(t)                   # solo re-id de ids reales
            else:
                sobreviven.append(t)
        self.activos = sobreviven

        # 4) detecciones libres: re-id contra el limbo (B4) o track nuevo
        for t in self.limbo:
            t.kf.predict(dt); t.lost += 1                  # vel=0 (congelada) -> no deriva
        re_parejas, _, libres = self._match(self.limbo, detecciones, libres, dt, self.reid_dist)
        reusadas = set()
        for t, j in re_parejas:
            t.kf.update(detecciones[j])
            t.last_xy = detecciones[j]
            t.misses = 0; t.lost = 0
            self.activos.append(t); reusadas.add(id(t))    # recupera el MISMO id
        self.limbo = [t for t in self.limbo
                      if id(t) not in reusadas and t.lost <= self.reid_frames]
        for j in libres:                                   # nadie reidentificado -> nuevo
            self.activos.append(self._nuevo(detecciones[j]))

        # 5) de-duplicar duplicados persistentes (reflexiones / fragmentacion)
        self._dedup()

        # 6) salida: solo confirmados, vistos ahora y NO silenciados
        salida = []
        for t in self.activos:
            if t.confirmed and t.misses == 0 and not t.muted:
                x, y = t.kf.pos; vx, vy = t.kf.vel
                salida.append({"id": t.id, "x": x, "y": y, "vx": vx, "vy": vy})
        return salida
