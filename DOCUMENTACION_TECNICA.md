# FRadar — Documentación técnica

Documento de referencia interna: arquitectura, algoritmos, parámetros y datos
de despliegue del proyecto FRadar (conteo y permanencia de personas con LiDAR 2D
YDLidar X2L).

> **Nota de seguridad:** las contraseñas **no** se guardan en este repositorio.
> El acceso a los equipos es por **clave SSH**. Las credenciales sensibles se
> custodian fuera del repo (gestor de contraseñas / acceso por clave).

---

## 1. Visión general

FRadar transforma el flujo de puntos de un LiDAR 2D en métricas de negocio
(personas que pasan, personas que se quedan, tiempo de permanencia). El
*pipeline*, barrido a barrido, es:

```
LiDAR (≈400 pts/vuelta, ≈7 Hz)
        │  doProcessSimple()
        ▼
Clasificación de puntos por zona
  ├─ excluida  → se ignora
  ├─ fondo     → mobiliario/pared fijo (filtro de fondo)
  ├─ ROI       → primer plano (cuenta) + se dibuja grande
  └─ entorno   → primer plano (cuenta)
        │  primer plano (x,y) en metros
        ▼
DBSCAN  → centroides de clusters tamaño-persona
        │  lista de (x,y)
        ▼
Tracker (Kalman + húngaro + M-de-N + re-ID)  → IDs estables + velocidad
        │  [{id, x, y, vx, vy}]
        ▼
Lógica de negocio (ROI/cerca/dwell, contadores, CSV)
        │
        ▼
Render PNG (matplotlib) → panel web Flask (último frame)
```

---

## 2. Arquitectura del software

### 2.1 Modelo de hilos (importante)

El binding **SWIG de `ydlidar` NO libera el GIL** en `doProcessSimple()`. Si el
render de matplotlib se hiciera en otro hilo, el servidor web se bloquearía. Por
eso:

- **Un único hilo de fondo** (`hilo_lidar`) lee el sensor **y** renderiza el PNG.
- El servidor **Flask** solo sirve el **último PNG** ya generado (`/frame.png`).
- El estado compartido se protege con un `threading.Lock` (`_lock`): config,
  versión de config, último PNG, referencia de fondo, peticiones de
  captura/grabación, etc.

Consecuencia operativa: **solo un proceso puede abrir `/dev/ttyUSB0`**. Hay que
parar el servicio antes de lanzar cualquier script de texto contra el LiDAR.

### 2.2 Ficheros

| Fichero | Rol |
|---|---|
| `ydlidar_web.py` | Aplicación principal: config, hilo LiDAR, clasificación de puntos, render, panel web y página de ajustes. ~950 líneas. |
| `tracker.py` | Seguimiento independiente del LiDAR (se alimenta de listas `(x,y)`): `KalmanCV`, `Track`, `Tracker`. |
| `replay.py` | Validación offline: escenarios sintéticos (cruce, oclusión) y reproducción de grabaciones `.jsonl`. |
| `setup_fradar.sh` | Instalador (SDK, venv, dependencias, servicio systemd). |

### 2.3 Configuración

- Valores por defecto en el diccionario `DEFAULTS` de `ydlidar_web.py`.
- Se persisten en `config.json` (no versionado; se genera por equipo).
- La página `/settings` escribe `config.json` y sube `_cfg_version`; el hilo
  detecta el cambio y **reconstruye la figura** y/o **reinicia el sensor** si se
  tocó un parámetro de LiDAR (`scan_frequency`, `sample_rate`).

---

## 3. Algoritmos

### 3.1 Clasificación de puntos por zona

Para cada punto `(ang, range)` del barrido, en orden:

1. **Zona excluida** (`en_excluida`): si está activa y el punto cae dentro → se
   ignora por completo (no cuenta ni dibuja como persona).
2. **Fondo** (`es_fondo`): con el filtro activo, si la distancia coincide
   (± `bg_tolerance`) con la referencia aprendida para ese ángulo → es
   mobiliario/pared fijo.
3. **ROI** (`en_roi`): dentro del rango de ángulos y distancias → primer plano,
   se dibuja grande con borde blanco.
4. **Entorno**: el resto → primer plano también, puntos normales.

El "primer plano" (ROI + entorno) es lo que alimenta al *clustering*.

### 3.2 Filtrado de fondo

`capturar_fondo()` promedia ~25 barridos del stand **vacío** y guarda, por cada
grado (`N_BINS = 360`), la **mediana** de la distancia. Esa referencia
(`bg_reference.json`) se resta en caliente: todo punto que coincida con la
referencia se marca como fondo. Imprescindible en centros comerciales por
cristales/espejos/reflejos.

### 3.3 Clustering (DBSCAN)

`clusterizar()` agrupa los puntos de primer plano con `DBSCAN(eps=cluster_eps,
min_samples=cluster_min_samples)`. Cada cluster cuyo tamaño (`max(ptp_x, ptp_y)`)
no supere `cluster_max_size` se reduce a su **centroide** `(cx, cy)`. Clusters
demasiado grandes (paredes mal filtradas) se descartan.

### 3.4 Filtro de Kalman (`KalmanCV`)

Kalman 2D con **modelo de velocidad constante**. Estado:

```
x = [x, y, vx, vy]
```

- **Predicción** (`predict(dt)`): matriz de transición `F` con `dt`; ruido de
  proceso `Q` modelado como aceleración aleatoria con factor `q`.
- **Corrección** (`update(z)`): observamos solo la posición (`H` toma `x, y`);
  ruido de medida `R = r·I`. Ganancia de Kalman estándar.
- Parámetros: `q` (ruido de proceso = cuánto puede acelerar) y `r` (ruido de
  medida = cuánto se fía del centroide). Defaults `q=0.04`, `r=0.05`.

El Kalman aporta la **posición suavizada** que se usa tanto para asociar como
para dibujar (incluida la estela), y la **velocidad** `(vx, vy)`.

### 3.5 Asociación óptima (húngaro)

`Tracker._asignar()` construye la matriz de costes = distancia euclídea entre
la posición **predicha** de cada track y cada detección, y resuelve con
`scipy.optimize.linear_sum_assignment`. Solo se aceptan parejas dentro de una
**puerta** (`gate_dist`). Esto evita que se intercambien IDs cuando dos personas
se cruzan.

### 3.6 Confirmación M-de-N

Un track nuevo no es "oficial" hasta acumular `n_confirm` aciertos seguidos
(`hits >= n_confirm`). Mata los IDs espurios de un solo frame (ruido).

### 3.7 Limbo y re-identificación

- Un track que falla más de `max_misses` veces seguidas pasa al **limbo** (solo
  si estaba confirmado).
- En el limbo sigue **prediciendo** con Kalman durante `reid_frames` frames.
- Si una detección libre aparece cerca (`reid_dist`) de la posición prevista de
  un track del limbo → se **recupera el mismo ID** (re-ID tras oclusión).
- Si no, la detección libre crea un track nuevo.

### 3.8 Trayectorias (estelas)

Cada persona guarda en su estado un `deque(maxlen=trail_len)` de **posiciones
suavizadas por Kalman**, una por frame visto. En el render, esas posiciones se
dibujan con una `LineCollection` donde el alfa crece de las antiguas (tenues) a
las recientes (opacas), en el color del estado de la persona. La estela solo es
visible cuando hay movimiento real entre frames.

### 3.9 Lógica de negocio (conteo y dwell)

Por cada ID estable (`personas[tid]`) se mantiene estado de:

- **entrada a ROI** → incrementa `total` la primera vez (`entered_roi`).
- **cerca** = en ROI y a `< near_dist` → arranca cronómetro `near_dwell`.
- **se queda** = `near_dwell >= stay_near_s` → incrementa `quedan`, evento
  `se_queda_cerca`.
- **pasa** = entró en ROI, no se quedó, y desaparece > `track_max_missing`
  frames → incrementa `pasan`, evento `pasa`.

Cada evento se registra en `eventos.csv` (descargable en `/eventos.csv`). Los
contadores en pantalla se reinician al reiniciar el servicio; el CSV persiste.

---

## 4. Referencia de parámetros (`/settings`)

### ROI y presencia
| Clave | Def | Significado |
|---|---|---|
| `roi_angle_min` / `roi_angle_max` | -45 / 45 | rango angular de la ROI (grados) |
| `roi_dist_min` / `roi_dist_max` | 0.20 / 3.00 | rango de distancias de la ROI (m) |
| `min_points` | 8 | puntos mínimos en ROI para marcar presencia |

### Zona excluida
| Clave | Def | Significado |
|---|---|---|
| `excl_enabled` | False | activar exclusión |
| `excl_angle_min/max`, `excl_dist_min/max` | 120/180, 0/2.5 | zona ignorada |

### Filtrado de fondo
| Clave | Def | Significado |
|---|---|---|
| `bg_filter_enabled` | False | activar resta de fondo |
| `bg_tolerance` | 0.20 | margen (m); más alto filtra más agresivo |
| `show_background` | True | dibujar el fondo en gris tenue |

### Conteo / tracking
| Clave | Def | Significado |
|---|---|---|
| `track_enabled` | False | activar clustering + tracking |
| `cluster_eps` | 0.35 | radio DBSCAN (m) |
| `cluster_min_samples` | 4 | puntos mínimos por cluster |
| `cluster_max_size` | 0.90 | tamaño máx. de un cluster-persona (m) |
| `track_max_dist` | 0.75 | puerta de asociación (m) — `gate_dist` |
| `track_confirm` | 3 | barridos para confirmar un ID (M-de-N) |
| `track_max_misses` | 5 | fallos antes de mandar al limbo |
| `track_max_missing` | 40 | frames recuperable en el limbo (re-ID) |
| `track_reid_dist` | 1.0 | distancia máx. de re-ID (m) |
| `near_dist` | 1.0 | umbral "cerca" (m) |
| `stay_near_s` | 10.0 | segundos a <cerca para "se queda" |
| `log_csv` | True | registrar eventos en `eventos.csv` |

### Visualización
| Clave | Def | Significado |
|---|---|---|
| `view_max` | 4.0 | alcance mostrado (m) — solo zoom |
| `point_size` / `roi_point_size` | 7 / 22 | tamaño de puntos |
| `show_trails` | True | dibujar trayectorias (estelas) |
| `trail_len` | 40 | nº de posiciones de la estela |
| `cmap` | turbo | mapa de color por distancia |

### LiDAR (reinician el sensor)
| Clave | Def | Significado |
|---|---|---|
| `scan_frequency` | 7.0 | frecuencia de giro (Hz) — X2L ≈7 |
| `sample_rate` | 3 | sample rate (kHz) — X2L = 3 |

### Configuración fija del SDK (no tocar a la ligera)
```python
LidarPropSerialPort      = "/dev/ttyUSB0"
LidarPropSerialBaudrate  = 115200
LidarPropLidarType       = TYPE_TRIANGLE
LidarPropDeviceType      = YDLIDAR_TYPE_SERIAL
LidarPropSingleChannel   = True       # X2/X2L son single channel
LidarPropIntenstiy       = False
```

---

## 5. Equipos de despliegue

> Sin contraseñas en el repo. Acceso por **clave SSH**.

### 5.1 Raspberry Pi 5 — despliegue actual de FRadar

| Campo | Valor |
|---|---|
| Hostname | `RaspBerryFRadar` |
| IP | `10.10.10.201` |
| Usuario | `fradar` |
| SO / Python | Debian 13 (trixie) / Python 3.13 / ARM64 |
| Acceso | **clave SSH** `~/.ssh/id_fradar` (contraseña fuera del repo) |
| sudo | `NOPASSWD` para `fradar` |
| Proyecto | `~/fradar` (`/home/fradar/fradar/`) |
| LiDAR | `/dev/ttyUSB0` |
| Servicio | `fradar-web.service` (systemd, enable --now) |
| Panel | `http://10.10.10.201:8080` |

```bash
# acceso
ssh -i ~/.ssh/id_fradar fradar@10.10.10.201
# gestión del servicio
sudo systemctl {status,restart,stop} fradar-web
journalctl -u fradar-web -f
```

### 5.2 Mini PC "FaceMe" — primer prototipo

| Campo | Valor |
|---|---|
| IP | `10.10.10.22` |
| Usuario | `rodrigo` |
| SO | Ubuntu x86_64 (headless, sin DISPLAY) |
| Proyecto | `~/ydlidar-pruebas` (venv Python 3.10, SDK 1.2.20 desde fuente) |
| LiDAR | `/dev/ttyUSB0` (CP210x); usuario en grupo `dialout` |
| Servicio | `ydlidar-web.service` · panel en `http://10.10.10.22:8080` |

---

## 6. Operación y resolución de problemas

- **`Checksum error` al arrancar**: negociación normal del X2L single-channel.
  No es un fallo.
- **El panel se queda congelado**: comprobar que el hilo del LiDAR vive
  (`journalctl`). Recordar que render y lectura comparten hilo por el GIL.
- **No arranca el LiDAR / `/dev/ttyUSB0` ocupado**: solo un proceso puede abrir
  el puerto. Parar el servicio antes de lanzar scripts de texto.
- **Variables globales en `hilo_lidar`**: las globales que se asignan dentro del
  hilo deben declararse `global` (si no, `UnboundLocalError`, p. ej. con
  `_rec_request`).
- **numpy 2.x**: usar `np.ptp(arr)`, no `arr.ptp()`.
- **Distancias fiables**: el X2L distingue personas hasta ~4–5 m. Recomendado
  vista 3–4 m y ROI ≤3 m.

---

## 7. Afinado offline

```bash
python replay.py --test                 # cruce de 2 personas + oclusión; objetivo 0 saltos de ID
python replay.py --file grabacion.jsonl # reproduce barridos reales grabados desde /settings
```

El panel puede **grabar** N segundos de barridos crudos (`grabacion_*.jsonl`)
desde `/settings`, descargables en `/grabaciones`, para reproducirlos por el
pipeline de clustering+tracking y ajustar parámetros sin estar delante del radar.

---

## 8. Roadmap pendiente

1. Capturar fondo en el sitio definitivo del stand.
2. Grabar tráfico real y afinar parámetros con `replay.py`.
3. Validar conteo "pasa/se queda" contra observación manual.
4. Endurecer el despliegue (arranque al boot ya cubierto por systemd).
