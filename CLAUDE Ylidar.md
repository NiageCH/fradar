# Proyecto: Análisis de tráfico de personas con YDLidar X2L

## Objetivo
Usar un lidar 2D **YDLidar X2L** en stands de centros comerciales para medir:
- **Cuánta gente pasa** cerca del stand (conteo).
- **Cuánto tiempo se quedan** (tiempo de permanencia / dwell time).

Fase actual: **pruebas y validación del prototipo**. Todavía no es despliegue
de producción.

## Hardware
- **Sensor:** YDLidar X2L (lidar 2D, 360°, triangulación).
  - Single-channel, baudrate **115200**.
  - ~3000 muestras/s, ~7 Hz de giro → **~400 puntos por vuelta** (~0.9° de
    resolución angular).
  - Alcance nominal hasta ~8 m; detección fiable de personas ~4–5 m.
  - Se conecta por una placa adaptadora USB-serie (chip CP210x o CH340).
- **Ordenador:** mini PC con **Ubuntu**, IP `10.10.10.22`, acceso por SSH.
  - El lidar está conectado físicamente a este mini PC, normalmente en
    `/dev/ttyUSB0`.

> IMPORTANTE para Claude Code: tú puedes editar y ejecutar código en este
> mini PC, pero NO tienes acceso a los datos del sensor en vivo. Las pruebas
> con el lidar real las lanza el usuario. Propón comandos, no asumas que
> "ves" lo que devuelve el lidar.

## Entorno de desarrollo
- Carpeta de trabajo: `~/ydlidar-pruebas`
- Entorno virtual de Python en `./venv` (Ubuntu 24.04 bloquea pip global).
  Activarlo siempre antes de trabajar: `source venv/bin/activate`
- Dependencias Python: `ydlidar` (binding del SDK), `matplotlib`, `numpy`.
- SDK compilado e instalado desde fuente: https://github.com/YDLIDAR/YDLidar-SDK
  (incluye `tri_test` en `build/` para validar hardware sin Python).

## Configuración del lidar (parámetros del SDK)
Estos valores son los correctos para el X2L y NO deben cambiarse a la ligera:
```python
LidarPropSerialPort      = "/dev/ttyUSB0"
LidarPropSerialBaudrate  = 115200
LidarPropLidarType       = TYPE_TRIANGLE
LidarPropDeviceType      = YDLIDAR_TYPE_SERIAL
LidarPropScanFrequency   = 7.0
LidarPropSampleRate      = 3          # kHz
LidarPropSingleChannel   = True       # X2/X2L son single channel
LidarPropIntenstiy       = False
```

## Estado del código
Archivo principal: **`ydlidar_stand_test.py`**

Qué hace ya:
1. Conecta con el lidar y lee barridos (`doProcessSimple`).
2. Dibuja los puntos en vivo (vista cenital, lidar en el origen) con matplotlib.
3. Define una **zona de interés (ROI)** delante del stand mediante rango de
   ángulos y de distancia, y detecta presencia + mide tiempo de permanencia
   básico (entrada/salida de la zona).

Parámetros de la ROI ajustables al principio del archivo:
`ROI_ANGLE_MIN_DEG`, `ROI_ANGLE_MAX_DEG`, `ROI_DIST_MIN_M`, `ROI_DIST_MAX_M`,
`MIN_POINTS_PRESENCE`.

Limitación actual: solo detecta "hay algo en la zona o no". NO distingue
personas individuales todavía.

## Montaje físico para las pruebas
- Lidar en horizontal, a altura de cintura/torso (~1.0–1.2 m).
- Colocado en el borde del stand mirando al pasillo.
- La ROI se calibra mirando el gráfico en vivo hasta cubrir "delante del stand".

## Roadmap (próximos pasos, en orden)
1. **Filtrado de fondo:** capturar un barrido de referencia con el stand vacío
   y restar los puntos fijos (paredes, mostrador) para quedarse solo con lo que
   se mueve. Imprescindible en centros comerciales por cristales/espejos/reflejos.
2. **Clustering:** agrupar los puntos de cada barrido en objetos (DBSCAN de
   scikit-learn sobre coordenadas x,y). Un cluster del tamaño de una persona =
   candidato a persona.
3. **Tracking:** asignar un ID estable a cada cluster entre barridos
   consecutivos (nearest-neighbour o filtro tipo Kalman/SORT) para seguir a
   cada persona.
4. **Conteo y dwell por persona:** con IDs estables, contar entradas únicas a
   la ROI y medir el tiempo que cada ID permanece dentro.
5. **Producción:** quitar el gráfico en vivo (el mini PC irá sin pantalla),
   registrar eventos a CSV/log, y arrancar como servicio.

## Consideraciones y limitaciones a recordar
- Es un lidar **2D**: solo ve un plano. Personas en fila india se ocultan entre sí.
- Resolución baja: mantener la ROI corta (<3–4 m) para distinguir individuos.
- Cristal, espejos y luz solar directa generan lecturas falsas → de ahí la
  importancia del filtrado de fondo.

## Convenciones
- Comentarios y mensajes de salida en español.
- Antes de tocar la configuración del SDK, avisar: cambiarla puede romper la
  lectura del sensor.
- Para probar cambios que dependan del hardware, dejar el comando listo para
  que lo ejecute el usuario.
