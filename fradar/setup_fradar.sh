#!/usr/bin/env bash
# =============================================================================
# FRadar - instalacion en Raspberry Pi 5 (Raspberry Pi OS Bookworm 64-bit o
# Ubuntu 24.04 ARM64). Deja el panel web corriendo como servicio systemd.
#
# Uso:
#   1) copia este script y ydlidar_web.py a la Raspberry (misma carpeta)
#   2) chmod +x setup_fradar.sh && ./setup_fradar.sh
#   3) abre http://<IP-de-la-pi>:8080
# =============================================================================
set -e
exec > "$HOME/setup.log" 2>&1     # toda la salida al log (sudo pide pass por la tty)

USUARIO="${SUDO_USER:-$USER}"
HOME_DIR="$(eval echo ~"$USUARIO")"
PROY="$HOME_DIR/fradar"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> Usuario: $USUARIO   Carpeta proyecto: $PROY"

echo "==> 1/6 Dependencias del sistema (apt)"
sudo apt-get update -qq
sudo apt-get install -y git cmake build-essential swig python3-dev python3-venv

echo "==> 2/6 Permiso del puerto serie (dialout)"
sudo usermod -aG dialout "$USUARIO"

echo "==> 3/6 Compilar e instalar YDLidar-SDK (ARM64)"
mkdir -p "$PROY"
cd "$PROY"
if [ ! -d YDLidar-SDK ]; then
    git clone --depth 1 https://github.com/YDLIDAR/YDLidar-SDK.git
fi
cd YDLidar-SDK
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"
sudo make install
sudo ldconfig

echo "==> 4/6 Entorno Python (venv) + dependencias"
cd "$PROY"
python3 -m venv venv
./venv/bin/pip install -q --upgrade pip
./venv/bin/pip install -q numpy matplotlib scikit-learn flask
echo "    (instalando binding python ydlidar desde el SDK)"
./venv/bin/pip install ./YDLidar-SDK

echo "==> 5/6 Copiar la app"
cp "$SRC_DIR/ydlidar_web.py" "$PROY/ydlidar_web.py"

echo "==> 6/6 Servicio systemd (fradar-web)"
sudo tee /etc/systemd/system/fradar-web.service >/dev/null <<UNIT
[Unit]
Description=FRadar - panel web YDLidar X2L
After=network.target

[Service]
User=$USUARIO
WorkingDirectory=$PROY
ExecStart=$PROY/venv/bin/python $PROY/ydlidar_web.py
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now fradar-web

echo
echo "================================================================"
echo " FRadar instalado. Servicio: fradar-web"
echo "  - Estado:   sudo systemctl status fradar-web"
echo "  - Log:      journalctl -u fradar-web -f"
echo "  - Panel:    http://<IP-de-la-pi>:8080"
echo
echo " NOTA: el grupo 'dialout' se aplica al reiniciar sesion. Si el"
echo " lidar no lee a la primera, reinicia la Pi:  sudo reboot"
echo "================================================================"
