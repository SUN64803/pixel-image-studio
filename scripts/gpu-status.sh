#!/usr/bin/env bash
# Read-only GPU health check for PixelSmith Image Studio.
#
# Nothing here changes system state: no sudo, no module loads, no writes.
# Run it after a reboot, or right before rebooting after an Xid 79 crash:
#
#   bash scripts/gpu-status.sh
#   bash scripts/gpu-status.sh > ~/gpu-status-$(date +%F-%H%M).txt
#
# Exit code: 0 = no Xid 79 in this boot, 1 = Xid 79 seen in this boot.
set -u

ok()   { printf '  \033[32mOK\033[0m   %s\n' "$*"; }
warn() { printf '  \033[33mWARN\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$*"; }
info() { printf '  ..   %s\n' "$*"; }
head1() { printf '\n== %s ==\n' "$*"; }

have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------- kernel
head1 "Kernel"
info "running: $(uname -r)"
if [ -r /proc/cmdline ]; then
  info "cmdline: $(cat /proc/cmdline)"
fi

# ---------------------------------------------------------------- driver
head1 "NVIDIA driver"
if [ -r /proc/driver/nvidia/version ]; then
  ok "$(grep -m1 NVRM /proc/driver/nvidia/version)"
else
  bad "/proc/driver/nvidia/version missing — module not loaded"
fi

if have dkms; then
  dkms_out="$(dkms status 2>/dev/null | grep -i '^nvidia/' || true)"
  if [ -n "$dkms_out" ]; then
    info "dkms: $dkms_out"
  else
    warn "dkms: no nvidia module registered"
  fi
fi

if [ -d /usr/src/nvidia-580.196.07 ]; then
  ok "patched source present: /usr/src/nvidia-580.196.07"
else
  warn "patched source missing: /usr/src/nvidia-580.196.07"
fi

# A dpkg-managed nvidia driver would fight the .run/DKMS build.
if have dpkg-query; then
  pkg="$(dpkg-query -W -f='${Status}\n' 'nvidia-driver-*' 2>/dev/null | grep -c 'install ok installed' || true)"
  if [ "${pkg:-0}" -gt 0 ]; then
    warn "dpkg reports $pkg nvidia-driver-* package(s) installed — may conflict with .run"
  else
    ok "no dpkg nvidia-driver-* packages (expected: .run + DKMS)"
  fi
fi

head1 "APT pin"
pin=/etc/apt/preferences.d/99-block-nvidia-driver
if [ -f "$pin" ]; then
  ok "pin file present: $pin"
  grep -q 'Pin-Priority: -1' "$pin" && ok "Pin-Priority: -1 set" \
    || warn "pin file exists but has no 'Pin-Priority: -1'"
else
  bad "pin file missing: $pin"
fi

# ---------------------------------------------------------------- GPU state
head1 "GPU state"
if have nvidia-smi; then
  if nvidia-smi -L >/dev/null 2>&1; then
    nvidia-smi -L | sed 's/^/  ..   /'
    if have nvidia-smi; then
      info "power/temp: $(nvidia-smi --query-gpu=name,pstate,power.draw,power.limit,power.default_limit,temperature.gpu,memory.used,memory.total \
        --format=csv,noheader 2>/dev/null)"
      info "persistence: $(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null)"
    fi
  else
    bad "nvidia-smi cannot talk to the GPU (driver/GPU down?)"
  fi
else
  warn "nvidia-smi not found"
fi

# ---------------------------------------------------------------- PCIe link
head1 "PCIe link (P102-100 @ 0000:07:00.0)"
gpu=0000:07:00.0
sys=/sys/bus/pci/devices/$gpu
if [ -d "$sys" ]; then
  info "current speed: $(cat "$sys/current_link_speed" 2>/dev/null || echo '?')"
  info "current width: $(cat "$sys/current_link_width" 2>/dev/null || echo '?')"
  info "max speed:     $(cat "$sys/max_link_speed" 2>/dev/null || echo '?')"
  info "max width:     $(cat "$sys/max_link_width" 2>/dev/null || echo '?')"
  drv="$(readlink -f "$sys/driver" 2>/dev/null || true)"
  [ -n "$drv" ] && drv="$(basename "$drv")" || drv="none"
  info "driver:        $drv"
  if [ -r "$sys/current_link_speed" ]; then
    case "$(cat "$sys/current_link_speed" 2>/dev/null)" in
      *2.5*|*"1.0"*) warn "link is at Gen1 — expected on this M.2 x4 slot, but check the adapter" ;;
    esac
  fi
else
  bad "$gpu not present on the PCI bus (GPU fell off?)"
fi

# ---------------------------------------------------------------- DRM nodes
head1 "DRM nodes"
if [ -d /dev/dri ]; then
  ls -l /dev/dri/ | sed 's/^/  ..   /'
  cards=$(find /dev/dri -maxdepth 1 -name 'card[0-9]*' 2>/dev/null | wc -l)
  info "$cards card node(s) — 2 means nvidia-drm is still exposing a node"
else
  bad "/dev/dri missing"
fi

# ---------------------------------------------------------------- Xorg
head1 "Xorg"
oc=/usr/share/X11/xorg.conf.d/nvidia-drm-outputclass.conf
[ -f "$oc" ] && warn "nvidia OutputClass present: $oc (Xorg will bind the GPU)" \
             || ok "no nvidia OutputClass — Xorg will not bind the GPU"
off=/usr/share/X11/xorg.conf.d/11-nvidia-offload.conf
[ -f "$off" ] && warn "AllowNVIDIAGPUScreens present: $off" \
              || ok "no AllowNVIDIAGPUScreens override"
if [ -n "${DISPLAY:-}" ] && have xrandr; then
  providers="$(xrandr --listproviders 2>/dev/null | head -1)"
  [ -n "$providers" ] && info "$providers"
fi

head1 "nvidia-drm modeset"
ms_conf=/etc/modprobe.d/nvidia-drm-modeset.conf
if [ -f "$ms_conf" ]; then
  ok "config present: $ms_conf -> $(tr '\n' ' ' < "$ms_conf")"
else
  warn "no $ms_conf (modeset defaults to on; Xorg can bind the GPU)"
fi
if [ -r /sys/module/nvidia_drm/parameters/modeset ]; then
  info "live value: $(cat /sys/module/nvidia_drm/parameters/modeset 2>/dev/null)"
else
  info "live value: unreadable without root (expected)"
fi

# ---------------------------------------------------------------- Xid 79
head1 "Xid 79 in this boot"
xid_found=0
if have journalctl; then
  xid="$(journalctl -k -b 0 --no-pager 2>/dev/null | grep -E 'Xid.*79|fallen off the bus' || true)"
  if [ -n "$xid" ]; then
    xid_found=1
    bad "Xid 79 found in current boot:"
    printf '%s\n' "$xid" | tail -5 | sed 's/^/       /'
  else
    ok "no Xid 79 in the current boot"
  fi
  nvrm="$(journalctl -k -b 0 --no-pager 2>/dev/null | grep -cE 'NVRM' || true)"
  info "$nvrm NVRM line(s) in this boot"
else
  warn "journalctl not available"
fi

# ---------------------------------------------------------------- firmware/meta
head1 "Kernel packages"
if have dpkg-query; then
  for p in "linux-image-$(uname -r)" "linux-modules-$(uname -r)"; do
    st="$(dpkg-query -W -f='${Status}' "$p" 2>/dev/null || true)"
    case "$st" in
      *"install ok installed"*) ok "$p installed" ;;
      *) bad "$p NOT installed — running kernel may be inconsistent" ;;
    esac
  done
  # Removed by the old purge; not needed for a consistent kernel.
  for p in linux-firmware linux-generic-7.0 linux-image-generic-7.0; do
    st="$(dpkg-query -W -f='${Status}' "$p" 2>/dev/null || true)"
    case "$st" in
      *"install ok installed"*) ok "$p installed" ;;
      *) info "$p not installed (removed by purge; meta/firmware only)" ;;
    esac
  done
  holds="$(apt-mark showhold 2>/dev/null | tr '\n' ' ')"
  [ -n "$holds" ] && info "apt holds: $holds" || warn "no apt holds (kernel may upgrade and break DKMS)"
fi

# ---------------------------------------------------------------- Ollama
head1 "Ollama GPU usage"
if have ollama; then
  ps_out="$(ollama ps 2>/dev/null || true)"
  [ -n "$ps_out" ] && printf '%s\n' "$ps_out" | sed 's/^/  ..   /'
fi
if systemctl is-active --quiet ollama 2>/dev/null; then
  envs="$(systemctl show ollama -p Environment --value 2>/dev/null || true)"
  case "$envs" in
    *CUDA_VISIBLE_DEVICES=*|*CUDA_VISIBLE_DEVICES=\"\"*) ok "ollama forced to CPU" ;;
    *) warn "ollama may still use CUDA (see docs/GPU-XID79.md §5.3)" ;;
  esac
fi

# ---------------------------------------------------------------- verdict
head1 "Verdict"
if [ "$xid_found" -eq 1 ]; then
  bad "This boot already lost the GPU. Reboot to recover; do not purge the driver."
  exit 1
fi
ok "No Xid 79 in this boot."
printf '  ..   Reminder: apt purge does not touch the .run/DKMS driver; reboot is the fix.\n'
exit 0
