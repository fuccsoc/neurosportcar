import socket
import threading
import time
import sys
from collections import deque
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from CapsuleSDK.Capsule import Capsule
from CapsuleSDK.DeviceLocator import DeviceLocator
from CapsuleSDK.DeviceType import DeviceType
from CapsuleSDK.Device import Device
from CapsuleSDK.EEGTimedData import EEGTimedData
from CapsuleSDK.Resistances import Resistances


ESP32_IP = "172.20.10.12"
UDP_PORT = 9999
KEEP_ALIVE_SEC = 0.8

LIB_PATH = "./libCapsuleClient.dylib"
BIPOLAR_CHANNELS = True
N_CH = 2 if BIPOLAR_CHANNELS else 4

SAMPLE_RATE = 250.0
WINDOW_SECONDS = 2.0
WINDOW_LEN = int(SAMPLE_RATE * WINDOW_SECONDS)

CALIBRATION_SECONDS = 10.0

SPEED_ALWAYS = 100

ALPHA_BAND = (8.0, 13.0)
BETA_BAND = (14.0, 30.0)

RES_HISTORY_SECONDS = 60.0
RES_MAX_POINTS = int(RES_HISTORY_SECONDS * 2)

is_running = True

current_direction = "S"  # "F" | "B" | "S"
current_speed = 0

alpha_power = 0.0
beta_power = 0.0

status_lock = threading.Lock()
buf_lock = threading.Lock()
res_lock = threading.Lock()

eeg_buf = np.zeros((N_CH, WINDOW_LEN), dtype=float)
eeg_idx = 0

last_freqs = None
last_psd = None

ab_t = deque(maxlen=800)
ab_alpha = deque(maxlen=800)
ab_beta = deque(maxlen=800)

res_hist = {}
res_latest = {}

device_locator = None
device = None


def send_command():
    global current_direction, current_speed
    if current_direction == "S":
        cmd = "S"
    else:
        cmd = f"{current_direction},{current_speed}"

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto((cmd + "\n").encode("utf-8"), (ESP32_IP, UDP_PORT))
    except OSError as e:
        print(f"\n[!] Ошибка отправки UDP: {e}")
    finally:
        sock.close()


def clear_line():
    if sys.platform.startswith("win"):
        print("\r", end="", flush=True)
    else:
        print("\r\033[K", end="", flush=True)

def update_status(extra: str = ""):
    with status_lock:
        direction_str = {"F": "ВПЕРЁД", "B": "НАЗАД", "S": "СТОП"}[current_direction]
        clear_line()
        print(
            f"Скорость: {current_speed:3d}% | Направление: {direction_str} | "
            f"alpha={alpha_power:.3e} beta={beta_power:.3e} {extra}",
            end="",
            flush=True,
        )


def push_block(block: np.ndarray):
    global eeg_idx, eeg_buf
    block = np.asarray(block, dtype=float)
    ch, n_s = block.shape

    if ch != N_CH:
        if ch > N_CH:
            block = block[:N_CH]
        else:
            pad = np.zeros((N_CH, n_s), dtype=float)
            pad[:ch] = block
            block = pad

    with buf_lock:
        for i in range(n_s):
            eeg_buf[:, eeg_idx] = block[:, i]
            eeg_idx = (eeg_idx + 1) % WINDOW_LEN

def get_window_copy() -> np.ndarray:
    with buf_lock:
        idx = eeg_idx
        win = np.concatenate([eeg_buf[:, idx:], eeg_buf[:, :idx]], axis=1)
    return win


def compute_psd_welch(x: np.ndarray, sfreq: float):
    from scipy.signal import welch
    freqs, psd = welch(
        x,
        fs=sfreq,
        nperseg=min(x.shape[1], int(sfreq * 2)),
        axis=1,
    )
    return freqs, psd

def bandpower(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    p_ch = np.trapz(psd[:, mask], freqs[mask], axis=1)
    return float(np.mean(p_ch))

def decide_command(alpha: float, beta: float):
    if alpha > beta:
        return "F", SPEED_ALWAYS
    else:
        return "B", SPEED_ALWAYS


class EventFlag:
    def __init__(self):
        self._v = False
    def set(self): self._v = True
    def is_set(self): return self._v

found_flag = EventFlag()
conn_flag = EventFlag()

def on_device_list(locator, info, fail_reason):
    global device
    if not info:
        print("Устройства не найдены.")
        return
    chosen = info[0]
    print("Найдено устройство:", chosen.get_name(), "serial:", chosen.get_serial())
    device = Device(locator, chosen.get_serial(), locator.get_lib())
    found_flag.set()

def on_connection_status_changed(d, status):
    print("Статус подключения:", status)
    conn_flag.set()

def on_eeg(d, eeg: EEGTimedData):
    samples = eeg.get_samples_count()
    ch = eeg.get_channels_count()
    if samples <= 0:
        return

    block = np.zeros((ch, samples), dtype=float)
    for i in range(samples):
        for c in range(ch):
            block[c, i] = eeg.get_processed_value(c, i)

    push_block(block)

def on_resistances(d, res: Resistances):
    t = time.time()
    with res_lock:
        for i in range(len(res)):
            ch_name = res.get_channel_name(i)
            kohm = float(res.get_value(i)) / 1e3

            if not np.isfinite(kohm) or kohm <= 0:
                continue

            res_latest[ch_name] = kohm
            if ch_name not in res_hist:
                res_hist[ch_name] = deque(maxlen=RES_MAX_POINTS)
            res_hist[ch_name].append((t, kohm))


def control_loop():
    global current_direction, current_speed
    global alpha_power, beta_power, last_freqs, last_psd
    global is_running

    calibration_start = time.time()
    calibrated = False
    update_status("| калибровка...")

    while is_running:
        time.sleep(0.15)

        win = get_window_copy()
        elapsed = time.time() - calibration_start

        if (not calibrated) and elapsed < CALIBRATION_SECONDS:
            if current_direction != "S":
                current_direction = "S"
                current_speed = 0
                send_command()
            update_status(f"| калибровка {elapsed:4.1f}/{CALIBRATION_SECONDS:.1f}s")
            continue

        if not calibrated:
            calibrated = True
            update_status("| калибровка завершена ✅")

        try:
            freqs, psd = compute_psd_welch(win, SAMPLE_RATE)
            a = bandpower(freqs, psd, ALPHA_BAND[0], ALPHA_BAND[1])
            b = bandpower(freqs, psd, BETA_BAND[0], BETA_BAND[1])

            alpha_power = a
            beta_power = b
            last_freqs = freqs
            last_psd = psd

            now = time.time()
            ab_t.append(now)
            ab_alpha.append(a)
            ab_beta.append(b)

        except Exception as e:
            update_status(f"| PSD error: {e}")
            continue

        new_dir, new_speed = decide_command(alpha_power, beta_power)
        changed = (new_dir != current_direction) or (new_speed != current_speed)

        if changed:
            current_direction = new_dir
            current_speed = new_speed
            send_command()

        with res_lock:
            if res_latest:
                items = list(res_latest.items())[:2]
                res_str = " | " + " ".join([f"{k}={v:.1f}кОм" for k, v in items])
            else:
                res_str = ""
        update_status(res_str)


fig, axs = plt.subplots(4, 1, figsize=(12, 12), constrained_layout=True)
ax_eeg, ax_psd, ax_ab, ax_res = axs

# 1) EEG
t_eeg = np.linspace(-WINDOW_SECONDS, 0.0, WINDOW_LEN)
eeg_lines = []
for i in range(N_CH):
    ln, = ax_eeg.plot([], [], lw=1, label=f"Ch{i}")
    eeg_lines.append(ln)
ax_eeg.set_title("EEG (последнее окно)")
ax_eeg.set_xlabel("Time (s)")
ax_eeg.set_ylabel("Amplitude")
ax_eeg.grid(True)
ax_eeg.set_xlim(-WINDOW_SECONDS, 0.0)
ax_eeg.legend(loc="upper right")

# 2) PSD
psd_lines = []
for i in range(N_CH):
    ln, = ax_psd.plot([], [], lw=1, label=f"PSD Ch{i}")
    psd_lines.append(ln)
ax_psd.set_title("PSD (Welch) + alpha/beta bands")
ax_psd.set_xlabel("Frequency (Hz)")
ax_psd.set_ylabel("Power")
ax_psd.set_xlim(0, 40)
ax_psd.grid(True)
ax_psd.axvspan(ALPHA_BAND[0], ALPHA_BAND[1], alpha=0.15)
ax_psd.axvspan(BETA_BAND[0], BETA_BAND[1], alpha=0.15)
psd_text = ax_psd.text(0.02, 0.95, "", transform=ax_psd.transAxes, va="top")
ax_psd.legend(loc="upper right")

# 3) alpha/beta history
ab_alpha_line, = ax_ab.plot([], [], lw=1, label="alpha power")
ab_beta_line, = ax_ab.plot([], [], lw=1, label="beta power")
ax_ab.set_title("Alpha/Beta во времени")
ax_ab.set_xlabel("Time (s, relative)")
ax_ab.set_ylabel("Band power")
ax_ab.grid(True)
ax_ab.legend(loc="upper right")

# 4) resistances history
res_lines = {}  # ch_name -> line2d
ax_res.set_title("Resistances (кОм) во времени")
ax_res.set_xlabel("Time (s, relative)")
ax_res.set_ylabel("kOhm")
ax_res.grid(True)
res_text = ax_res.text(0.02, 0.95, "", transform=ax_res.transAxes, va="top")

def update_plot(_):
    # EEG
    win = get_window_copy()

    for i in range(N_CH):
        eeg_lines[i].set_data(t_eeg, win[i])

    ymin = float(np.nanmin(win))
    ymax = float(np.nanmax(win))
    if (not np.isfinite(ymin)) or (not np.isfinite(ymax)) or ymin == ymax:
        pass
    else:
        pad = 0.1 * (ymax - ymin)
        ax_eeg.set_ylim(ymin - pad, ymax + pad)

    # PSD
    if last_freqs is not None and last_psd is not None:
        freqs = last_freqs
        psd = last_psd

        mask = freqs <= 40
        f = freqs[mask]
        p = psd[:, mask]

        for i in range(N_CH):
            psd_lines[i].set_data(f, p[i])

        pmax = float(np.nanmax(p))
        if not np.isfinite(pmax) or pmax <= 0:
            pmax = 1.0
        ax_psd.set_ylim(0, pmax * 1.15)

        psd_text.set_text(
            f"alpha={alpha_power:.3e}\n"
            f"beta ={beta_power:.3e}\n"
            f"alpha>beta => F, иначе => B\n"
            f"speed={SPEED_ALWAYS}%"
        )

    # alpha/beta history
    if len(ab_t) >= 2:
        t0 = ab_t[-1]
        tx = np.array([x - t0 for x in ab_t], dtype=float)
        ay = np.array(ab_alpha, dtype=float)
        by = np.array(ab_beta, dtype=float)

        ab_alpha_line.set_data(tx, ay)
        ab_beta_line.set_data(tx, by)

        ax_ab.set_xlim(tx.min(), 0.0)
        y_max = float(max(np.nanmax(ay), np.nanmax(by), 1e-12))
        if np.isfinite(y_max) and y_max > 0:
            ax_ab.set_ylim(0, y_max * 1.15)

    # resistances history
    with res_lock:
        if res_hist:
            t_now = time.time()
            latest_lines = []

            for ch_name, dq in res_hist.items():
                if len(dq) < 2:
                    continue

                arr = np.array(dq, dtype=float)  # (n,2)
                x = arr[:, 0] - t_now
                y = arr[:, 1]

                m = np.isfinite(y)
                x = x[m]
                y = y[m]
                if len(x) < 2:
                    continue

                if ch_name not in res_lines:
                    ln, = ax_res.plot([], [], lw=1, label=ch_name)
                    res_lines[ch_name] = ln
                    ax_res.legend(loc="upper right")

                res_lines[ch_name].set_data(x, y)
                latest_lines.append((ch_name, res_latest.get(ch_name)))

            all_y = []
            for dq in res_hist.values():
                for _, v in dq:
                    if np.isfinite(v):
                        all_y.append(v)

            if all_y:
                y_max = float(max(all_y))
                if np.isfinite(y_max) and y_max > 0:
                    ax_res.set_ylim(0, max(5.0, y_max * 1.2))

            ax_res.set_xlim(-RES_HISTORY_SECONDS, 0.0)

            latest_lines = [(k, v) for k, v in latest_lines if v is not None and np.isfinite(v)]
            latest_lines = latest_lines[:6]
            if latest_lines:
                res_text.set_text("Последние:\n" + "\n".join([f"{k}: {v:.1f} кОм" for k, v in latest_lines]))
            else:
                res_text.set_text("Последние:\n(нет валидных значений)")
        else:
            res_text.set_text("Жду resistances...")

    return eeg_lines + psd_lines + [ab_alpha_line, ab_beta_line] + list(res_lines.values())


def main():
    global device_locator, is_running, current_direction, current_speed

    print("Подключение к CapsuleSDK...")
    capsule = Capsule(LIB_PATH)
    device_locator = DeviceLocator(capsule.get_lib())
    device_locator.set_on_devices_list(on_device_list)
    device_locator.request_devices(device_type=DeviceType.Band, seconds_to_search=10)

    t0 = time.time()
    while not found_flag.is_set():
        device_locator.update()
        time.sleep(0.02)
        if time.time() - t0 > 15:
            print("Не удалось найти устройство.")
            return

    device.set_on_connection_status_changed(on_connection_status_changed)
    device.set_on_eeg(on_eeg)

    device.set_on_resistances(on_resistances)

    print("Подключаемся к устройству...")
    device.connect(bipolarChannels=BIPOLAR_CHANNELS)

    t0 = time.time()
    while not conn_flag.is_set():
        device_locator.update()
        time.sleep(0.02)
        if time.time() - t0 > 40:
            print("Не удалось подключиться.")
            return

    device.start()
    print("Готово! Управление + визуализации запущены.\n")

    if not sys.platform.startswith("win"):
        print("\033[?25l", end="", flush=True)

    current_direction = "S"
    current_speed = 0
    send_command()

    ctrl_thread = threading.Thread(target=control_loop, daemon=True)
    ctrl_thread.start()

    def keep_alive_loop():
        while is_running:
            time.sleep(KEEP_ALIVE_SEC)
            try:
                device_locator.update()
            except Exception:
                pass
            send_command()

    ka_thread = threading.Thread(target=keep_alive_loop, daemon=True)
    ka_thread.start()

    ani = FuncAnimation(fig, update_plot, interval=100, blit=False, cache_frame_data=False)

    try:
        plt.show()
    finally:
        is_running = False
        current_direction = "S"
        current_speed = 0
        send_command()

        if not sys.platform.startswith("win"):
            print("\033[?25h", end="", flush=True)
        clear_line()
        print("Остановлено.")

        try:
            device.stop()
            device.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
