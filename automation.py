from __future__ import annotations

import argparse
import time
from pathlib import Path
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm

try:
    from RsInstrument import RsInstrument
except ImportError:
    try:
        from rsinstrument import RsInstrument
    except ImportError as exc:  # pragma: no cover - import error surfaced at runtime
        raise RuntimeError(
            "RSInstrument is required for this script. Install it with: pip install RsInstrument"
        ) from exc

SAMPLE_RATE = 48_000
SUPPORTED_SAMPLE_RATES = (44_100, 48_000)
MAX_AUDIO_SECONDS = 10.0
MIN_SCOPE_WAVEFORM_POINTS = 256
SCOPE_TIMEOUT = 20000


def normalize_pcm(samples: np.ndarray) -> np.ndarray:
    """Normalize PCM-like arrays to float32 in [-1, 1]."""
    if samples.dtype.kind in {"i", "u"}:
        max_value = float(np.iinfo(samples.dtype).max)
        if samples.dtype.kind == "i":
            return samples.astype(np.float32) / max_value
        return (samples.astype(np.float32) - max_value / 2.0) / (max_value / 2.0)
    if samples.dtype.kind == "f":
        return samples.astype(np.float32)
    raise TypeError(f"Unsupported sample dtype: {samples.dtype!r}")


def load_wav(path: str | Path, target_sample_rate: int = SAMPLE_RATE) -> tuple[int, np.ndarray]:
    """Load a WAV file, convert to mono if needed, and resample to a supported rate."""
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    sr, data = wavfile.read(audio_path)

    if data.ndim == 2:
        data = data.mean(axis=1)
    elif data.ndim == 0:
        raise ValueError(f"Unsupported audio array shape in {audio_path}: {data.shape}")

    samples = normalize_pcm(np.asarray(data))

    if sr != target_sample_rate:
        samples = resample_audio(samples, sr, target_sample_rate)

    max_record_length = int(target_sample_rate * MAX_AUDIO_SECONDS)
    if samples.size > max_record_length:
        raise ValueError(
            f"The selected audio file is too long for a single acquisition: "
            f"{samples.size / target_sample_rate:.2f}s exceeds the practical limit of "
            f"{MAX_AUDIO_SECONDS:.1f}s at {target_sample_rate} Hz."
        )

    return target_sample_rate, samples.astype(np.float32, copy=False)


def resample_audio(samples: np.ndarray, input_sr: int, target_sr: int) -> np.ndarray:
    """Resample a waveform from input_sr to target_sr using polyphase filtering."""
    if input_sr == target_sr:
        return samples.astype(np.float32, copy=False)

    gcd = np.gcd(input_sr, target_sr)
    up = target_sr // gcd
    down = input_sr // gcd
    y = signal.resample_poly(samples, up, down)
    return y.astype(np.float32, copy=False)


def open_scope(resource: str):
    """Open an RSInstrument connection to the oscilloscope."""
    scope = RsInstrument(resource)
    scope.timeout = SCOPE_TIMEOUT
    idn = scope.query_str("*IDN?")
    print(f"Connected to scope: {idn.strip()}")
    return scope


def _write_any(scope, commands, required: bool = True) -> bool:
    """Try a list of SCPI commands until one is accepted by the instrument."""
    last_error = None
    for command in commands:
        try:
            scope.write_str(command)
            return True
        except Exception as exc:  # pragma: no cover - depends on instrument capability
            last_error = exc
    if required and last_error is not None:
        raise last_error
    return False


def write_any(scope, commands, required: bool = True) -> bool:
    """Backward-compatible alias for older call sites."""
    return _write_any(scope, commands, required=required)


def release_scope_front_panel(scope) -> None:
    """Return control to the scope front panel after remote capture."""
    write_any(scope, [
        "SYST:KLOC OFF",
        "SYST:LOCK OFF",
        "SYST:LOC",
        "SYSTEM:LOCAL",
    ], required=False)

    # Some VISA backends expose a local-control API rather than SCPI LOCAL.
    for method_name in ["go_to_local", "go_to_local_control", "visa_go_to_local"]:
        method = getattr(scope, method_name, None)
        if callable(method):
            try:
                method()
                break
            except Exception:
                pass


def configure_scope(scope, duration_s: float, sample_rate: int, channel: str = "CH1", scope_vpp=8, scope_offset=0):
    """Configure the RTB2004 for a single-sequence capture with R&S-compatible SCPI."""
    if duration_s > MAX_AUDIO_SECONDS:
        raise ValueError(f"Duration exceeds the practical acquisition limit of {MAX_AUDIO_SECONDS:.1f}s.")

    channel = channel.upper()
    if not channel.startswith("CH"):
        channel = f"CH{channel}"

    try:
        scope.write_str("*RST")
    except Exception:
        pass

    volts_per_div = scope_vpp / 10.0
    scope.write(f"CHANnel{channel[2:]}:SCALe {volts_per_div}")
    scope.write(f"CHANnel{channel[2:]}:OFFSet {scope_offset}")

    scope.write(f"TRIGger:A:SOURce CH4")
    # make it so the trigger doesnt happen naturally, only from software trigger
    # necessary bc the scope doesnt support exclusive bus source
    #scope.write(f"TRIGger:A:LEVel 1.6")

    target_points = int(duration_s * SAMPLE_RATE)
    scope.write(f"ACQuire:POINts {target_points}")

    scope.write(f"ACQuire:SRATE {SAMPLE_RATE}");
    
    # scope.write(f"ACQuire:MODe: HIGHRES");
    '''
    write_any(scope, [
        f"ACQ:SRAT {sample_rate}",
        f"ACQUIRE:SRATE {sample_rate}",
        f"ACQ:SRATE {sample_rate}",
    ], required=False)
    '''

    # Match horizontal span to recording duration and keep 0 s at the left edge.
    scope.write(f"TIMebase:RANGe {duration_s}")
    write_any(scope, [
        f"TIM:RANG {duration_s}",
        f"TIMEBASE:RANG {duration_s}",
        f"TIMEBASE:RANGE {duration_s}",
    ], required=False)

    time_per_div = duration_s / 10.0
    write_any(scope, [
        f"TIMEBASE:SCAL {time_per_div}",
        f"TIM:SCAL {time_per_div}",
        f"TIMEBASE:MAIN:SCALE {time_per_div}",
    ])
    #scope.write_str(f"TIMEBASE:POSition {duration_s / 2.0}")
    scope.write("SING")  


def fetch_waveform(scope, channel: str = "CH1") -> np.ndarray:
    """Read the current waveform from the RTB2004 using the channel-scoped waveform query."""
    channel = channel.upper()
    if not channel.startswith("CH"):
        channel = f"CH{channel}"
    channel_index = channel[2:]

    # RTB2004 uses channel-specific waveform queries, not Tektronix DATA:SOURCE commands.
    original_timeout = getattr(scope, "timeout", None)
    if original_timeout is not None:
        scope.timeout = min(int(original_timeout), 3000)
    
    data = None
    # Try plain string query first; some RTB firmware returns CSV-like waveform payloads.
   
    data = np.asarray(scope.query_bin_or_ascii_float_list(f"CHANnel{channel_index}:DATA?"), dtype=np.float32)
    if not _is_usable_waveform(data):
        raise ValueError(f"Scope returned unusable waveform data: {data}")
    
    return condition_scope_waveform(np.asarray(data, dtype=np.float32))


def _is_usable_waveform(data: np.ndarray) -> bool:
    """Return True when returned data looks like an actual waveform, not a stub value."""
    arr = np.asarray(data, dtype=np.float32).reshape(-1)
    if arr.size < MIN_SCOPE_WAVEFORM_POINTS:
        return False
    finite = arr[np.isfinite(arr)]
    if finite.size < MIN_SCOPE_WAVEFORM_POINTS:
        return False
    return float(np.std(finite)) > 1e-12


def condition_scope_waveform(data: np.ndarray) -> np.ndarray:
    """Condition scope data to a plottable 1-D waveform.

    Handles common RTB responses where time and amplitude can be interleaved.
    """
    y = np.asarray(data, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return y

    finite_mask = np.isfinite(y)
    if not np.all(finite_mask):
        y = y[finite_mask]
    if y.size == 0:
        return np.zeros(0, dtype=np.float32)

    # TODO - CHECK IF THIS IS EVEN USEFUL
    # Heuristic: if data is interleaved [time0, value0, time1, value1, ...],
    # pick the lane that looks most like a waveform (non-monotonic, higher detail).
    '''
    if y.size >= 20 and y.size % 2 == 0:
        lane_a = y[0::2]
        lane_b = y[1::2]
        diff_a = np.diff(lane_a)
        diff_b = np.diff(lane_b)
        mono_a_up = diff_a.size > 0 and float(np.mean(diff_a >= 0.0))
        mono_b_up = diff_b.size > 0 and float(np.mean(diff_b >= 0.0))
        mono_a_dn = diff_a.size > 0 and float(np.mean(diff_a <= 0.0))
        mono_b_dn = diff_b.size > 0 and float(np.mean(diff_b <= 0.0))
        mono_a = max(mono_a_up, mono_a_dn)
        mono_b = max(mono_b_up, mono_b_dn)

        if mono_a > 0.995 and mono_b < 0.995:
            y = lane_b
        elif mono_b > 0.995 and mono_a < 0.995:
            y = lane_a
        else:
            # Choose the lane with more waveform-like detail.
            curv_a = float(np.std(np.diff(diff_a))) if diff_a.size > 1 else 0.0
            curv_b = float(np.std(np.diff(diff_b))) if diff_b.size > 1 else 0.0
            var_a = float(np.std(lane_a - np.median(lane_a)))
            var_b = float(np.std(lane_b - np.median(lane_b)))
            score_a = (var_a + curv_a) * (1.0 - 0.5 * mono_a)
            score_b = (var_b + curv_b) * (1.0 - 0.5 * mono_b)
            y = lane_a if score_a >= score_b else lane_b
    '''

    if y.size > 8:
        y = y - np.median(y)
    y = y/np.max(y)

    return y.astype(np.float32, copy=False)


def resample_to_length(samples: np.ndarray, target_len: int) -> np.ndarray:
    """Resample one array to a target length for side-by-side comparison plots."""
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if samples.size == target_len:
        return samples.astype(np.float32, copy=False)
    old_x = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
    new_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(new_x, old_x, samples).astype(np.float32, copy=False)


def save_outputs(prefix: str | Path, sample_rate: int, audio: np.ndarray, scope_waveform: np.ndarray) -> tuple[Path, Path, Path]:
    """Save aligned arrays, raw scope waveform, and scope audio WAV to disk."""
    base = Path(prefix)
    npz_path = base.with_suffix(".npz")
    raw_path = base.with_name(f"{base.name}_scope_raw.npy")
    wav_path = base.with_name(f"{base.name}_scope.wav")
    aligned_scope = resample_to_length(scope_waveform, audio.size)

    np.save(raw_path, scope_waveform)
    np.savez(npz_path, sample_rate=sample_rate, audio=audio, scope_aligned=aligned_scope)
    wavfile.write(wav_path, sample_rate, aligned_scope.astype(np.float32, copy=False))

    return npz_path, raw_path, wav_path


def print_waveform_stats(name: str, samples: np.ndarray) -> None:
    """Print quick stats to help diagnose noisy or malformed captures."""
    x = np.asarray(samples, dtype=np.float32)
    if x.size == 0:
        print(f"{name}: empty")
        return
    print(
        f"{name}: n={x.size}, min={float(np.min(x)):.6g}, max={float(np.max(x)):.6g}, "
        f"mean={float(np.mean(x)):.6g}, std={float(np.std(x)):.6g}"
    )


def plot_side_by_side(audio: np.ndarray, scope_waveform: np.ndarray, sample_rate: int, output_png: Path) -> None:
    """Plot original audio and captured scope waveform side-by-side and save the figure."""
    aligned_scope = resample_to_length(scope_waveform, audio.size)
    t = np.arange(audio.size, dtype=np.float32) / float(sample_rate)

    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        axes[0].plot(t, audio, linewidth=0.8)
        axes[0].set_title("Original Audio")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(alpha=0.3)

        axes[1].plot(t, aligned_scope, linewidth=0.8, color="tab:orange")
        axes[1].set_title("Scope Capture (Aligned)")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("ADC Counts")
        axes[1].grid(alpha=0.3)

        fig.savefig(output_png, dpi=150)
        plt.show()
        input("Press any key to exit")
        plt.close(fig)
    except Exception as exc:
        print(f"Warning: Unable to generate plot: {exc}")
    
def play_audio_with_scope(scope, samples: np.ndarray, sample_rate: int, channel: str, scope_vpp: float, scope_offset:float):
    """Play the audio through the default output while arming the scope to trigger at playback start."""
    duration_s = len(samples) / sample_rate
    configure_scope(
        scope,
        duration_s=duration_s,
        sample_rate=sample_rate,
        channel=channel,
        scope_vpp=scope_vpp,
        scope_offset=scope_offset
    )

    def audio_worker():
        sd.play(samples, samplerate=sample_rate, blocking=True)

    audio_thread = Thread(target=audio_worker, daemon=True)

    status = scope.query("STATus:OPERation:CONDition?")

    #RESTORE LATER
    '''
    while status != '8':
        time.sleep(0.05)
        status = scope.query("STATus:OPERation:CONDition?")
        print(f"Waiting for scope to be ready, status={status}")
    '''

    audio_thread.start()

    # RESTORE LATER
    # scope.write("*TRG")

    start_time = time.monotonic()
    with tqdm(total=duration_s, desc="Audio + scope capture", unit="s") as pbar:
        while audio_thread.is_alive():
            elapsed = min(time.monotonic() - start_time, duration_s)
            pbar.n = elapsed
            pbar.refresh()
            time.sleep(0.05)

    # Freeze the display on the captured acquisition after playback completes.
    scope.write_str(":STOP")
    release_scope_front_panel(scope)

    waveform = fetch_waveform(scope, channel=channel)
    return waveform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_file", type=str, help="Path to the WAV file to play.")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE, choices=SUPPORTED_SAMPLE_RATES,
                        help="Audio and scope-trigger timing rate. 44.1 kHz or 48 kHz are both accepted.")
    parser.add_argument(
        "--scope-vpp", type=float, default=1,
        help="Approximate vertical full-scale target in volts peak-to-peak for the selected channel.",
    )
    parser.add_argument(
        "--scope-offset", type=float, default=0,
        help="Approximate voltage offset for the selected channel.",
    )
    parser.add_argument("--channel", default="CH1", help="Scope channel to trigger and capture.")
    parser.add_argument("--save-prefix", default="capture", help="Output prefix for .npz/.npy and plot files.")
    parser.add_argument("--create-plot", action="store_true", help="create-plot", default=False)
    parser.add_argument('--use-trigger', action="store_true",default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sample_rate, samples = load_wav(args.audio_file, target_sample_rate=SAMPLE_RATE)
    duration_s = len(samples) / sample_rate
    print(f"Loaded {len(samples)} samples at {sample_rate} Hz ({duration_s:.3f}s)")
    print(f"Default audio device: {sd.default.device}")

    scope = open_scope(RsInstrument.list_resources()[0]) # may need to hardcode this
    try:
        waveform = play_audio_with_scope(
            scope,
            samples,
            sample_rate=sample_rate,
            channel=args.channel,
            scope_vpp=args.scope_vpp,
            scope_offset=args.scope_offset
        )
    finally:
        release_scope_front_panel(scope)
        scope.close()

    npz_path, raw_path, wav_path = save_outputs(args.save_prefix, SAMPLE_RATE, samples, waveform)
    print_waveform_stats("Scope waveform", waveform)
    if (args.create_plot):
        plot_path = Path(args.save_prefix).with_name(f"{Path(args.save_prefix).name}_comparison.png")
        plot_side_by_side(samples, waveform, SAMPLE_RATE, plot_path)
        print(f"Saved side-by-side plot: {plot_path}")
        

    print(f"Captured waveform with {waveform.size} points")
    print(f"Saved aligned arrays: {npz_path}")
    print(f"Saved raw scope waveform: {raw_path}")
    print(f"Saved scope WAV: {wav_path}")


if __name__ == "__main__":
    main()
