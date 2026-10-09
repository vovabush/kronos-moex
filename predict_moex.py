"""Проверка Kronos-base на дневных акциях Московской биржи (режим TQBR).

Модель видит 400 торговых дней и не видит последние 90. Прогноз на 120 дней:
первые 90 лежат поверх факта, последние 30 уходят дальше последней известной
свечи. На графике синяя линия — факт, красная — прогноз.

Запуск из этой папки:

    .venv\\Scripts\\python predict_moex.py SBER
    .venv\\Scripts\\python predict_moex.py TATN
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# venv на Windows не находит Tcl сам, и окно matplotlib тогда не открывается.
_tcl_root = Path(sys.base_prefix) / "tcl"
_tcl_library = _tcl_root / "tcl8.6"
_tk_library = _tcl_root / "tk8.6"
if (_tcl_library / "init.tcl").is_file():
    os.environ.setdefault("TCL_LIBRARY", str(_tcl_library))
if (_tk_library / "tk.tcl").is_file():
    os.environ.setdefault("TK_LIBRARY", str(_tk_library))

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
KRONOS_ROOT = ROOT / "Kronos"
OUTPUT_DIR = ROOT / "outputs"

LOOKBACK = 400
HELD_OUT = 90
FUTURE = 30
PRED_LEN = HELD_OUT + FUTURE
ISS_PAGE = 500
ISS_URL = (
    "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR"
    "/securities/{ticker}/candles.json"
)
HISTORY_FROM = "2022-01-01"


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "kronos-moex-test/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"MOEX ISS вернул ошибку {exc.code} для {url}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Нет доступа к MOEX ISS: {exc.reason}") from exc


def fetch_tqbr_daily(ticker: str) -> pd.DataFrame:
    """Дневные свечи TQBR: open, high, low, close, volume, amount."""
    rows: list[list] = []
    columns: list[str] | None = None
    start = 0
    while True:
        url = (
            f"{ISS_URL.format(ticker=ticker)}"
            f"?interval=24&from={HISTORY_FROM}&iss.meta=off&start={start}"
        )
        payload = fetch_json(url)["candles"]
        if columns is None:
            columns = payload["columns"]
        chunk = payload["data"]
        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < ISS_PAGE:
            break
        start += len(chunk)

    if not rows:
        raise SystemExit(explain_missing_ticker(ticker))

    frame = pd.DataFrame(rows, columns=columns)
    frame = frame.rename(columns={"value": "amount", "begin": "timestamps"})
    frame["timestamps"] = pd.to_datetime(frame["timestamps"])
    for column in ("open", "high", "low", "close", "volume", "amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["open", "high", "low", "close", "volume", "amount"])
    frame = frame.sort_values("timestamps").drop_duplicates("timestamps")

    # Сегодняшняя дневная свеча ещё не закрыта, пока идёт сессия.
    now_moscow = pd.Timestamp.now(tz="Europe/Moscow")
    today = pd.Timestamp(now_moscow.year, now_moscow.month, now_moscow.day)
    frame = frame.loc[frame["timestamps"] < today]
    return frame.reset_index(drop=True)


def next_trading_days(after: pd.Timestamp, count: int) -> pd.Series:
    """Следующие торговые дни фондового рынка после даты after.

    Берётся производственный календарь РФ: 0 и 2 — рабочий день, 1 — выходной.
    31 декабря биржа не проводит обычную сессию акций, даже если день рабочий.
    """
    after = pd.Timestamp(after).normalize()
    collected: list[pd.Timestamp] = []
    cursor = after + pd.Timedelta(days=1)
    while len(collected) < count:
        window_end = cursor + pd.Timedelta(days=max(count * 3, 45))
        for day in trading_days_between(cursor, window_end):
            if day.month == 12 and day.day == 31:
                continue
            collected.append(day)
            if len(collected) == count:
                break
        cursor = window_end + pd.Timedelta(days=1)
    return pd.Series(collected)


def trading_days_between(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    url = f"https://isdayoff.ru/api/getdata?date1={start:%Y%m%d}&date2={end:%Y%m%d}&cc=ru"
    try:
        flags = fetch_text(url).strip()
    except SystemExit as exc:
        raise SystemExit(f"Не удалось получить календарь рабочих дней: {exc}") from exc

    expected = (end - start).days + 1
    if len(flags) != expected or any(flag not in "012" for flag in flags):
        raise SystemExit(
            "Календарь рабочих дней вернул неожиданный ответ. "
            f"Ожидалось {expected} отметок, получено {flags[:40]!r}."
        )

    days: list[pd.Timestamp] = []
    for offset, flag in enumerate(flags):
        if flag in "02":
            days.append(start + pd.Timedelta(days=offset))
    return days


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "kronos-moex-test/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Сервис календаря вернул ошибку {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Нет доступа к календарю рабочих дней: {exc.reason}") from exc


def explain_missing_ticker(ticker: str) -> str:
    url = f"https://iss.moex.com/iss/securities/{ticker}.json?iss.meta=off&iss.only=boards"
    try:
        boards = fetch_json(url)["boards"]
    except SystemExit:
        return f"Тикер {ticker} не найден на Московской бирже, либо ISS сейчас недоступен."

    if not boards["data"]:
        return f"Тикер {ticker} не найден на Московской бирже."

    board_ids = {row[boards["columns"].index("boardid")] for row in boards["data"]}
    shown = ", ".join(sorted(board_ids)[:8])
    return (
        f"{ticker} не торгуется в режиме TQBR (акции основного рынка). "
        f"Скрипт принимает только такие акции, например SBER или TATN. "
        f"Режимы этого тикера: {shown}."
    )


def load_predictor():
    if not KRONOS_ROOT.is_dir():
        raise SystemExit(
            "Рядом со скриптом нет папки Kronos. "
            "Код модели ожидается в .\\Kronos (репозиторий shiyu-coder/Kronos)."
        )

    sys.path.insert(0, str(KRONOS_ROOT))
    import torch
    from model import Kronos, KronosPredictor, KronosTokenizer

    if torch.cuda.is_available():
        device = "cuda:0"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Устройство: {device} ({gpu_name})", flush=True)
    else:
        device = "cpu"
        print("CUDA недоступна, прогноз пойдёт на процессоре и будет заметно дольше.", flush=True)

    print("Загрузка Kronos-base и токенизатора. При первом запуске качаются веса, около 425 МБ.", flush=True)
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    tokenizer.eval()
    model.eval()
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    return predictor, torch


def plot_prediction(
    ticker: str,
    kline_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    split_at: pd.Timestamp,
    future_start: pd.Timestamp,
) -> Path:
    """Две панели, как в examples/prediction_example.py: цена закрытия и объём.

    Индекс прогноза длиннее факта на FUTURE дней, поэтому красная линия
    продолжается после конца синей.
    """
    close_df = pd.concat(
        [kline_df["close"].rename("Факт"), pred_df["close"].rename("Прогноз")],
        axis=1,
    )
    volume_df = pd.concat(
        [kline_df["volume"].rename("Факт"), pred_df["volume"].rename("Прогноз")],
        axis=1,
    )
    future_end = pred_df.index[-1]

    fig, (ax_price, ax_volume) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax_price.axvspan(future_start, future_end, color="tab:red", alpha=0.08, zorder=0)
    ax_price.plot(close_df["Факт"], label="Факт", color="tab:blue", linewidth=1.5)
    ax_price.plot(close_df["Прогноз"], label="Прогноз Kronos-base", color="tab:red", linewidth=1.5)
    ax_price.axvline(split_at, color="0.4", linestyle="--", linewidth=1)
    ax_price.set_ylabel("Цена закрытия, руб.")
    ax_price.set_title(
        f"{ticker}, дневные свечи TQBR. Скрыто {HELD_OUT} дней, "
        f"дальше {FUTURE} дней без факта. Контекст {LOOKBACK} дней"
    )
    ax_price.legend(loc="upper left")
    ax_price.grid(True, alpha=0.3)

    ax_volume.axvspan(future_start, future_end, color="tab:red", alpha=0.08, zorder=0)
    ax_volume.plot(volume_df["Факт"], label="Факт", color="tab:blue", linewidth=1.2)
    ax_volume.plot(volume_df["Прогноз"], label="Прогноз Kronos-base", color="tab:red", linewidth=1.2)
    ax_volume.axvline(split_at, color="0.4", linestyle="--", linewidth=1)
    ax_volume.set_ylabel("Объём, шт.")
    ax_volume.set_xlabel("Дата")
    ax_volume.legend(loc="upper left")
    ax_volume.grid(True, alpha=0.3)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    image_path = OUTPUT_DIR / f"{ticker}_kronos_base.png"
    fig.savefig(image_path, dpi=120)
    return image_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Проверка прогноза Kronos-base по акции TQBR.")
    parser.add_argument("ticker", help="Тикер акции, например SBER или TATN")
    return parser.parse_args()


def main() -> None:
    ticker = parse_args().ticker.strip().upper()
    if not ticker.isalnum():
        raise SystemExit(f"Некорректный тикер: {ticker!r}. Ожидается код вроде SBER или TATN.")

    print(f"Загрузка дневных свечей {ticker} с MOEX ISS...", flush=True)
    history = fetch_tqbr_daily(ticker)
    needed = LOOKBACK + HELD_OUT
    if len(history) < needed:
        raise SystemExit(
            f"У {ticker} только {len(history)} дневных свечей с {HISTORY_FROM}, "
            f"а для проверки нужно {needed}: {LOOKBACK} дней контекста и {HELD_OUT} скрытых."
        )

    window = history.iloc[-needed:].reset_index(drop=True)
    context = window.iloc[:LOOKBACK]
    actual = window.iloc[LOOKBACK:]
    future = next_trading_days(actual["timestamps"].iloc[-1], FUTURE)
    forecast_timestamps = pd.concat([actual["timestamps"], future], ignore_index=True)
    split_at = actual["timestamps"].iloc[0]
    future_start = future.iloc[0]

    print(
        f"Контекст: {context['timestamps'].iloc[0]:%Y-%m-%d} — {context['timestamps'].iloc[-1]:%Y-%m-%d} "
        f"({LOOKBACK} дней)."
    )
    print(
        f"Скрытый участок: {actual['timestamps'].iloc[0]:%Y-%m-%d} — {actual['timestamps'].iloc[-1]:%Y-%m-%d} "
        f"({HELD_OUT} дней)."
    )
    print(
        f"Прогноз без факта: {future.iloc[0]:%Y-%m-%d} — {future.iloc[-1]:%Y-%m-%d} "
        f"({FUTURE} дней)."
    )

    predictor, torch = load_predictor()
    torch.manual_seed(42)

    print(f"Прогноз {PRED_LEN} дневных свечей...", flush=True)
    prediction = predictor.predict(
        df=context.loc[:, ["open", "high", "low", "close", "volume", "amount"]],
        x_timestamp=context["timestamps"],
        y_timestamp=forecast_timestamps,
        pred_len=PRED_LEN,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    )
    prediction.index = pd.DatetimeIndex(forecast_timestamps)

    compared = prediction.iloc[:HELD_OUT]
    actual_close = actual["close"].to_numpy()
    predicted_close = compared["close"].to_numpy()
    mae = float(abs(predicted_close - actual_close).mean())
    mape = float((abs(predicted_close - actual_close) / actual_close).mean() * 100)
    print(f"Ошибка закрытия на скрытых {HELD_OUT} днях: MAE {mae:.2f} руб., MAPE {mape:.1f}%.")
    print(
        f"Последний известный день {actual['timestamps'].iloc[-1]:%Y-%m-%d}: "
        f"факт {actual_close[-1]:.2f}, прогноз {predicted_close[-1]:.2f}."
    )
    print(
        f"Прогноз закрытия на {future.iloc[-1]:%Y-%m-%d}: {prediction['close'].iloc[-1]:.2f} руб."
    )

    plotted = window.set_index("timestamps")
    image_path = plot_prediction(ticker, plotted, prediction, split_at, future_start)
    print(f"График сохранён: {image_path}")
    print("Открываю окно. Закройте его, чтобы завершить скрипт.")
    plt.show()


if __name__ == "__main__":
    main()
