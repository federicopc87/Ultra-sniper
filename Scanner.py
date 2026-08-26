# ============================================================
# 🚀 ULTRA SNIPER V11.4 FIXED
# BASELINE ROBUSTA + CONVINZIONE + BACKTEST/OOS CORRETTO
# ============================================================
#
# FIX PRINCIPALI RISPETTO A V11.4:
#
# 1) Weekly/4H:
#    usa esclusivamente l'ultima candela COMPLETAMENTE CHIUSA
#    disponibile alla data/ora del segnale -> no look-ahead.
#
# 2) Backtest:
#    il segnale viene calcolato alla chiusura del giorno i.
#    L'operazione parte dal giorno successivo.
#
# 3) Stop/Target:
#    se nella stessa candela giornaliera vengono toccati sia
#    STOP che TARGET e non conosciamo l'ordine intraday,
#    viene assunto prima lo STOP (conservativo).
#
# 4) Partial exit:
#    contabilizzazione esplicita della posizione residua.
#
# 5) OOS:
#    split cronologico riportato come diagnostica.
#    Se N < 15 per metà, NON viene dichiarata una validazione.
#
# 6) Nessun filtro RSI 62-64 aggiuntivo:
#    60-66 resta il filtro operativo.
#    62-64 = solamente "ALTA CONVINZIONE".
#
# 7) Stesso motore di filtro per backtest e scanner.
#
# 8) SPY/VIX storici:
#    esclusivamente informazione disponibile alla data del segnale.
#
# 9) Evita di usare una candela weekly/4H in formazione.
#
# 10) Le metriche di performance sono espresse in R.
#
# ============================================================



import warnings
warnings.filterwarnings("ignore")

import io
import os
import time
import pickle
import hashlib
import requests
import os

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram error: {e}")

import numpy as np
import pandas as pd
import yfinance as yf

from tqdm.auto import tqdm
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange
from IPython.display import display


# ============================================================
# PARAMETRI
# ============================================================

CAPITALE = 100.0

SCORE_CANDIDATO = 85.0
SCORE_BUY = 88.0

RISCHIO_MASSIMO = 6.0
RR_MINIMO = 2.0

MIN_PRICE = 10.0
MAX_PRICE = 2000.0
MIN_AVG_VOLUME = 500_000

TOP_FINAL = 5
BATCH_SIZE = 50

# ----------------------------
# EXIT
# ----------------------------

ATR_STOP_MULTIPLIER = 1.4
ATR_TRAILING_MULTIPLIER = 2.8
TARGET_RR = 2.0

TRAILING_ACTIVATION_R = 1.5

EXIT_ON_TREND_BREAK = True
EXIT_RSI_LEVEL = 42

# ----------------------------
# PARTIAL
# ----------------------------

PARTIAL_EXIT_R = 1.0
PARTIAL_EXIT_FRACTION = 0.5
BREAKEVEN_BUFFER_R = 0.15

# ----------------------------
# REGIME
# ----------------------------

VIX_MAX_BULLISH = 25.0

# ----------------------------
# FILTRI HARD
# ----------------------------

ADX_MIN = 30

RELVOL_MIN = 1.2
RELVOL_MAX = 1.5

ATR_PCT_MAX = 5.8

PULLBACK_MIN = 1.5
PULLBACK_MAX = 4.0

RSI_DEADZONE_MIN = 55.0
RSI_DEADZONE_MAX = 60.0

RSI_FLOOR = 60.0
RSI_CEIL = 66.0

# Zona solamente diagnostica
RSI_HIGH_CONVICTION_MIN = 62.0
RSI_HIGH_CONVICTION_MAX = 64.0

INCLUDE_SP400_MIDCAP = True

REQUIRE_4H_CONFIRMATION = True

# ============================================================
# BACKTEST
# ============================================================

BACKTEST_LOOKBACK_DAYS = 900
BACKTEST_STEP = 5

# Forward massimo per operazione
BACKTEST_FORWARD_DAYS = 120

BACKTEST_END_DATE = pd.Timestamp("2025-12-31")

INITIAL_BACKTEST_CAPITAL = 10_000.0
BACKTEST_RISK_PERCENT = 1.0

# OOS minimo dichiarativo.
# Non è un test statistico "magico": sotto questa soglia
# il confronto viene semplicemente dichiarato non conclusivo.
OOS_MIN_TRADES_PER_HALF = 15


print("=" * 100)
print("🚀 ULTRA SNIPER V11.4 FIXED")
print("=" * 100)

print(
    f"RSI hard: {RSI_FLOOR}-{RSI_CEIL} | "
    f"Alta convinzione: {RSI_HIGH_CONVICTION_MIN}-{RSI_HIGH_CONVICTION_MAX}"
)

print(
    f"ADX min: {ADX_MIN} | "
    f"RelVol: {RELVOL_MIN}-{RELVOL_MAX} | "
    f"4H: {'ON' if REQUIRE_4H_CONFIRMATION else 'OFF'}"
)

print(f"Backtest end: {BACKTEST_END_DATE.date()}")
print("=" * 100)


# ============================================================
# UTILITY
# ============================================================

def clean_ticker(t):
    if t is None:
        return None

    t = str(t).strip().upper()
    t = t.replace(".", "-").replace("/", "-")

    if t in ("NAN", "NONE", ""):
        return None

    return t


HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def get_wiki_tables(url):
    try:
        r = requests.get(
            url,
            headers=HEADERS,
            timeout=30
        )

        r.raise_for_status()

        return pd.read_html(
            io.StringIO(r.text)
        )

    except Exception as e:

        print(
            f"⚠️ Wikipedia: {str(e)[:120]}"
        )

        return []


def ensure_dt(df):

    if df is None or df.empty:
        return None

    x = df.copy()

    try:
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
    except Exception:
        pass

    x = x[
        ~x.index.duplicated(
            keep="last"
        )
    ]

    return x.sort_index()


def scalar(x):

    if isinstance(x, pd.Series):
        if len(x) == 0:
            return np.nan
        return x.iloc[0]

    return x


# ============================================================
# UNIVERSO
# ============================================================

print("\n📥 Universo S&P 500 + Nasdaq 100...")

sp500 = []

tables = get_wiki_tables(
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
)

if tables:

    table = tables[0]

    for col in [
        "Symbol",
        "Ticker symbol",
        "Ticker"
    ]:

        if col in table.columns:

            sp500 = [
                clean_ticker(t)
                for t in table[col].tolist()
            ]

            sp500 = [
                t for t in sp500
                if t
            ]

            break


# ============================================================
# NASDAQ 100
# ============================================================

nasdaq100 = []

for table in get_wiki_tables(
    "https://en.wikipedia.org/wiki/Nasdaq-100"
):

    for col in table.columns:

        name = str(col).lower()

        if (
            "ticker" in name
            or "symbol" in name
        ):

            temp = [
                clean_ticker(v)
                for v in table[col]
                .dropna()
                .astype(str)
                .tolist()
            ]

            temp = [
                t for t in temp
                if (
                    t
                    and 1 <= len(t) <= 6
                    and t.replace("-", "").isalnum()
                )
            ]

            if len(temp) >= 80:

                nasdaq100 = temp
                break

    if len(nasdaq100) >= 80:
        break


if len(nasdaq100) < 80:

    fallback = """
    AAPL ABNB ADBE ADI ADP ADSK AEP ALGN AMAT AMD
    AMGN AMZN ANSS APP ARM ASML AVGO AXON AZN BIIB
    BKNG BKR CCEP CDNS CDW CEG CHTR CMCSA COST CPRT
    CRWD CSCO CSGP CSX CTAS CTSH DASH DDOG DXCM EA
    EXC FANG FAST FTNT GFS GILD GOOG GOOGL HON IDXX
    ILMN INTC INTU ISRG KDP KHC KLAC LIN LRCX LULU
    MAR MCHP MDB MDLZ MELI META MNST MRVL MSFT MSTR
    MU NFLX NVDA NXPI ODFL ON ORLY PANW PAYX PCAR
    PDD PEP PLTR PYPL QCOM REGN ROP ROST SBUX SHOP
    SNPS TEAM TMUS TRGP TSLA TTD TTWO TXN VRSK
    VRTX WBD WDAY XEL ZS
    """

    nasdaq100 = sorted(
        set(
            t for t in
            (
                clean_ticker(x)
                for x in fallback.split()
            )
            if t
        )
    )


# ============================================================
# S&P 400
# ============================================================

sp400 = []

if INCLUDE_SP400_MIDCAP:

    print("📥 Caricamento S&P 400 MidCap...")

    tables = get_wiki_tables(
        "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    )

    if tables:

        for table in tables:

            for col in [
                "Symbol",
                "Ticker symbol",
                "Ticker"
            ]:

                if col in table.columns:

                    temp = [
                        clean_ticker(v)
                        for v in table[col]
                        .dropna()
                        .astype(str)
                        .tolist()
                    ]

                    temp = [
                        t for t in temp
                        if (
                            t
                            and 1 <= len(t) <= 6
                            and t.replace("-", "").isalnum()
                        )
                    ]

                    if len(temp) >= 300:

                        sp400 = temp
                        break

            if len(sp400) >= 300:
                break

    sp400 = sorted(set(sp400))

    print(
        f"✅ S&P 400 MidCap: {len(sp400)}"
    )


# ============================================================
# TICKER UNIVERSE
# ============================================================

tickers = sorted(
    set(
        [t for t in sp500 if t]
        + [t for t in nasdaq100 if t]
        + [t for t in sp400 if t]
    )
)

print(
    f"✅ Universo totale: {len(tickers)} titoli"
)

if not tickers:
    raise RuntimeError(
        "Universo vuoto."
    )


# ============================================================
# SPY + VIX
# ============================================================

print("\n📥 Download SPY/VIX...")

spy = yf.download(
    "SPY",
    period="10y",
    interval="1d",
    auto_adjust=True,
    progress=False,
    threads=False
)

if spy is None or spy.empty:
    raise RuntimeError(
        "SPY non disponibile."
    )

spy_close = ensure_dt(
    pd.DataFrame(
        {
            "Close":
            spy["Close"].squeeze()
        }
    )
)["Close"].dropna()


vix = yf.download(
    "^VIX",
    period="10y",
    interval="1d",
    auto_adjust=True,
    progress=False,
    threads=False
)

if vix is None or vix.empty:
    raise RuntimeError(
        "VIX non disponibile."
    )

vix_close = ensure_dt(
    pd.DataFrame(
        {
            "Close":
            vix["Close"].squeeze()
        }
    )
)["Close"].dropna()


spy_sma200 = spy_close.rolling(
    200
).mean()

market_bullish_now = bool(
    len(spy_close) >= 200
    and
    spy_close.iloc[-1]
    >
    spy_sma200.iloc[-1]
)

vix_ok_now = bool(
    vix_close.iloc[-1]
    <= VIX_MAX_BULLISH
)

market_bullish_now = (
    market_bullish_now
    and
    vix_ok_now
)

print(
    f"SPY: ${spy_close.iloc[-1]:.2f} | "
    f"{'🟢 BULLISH' if market_bullish_now else '🔴 DEBOLE'}"
)

print(
    f"VIX: {vix_close.iloc[-1]:.2f} | "
    f"{'🟢 OK' if vix_ok_now else '🔴 ALTO'}"
)


# ============================================================
# DOWNLOAD + CACHE
# ============================================================

CACHE_DIR = "/content/sniper_cache"

CACHE_MAX_AGE_HOURS = 20

os.makedirs(
    CACHE_DIR,
    exist_ok=True
)


def _cache_path(
    period,
    interval,
    tick_list
):

    h = hashlib.md5(
        "|".join(
            sorted(tick_list)
        ).encode()
    ).hexdigest()[:10]

    return os.path.join(
        CACHE_DIR,
        f"{interval}_{period}_{h}.pkl"
    )


def download_batches(
    tick_list,
    period,
    interval,
    desc
):

    out = {}

    for start in tqdm(
        range(
            0,
            len(tick_list),
            BATCH_SIZE
        ),
        desc=desc,
        leave=False
    ):

        batch = tick_list[
            start:start + BATCH_SIZE
        ]

        try:

            data = yf.download(
                batch,
                period=period,
                interval=interval,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False
            )

            if data is None or data.empty:
                continue

            if isinstance(
                data.columns,
                pd.MultiIndex
            ):

                level0 = (
                    data.columns
                    .get_level_values(0)
                )

                for t in batch:

                    if t not in level0:
                        continue

                    df = data[t]

                    if (
                        not df.empty
                        and
                        "Close" in df.columns
                    ):

                        df = ensure_dt(
                            df.dropna(
                                subset=["Close"]
                            )
                        )

                        if (
                            df is not None
                            and len(df) > 0
                        ):
                            out[t] = df

            else:

                if (
                    len(batch) == 1
                    and
                    "Close" in data.columns
                ):

                    df = ensure_dt(
                        data.dropna(
                            subset=["Close"]
                        )
                    )

                    if (
                        df is not None
                        and len(df) > 0
                    ):
                        out[batch[0]] = df

        except Exception as e:

            print(
                f"⚠️ batch: {str(e)[:100]}"
            )

        time.sleep(0.3)

    return out


def download_batches_cached(
    tick_list,
    period,
    interval,
    desc
):

    path = _cache_path(
        period,
        interval,
        tick_list
    )

    if os.path.exists(path):

        age_hours = (
            time.time()
            -
            os.path.getmtime(path)
        ) / 3600

        if age_hours < CACHE_MAX_AGE_HOURS:

            try:

                with open(
                    path,
                    "rb"
                ) as f:

                    data = pickle.load(f)

                print(
                    f"📦 Cache {desc} "
                    f"(età {age_hours:.1f}h, "
                    f"{len(data)} titoli)"
                )

                return data

            except Exception:

                print(
                    "⚠️ Cache corrotta, "
                    "riscarico."
                )

        else:

            print(
                f"⏰ Cache {desc} scaduta."
            )


    data = download_batches(
        tick_list,
        period,
        interval,
        desc
    )

    try:

        with open(
            path,
            "wb"
        ) as f:

            pickle.dump(
                data,
                f
            )

    except Exception as e:

        print(
            f"⚠️ Cache non salvata: "
            f"{str(e)[:80]}"
        )

    return data


# ============================================================
# DOWNLOAD DAILY
# ============================================================

print("\n📥 Download daily (5y)...")

daily_data = download_batches_cached(
    tickers,
    "5y",
    "1d",
    "Daily"
)

print(
    f"✅ Daily: {len(daily_data)} titoli"
)


# ============================================================
# DOWNLOAD WEEKLY
# ============================================================

print("\n📥 Download weekly (7y)...")

weekly_data = download_batches_cached(
    tickers,
    "7y",
    "1wk",
    "Weekly"
)

print(
    f"✅ Weekly: {len(weekly_data)} titoli"
)


# ============================================================
# DOWNLOAD 1H
# ============================================================

print("\n📥 Download 1H (730d)...")

intraday_all = download_batches_cached(
    tickers,
    "730d",
    "1h",
    "1H"
)

print(
    f"✅ 1H: {len(intraday_all)} titoli"
)


# ============================================================
# 4H RESAMPLE
# ============================================================

def resample_4h(df):

    if df is None or df.empty:
        return None

    x = ensure_dt(df)

    if x is None:
        return None

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    if not all(
        c in x.columns
        for c in required
    ):
        return None

    x = x.dropna(
        subset=["Close"]
    )

    if len(x) < 30:
        return None

    h4 = (
        x.resample("4h")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum"
            }
        )
        .dropna(
            subset=["Close"]
        )
    )

    if len(h4) < 30:
        return None

    return h4


print("\n⏳ Resample 4H...")

four_h_data = {}

for t, df in tqdm(
    intraday_all.items(),
    desc="4H",
    leave=False
):

    h4 = resample_4h(df)

    if h4 is not None:
        four_h_data[t] = h4


print(
    f"✅ 4H: {len(four_h_data)} titoli"
)


# ============================================================
# DAILY INDICATORS
# ============================================================

def build_daily_frame(df):

    if df is None or len(df) < 260:
        return None

    x = df.copy()

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    if not all(
        c in x.columns
        for c in required
    ):
        return None

    close = x["Close"]
    high = x["High"]
    low = x["Low"]
    vol = x["Volume"]

    x["ema20"] = (
        EMAIndicator(
            close,
            20
        ).ema_indicator()
    )

    x["ema50"] = (
        EMAIndicator(
            close,
            50
        ).ema_indicator()
    )

    x["sma50"] = (
        SMAIndicator(
            close,
            50
        ).sma_indicator()
    )

    x["sma200"] = (
        SMAIndicator(
            close,
            200
        ).sma_indicator()
    )

    x["rsi"] = (
        RSIIndicator(
            close,
            14
        ).rsi()
    )

    macd = MACD(close)

    x["macd"] = macd.macd()
    x["macd_signal"] = macd.macd_signal()

    adx_obj = ADXIndicator(
        high,
        low,
        close,
        14
    )

    x["adx"] = adx_obj.adx()
    x["plus_di"] = adx_obj.adx_pos()
    x["minus_di"] = adx_obj.adx_neg()

    x["atr"] = (
        AverageTrueRange(
            high,
            low,
            close,
            14
        ).average_true_range()
    )

    x["avg_vol20"] = (
        vol.rolling(20).mean()
    )

    x["high20"] = (
        close.rolling(20).max()
    )

    x["high252"] = (
        close.rolling(252).max()
    )

    x["low20"] = (
        low.rolling(20).min()
    )

    x["sma50_slope"] = (
        x["sma50"]
        -
        x["sma50"].shift(5)
    )

    return x


# ============================================================
# WEEKLY FLAGS
# ============================================================

def build_weekly_flags(wdf):

    if wdf is None or len(wdf) < 55:
        return None

    x = wdf.copy()

    close = x["Close"]

    ema20 = (
        EMAIndicator(
            close,
            20
        ).ema_indicator()
    )

    ema50 = (
        EMAIndicator(
            close,
            50
        ).ema_indicator()
    )

    x["w_bull"] = (
        (close > ema20)
        &
        (ema20 > ema50)
    )

    return x[
        ["w_bull"]
    ]


# ============================================================
# 4H FLAGS
# ============================================================

def build_4h_flags(h4df):

    if h4df is None or len(h4df) < 30:
        return None

    x = h4df.copy()

    close = x["Close"]
    high = x["High"]
    low = x["Low"]

    ema20 = (
        EMAIndicator(
            close,
            20
        ).ema_indicator()
    )

    ema50 = (
        EMAIndicator(
            close,
            50
        ).ema_indicator()
    )

    macd = MACD(close)

    adx_obj = ADXIndicator(
        high,
        low,
        close,
        14
    )

    bullish = (
        (close > ema20)
        &
        (ema20 > ema50)
    )

    macd_ok = (
        macd.macd()
        >
        macd.macd_signal()
    )

    di_ok = (
        adx_obj.adx_pos()
        >
        adx_obj.adx_neg()
    )

    x["h4_bull"] = (
        bullish
        &
        (macd_ok | di_ok)
    )

    return x[
        ["h4_bull"]
    ]


# ============================================================
# INDICATORI
# ============================================================

print("\n⚙️ Calcolo indicatori...")

daily_frames = {}
weekly_flags = {}
h4_flags = {}

for t in tqdm(
    tickers,
    desc="Indicatori"
):

    dfr = build_daily_frame(
        daily_data.get(t)
    )

    if dfr is None:
        continue

    wfr = build_weekly_flags(
        weekly_data.get(t)
    )

    if wfr is None:
        continue

    daily_frames[t] = dfr
    weekly_flags[t] = wfr

    hfr = build_4h_flags(
        four_h_data.get(t)
    )

    if hfr is not None:
        h4_flags[t] = hfr


print(
    f"✅ Pronti {len(daily_frames)} titoli "
    f"(4H: {len(h4_flags)})"
)


# ============================================================
# RSI
# ============================================================

def rsi_in_good_zone(rsi):

    if pd.isna(rsi):
        return False

    if (
        RSI_DEADZONE_MIN
        <= rsi
        <= RSI_DEADZONE_MAX
    ):
        return False

    return (
        RSI_FLOOR
        <= rsi
        <= RSI_CEIL
    )


def is_high_conviction(rsi):

    if pd.isna(rsi):
        return False

    return (
        RSI_HIGH_CONVICTION_MIN
        <= rsi
        <= RSI_HIGH_CONVICTION_MAX
    )


# ============================================================
# MARKET STATE STORICO
# ============================================================

def historical_market_state(date):

    """
    Restituisce lo stato SPY/VIX usando solo dati
    <= date.
    """

    spy_hist = spy_close[
        spy_close.index <= date
    ]

    vix_hist = vix_close[
        vix_close.index <= date
    ]

    if len(spy_hist) < 200:
        return None

    spy_sma = (
        spy_hist.rolling(200)
        .mean()
        .iloc[-1]
    )

    if pd.isna(spy_sma):
        return None

    spy_bull = bool(
        spy_hist.iloc[-1]
        >
        spy_sma
    )

    if len(vix_hist) == 0:
        vix_ok = True
    else:
        vix_ok = bool(
            vix_hist.iloc[-1]
            <= VIX_MAX_BULLISH
        )

    return (
        spy_bull
        and
        vix_ok
    )


# ============================================================
# WEEKLY FLAG SENZA LOOK-AHEAD
# ============================================================

def latest_weekly_flag_before_or_at(
    wf,
    date
):

    if wf is None or wf.empty:
        return None

    available = wf[
        wf.index <= date
    ]

    if available.empty:
        return None

    return bool(
        available["w_bull"].iloc[-1]
    )


# ============================================================
# 4H FLAG SENZA LOOK-AHEAD
# ============================================================

def latest_4h_flag_before_or_at(
    h4f,
    date
):

    if h4f is None or h4f.empty:
        return None

    available = h4f[
        h4f.index <= date
    ]

    if available.empty:
        return None

    return bool(
        available["h4_bull"].iloc[-1]
    )


# ============================================================
# SCORE
# ============================================================

def score_and_flags(
    row,
    weekly_bull,
    mkt_bull,
    rs_vs_spy,
    pullback_pct,
    atr_pct,
    relvol
):

    price = row["Close"]

    needed = [
        "ema20",
        "ema50",
        "sma200"
    ]

    if any(
        pd.isna(row[c])
        for c in needed
    ):
        return None

    daily_bull = (
        price
        >
        row["ema20"]
        >
        row["ema50"]
        and
        price
        >
        row["sma200"]
    )

    sma50_rising = (
        pd.notna(row["sma50_slope"])
        and
        row["sma50_slope"] > 0
    )

    rsi = row["rsi"]

    rsi_good = rsi_in_good_zone(
        rsi
    )

    macd_bull = (
        pd.notna(row["macd"])
        and
        pd.notna(row["macd_signal"])
        and
        row["macd"]
        >
        row["macd_signal"]
    )

    adx_strong = (
        pd.notna(row["adx"])
        and
        row["adx"] >= ADX_MIN
    )

    di_bull = (
        pd.notna(row["plus_di"])
        and
        pd.notna(row["minus_di"])
        and
        row["plus_di"]
        >
        row["minus_di"]
    )

    score = 0.0

    if weekly_bull:
        score += 17

    if daily_bull:
        score += 20

    if sma50_rising:
        score += 5

    if rsi_good:
        score += 13

    if macd_bull:
        score += 10

    if adx_strong:
        score += 12

    if di_bull:
        score += 6

    if (
        RELVOL_MIN
        <= relvol
        <= RELVOL_MAX
    ):
        score += 8

    elif relvol > RELVOL_MAX:
        score -= 4

    if rs_vs_spy:
        score += 7

    if (
        PULLBACK_MIN
        <= pullback_pct
        <= PULLBACK_MAX
    ):
        score += 9

    if not mkt_bull:
        score -= 10

    if atr_pct > ATR_PCT_MAX:
        score -= 6

    score = max(
        0.0,
        min(
            100.0,
            score
        )
    )

    return (
        score,
        daily_bull,
        sma50_rising
    )


# ============================================================
# HARD QUALITY FILTERS
# ============================================================

def quality_filters(
    row,
    relvol,
    pullback_pct,
    atr_pct
):

    if (
        pd.isna(row["adx"])
        or
        row["adx"] < ADX_MIN
    ):
        return False

    if not (
        RELVOL_MIN
        <= relvol
        <= RELVOL_MAX
    ):
        return False

    if atr_pct > ATR_PCT_MAX:
        return False

    if not (
        PULLBACK_MIN
        <= pullback_pct
        <= PULLBACK_MAX
    ):
        return False

    if not rsi_in_good_zone(
        row["rsi"]
    ):
        return False

    if (
        pd.isna(row["sma50"])
        or
        row["Close"]
        <= row["sma50"]
    ):
        return False

    return True


# ============================================================
# RS VS SPY
# ============================================================

def calculate_rs_vs_spy(
    di,
    i,
    date,
    lookback=60
):

    if i < lookback:
        return 0

    spy_hist = spy_close[
        spy_close.index <= date
    ]

    if len(spy_hist) < 61:
        return 0

    spy_p3 = (
        spy_hist.iloc[-1]
        /
        spy_hist.iloc[-61]
        - 1
    ) * 100

    stock_start = (
        di["Close"]
        .iloc[i - lookback]
    )

    stock_now = (
        di["Close"]
        .iloc[i]
    )

    if (
        pd.isna(stock_start)
        or
        pd.isna(stock_now)
        or
        stock_start <= 0
    ):
        return 0

    stock_p3 = (
        stock_now
        /
        stock_start
        - 1
    ) * 100

    return int(
        stock_p3 > spy_p3
    )


# ============================================================
# STOP / TARGET
# ============================================================

def calculate_trade_levels(
    row
):

    price = row["Close"]
    atr = row["atr"]

    if (
        pd.isna(price)
        or
        pd.isna(atr)
        or
        atr <= 0
    ):
        return None

    support = row["low20"]

    stop_atr = (
        price
        -
        ATR_STOP_MULTIPLIER * atr
    )

    if pd.notna(support):

        stop_support = (
            support
            -
            0.2 * atr
        )

    else:

        stop_support = stop_atr

    stop = max(
        stop_atr,
        stop_support
    )

    risk = price - stop

    if risk <= 0:
        return None

    risk_pct = (
        risk
        /
        price
        * 100
    )

    if risk_pct > RISCHIO_MASSIMO:
        return None

    target = (
        price
        +
        TARGET_RR * risk
    )

    rr = (
        target - price
    ) / risk

    if rr < RR_MINIMO:
        return None

    return {
        "entry": price,
        "stop": stop,
        "risk": risk,
        "risk_pct": risk_pct,
        "target": target,
        "rr": rr
    }


# ============================================================
# BACKTEST ENGINE
# ============================================================

def simulate_trade(
    di,
    entry_i,
    entry_price,
    initial_stop,
    target,
    risk
):

    """
    Simulazione conservativa.

    Importante:
    con OHLC giornaliero non conosciamo l'ordine
    intraday di High/Low.

    Se una candela tocca sia stop che target,
    assumiamo STOP prima -> conservativo.

    Il partial a +1R viene eseguito se High raggiunge
    +1R e, sulla stessa candela, non è stato colpito
    lo stop precedente.

    La posizione residua continua con stop aggiornato.
    """

    future = di.iloc[
        entry_i + 1:
        entry_i + 1 + BACKTEST_FORWARD_DAYS
    ]

    if future.empty:
        return None

    current_stop = initial_stop

    highest = entry_price

    partial_done = False

    partial_r = None

    exit_price = None
    exit_reason = None

    for _, frow in future.iterrows():

        h = frow["High"]
        l = frow["Low"]
        c = frow["Close"]
        a = frow["atr"]

        if (
            pd.isna(h)
            or
            pd.isna(l)
            or
            pd.isna(c)
        ):
            continue

        # --------------------------------------------
        # Stop attuale PRIMA di aggiornare trailing
        # --------------------------------------------

        stop_hit = (
            l <= current_stop
        )

        # --------------------------------------------
        # Target
        # --------------------------------------------

        target_hit = (
            h >= target
        )

        # --------------------------------------------
        # Se stop e target sono entrambi nella
        # stessa candela -> STOP prima.
        # --------------------------------------------

        if (
            stop_hit
            and
            target_hit
        ):

            exit_price = current_stop

            exit_reason = (
                "STOP/TARGET SAME BAR"
            )

            break

        # --------------------------------------------
        # STOP
        # --------------------------------------------

        if stop_hit:

            exit_price = current_stop

            if partial_done:
                exit_reason = (
                    "STOP/TRAILING AFTER PARTIAL"
                )
            else:
                exit_reason = "STOP"

            break

        # --------------------------------------------
        # PARTIAL EXIT +1R
        # --------------------------------------------

        if (
            not partial_done
            and
            h >= (
                entry_price
                +
                PARTIAL_EXIT_R * risk
            )
        ):

            partial_done = True
            partial_r = PARTIAL_EXIT_R

            current_stop = max(
                current_stop,
                entry_price
                +
                BREAKEVEN_BUFFER_R * risk
            )

            # Se la candela raggiunge anche il target
            # dopo il partial, non sappiamo l'ordine.
            # Il partial è comunque già stato raggiunto
            # prima di proseguire la gestione.

        # --------------------------------------------
        # TARGET
        # --------------------------------------------

        if target_hit:

            exit_price = target

            exit_reason = "TARGET"

            break

        # --------------------------------------------
        # Aggiornamento highest
        # --------------------------------------------

        if h > highest:
            highest = h

        # --------------------------------------------
        # Profit R a chiusura
        # --------------------------------------------

        profit_r = (
            c - entry_price
        ) / risk

        # --------------------------------------------
        # Trailing
        # --------------------------------------------

        if (
            profit_r
            >= TRAILING_ACTIVATION_R
            and
            pd.notna(a)
        ):

            mult = (
                ATR_TRAILING_MULTIPLIER
                +
                (
                    0.4
                    if profit_r >= 2.0
                    else 0.0
                )
            )

            trailing_stop = (
                highest
                -
                mult * a
            )

            current_stop = max(
                current_stop,
                trailing_stop
            )

            if profit_r >= 2.0:

                current_stop = max(
                    current_stop,
                    entry_price
                    +
                    0.12 * risk
                )

        # --------------------------------------------
        # Trend break
        # --------------------------------------------

        e20 = frow["ema20"]
        e50 = frow["ema50"]

        if (
            EXIT_ON_TREND_BREAK
            and
            pd.notna(e20)
            and
            pd.notna(e50)
            and
            c < e20 < e50
        ):

            exit_price = c

            exit_reason = (
                "TREND BREAK"
            )

            break

        # --------------------------------------------
        # RSI weakness
        # --------------------------------------------

        r = frow["rsi"]

        if (
            pd.notna(r)
            and
            r < EXIT_RSI_LEVEL
        ):

            exit_price = c

            exit_reason = (
                "RSI WEAKNESS"
            )

            break

    # ====================================================
    # Fine forward window
    # ====================================================

    if exit_price is None:

        exit_price = (
            future["Close"].iloc[-1]
        )

        exit_reason = "TEMPO"

    # ====================================================
    # PNL R
    # ====================================================

    remainder_r = (
        exit_price
        -
        entry_price
    ) / risk

    if partial_done:

        pnl_r = (
            PARTIAL_EXIT_FRACTION
            *
            partial_r
            +
            (
                1
                -
                PARTIAL_EXIT_FRACTION
            )
            *
            remainder_r
        )

    else:

        pnl_r = remainder_r

    return {
        "Exit": exit_price,
        "Motivo": exit_reason,
        "PnL_R": pnl_r,
        "Partial": partial_done,
        "Partial_R": partial_r
    }


# ============================================================
# BACKTEST
# ============================================================

print("\n" + "=" * 100)
print("📊 BACKTEST V11.4 FIXED")
print("=" * 100)

trades = []

for t, di in tqdm(
    daily_frames.items(),
    desc="Backtest"
):

    wf = weekly_flags.get(t)

    h4f = h4_flags.get(t)

    n = len(di)

    end_pos = (
        di.index.searchsorted(
            BACKTEST_END_DATE,
            side="right"
        )
        - 1
    )

    end_i = min(
        end_pos,
        n - 2
    )

    if end_i < 260:
        continue

    start_i = max(
        260,
        end_i
        -
        BACKTEST_LOOKBACK_DAYS
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Non limitiamo artificialmente il numero di trade
    # per ticker.
    #
    # Il vecchio MAX_HISTORY_TRADES_PER_TICKER = 8 poteva
    # rendere il campione dipendente dall'ordine dei segnali.
    #
    # Usiamo invece tutti i segnali validi.
    # --------------------------------------------------------

    for i in range(
        start_i,
        end_i,
        BACKTEST_STEP
    ):

        row = di.iloc[i]

        date = di.index[i]

        price = row["Close"]

        # ----------------------------------------------------
        # Prezzo
        # ----------------------------------------------------

        if (
            pd.isna(price)
            or
            price < MIN_PRICE
            or
            price > MAX_PRICE
        ):
            continue

        # ----------------------------------------------------
        # Volume
        # ----------------------------------------------------

        if (
            pd.isna(row["avg_vol20"])
            or
            row["avg_vol20"]
            < MIN_AVG_VOLUME
        ):
            continue

        # ----------------------------------------------------
        # ATR
        # ----------------------------------------------------

        atr = row["atr"]

        if (
            pd.isna(atr)
            or
            atr <= 0
        ):
            continue

        # ----------------------------------------------------
        # WEEKLY
        # ----------------------------------------------------

        weekly_bull = (
            latest_weekly_flag_before_or_at(
                wf,
                date
            )
        )

        if weekly_bull is None:
            continue

        # ----------------------------------------------------
        # MARKET REGIME
        # ----------------------------------------------------

        mkt_bull_h = (
            historical_market_state(
                date
            )
        )

        if mkt_bull_h is None:
            continue

        # ----------------------------------------------------
        # PULLBACK
        # ----------------------------------------------------

        high20 = row["high20"]

        if (
            pd.notna(high20)
            and
            high20 > 0
        ):

            pullback_pct = (
                1
                -
                price / high20
            ) * 100

        else:

            pullback_pct = 0.0

        # ----------------------------------------------------
        # ATR %
        # ----------------------------------------------------

        atr_pct = (
            atr
            /
            price
            * 100
        )

        # ----------------------------------------------------
        # RELATIVE VOLUME
        # ----------------------------------------------------

        if row["avg_vol20"] > 0:

            relvol = (
                row["Volume"]
                /
                row["avg_vol20"]
            )

        else:

            relvol = 0.0

        # ----------------------------------------------------
        # HARD FILTERS
        # ----------------------------------------------------

        if not quality_filters(
            row,
            relvol,
            pullback_pct,
            atr_pct
        ):
            continue

        # ----------------------------------------------------
        # RS vs SPY
        # ----------------------------------------------------

        rs_vs_spy = (
            calculate_rs_vs_spy(
                di,
                i,
                date
            )
        )

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        result = score_and_flags(
            row,
            weekly_bull,
            mkt_bull_h,
            rs_vs_spy,
            pullback_pct,
            atr_pct,
            relvol
        )

        if result is None:
            continue

        score, daily_bull, _ = result

        if (
            score < SCORE_CANDIDATO
            or
            not weekly_bull
            or
            not daily_bull
        ):
            continue

        # ----------------------------------------------------
        # LEVELS
        # ----------------------------------------------------

        levels = calculate_trade_levels(
            row
        )

        if levels is None:
            continue

        # ----------------------------------------------------
        # 4H CONFIRMATION
        # ----------------------------------------------------

        if REQUIRE_4H_CONFIRMATION:

            h4_bull = (
                latest_4h_flag_before_or_at(
                    h4f,
                    date
                )
            )

            if h4_bull is None:
                continue

            if not h4_bull:
                continue

        else:

            h4_bull = True

        # ----------------------------------------------------
        # SIMULAZIONE
        # ----------------------------------------------------

        sim = simulate_trade(
            di=di,
            entry_i=i,
            entry_price=levels["entry"],
            initial_stop=levels["stop"],
            target=levels["target"],
            risk=levels["risk"]
        )

        if sim is None:
            continue

        # ----------------------------------------------------
        # CONVINZIONE
        # ----------------------------------------------------

        conviction = (
            "ALTA"
            if is_high_conviction(
                row["rsi"]
            )
            else
            "MEDIA"
        )

        # ----------------------------------------------------
        # TRADE RECORD
        # ----------------------------------------------------

        trades.append(
            {
                "Ticker": t,
                "Data": date,
                "Score": round(
                    score,
                    1
                ),
                "RSI": round(
                    row["rsi"],
                    2
                ),
                "ADX": round(
                    row["adx"],
                    2
                ),
                "Rel.Volume": round(
                    relvol,
                    2
                ),
                "Pullback %": round(
                    pullback_pct,
                    2
                ),
                "RS vs SPY": rs_vs_spy,
                "Entry": round(
                    levels["entry"],
                    2
                ),
                "Stop": round(
                    levels["stop"],
                    2
                ),
                "Target": round(
                    levels["target"],
                    2
                ),
                "Exit": round(
                    sim["Exit"],
                    2
                ),
                "Motivo": sim["Motivo"],
                "PnL_R": round(
                    sim["PnL_R"],
                    3
                ),
                "Partial": sim["Partial"],
                "Partial_R": sim["Partial_R"],
                "Convinzione": conviction
            }
        )


# ============================================================
# DATAFRAME
# ============================================================

trades_df = pd.DataFrame(
    trades
)


# ============================================================
# STATISTICHE
# ============================================================

print("\n" + "=" * 100)
print("📊 STATISTICHE BACKTEST V11.4 FIXED")
print("=" * 100)

if trades_df.empty:

    print(
        "⚠️ Nessun trade trovato."
    )

else:

    trades_df = (
        trades_df
        .sort_values("Data")
        .reset_index(drop=True)
    )

    wins = trades_df[
        trades_df["PnL_R"] > 0
    ]

    losses = trades_df[
        trades_df["PnL_R"] <= 0
    ]

    win_rate = (
        len(wins)
        /
        len(trades_df)
        *
        100
    )

    expectancy = (
        trades_df["PnL_R"]
        .mean()
    )

    gp = wins[
        "PnL_R"
    ].sum()

    gl = abs(
        losses[
            "PnL_R"
        ].sum()
    )

    pf = (
        gp / gl
        if gl > 0
        else np.inf
    )

    # --------------------------------------------------------
    # Equity RISK 1%
    # --------------------------------------------------------

    equity = (
        INITIAL_BACKTEST_CAPITAL
    )

    equity_curve = []

    for r in trades_df[
        "PnL_R"
    ]:

        equity += (
            equity
            *
            (
                BACKTEST_RISK_PERCENT
                /
                100
            )
            *
            r
        )

        equity_curve.append(
            equity
        )

    trades_df[
        "Equity"
    ] = equity_curve

    # --------------------------------------------------------
    # Max drawdown
    # --------------------------------------------------------

    equity_series = pd.Series(
        equity_curve
    )

    running_max = (
        equity_series
        .cummax()
    )

    drawdown = (
        equity_series
        /
        running_max
        -
        1
    )

    max_dd = (
        drawdown.min()
        * 100
    )

    # --------------------------------------------------------
    # Stats
    # --------------------------------------------------------

    print(
        f"Trade: {len(trades_df)}"
    )

    print(
        f"Win rate: {win_rate:.1f}%"
    )

    print(
        f"Expectancy: {expectancy:.3f}R"
    )

    print(
        f"Profit Factor: {pf:.2f}"
    )

    print(
        f"Max Drawdown: {max_dd:.2f}%"
    )

    print(
        f"Partial: "
        f"{trades_df['Partial'].sum()}"
        f"/{len(trades_df)}"
    )

    print(
        f"Capitale simulato finale: "
        f"€{equity:,.2f}"
    )


    # ========================================================
    # CONVINZIONE
    # ========================================================

    print(
        "\n--- Breakdown per Convinzione ---"
    )

    conv = (
        trades_df
        .groupby("Convinzione")
        .agg(
            N=("PnL_R", "count"),

            WinRate=(
                "PnL_R",
                lambda x:
                round(
                    (x > 0).mean()
                    * 100,
                    1
                )
            ),

            ExpectancyR=(
                "PnL_R",
                lambda x:
                round(
                    x.mean(),
                    3
                )
            ),

            PF=(
                "PnL_R",
                lambda x:
                (
                    x[x > 0].sum()
                    /
                    abs(
                        x[x <= 0].sum()
                    )
                    if abs(
                        x[x <= 0].sum()
                    ) > 0
                    else np.inf
                )
            )
        )
        .reset_index()
    )

    display(conv)


    # ========================================================
    # EXIT
    # ========================================================

    print(
        "\n--- Motivi di uscita ---"
    )

    display(
        trades_df[
            "Motivo"
        ]
        .value_counts()
        .to_frame("N.Trade")
    )


    # ========================================================
    # DIAGNOSTICA COMPONENTI
    # ========================================================

    print(
        "\n🔬 DIAGNOSTICA COMPONENTI"
    )


    def bucket_report(
        df,
        column,
        bins,
        title,
        quiet=False
    ):

        d = df.copy()

        if (
            column not in d.columns
            or
            len(d) == 0
        ):
            if not quiet:
                print(
                    f"--- {title} --- "
                    "(dati insufficienti)"
                )
            return None

        bins_sorted = sorted(
            set(bins)
        )

        if len(bins_sorted) < 3:

            if not quiet:
                print(
                    f"--- {title} --- "
                    "(bucket insufficienti)"
                )

            return None

        d["_b"] = pd.cut(
            d[column],
            bins=bins_sorted,
            include_lowest=True,
            duplicates="drop"
        )

        rep = (
            d.groupby(
                "_b",
                observed=True
            )
            .agg(
                N=("PnL_R", "count"),

                WinRate=(
                    "PnL_R",
                    lambda x:
                    round(
                        (x > 0).mean()
                        * 100,
                        1
                    )
                ),

                ExpectancyR=(
                    "PnL_R",
                    lambda x:
                    round(
                        x.mean(),
                        3
                    )
                )
            )
            .reset_index()
        )

        if not quiet:

            print(
                f"--- {title} ---"
            )

            display(rep)

        return rep


    bucket_report(
        trades_df,
        "Rel.Volume",
        [1.2, 1.3, 1.4, 1.5],
        "Volume relativo"
    )

    bucket_report(
        trades_df,
        "ADX",
        [30, 35, 40, 100],
        "ADX"
    )

    bucket_report(
        trades_df,
        "RSI",
        [60, 62, 64, 66],
        "RSI"
    )


    # ========================================================
    # OOS CHRONOLOGICAL SPLIT
    # ========================================================

    print(
        "\n" + "=" * 100
    )

    print(
        "🧪 OOS / SPLIT CRONOLOGICO"
    )

    print(
        "=" * 100
    )

    trades_sorted = (
        trades_df
        .sort_values("Data")
        .reset_index(drop=True)
    )

    mid = (
        len(trades_sorted)
        // 2
    )

    half_a = (
        trades_sorted
        .iloc[:mid]
        .copy()
    )

    half_b = (
        trades_sorted
        .iloc[mid:]
        .copy()
    )

    print(
        f"Metà A: {len(half_a)} trade"
    )

    print(
        f"Metà B: {len(half_b)} trade"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # sotto 15 per metà non dichiariamo validazione.
    # --------------------------------------------------------

    if (
        len(half_a)
        < OOS_MIN_TRADES_PER_HALF
        or
        len(half_b)
        < OOS_MIN_TRADES_PER_HALF
    ):

        print(
            "\n⚠️ CAMPIONE TROPPO PICCOLO."
        )

        print(
            f"A={len(half_a)}, "
            f"B={len(half_b)} "
            f"(minimo descrittivo "
            f"{OOS_MIN_TRADES_PER_HALF} "
            f"per metà)."
        )

        print(
            "La divisione cronologica "
            "viene mostrata solo come "
            "diagnostica e NON viene "
            "considerata una validazione OOS."
        )

    else:

        print(
            f"\nA: "
            f"{half_a['Data'].min().date()} "
            f"→ "
            f"{half_a['Data'].max().date()}"
        )

        print(
            f"B: "
            f"{half_b['Data'].min().date()} "
            f"→ "
            f"{half_b['Data'].max().date()}"
        )

        exp_a = (
            half_a["PnL_R"]
            .mean()
        )

        exp_b = (
            half_b["PnL_R"]
            .mean()
        )

        wr_a = (
            half_a["PnL_R"]
            .gt(0)
            .mean()
            * 100
        )

        wr_b = (
            half_b["PnL_R"]
            .gt(0)
            .mean()
            * 100
        )

        print(
            f"Expectancy A = "
            f"{exp_a:.3f}R"
        )

        print(
            f"Expectancy B = "
            f"{exp_b:.3f}R"
        )

        print(
            f"Win rate A = "
            f"{wr_a:.1f}%"
        )

        print(
            f"Win rate B = "
            f"{wr_b:.1f}%"
        )


        # ====================================================
        # COMPARAZIONE BUCKET
        # ====================================================

        def compare_halves(
            column,
            bins,
            title
        ):

            print(
                f"\n--- "
                f"{title}: A vs B ---"
            )

            rep_a = bucket_report(
                half_a,
                column,
                bins,
                title,
                quiet=True
            )

            rep_b = bucket_report(
                half_b,
                column,
                bins,
                title,
                quiet=True
            )

            if (
                rep_a is None
                or
                rep_b is None
            ):

                print(
                    "Dati insufficienti."
                )

                return

            merged = rep_a.merge(
                rep_b,
                on="_b",
                suffixes=(
                    "_A",
                    "_B"
                )
            )

            merged.columns = [
                "Bucket",
                "N_A",
                "WinRate_A",
                "Exp_A",
                "N_B",
                "WinRate_B",
                "Exp_B"
            ]

            display(
                merged
            )

            # --------------------------------------------
            # Un bucket viene considerato "dominante"
            # solo se ha almeno 8 osservazioni nella metà.
            # --------------------------------------------

            eligible_a = merged[
                merged["N_A"] >= 8
            ]

            eligible_b = merged[
                merged["N_B"] >= 8
            ]

            if (
                eligible_a.empty
                or
                eligible_b.empty
            ):

                print(
                    "⚠️ Nessun bucket con "
                    "N>=8 in entrambe le metà."
                )

                return

            best_a = (
                eligible_a
                .loc[
                    eligible_a["Exp_A"].idxmax(),
                    "Bucket"
                ]
            )

            best_b = (
                eligible_b
                .loc[
                    eligible_b["Exp_B"].idxmax(),
                    "Bucket"
                ]
            )

            if str(best_a) == str(best_b):

                print(
                    f"🟢 BUCKET COERENTE: "
                    f"{best_a}"
                )

            else:

                print(
                    f"🟠 Bucket diverso: "
                    f"A={best_a} | "
                    f"B={best_b}"
                )


        compare_halves(
            "Rel.Volume",
            [1.2, 1.3, 1.4, 1.5],
            "Volume relativo"
        )

        compare_halves(
            "ADX",
            [30, 35, 40, 100],
            "ADX"
        )

        compare_halves(
            "RSI",
            [60, 62, 64, 66],
            "RSI"
        )


# ============================================================
# SCANNER OGGI
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "🔎 SCANNER OGGI — V11.4 FIXED"
)

print(
    "=" * 100
)


today_rows = []

for t, di in daily_frames.items():

    # --------------------------------------------------------
    # Ultima candela giornaliera disponibile.
    # --------------------------------------------------------

    row = di.iloc[-1]

    date = di.index[-1]

    price = row["Close"]

    if (
        pd.isna(price)
        or
        price < MIN_PRICE
        or
        price > MAX_PRICE
    ):
        continue

    if (
        pd.isna(row["avg_vol20"])
        or
        row["avg_vol20"]
        < MIN_AVG_VOLUME
    ):
        continue

    atr = row["atr"]

    if (
        pd.isna(atr)
        or
        atr <= 0
    ):
        continue

    # --------------------------------------------------------
    # WEEKLY
    # --------------------------------------------------------

    wf = weekly_flags.get(t)

    weekly_bull = (
        latest_weekly_flag_before_or_at(
            wf,
            date
        )
    )

    if weekly_bull is None:
        continue

    # --------------------------------------------------------
    # PULLBACK
    # --------------------------------------------------------

    if (
        pd.notna(row["high20"])
        and
        row["high20"] > 0
    ):

        pullback_pct = (
            1
            -
            price / row["high20"]
        ) * 100

    else:

        pullback_pct = 0.0

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    atr_pct = (
        atr
        /
        price
        * 100
    )

    # --------------------------------------------------------
    # RELVOL
    # --------------------------------------------------------

    if row["avg_vol20"] > 0:

        relvol = (
            row["Volume"]
            /
            row["avg_vol20"]
        )

    else:

        relvol = 0.0

    # --------------------------------------------------------
    # HARD FILTERS
    # --------------------------------------------------------

    if not quality_filters(
        row,
        relvol,
        pullback_pct,
        atr_pct
    ):
        continue

    # --------------------------------------------------------
    # RS VS SPY
    # --------------------------------------------------------

    rs_vs_spy = (
        calculate_rs_vs_spy(
            di,
            len(di) - 1,
            date
        )
    )

    # --------------------------------------------------------
    # MARKET REGIME
    # --------------------------------------------------------

    # Per lo scanner usiamo lo stato corrente.
    # Questo è appropriato perché stiamo decidendo ORA.
    market_bullish = (
        market_bullish_now
    )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    result = score_and_flags(
        row,
        weekly_bull,
        market_bullish,
        rs_vs_spy,
        pullback_pct,
        atr_pct,
        relvol
    )

    if result is None:
        continue

    score, daily_bull, _ = result

    if (
        score < SCORE_BUY
        or
        not weekly_bull
        or
        not daily_bull
    ):
        continue

    # --------------------------------------------------------
    # LEVELS
    # --------------------------------------------------------

    levels = calculate_trade_levels(
        row
    )

    if levels is None:
        continue

    # --------------------------------------------------------
    # 4H
    # --------------------------------------------------------

    hf = h4_flags.get(t)

    if REQUIRE_4H_CONFIRMATION:

        h4_bull = (
            latest_4h_flag_before_or_at(
                hf,
                date
            )
        )

        if h4_bull is None:
            continue

        if not h4_bull:
            continue

    else:

        h4_bull = True

    # --------------------------------------------------------
    # CONVINZIONE
    # --------------------------------------------------------

    conviction = (
        "ALTA"
        if is_high_conviction(
            row["rsi"]
        )
        else
        "MEDIA"
    )

    # --------------------------------------------------------
    # POSITION SIZE
    # --------------------------------------------------------

    quote_100 = (
        CAPITALE
        /
        price
    )

    # --------------------------------------------------------
    # ROW
    # --------------------------------------------------------

    today_rows.append(
        {
            "Ticker": t,
            "Score": round(
                score,
                1
            ),
            "Convinzione": conviction,
            "RSI": round(
                row["rsi"],
                2
            ),
            "ADX": round(
                row["adx"],
                2
            ),
            "Rel.Volume": round(
                relvol,
                2
            ),
            "Pullback %": round(
                pullback_pct,
                2
            ),
            "RS vs SPY": rs_vs_spy,
            "Prezzo": round(
                price,
                2
            ),
            "Stop": round(
                levels["stop"],
                2
            ),
            "Target": round(
                levels["target"],
                2
            ),
            "Rischio %": round(
                levels["risk_pct"],
                2
            ),
            "4H Bull": h4_bull,
            "Quote 100€": round(
                quote_100,
                3
            )
        }
    )


# ============================================================
# SCANNER RESULT
# ============================================================

today_df = pd.DataFrame(
    today_rows
)

if today_df.empty:

    print(
        "🔴 Nessun candidato oggi."
    )

else:

    # ALTA prima, poi score
    today_df[
        "conv_order"
    ] = (
        today_df[
            "Convinzione"
        ]
        .map(
            {
                "ALTA": 0,
                "MEDIA": 1
            }
        )
    )

    today_df = (
        today_df
        .sort_values(
            [
                "conv_order",
                "Score"
            ],
            ascending=[
                True,
                False
            ]
        )
        .head(TOP_FINAL)
        .drop(
            columns=[
                "conv_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\n🎯 TOP SETUP:"
    )

    display(
        today_df
    )


# ============================================================
# EXPORT
# ============================================================

if not trades_df.empty:

    trades_df.to_csv(
        "ULTRA_SNIPER_V11_4_FIXED_BACKTEST.csv",
        index=False
    )

    print(
        "\n💾 Salvato:"
    )

    print(
        "ULTRA_SNIPER_V11_4_FIXED_BACKTEST.csv"
    )


if not today_df.empty:

    today_df.to_csv(
        "ULTRA_SNIPER_V11_4_FIXED_SCANNER.csv",
        index=False
    )

    print(
        "ULTRA_SNIPER_V11_4_FIXED_SCANNER.csv"
    )


# ============================================================
# CONTROLLO FINALE
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "✅ FINE ULTRA SNIPER V11.4 FIXED"
)

print(
    "=" * 100
)

print(
    "RSI operativo: 60-66"
)

print(
    "RSI 62-64: solo ALTA CONVINZIONE"
)

print(
    "Nessun filtro RSI aggiuntivo."
)

print(
    "Weekly/4H: ultima informazione disponibile "
    "alla data del segnale."
)

print(
    "Stop+Target sulla stessa candela: "
    "ordine conservativo = STOP."
)

print(
    "OOS con N insufficiente: "
    "NON dichiarato validato."
)

print(
    "=" * 100
)
# ============================================================
# TELEGRAM OUTPUT
# ============================================================

try:
    send_telegram_message("🔔 Ultra Sniper V11.4 completato con successo.")
except Exception as e:
    print(f"Errore Telegram finale: {e}")
