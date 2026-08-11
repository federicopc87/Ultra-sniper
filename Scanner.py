import os
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURAZIONE
# ============================================================

CAPITAL_PER_TRADE = 100.0

MIN_CANDIDATE_SCORE = 82
MIN_BUY_SCORE = 90

MAX_RISK_PERCENT = 7.0
MIN_RR = 2.0

MAX_TELEGRAM_ALERTS = 3

MIN_PRICE = 10.0
MIN_AVG_VOLUME = 500_000

SPY_TICKER = "SPY"

# Token e Chat ID vengono letti dai Secrets di GitHub
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


# ============================================================
# STAMPA INIZIALE
# ============================================================

print("=" * 100)
print("                 🚀 ULTRA SNIPER V8 — GITHUB")
print("=" * 100)
print()
print(f"💰 Capitale per operazione: €{CAPITAL_PER_TRADE:.2f}")
print(f"🎯 Score minimo candidato: {MIN_CANDIDATE_SCORE}")
print(f"🟢 Score minimo BUY: {MIN_BUY_SCORE}")
print(f"🛑 Rischio massimo: {MAX_RISK_PERCENT:.1f}%")
print(f"📈 R/R minimo: {MIN_RR:.1f}")
print()


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):
    """
    Invia un messaggio Telegram usando i GitHub Secrets.
    """

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram non configurato.")
        return False

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    try:
        response = requests.post(
            url,
            data=payload,
            timeout=20
        )

        if response.ok:
            print("📲 Alert Telegram inviato.")
            return True

        print(
            "⚠️ Errore Telegram:",
            response.status_code,
            response.text
        )

    except Exception as exc:
        print("⚠️ Errore connessione Telegram:", exc)

    return False


# ============================================================
# UTILITÀ
# ============================================================

def clean_ticker(value):
    """
    Converte i ticker Yahoo in formato compatibile.
    """
    return (
        str(value)
        .strip()
        .replace(".", "-")
        .upper()
    )


def safe_float(value):
    try:
        if value is None:
            return None

        value = float(value)

        if np.isnan(value) or np.isinf(value):
            return None

        return value

    except Exception:
        return None


def normalize_downloaded_frame(df, ticker):
    """
    Estrae il DataFrame di un singolo ticker da yfinance,
    gestendo colonne normali e MultiIndex.
    """

    if df is None or df.empty:
        return None

    try:

        if isinstance(df.columns, pd.MultiIndex):

            levels = df.columns.nlevels

            # Caso tipico:
            # Price / Ticker
            if ticker in df.columns.get_level_values(-1):
                out = df.xs(
                    ticker,
                    axis=1,
                    level=-1,
                    drop_level=True
                )

            elif ticker in df.columns.get_level_values(0):
                out = df.xs(
                    ticker,
                    axis=1,
                    level=0,
                    drop_level=True
                )

            else:
                return None

        else:
            out = df.copy()

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        if not all(
            column in out.columns
            for column in required
        ):
            return None

        out = out[required].copy()

        out = out.apply(
            pd.to_numeric,
            errors="coerce"
        )

        out = out.dropna()

        if out.empty:
            return None

        return out

    except Exception:
        return None


def download_one_ticker(
    ticker,
    period,
    interval
):
    """
    Download singolo ticker.
    Il download individuale è più lento del bulk,
    ma è più robusto quando Yahoo restituisce errori
    su singoli simboli.
    """

    try:

        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False
        )

        return normalize_downloaded_frame(
            data,
            ticker
        )

    except Exception:
        return None


# ============================================================
# UNIVERSO
# ============================================================

def get_sp500():
    """
    Scarica l'elenco S&P 500 da Wikipedia.
    """

    print("📥 Caricamento S&P 500...")

    urls = [
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    ]

    for url in urls:

        try:

            tables = pd.read_html(url)

            for table in tables:

                if "Symbol" in table.columns:

                    symbols = [
                        clean_ticker(x)
                        for x in table["Symbol"].dropna()
                    ]

                    if len(symbols) >= 400:
                        print(
                            f"✅ S&P 500: {len(symbols)}"
                        )
                        return sorted(set(symbols))

        except Exception as exc:

            print(
                "⚠️ Errore S&P 500:",
                exc
            )

    print("⚠️ S&P 500 non disponibile.")

    return []


def get_nasdaq100():
    """
    Lista di sicurezza del Nasdaq 100.
    Viene usata solo per ampliare l'universo.
    """

    print()
    print("📥 Caricamento Nasdaq 100...")

    # Lista incorporata di sicurezza.
    # Alcuni ticker possono essere cambiati nel tempo;
    # gli eventuali ticker non validi vengono semplicemente ignorati.

    symbols = """
AAPL
ABNB
ADBE
ADI
ADP
ADSK
AEP
AMAT
AMD
AMGN
AMZN
ANSS
ARM
ASML
AVGO
AXON
AZN
BIIB
BKNG
BKR
CCEP
CDNS
CDW
CEG
CHTR
CMCSA
COST
CPRT
CRWD
CSCO
CSGP
CSX
CTAS
CTSH
DASH
DDOG
DXCM
EA
EXC
FANG
FAST
FTNT
GEHC
GFS
GILD
GOOG
GOOGL
HON
IDXX
ILMN
INTC
INTU
ISRG
KDP
KHC
KLAC
LIN
LRCX
LULU
MAR
MCHP
MDB
MDLZ
MELI
META
MNST
MRNA
MRVL
MSFT
MSTR
MU
NFLX
NVDA
NXPI
ODFL
ON
ORLY
PANW
PAYX
PCAR
PDD
PEP
PLTR
PYPL
QCOM
REGN
ROP
ROST
SBUX
SNPS
TEAM
TMUS
TSLA
TTD
TTWO
TXN
VRSK
VRTX
WBD
WDAY
XEL
ZS
"""

    result = sorted(
        set(
            clean_ticker(x)
            for x in symbols.split()
            if x.strip()
        )
    )

    print(
        f"✅ Nasdaq 100 fallback: {len(result)}"
    )

    return result


def build_universe():

    sp500 = get_sp500()
    nasdaq = get_nasdaq100()

    tickers = sorted(
        set(sp500 + nasdaq)
    )

    print()
    print("=" * 100)
    print("                         UNIVERSO FINALE")
    print("=" * 100)
    print()
    print(
        f"📊 Titoli unici da scannerizzare: {len(tickers)}"
    )
    print()

    if not tickers:
        raise RuntimeError(
            "❌ Universo vuoto."
        )

    return tickers


# ============================================================
# INDICATORI
# ============================================================

def calculate_indicators(df):

    if df is None:
        return None

    if len(df) < 220:
        return None

    try:

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        ema20 = EMAIndicator(
            close=close,
            window=20
        ).ema_indicator()

        ema50 = EMAIndicator(
            close=close,
            window=50
        ).ema_indicator()

        sma200 = SMAIndicator(
            close=close,
            window=200
        ).sma_indicator()

        rsi_obj = RSIIndicator(
            close=close,
            window=14
        )

        rsi = rsi_obj.rsi()

        macd_obj = MACD(
            close=close,
            window_slow=26,
            window_fast=12,
            window_sign=9
        )

        macd = macd_obj.macd()
        macd_signal = macd_obj.macd_signal()
        macd_hist = macd_obj.macd_diff()

        adx_obj = ADXIndicator(
            high=high,
            low=low,
            close=close,
            window=14
        )

        adx = adx_obj.adx()
        plus_di = adx_obj.adx_pos()
        minus_di = adx_obj.adx_neg()

        atr = AverageTrueRange(
            high=high,
            low=low,
            close=close,
            window=14
        ).average_true_range()

        avg_volume = volume.rolling(
            20
        ).mean()

        values = {
            "price": safe_float(close.iloc[-1]),

            "ema20": safe_float(ema20.iloc[-1]),
            "ema50": safe_float(ema50.iloc[-1]),
            "sma200": safe_float(sma200.iloc[-1]),

            "rsi": safe_float(rsi.iloc[-1]),
            "rsi_previous": safe_float(rsi.iloc[-2]),

            "macd": safe_float(macd.iloc[-1]),
            "macd_signal": safe_float(
                macd_signal.iloc[-1]
            ),
            "macd_hist": safe_float(
                macd_hist.iloc[-1]
            ),

            "adx": safe_float(adx.iloc[-1]),
            "plus_di": safe_float(
                plus_di.iloc[-1]
            ),
            "minus_di": safe_float(
                minus_di.iloc[-1]
            ),

            "atr": safe_float(atr.iloc[-1]),

            "volume": safe_float(volume.iloc[-1]),
            "avg_volume": safe_float(
                avg_volume.iloc[-1]
            )
        }

        if any(
            value is None
            for value in values.values()
        ):
            return None

        return values

    except Exception:
        return None


# ============================================================
# CONVERSIONE 1H → 4H
# ============================================================

def convert_to_4h(df):

    if df is None or df.empty:
        return None

    try:

        frame = df.copy()

        frame.index = pd.to_datetime(
            frame.index
        )

        frame = frame.resample(
            "4h"
        ).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        })

        frame = frame.dropna()

        return frame

    except Exception:
        return None


# ============================================================
# SPY MARKET FILTER
# ============================================================

def get_market_filter():

    print("📥 Scaricamento SPY...")

    spy = download_one_ticker(
        SPY_TICKER,
        "2y",
        "1d"
    )

    if spy is None or len(spy) < 220:

        print(
            "⚠️ SPY non disponibile."
        )

        # Fail-safe:
        # non blocchiamo completamente lo scanner.
        return None, False

    try:

        close = spy["Close"]

        sma200 = (
            close
            .rolling(200)
            .mean()
            .iloc[-1]
        )

        price = close.iloc[-1]

        market_bullish = bool(
            float(price) > float(sma200)
        )

        print()

        if market_bullish:

            print(
                "📈 Market filter SPY: 🟢 BULLISH"
            )

        else:

            print(
                "📉 Market filter SPY: 🔴 NOT BULLISH"
            )

        return spy, market_bullish

    except Exception as exc:

        print(
            "⚠️ Errore filtro SPY:",
            exc
        )

        return spy, False


# ============================================================
# RELATIVE STRENGTH VS SPY
# ============================================================

def relative_strength_vs_spy(
    daily,
    spy
):

    if spy is None:
        return False

    try:

        stock_close = daily["Close"]
        spy_close = spy["Close"]

        if (
            len(stock_close) < 61
            or
            len(spy_close) < 61
        ):
            return False

        stock_perf = (
            float(stock_close.iloc[-1])
            /
            float(stock_close.iloc[-61])
            - 1
        ) * 100

        spy_perf = (
            float(spy_close.iloc[-1])
            /
            float(spy_close.iloc[-61])
            - 1
        ) * 100

        return bool(
            stock_perf > spy_perf
        )

    except Exception:
        return False


# ============================================================
# ANALISI TITOLO
# ============================================================

def analyze_stock(
    ticker,
    spy,
    market_bullish
):

    daily = download_one_ticker(
        ticker,
        "2y",
        "1d"
    )

    if daily is None:
        return None

    if len(daily) < 220:
        return None

    weekly = download_one_ticker(
        ticker,
        "3y",
        "1wk"
    )

    if weekly is None:
        return None

    if len(weekly) < 50:
        return None

    daily_ind = calculate_indicators(
        daily
    )

    weekly_ind = calculate_indicators(
        weekly
    )

    if (
        daily_ind is None
        or
        weekly_ind is None
    ):
        return None

    price = daily_ind["price"]

    if price is None:
        return None

    avg_volume = daily_ind["avg_volume"]

    if avg_volume is None:
        return None

    # --------------------------------------------------------
    # LIQUIDITÀ
    # --------------------------------------------------------

    if price < MIN_PRICE:
        return None

    if avg_volume < MIN_AVG_VOLUME:
        return None

    # --------------------------------------------------------
    # TREND WEEKLY
    # --------------------------------------------------------

    weekly_bullish = (
        weekly_ind["price"]
        >
        weekly_ind["ema20"]
        and
        weekly_ind["ema20"]
        >
        weekly_ind["ema50"]
    )

    weekly_long_term = (
        weekly_ind["price"]
        >
        weekly_ind["sma200"]
    )

    # --------------------------------------------------------
    # TREND DAILY
    # --------------------------------------------------------

    daily_bullish = (
        price > daily_ind["ema20"]
        and
        daily_ind["ema20"]
        >
        daily_ind["ema50"]
        and
        price > daily_ind["sma200"]
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi = daily_ind["rsi"]

    rsi_previous = daily_ind[
        "rsi_previous"
    ]

    rsi_recovery = (
        rsi_previous < rsi
        and
        rsi >= 45
    )

    rsi_good = (
        50 <= rsi <= 70
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    macd_bullish = (
        daily_ind["macd"]
        >
        daily_ind["macd_signal"]
    )

    macd_hist_positive = (
        daily_ind["macd_hist"] > 0
    )

    # --------------------------------------------------------
    # ADX / DI
    # --------------------------------------------------------

    adx_strong = (
        daily_ind["adx"] >= 25
    )

    di_bullish = (
        daily_ind["plus_di"]
        >
        daily_ind["minus_di"]
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    relative_volume = (
        daily_ind["volume"]
        /
        daily_ind["avg_volume"]
    )

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    close = daily["Close"]

    perf_1m = (
        float(close.iloc[-1])
        /
        float(close.iloc[-21])
        - 1
    ) * 100

    perf_3m = (
        float(close.iloc[-1])
        /
        float(close.iloc[-61])
        - 1
    ) * 100

    perf_6m = (
        float(close.iloc[-1])
        /
        float(close.iloc[-126])
        - 1
    ) * 100

    # --------------------------------------------------------
    # DISTANCE HIGH
    # --------------------------------------------------------

    high_1y = float(
        close.tail(252).max()
    )

    distance_high = (
        1 -
        price / high_1y
    ) * 100

    # --------------------------------------------------------
    # PULLBACK
    # --------------------------------------------------------

    high_20 = float(
        close.tail(20).max()
    )

    pullback = (
        1 -
        price / high_20
    ) * 100

    # --------------------------------------------------------
    # SUPPORT
    # --------------------------------------------------------

    recent_support = float(
        daily["Low"]
        .tail(20)
        .min()
    )

    support_distance = (
        price /
        recent_support
        - 1
    ) * 100

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    atr = daily_ind["atr"]

    atr_percent = (
        atr / price
    ) * 100

    # --------------------------------------------------------
    # GAP
    # --------------------------------------------------------

    previous_close = float(
        close.iloc[-2]
    )

    current_open = float(
        daily["Open"].iloc[-1]
    )

    gap_percent = (
        current_open /
        previous_close
        - 1
    ) * 100

    # --------------------------------------------------------
    # RELATIVE STRENGTH
    # --------------------------------------------------------

    rs_bullish = relative_strength_vs_spy(
        daily,
        spy
    )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    score = 0.0

    if weekly_bullish:
        score += 12

    if weekly_long_term:
        score += 5

    if daily_bullish:
        score += 15

    if price > daily_ind["ema20"]:
        score += 5

    if rsi_good:
        score += 8

    elif rsi_recovery:
        score += 5

    if macd_bullish:
        score += 8

    if macd_hist_positive:
        score += 4

    if adx_strong:
        score += 7

    if di_bullish:
        score += 4

    if relative_volume >= 1.2:
        score += 5

    elif relative_volume >= 1.0:
        score += 2

    if perf_6m > 15:
        score += 4

    if rs_bullish:
        score += 5

    if 2 <= pullback <= 15:
        score += 5

    # --------------------------------------------------------
    # PENALITÀ
    # --------------------------------------------------------

    if rsi > 75:
        score -= 8

    if distance_high > 30:
        score -= 8

    if atr_percent > 8:
        score -= 5

    if abs(gap_percent) > 8:
        score -= 5

    score = max(
        0,
        min(100, score)
    )

    # --------------------------------------------------------
    # FILTRO CANDIDATO
    # --------------------------------------------------------

    if score < MIN_CANDIDATE_SCORE:
        return None

    # --------------------------------------------------------
    # CONTROLLO 4H
    # --------------------------------------------------------

    hourly = download_one_ticker(
        ticker,
        "60d",
        "1h"
    )

    if hourly is None:
        return None

    four_hour = convert_to_4h(
        hourly
    )

    if four_hour is None:
        return None

    if len(four_hour) < 60:
        return None

    four_hour_ind = calculate_indicators(
        four_hour
    )

    if four_hour_ind is None:
        return None

    four_hour_bullish = (
        four_hour_ind["price"]
        >
        four_hour_ind["ema20"]
        and
        four_hour_ind["ema20"]
        >
        four_hour_ind["ema50"]
    )

    # --------------------------------------------------------
    # ENTRY / STOP
    # --------------------------------------------------------

    entry = price

    stop_atr = (
        entry -
        1.5 * atr
    )

    stop_support = (
        recent_support -
        0.25 * atr
    )

    stop = max(
        stop_atr,
        stop_support
    )

    risk = (
        entry -
        stop
    )

    if risk <= 0:
        return None

    risk_percent = (
        risk / entry
    ) * 100

    if risk_percent > MAX_RISK_PERCENT:
        return None

    # --------------------------------------------------------
    # TARGET / R:R
    # --------------------------------------------------------

    target = (
        entry +
        2.0 * risk
    )

    rr = (
        target - entry
    ) / risk

    if rr < MIN_RR:
        return None

    # --------------------------------------------------------
    # SETUP FINALE
    # --------------------------------------------------------

    valid_setup = bool(
        score >= MIN_BUY_SCORE
        and
        weekly_bullish
        and
        daily_bullish
        and
        four_hour_bullish
        and
        rr >= MIN_RR
        and
        risk_percent <= MAX_RISK_PERCENT
    )

    if valid_setup:
        quality = "A"
        signal = "BUY CANDIDATE"

    else:
        quality = "B"
        signal = "WATCHLIST"

    return {
        "Ticker": ticker,
        "Score": round(score, 2),
        "Quality": quality,
        "Signal": signal,
        "Valid Setup": valid_setup,
        "Price": round(price, 2),
        "RSI": round(rsi, 2),
        "ADX": round(
            daily_ind["adx"],
            2
        ),
        "Rel.Volume": round(
            relative_volume,
            2
        ),
        "Perf.3M %": round(
            perf_3m,
            2
        ),
        "Perf.6M %": round(
            perf_6m,
            2
        ),
        "Pullback %": round(
            pullback,
            2
        ),
        "ATR %": round(
            atr_percent,
            2
        ),
        "Entry": round(
            entry,
            2
        ),
        "Stop": round(
            stop,
            2
        ),
        "Target": round(
            target,
            2
        ),
        "Risk %": round(
            risk_percent,
            2
        ),
        "R/R": round(
            rr,
            2
        ),
        "4H Bull": four_hour_bullish,
        "SPY Bull": market_bullish,
        "RS vs SPY": rs_bullish
    }


# ============================================================
# TELEGRAM FORMAT
# ============================================================

def format_alert(row, rank):

    ticker = row["Ticker"]

    return (
        "🚀 ULTRA SNIPER — SETUP A\n\n"
        f"🏆 Ranking: #{rank}\n"
        f"📌 Ticker: {ticker}\n"
        f"⭐ Score: {row['Score']:.0f}/100\n\n"

        f"💰 Prezzo: {row['Price']:.2f}\n"
        f"🎯 Entry: {row['Entry']:.2f}\n"
        f"🛑 Stop: {row['Stop']:.2f}\n"
        f"🏁 Target: {row['Target']:.2f}\n"
        f"📊 R/R: {row['R/R']:.2f}\n"
        f"⚠️ Rischio: {row['Risk %']:.2f}%\n\n"

        f"RSI: {row['RSI']:.1f}\n"
        f"ADX: {row['ADX']:.1f}\n"
        f"Rel.Volume: {row['Rel.Volume']:.2f}\n"
        f"Perf. 3M: {row['Perf.3M %']:.1f}%\n"
        f"Perf. 6M: {row['Perf.6M %']:.1f}%\n"
        f"Pullback: {row['Pullback %']:.1f}%\n\n"

        "🟢 Daily + Weekly + 4H confermati\n"
        "📈 Controllare TradingView prima di operare.\n"
        "⚠️ Segnale tecnico, non garanzia di profitto."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    tickers = build_universe()

    spy, market_bullish = get_market_filter()

    print()
    print("=" * 100)
    print("                 🔎 PASSAGGIO 1 — SCANSIONE")
    print("=" * 100)
    print()

    results = []

    total = len(tickers)

    for index, ticker in enumerate(
        tickers,
        start=1
    ):

        try:

            result = analyze_stock(
                ticker,
                spy,
                market_bullish
            )

            if result is not None:
                results.append(result)

        except Exception as exc:

            print(
                f"⚠️ {ticker}: errore ignorato → {exc}"
            )

        if (
            index % 20 == 0
            or
            index == total
        ):

            print(
                f"Analizzati {index}/{total}"
            )

    print()
    print(
        f"✅ Candidati validi: {len(results)}"
    )

    # --------------------------------------------------------
    # NESSUN RISULTATO
    # --------------------------------------------------------

    if not results:

        print()
        print(
            "🔴 Nessun candidato sopra "
            f"{MIN_CANDIDATE_SCORE}/100."
        )

        print(
            "Nessun alert Telegram inviato."
        )

        return

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    df = pd.DataFrame(results)

    df = df.sort_values(
        by=[
            "Valid Setup",
            "Score",
            "R/R"
        ],
        ascending=[
            False,
            False,
            False
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # SALVATAGGIO
    # --------------------------------------------------------

    df.to_csv(
        "ULTRA_SNIPER_RESULTS.csv",
        index=False
    )

    # --------------------------------------------------------
    # TOP 5
    # --------------------------------------------------------

    top5 = df.head(5)

    print()
    print("=" * 100)
    print("                         🚀 TOP 5")
    print("=" * 100)
    print()

    print(
        top5.to_string(index=False)
    )

    # --------------------------------------------------------
    # SETUP A
    # --------------------------------------------------------

    setup_a = df[
        df["Valid Setup"] == True
    ].copy()

    print()
    print("=" * 100)
    print("                         🟢 SETUP A")
    print("=" * 100)
    print()

    if setup_a.empty:

        print(
            "🔴 NESSUN SETUP A."
        )

        print(
            "Nessun alert Telegram inviato."
        )

        return

    print(
        setup_a.to_string(index=False)
    )

    # --------------------------------------------------------
    # TELEGRAM
    # --------------------------------------------------------

    alerts_sent = 0

    for rank, (_, row) in enumerate(
        setup_a.head(
            MAX_TELEGRAM_ALERTS
        ).iterrows(),
        start=1
    ):

        message = format_alert(
            row,
            rank
        )

        if send_telegram(message):
            alerts_sent += 1

    print()
    print("=" * 100)
    print("                         ✅ COMPLETATO")
    print("=" * 100)
    print()

    print(
        f"📲 Alert Telegram inviati: {alerts_sent}"
    )

    print(
        "📄 Risultati: ULTRA_SNIPER_RESULTS.csv"
    )

    print()
    print(
        "⚠️ Controllare sempre il grafico "
        "su TradingView prima di qualsiasi decisione."
    )


if __name__ == "__main__":
    main()
