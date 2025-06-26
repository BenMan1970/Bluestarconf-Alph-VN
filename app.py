# =========================================================
#  1. IMPORTS (TOUJOURS EN PREMIER)
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone
import oandapyV20
import oandapyV20.endpoints.instruments as instruments

# =========================================================
#  2. CONFIGURATION DE L'API (APRÈS LES IMPORTS)
# =========================================================
try:
    # Lecture des secrets stockés dans Streamlit Cloud
    ACCOUNT_ID = st.secrets["OANDA_ACCOUNT_ID"]
    ACCESS_TOKEN = st.secrets["OANDA_ACCESS_TOKEN"]
    
    # Définition de l'environnement : 'practice' pour démo, 'live' pour réel
    # Assurez-vous que cela correspond à votre clé API.
    API_ENVIRONMENT = "practice"

except KeyError as e:
    # Affiche une erreur claire si un secret est manquant
    st.error(f"Secret OANDA manquant : '{e.args[0]}'. Veuillez le configurer dans les paramètres de l'application.")
    st.stop()

# Initialisation du client API OANDA
api = oandapyV20.API(access_token=ACCESS_TOKEN, environment=API_ENVIRONMENT)


# =========================================================
#  3. FONCTIONS HELPER
# =========================================================

# --- Fonctions de calcul mathématique ---
def wma(series: pd.Series, length: int) -> pd.Series:
    weights = pd.Series(range(1, length + 1))
    return series.rolling(length).apply(lambda prices: (prices * weights).sum() / weights.sum(), raw=True)

def hma(series: pd.Series, length: int) -> pd.Series:
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    return wma(2 * wma(series, half_length) - wma(series, length), sqrt_length)

# --- Fonction de récupération des données via OANDA ---
@st.cache_data(ttl=300) # Cache de 5 minutes pour éviter de surcharger l'API
def get_data(symbol, interval, output_size):
    """Récupère les données de marché depuis OANDA v20 API."""
    
    # Mapping des intervalles vers la nomenclature OANDA ("granularity")
    granularity_map = {
        "1day": "D",
        "4h": "H4",
        "1h": "H1",
    }
    if interval not in granularity_map:
        st.error(f"Intervalle non supporté par le code pour OANDA: {interval}")
        return None
    
    # Le symbole doit être au format OANDA (ex: EUR_USD)
    instrument_name = symbol.replace("/", "_")
    
    params = {
        "count": output_size,
        "granularity": granularity_map[interval],
        "price": "M"  # 'M' pour Midpoint, 'B' pour Bid, 'A' pour Ask
    }
    
    r = instruments.InstrumentsCandles(instrument=instrument_name, params=params)
    
    try:
        api.request(r)
        
        # Formater la réponse JSON en DataFrame pandas
        data = []
        for candle in r.response.get('candles', []):
            if candle['complete']:
                data.append({
                    'datetime': pd.to_datetime(candle['time']),
                    'Open': float(candle['mid']['o']),
                    'High': float(candle['mid']['h']),
                    'Low': float(candle['mid']['l']),
                    'Close': float(candle['mid']['c']),
                })
        
        if not data:
            st.warning(f"Aucune donnée retournée par OANDA pour {symbol} avec les paramètres actuels.")
            return None
            
        df = pd.DataFrame(data)
        df.set_index('datetime', inplace=True)
        # OANDA fournit déjà les timestamps en UTC
        return df

    except oandapyV20.exceptions.V20Error as e:
        st.warning(f"Erreur API OANDA pour {symbol}: {e}")
        return None
    except Exception as e:
        st.error(f"Une erreur inattendue est survenue avec OANDA pour {symbol}: {e}")
        return None

# =========================================================
#  4. PAIRES ET LOGIQUE DE TRADING
# =========================================================

# --- Liste des paires à scanner ---
FOREX_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD", "EUR/JPY", "GBP/JPY", "EUR/GBP", "XAU/USD"]

# --- Fonctions de calcul des indicateurs et filtres ---
def ema(s, p): return s.ewm(span=p, adjust=False).mean()

def check_directional_filters(symbol):
    df_d1 = get_data(symbol, interval="1day", output_size=100)
    time.sleep(1) # Petite pause pour respecter l'API
    df_h4 = get_data(symbol, interval="4h", output_size=100)
    time.sleep(1)
    if df_d1 is None or df_h4 is None or len(df_d1) < 51 or len(df_h4) < 21: return None 
    df_d1['ema20'], df_d1['ema50'] = ema(df_d1['Close'], 20), ema(df_d1['Close'], 50)
    df_h4['ema9'], df_h4['ema20'] = ema(df_h4['Close'], 9), ema(df_h4['Close'], 20)
    last_d1, last_h4 = df_d1.iloc[-1], df_h4.iloc[-1]
    if (last_d1['ema20'] > last_d1['ema50']) and (last_h4['ema9'] > last_h4['ema20']): return "HAUSSIER"
    if (last_d1['ema20'] < last_d1['ema50']) and (last_h4['ema9'] < last_h4['ema20']): return "BAISSIER"
    return None

def rma(s, p): return s.ewm(alpha=1/p, adjust=False).mean()

def adx(h, l, c, di_len, adx_len):
    tr = pd.concat([h-l, abs(h-c.shift()), abs(l-c.shift())], axis=1).max(axis=1); atr = rma(tr, di_len)
    up = h.diff(); down = -l.diff(); plus_dm = np.where((up > down) & (up > 0), up, 0.0); minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_di = 100 * rma(pd.Series(plus_dm, index=h.index), di_len) / atr.replace(0, 1e-9); minus_di = 100 * rma(pd.Series(minus_dm, index=h.index), di_len) / atr.replace(0, 1e-9)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9)
    return rma(dx, adx_len)

def rsi(src, p):
    d = src.diff(); g = d.where(d > 0, 0.0); l = -d.where(d < 0, 0.0); rs = rma(g, p) / rma(l, p).replace(0, 1e-9)
    return 100 - 100 / (1 + rs)

def calculate_signals(df, aligned_direction=None):
    if df is None or len(df) < 100: return None
    
    # Paramètres
    hmaLength=20; adxThreshold=20; rsiLength=10; adxLength=14; diLength=14; ichimokuTenkan=9; sha_len1=10; sha_len2=10
    signals={}; bullConfluences=0; bearConfluences=0

    # Calcul des indicateurs
    hma_series=hma(df['Close'], hmaLength)
    hmaSlope=1 if hma_series.iloc[-1]>hma_series.iloc[-2] else -1
    signals['HMA']="▲" if hmaSlope==1 else "▼"

    ha_close=df[['Open','High','Low','Close']].mean(axis=1)
    ha_open=pd.Series(np.nan,index=df.index)
    ha_open.iloc[0]=(df['Open'].iloc[0]+df['Close'].iloc[0])/2
    for i in range(1,len(df)): ha_open.iloc[i]=(ha_open.iloc[i-1]+ha_close.iloc[i-1])/2
    haSignal=1 if ha_close.iloc[-1]>ha_open.iloc[-1] else -1
    signals['Heikin Ashi']="▲" if haSignal==1 else "▼"

    o=ema(df['Open'],sha_len1); c=ema(df['Close'],sha_len1); h=ema(df['High'],sha_len1); l=ema(df['Low'],sha_len1)
    haclose_s=(o+h+l+c)/4
    haopen_s=pd.Series(np.nan,index=df.index); haopen_s.iloc[0]=(o.iloc[0]+c.iloc[0])/2
    for i in range(1,len(df)): haopen_s.iloc[i]=(haopen_s.iloc[i-1]+haclose_s.iloc[i-1])/2
    o2=ema(haopen_s,sha_len2); c2=ema(haclose_s,sha_len2)
    smoothedHaSignal=1 if o2.iloc[-1]<=c2.iloc[-1] else -1
    signals['Smoothed HA']="▲" if smoothedHaSignal==1 else "▼"

    ohlc4=df[['Open','High','Low','Close']].mean(axis=1)
    rsi_series=rsi(ohlc4,rsiLength)
    rsiSignal=1 if rsi_series.iloc[-1]>50 else -1
    signals['RSI']=f"{int(rsi_series.iloc[-1])} {'▲' if rsiSignal==1 else '▼'}"

    adx_series=adx(df['High'],df['Low'],df['Close'],diLength,adxLength)
    adxHasMomentum=adx_series.iloc[-1]>=adxThreshold
    signals['ADX']=f"{int(adx_series.iloc[-1])} {'💪' if adxHasMomentum else '💤'}"

    tenkan=(df['High'].rolling(ichimokuTenkan).max()+df['Low'].rolling(ichimokuTenkan).min())/2
    kijun=(df['High'].rolling(26).max()+df['Low'].rolling(26).min())/2
    senkouA=(tenkan+kijun)/2
    senkouB=(df['High'].rolling(52).max()+df['Low'].rolling(52).min())/2
    price=df['Close'].iloc[-1]
    ichimokuSignal = 1 if price > senkouA.iloc[-1] and price > senkouB.iloc[-1] else -1 if price < senkouA.iloc[-1] and price < senkouB.iloc[-1] else 0
    signals['Ichimoku']="▲" if ichimokuSignal==1 else "▼" if ichimokuSignal==-1 else "─"

    # Comptage des confluences
    bullConfluences+=(hmaSlope==1); bullConfluences+=(haSignal==1); bullConfluences+=(smoothedHaSignal==1); bullConfluences+=(rsiSignal==1); bullConfluences+=adxHasMomentum; bullConfluences+=(ichimokuSignal==1)
    bearConfluences+=(hmaSlope==-1); bearConfluences+=(haSignal==-1); bearConfluences+=(smoothedHaSignal==-1); bearConfluences+=(rsiSignal==-1); bearConfluences+=adxHasMomentum; bearConfluences+=(ichimokuSignal==-1)
    
    # Logique de direction
    if aligned_direction == "HAUSSIER":
        direction_display, confluence = "↗ HAUSSIER (Filtré D1/H4)", bullConfluences
    elif aligned_direction == "BAISSIER":
        direction_display, confluence = "↘ BAISSIER (Filtré D1/H4)", bearConfluences
    else: 
        if bullConfluences > bearConfluences: direction_display, confluence = "▲ HAUSSIER (H1)", bullConfluences
        elif bearConfluences > bullConfluences: direction_display, confluence = "▼ BAISSIER (H1)", bearConfluences
        else: direction_display, confluence = "─ NEUTRE (H1)", bullConfluences
            
    stars="⭐"*confluence; return {"confluence":confluence,"direction":direction_display,"stars":stars,"signals":signals}


# =========================================================
#  5. INTERFACE UTILISATEUR STREAMLIT
# =========================================================
st.set_page_config(layout="wide", page_title="Scanner Canadian Star")
st.title("Scanner Canadian Star 🌠")
st.info("Cet outil scanne les paires pour trouver des opportunités basées sur une confluence de signaux techniques. Données fournies par OANDA.", icon="💡")
st.subheader("Configuration du Scan")
col1, col2 = st.columns([1, 2])
with col1:
    min_conf = st.slider("Confluence H1 minimale", 1, 6, 4, help="Seuil minimum de signaux concordants en H1.")
with col2:
    use_trend_filter = st.checkbox("✅ Activer le filtre de tendance D1/H4 (Recommandé, très strict)", value=True)

if st.button("Lancer le Scan 🚀", use_container_width=True):
    results = []
    total_pairs = len(FOREX_PAIRS)
    progress_bar = st.progress(0, text="Initialisation du scan...")

    for i, symbol in enumerate(FOREX_PAIRS):
        current_progress = (i + 1) / total_pairs
        aligned_direction = None
        
        # Logique du filtre de tendance D1/H4
        if use_trend_filter:
            progress_bar.progress(current_progress, text=f"({i+1}/{total_pairs}) Vérification D1/H4 pour {symbol}...")
            aligned_direction = check_directional_filters(symbol)
            if not aligned_direction:
                continue # Passe à la paire suivante si la tendance n'est pas alignée
            st.toast(f"{symbol} : Tendance {aligned_direction} alignée ! ✅", icon="📈")
        
        # Calcul de la confluence H1
        progress_bar.progress(current_progress, text=f"({i+1}/{total_pairs}) Calcul de la confluence H1 pour {symbol}...")
        df_h1 = get_data(symbol, interval="1h", output_size=200)
        time.sleep(1) # Pause pour éviter de spammer l'API

        if df_h1 is not None:
            res = calculate_signals(df_h1, aligned_direction)
            if res and res['confluence'] >= min_conf:
                row = {"Paire": symbol, "Confluences": res['stars'], "Direction": res['direction'], "confluence_score": res['confluence']}
                row.update(res['signals'])
                results.append(row)
    
    progress_bar.empty()

    # Affichage des résultats
    if results:
        df_res = pd.DataFrame(results).sort_values(by="confluence_score", ascending=False)
        column_order = ["Paire", "Confluences", "Direction", "HMA", "Heikin Ashi", "Smoothed HA", "RSI", "ADX", "Ichimoku"]
        df_display = df_res.drop(columns=['confluence_score'])[column_order]
        def style_direction(direction):
            if 'HAUSSIER' in direction: return 'color: #2ECC71; font-weight: bold;'
            if 'BAISSIER' in direction: return 'color: #E74C3C; font-weight: bold;'
            return ''
        st.dataframe(df_display.style.applymap(style_direction, subset=['Direction']), use_container_width=True)
        csv = df_res.to_csv(index=False).encode('utf-8')
        st.download_button(label="📂 Exporter en CSV", data=csv, file_name="resultats_canadian_star.csv", mime="text/csv")
    else:
        st.warning("Scan terminé. Aucun résultat trouvé avec les paramètres actuels.")
        if use_trend_filter:
            st.info("Le filtre de tendance D1/H4 est très sélectif. Essayez de relancer le scan en décochant la case 'Activer le filtre' pour voir plus de signaux potentiels (non filtrés).")

st.caption(f"Confluence calculée sur 1h. | Dernière MàJ : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
