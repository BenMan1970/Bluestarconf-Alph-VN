# Mettez ces imports en haut de votre fichier app.py
import oandapyV20
import oandapyV20.endpoints.instruments as instruments

# --- CONFIGURATION (modifiée) ---
try:
    ACCOUNT_ID = st.secrets["OANDA_ACCOUNT_ID"]
    ACCESS_TOKEN = st.secrets["OANDA_ACCESS_TOKEN"]
    # OANDA utilise 'practice' pour les comptes démo et 'live' pour les comptes réels
    # Assurez-vous que l'environnement correspond à votre clé
    API_ENVIRONMENT = "practice" # ou "live"
except KeyError as e:
    st.error(f"Secret '{e.args[0]}' non trouvé. Veuillez le configurer pour OANDA.")
    st.stop()

# Initialisation du client API
api = oandapyV20.API(access_token=ACCESS_TOKEN, environment=API_ENVIRONMENT)


@st.cache_data(ttl=300) # TTL plus court car les données sont plus "live"
def get_data(symbol, interval, output_size):
    """
    Récupère les données de marché depuis OANDA v20 API.
    """
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
        "price": "M" # 'M' pour Midpoint, 'B' pour Bid, 'A' pour Ask
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

# Le reste de votre code (fonctions d'indicateurs, UI, etc.)
# n'a pas besoin d'être modifié car il attend un DataFrame
# avec les colonnes 'Open', 'High', 'Low', 'Close' et un index datetime,
# ce que la nouvelle fonction get_data fournit.
#
# Assurez-vous juste que la liste des paires utilise le format "EUR/USD"
# car la fonction get_data s'occupe de la conversion en "EUR_USD".
FOREX_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD", "EUR/JPY", "GBP/JPY", "EUR/GBP", "XAU/USD"]
# Note : Pour XAU/USD, le nom de l'instrument OANDA est XAU_USD, la conversion fonctionnera.
