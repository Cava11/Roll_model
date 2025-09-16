
# Modello di Roll — simulazione & diagnostiche (Python)

Questo repository contiene un piccolo esperimento sul **modello di Roll** (1984) per microstruttura:
si simula un prezzo efficiente che segue un random walk e si aggiunge un **rimbalzo bid–ask** (cost-to-trade `c`), 
che introduce **autocorrelazione negativa** nei rendimenti a 1 passo. Il codice produce:
- la **Signature Plot**: \( C(\tau) = \mathbb{E}[(p_{t+\tau} - p_t)^2]/\tau \)  
- l’**ACF dei rendimenti** (lag 1 negativo)  
- una **stima dello spread** via la covarianza al primo ritardo  
- un semplice **filtro** tramite MA(1) (ARIMA(0,0,1)) sui rendimenti per avvicinare lo stato latente

> **Idea chiave (Roll)**: con \( p_t = m_t + u_t \), dove \( m_t \) è il “prezzo efficiente” (random walk) e \( u_t \in \{\pm c\} \) è il rimbalzo bid–ask,
i rendimenti \( \Delta p_t = p_t - p_{t-1} \) hanno **autocovarianza** al lag 1 pari a \( \gamma_1 = \operatorname{Cov}(\Delta p_t, \Delta p_{t-1}) = -c^2 \).  
Quindi **\( c = \sqrt{-\gamma_1} \)** e **lo spread effettivo** è circa **\( 2c \)**.

## Cosa fa lo script (`Roll_model.py`)
1. **Parametri**: `T` (lunghezza), `c` (costo per trade), `m0` (livello iniziale), `su` (deviazione std dell’innovazione del random walk).  
2. **Simulazione**  
   - `q_t ∈ {−1, +1}` direzione del trade (simmetrica).  
   - `m_t = m_{t-1} + ε_t`, con `ε_t ~ N(0, su)` (random walk).  
   - **Prezzo osservato**: `p_t = m_t + c⋅q_t`.  
3. **Signature plot**: calcola \( C(\tau) \) per \( \tau = 1..L_{max} \) e la confronta con la curva teorica \( su^2 + 2c^2/\tau \).  
4. **ACF dei rendimenti**: mostra il lag 1 negativo tipico del rimbalzo bid–ask.  
5. **Stima dello spread**: usa \( \gamma_1 \) (autocovarianza a lag 1) per ricavare \( c \approx \sqrt{-\gamma_1} \).  
   - **Nota pratica**: molte funzioni restituiscono l’**autocorrelazione** \( \rho_1 \) (non la covarianza). In tal caso:  
     \[ c = \sqrt{-\rho_1}\;\sigma_{\Delta p} \]  
     dove \( \sigma_{\Delta p}^2 = \operatorname{Var}(\Delta p_t) \).  
6. **Filtro MA(1)**: stima un **ARIMA(0,0,1)** su \( \Delta p_t \) (che teoricamente ha struttura MA(1)) e costruisce una traiettoria filtrata di riferimento.

## Requisiti
- Python 3.10+  
- `numpy`, `matplotlib`, `statsmodels`

Installa dipendenze:
```bash
pip install -r requirements.txt
```

## Esecuzione
```bash
python Roll_model.py
```
Genera i grafici: *Signature Plot*, *ACF*, confronto `p` vs `m`, stima spread, filtro.

## Parametri chiave (da codice)
- `T` — numero di osservazioni (default 20.000)  
- `c` — **mezzo-spread** (il **bid–ask spread** è ~ `2c`)  
- `su` — volatilità dell’innovazione del prezzo efficiente (sd per passo)  
- `Lmax` — orizzonte massimo per la signature plot

## Output attesi
- **Signature Plot**: punti simulati che si avvicinano a \( su^2 + 2c^2/\tau \) (asintoticamente vanno a \( su^2 \)).  
- **ACF(Δp)**: picco **negativo** al lag 1.  
- **Stima spread**: ricava \( c \) dal primo lag (vedi nota autocovarianza vs autocorrelazione).  
- **Filtro**: curva filtrata che attenua il rumore di microstruttura rispetto a `p` e si avvicina a `m`.

## Note & buone pratiche
- Imposta un **seed** (ad es. `np.random.seed(42)`) in testa per riproducibilità.  
- Se confronti stime teoriche/empiriche, assicurati della **scala** (covarianza vs correlazione).  
- La stima MA(1) è un proxy semplice; per un filtro “state-space” valuta un **Kalman filter** con `p_t = m_t + u_t`, `m_t = m_{t-1} + ε_t`.

## Riferimento
- Roll, R. (1984). *A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market*.

---

> Se vuoi, posso aggiungere una versione **Notebook** (`.ipynb`) con spiegazioni passo-passo e figure salvate in `figures/`.
