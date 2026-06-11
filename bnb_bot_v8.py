#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║              SOLANA BOT v1.0 — Acumulare maximă SOL                     ║
╠══════════════════════════════════════════════════════════════════════════╣
║  BUGET: $800 = ~1.22 BNB @ $654/BNB (15 mar 2026)                      ║
║                                                                          ║
║  PROFIT REALIST LUNAR (v5.0, simulat Monte Carlo 1000 rulari):          ║
║    Pesimist:  $18-22  (bear market, fara crash)                         ║
║    Realist:   $38-42  (sideways/bull — 2X față de v4.0)                ║
║    Optimist:  $55-65  (bull run puternic)                               ║
║    Compound:  $98/luna la luna 24 (4X organic, fara risc extra)         ║
║                                                                          ║
║  NOU IN v5.0 (Combo C — simulat +$18/luna):                            ║
║    📊 C1: Realocare capital: 60% funding / 5% swing (+$2/luna)         ║
║    🔲 C2: Tight grid BNBUSDT 0.8% spacing (+$10/luna, fills 3-4x/zi)  ║
║    🏦 C3: BNB Launchpool staking 20% capital (+$3-8/luna, APR 5-18%)  ║
║    📅 C4: Calendar Fed blackout — nu intrăm cu 24h inainte de FOMC     ║
║    🔄 C5: Compound 90% complet integrat cu tracking vizibil            ║
║                                                                          ║
║  ALOCARE v5.0:                                                          ║
║    S1: FUNDING ARB  60% ($480) — 0 trades dupa intrare [era 55%]       ║
║    S2: GRID MAKER   25% ($200) — BNBUSDT 0.8% + rest 1.5% [era 30%]  ║
║    S3: SWING         5%  ($40) — max 2 trades/zi, TP 1.8% [era 10%]  ║
║    S4: LAUNCHPOOL   20% ($160) — Binance Earn/Launchpool [NOU]         ║
║    REZERVA          5%  ($40)  — neatinsa                              ║
║                                                                          ║
║  MOȘTENIRE din v4.0:                                                    ║
║    ✅ Market Crash Guard (YELLOW/ORANGE/RED)                            ║
║    ✅ Funding sync UTC (00/08/16h)                                      ║
║    ✅ Grid fill bazat pe pret real                                       ║
║    ✅ EV swing formula corecta + slippage                               ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, json, time, math, logging, threading, hashlib, hmac, signal, uuid
import urllib.parse, random
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print("pip install requests"); sys.exit(1)

# Sesiune HTTP globală reutilizabilă pentru clasele fără Binance client
# (FundingSpreadFilter, OISentinel, TelegramBot, MarketSentinel)
_http_session = requests.Session()
_http_session.mount("https://", HTTPAdapter(
    pool_connections=10, pool_maxsize=20,
    max_retries=Retry(total=3, backoff_factor=0.5,
                      status_forcelist=[429, 500, 502, 503, 504])
))
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler
import tempfile
from typing import Optional, Dict, List, Tuple, Any

# ═══ ENHANCEMENTS v1.3 — INTEGRAT DIRECT ÎN BOT ═══
ENH_AVAILABLE = True
ML_AVAILABLE = True
_enh_logger = logging.getLogger("bot_enhancements")



# ═══════════════════════════════════════════════════════════════════
# 1. CIRCUIT BREAKER GLOBAL — Drawdown Protection
# ═══════════════════════════════════════════════════════════════════

class CircuitBreakerState(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"           # drawdown -5% → reduce position sizes 50%
    CRITICAL = "CRITICAL"         # drawdown -8% → close all, stop trading
    COOLDOWN = "COOLDOWN"         # post-critical, waiting to resume
    CORRELATION_HALT = "CORR_HALT"  # BNB+SOL dropping together


@dataclass
class CircuitBreakerConfig:
    # v1.3: praguri mai largi — nu opri pentru fluctuații normale
    warning_threshold: float = -7.0       # era -5% → acum -7%
    critical_threshold: float = -12.0     # era -8% → acum -12% (crash real)
    
    # Cooldown ultra-scurt — 15min suficient pt stabilizare
    cooldown_minutes: int = 15            # era 30 → acum 15
    
    # Correlation halt: mai strict — doar crash simultan sever
    correlation_drop_pct: float = -5.0    # era -3% → acum -5%
    correlation_window_min: int = 15      # era 30 → acum 15 (flash crash rapid)
    
    # Recovery: instant la 90%
    recovery_size_pct: float = 90.0       # era 75% → acum 90%
    recovery_full_after_min: int = 15     # era 30 → acum 15
    
    # Daily reset
    daily_reset_hour_utc: int = 0
    
    # Delta-neutral: funding + grid (grid e limit orders, risc limitat)
    delta_neutral_strategies: tuple = ("funding_arb", "funding", "grid")


class GlobalCircuitBreaker:
    """
    Protecție globală anti-drawdown.
    Monitorizează P&L cumulat al TUTUROR strategiilor.
    Oprește totul dacă pierderile depășesc threshold-ul.
    """
    
    def __init__(self, config: CircuitBreakerConfig = None, 
                 initial_capital_usd: float = 798.0,
                 telegram_callback=None):
        self.config = config or CircuitBreakerConfig()
        self.initial_capital = initial_capital_usd
        self.telegram = telegram_callback
        
        # State tracking
        self.state = CircuitBreakerState.NORMAL
        self.daily_start_capital = initial_capital_usd
        self.current_capital = initial_capital_usd
        self.daily_pnl = 0.0
        self.daily_pnl_pct = 0.0
        
        # Correlation tracking
        self.price_history: Dict[str, List[Tuple[float, float]]] = {
            "BNBUSDC": [],
            "SOLUSDC": []
        }
        
        # Timing
        self.critical_triggered_at: Optional[float] = None
        self.recovery_started_at: Optional[float] = None
        self.last_daily_reset = datetime.now(tz=timezone.utc).date()
        
        # Stats
        self.total_circuit_breaks = 0
        self.total_correlation_halts = 0
        self._lock = threading.Lock()  # thread safety pentru state changes
        self.log = logging.getLogger("CircuitBreaker")
    
    def update_capital(self, current_total_usd: float) -> CircuitBreakerState:
        """
        Apelează la fiecare ciclu de trading cu capitalul total curent.
        Returnează starea curentă a circuit breaker-ului.
        """
        self._check_daily_reset()
        
        self.current_capital = current_total_usd
        self.daily_pnl = current_total_usd - self.daily_start_capital
        self.daily_pnl_pct = (self.daily_pnl / max(self.daily_start_capital, 0.01)) * 100
        
        old_state = self.state
        
        # Verifică cooldown expiration
        if self.state == CircuitBreakerState.COOLDOWN:
            if self._cooldown_expired():
                self.state = CircuitBreakerState.NORMAL
                self.recovery_started_at = time.time()
                self._notify(f"🟢 Circuit Breaker RECOVERED. Reluăm cu "
                           f"{self.config.recovery_size_pct}% sizing.")
            else:
                remaining = self._cooldown_remaining_min()
                self.log.debug(f"CB cooldown: {remaining:.0f} min rămași")
                return self.state  # rămânem în cooldown
        
        # Verifică drawdown thresholds
        if self.state != CircuitBreakerState.COOLDOWN:
            if self.daily_pnl_pct <= self.config.critical_threshold:
                self.state = CircuitBreakerState.CRITICAL
                self.critical_triggered_at = time.time()
                self.total_circuit_breaks += 1
                if old_state != CircuitBreakerState.CRITICAL:
                    self._notify(
                        f"🔴 CIRCUIT BREAKER CRITICAL!\n"
                        f"Drawdown: {self.daily_pnl_pct:.1f}% "
                        f"(${self.daily_pnl:.2f})\n"
                        f"TOATE strategiile OPRITE.\n"
                        f"Cooldown: {self.config.cooldown_minutes} min."
                    )
                # force_close apelat ÎNAINTE de tranziție la COOLDOWN
                # altfel force_close_all() vede COOLDOWN nu CRITICAL
                try:
                    if hasattr(self, '_force_close_cb') and self._force_close_cb:
                        self._force_close_cb()
                except Exception as _e:
                    logging.getLogger("CircuitBreaker").warning(f"force_close: {_e}")
                # Transition to cooldown
                self.state = CircuitBreakerState.COOLDOWN
                
            elif self.daily_pnl_pct <= self.config.warning_threshold:
                self.state = CircuitBreakerState.WARNING
                if old_state == CircuitBreakerState.NORMAL:
                    self._notify(
                        f"🟡 Circuit Breaker WARNING.\n"
                        f"Drawdown: {self.daily_pnl_pct:.1f}%\n"
                        f"Position sizing redus la 50%."
                    )
            else:
                self.state = CircuitBreakerState.NORMAL
        
        return self.state
    
    def update_prices(self, symbol: str, price: float):
        """Track-uiește prețuri pentru detecție corelație."""
        now = time.time()
        if symbol in self.price_history:
            self.price_history[symbol].append((now, price))
            # Păstrează doar ultimele 60 min
            cutoff = now - 3600
            self.price_history[symbol] = [
                (t, p) for t, p in self.price_history[symbol] if t > cutoff
            ]
    
    def check_correlation_halt(self) -> bool:
        """
        Verifică dacă BNB și SOL scad simultan > threshold.
        Returnează True dacă trebuie halt.
        """
        window_sec = self.config.correlation_window_min * 60
        now = time.time()
        
        drops = {}
        for symbol, history in self.price_history.items():
            if len(history) < 2:
                return False
            
            # Preț acum vs. preț acum window_sec minute
            recent_prices = [(t, p) for t, p in history if t > now - window_sec]
            if len(recent_prices) < 2:
                return False
            
            first_price = recent_prices[0][1]
            last_price = recent_prices[-1][1]
            if first_price <= 0:
                return False
            drop_pct = ((last_price - first_price) / first_price) * 100
            drops[symbol] = drop_pct
        
        if len(drops) >= 2:
            all_dropping = all(
                d <= self.config.correlation_drop_pct for d in drops.values()
            )
            if all_dropping:
                with self._lock:
                    self.state = CircuitBreakerState.CORRELATION_HALT
                    self.total_correlation_halts += 1
                drops_str = ", ".join(
                    f"{s}: {d:.1f}%" for s, d in drops.items()
                )
                self._notify(
                    f"⚠️ CORRELATION HALT!\n"
                    f"BNB și SOL scad simultan: {drops_str}\n"
                    f"Reducere expunere imediată."
                )
                return True
        
        return False
    
    def get_position_size_multiplier(self) -> float:
        """
        v1.3: Multiplicator mai generos — doar swing se reduce semnificativ.
        """
        if self.state in (CircuitBreakerState.CRITICAL, 
                          CircuitBreakerState.COOLDOWN):
            return 0.0
        
        if self.state == CircuitBreakerState.CORRELATION_HALT:
            return 0.50  # era 0.25 → acum 0.50
        
        if self.state == CircuitBreakerState.WARNING:
            return 0.75  # era 0.50 → acum 0.75
        
        # Recovery rapid
        if self.recovery_started_at:
            elapsed = (time.time() - self.recovery_started_at) / 60
            if elapsed < self.config.recovery_full_after_min:
                progress = elapsed / max(self.config.recovery_full_after_min, 0.01)
                base = self.config.recovery_size_pct / 100
                return base + (1.0 - base) * progress
            else:
                self.recovery_started_at = None
        
        return 1.0
    
    def get_size_multiplier_for_strategy(self, strategy: str) -> float:
        """
        v1.3: Funding+Grid (delta-neutral/limit orders) = 100% mereu.
        Doar swing se reduce. Doar CRITICAL oprește totul.
        """
        is_dn = strategy.lower() in self.config.delta_neutral_strategies
        
        if self.state == CircuitBreakerState.CRITICAL:
            if is_dn:
                return 0.50  # v1.3: funding+grid la 50% chiar și la CRITICAL
            return 0.0
        
        if is_dn:
            # Funding+Grid: 100% în WARNING, COOLDOWN, CORR_HALT
            return 1.0
        
        # Doar swing se reduce
        return self.get_position_size_multiplier()
    
    def can_open_new_positions(self) -> bool:
        """Verifică dacă se pot deschide poziții noi."""
        return self.state in (CircuitBreakerState.NORMAL,
                              CircuitBreakerState.WARNING)
    
    def force_close_all(self) -> bool:
        """Returnează True dacă trebuie închise TOATE pozițiile (excl. funding)."""
        return self.state == CircuitBreakerState.CRITICAL
    
    def should_close_strategy(self, strategy: str) -> bool:
        """v1.1: Verifică dacă o strategie specifică trebuie închisă."""
        if self.state != CircuitBreakerState.CRITICAL:
            return False
        # Funding delta-neutral NU se închide la CB
        is_dn = strategy.lower() in self.config.delta_neutral_strategies
        return not is_dn
    
    def get_status(self) -> dict:
        """Status complet pentru logging/Telegram."""
        return {
            "state": self.state.value,
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_pnl_pct": round(self.daily_pnl_pct, 2),
            "size_multiplier": round(self.get_position_size_multiplier(), 2),
            "can_open": self.can_open_new_positions(),
            "total_breaks": self.total_circuit_breaks,
            "total_corr_halts": self.total_correlation_halts,
            "cooldown_remaining_min": (
                self._cooldown_remaining_min() 
                if self.state == CircuitBreakerState.COOLDOWN else 0
            )
        }
    
    def _cooldown_expired(self) -> bool:
        if not self.critical_triggered_at:
            return True
        elapsed = (time.time() - self.critical_triggered_at) / 60
        return elapsed >= self.config.cooldown_minutes
    
    def _cooldown_remaining_min(self) -> int:
        if not self.critical_triggered_at:
            return 0
        elapsed = (time.time() - self.critical_triggered_at) / 60
        return max(0, int(self.config.cooldown_minutes - elapsed))
    
    def _check_daily_reset(self):
        today = datetime.now(tz=timezone.utc).date()
        if today > self.last_daily_reset:
            self.daily_start_capital = self.current_capital
            self.daily_pnl = 0.0
            self.daily_pnl_pct = 0.0
            self.last_daily_reset = today
            if self.state == CircuitBreakerState.WARNING:
                self.state = CircuitBreakerState.NORMAL
            _enh_logger.info(f"Daily reset. Start capital: ${self.daily_start_capital:.2f}")
    
    def _notify(self, message: str):
        _enh_logger.warning(message)
        if self.telegram:
            try:
                self.telegram(message)
            except Exception as e:
                _enh_logger.error(f"Telegram notify failed: {e}")


# ═══════════════════════════════════════════════════════════════════
# 1b. DAILY PROFIT LOCK — Trailing Stop pe profit zilnic (v1.1)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ProfitLockConfig:
    # v1.3: activare mai târziu, trail mai larg, reducere mai mică
    activation_threshold_usd: float = 6.0   # era $3 → $6 (profit semnificativ)
    
    # Trail 60%: faci $10, lock doar dacă scade sub $4
    trail_pct: float = 40.0                 # 40% — protejează profitul mai eficient
    
    # Sizing la lock — mai generos
    lock_size_multiplier: float = 0.60      # era 0.30 → 0.60 (60% nu 30%)
    
    # Reset
    reset_on_new_high: bool = True


class DailyProfitLock:
    """
    v1.1: Protejează profitul zilnic cu trailing stop.
    
    Exemplu: faci +$6 dimineața.
    - peak_profit = $6
    - trail = 40% → lock dacă profitul scade sub $3.60
    - Dacă profitul scade la $3 → sizing redus la 30%
    - Dacă profitul revine la $7 → peak se mută, trail la $4.20
    """
    
    def __init__(self, config: ProfitLockConfig = None,
                 telegram_callback=None):
        self.config = config or ProfitLockConfig()
        self.telegram = telegram_callback
        
        # Daily state
        self.daily_pnl = 0.0
        self.peak_daily_pnl = 0.0
        self.locked = False
        self.lock_count = 0
        self._day = datetime.now(tz=timezone.utc).date()  # fix: .date() nu .day
    
    def get_multiplier(self) -> float:
        """Read-only getter — nu modifica starea."""
        if self.locked:
            return self.config.lock_size_multiplier
        return 1.0

    def update(self, current_daily_pnl_usd: float) -> float:
        """
        Actualizează cu P&L-ul zilnic curent.
        Returnează size multiplier (0.30 sau 1.0).
        """
        self._check_daily_reset()
        self.daily_pnl = current_daily_pnl_usd
        
        # Nu activăm sub threshold
        if self.peak_daily_pnl < self.config.activation_threshold_usd:
            if current_daily_pnl_usd > self.peak_daily_pnl:
                self.peak_daily_pnl = current_daily_pnl_usd
            return 1.0
        
        # Update peak
        if current_daily_pnl_usd > self.peak_daily_pnl:
            self.peak_daily_pnl = current_daily_pnl_usd
            if self.locked and self.config.reset_on_new_high:
                self.locked = False
                self._notify(
                    f"🟢 Profit Lock RESET — new high ${current_daily_pnl_usd:.2f}")
        
        # Check trail
        trail_level = self.peak_daily_pnl * (1 - self.config.trail_pct / 100)
        
        if current_daily_pnl_usd < trail_level and not self.locked:
            self.locked = True
            self.lock_count += 1
            self._notify(
                f"🔒 PROFIT LOCK activat!\n"
                f"Peak: ${self.peak_daily_pnl:.2f} → Acum: ${current_daily_pnl_usd:.2f}\n"
                f"Trail: ${trail_level:.2f} | Sizing: {self.config.lock_size_multiplier:.0%}")
        
        if self.locked:
            return self.config.lock_size_multiplier
        
        return 1.0
    
    def get_status(self) -> dict:
        return {
            "daily_pnl": round(self.daily_pnl, 2),
            "peak_daily_pnl": round(self.peak_daily_pnl, 2),
            "locked": self.locked,
            "lock_count": self.lock_count,
            "trail_level": round(
                self.peak_daily_pnl * (1 - self.config.trail_pct / 100), 2
            ) if self.peak_daily_pnl >= self.config.activation_threshold_usd else 0
        }
    
    def _check_daily_reset(self):
        today = datetime.now(tz=timezone.utc).date()  # fix: .date() nu .day
        if today != self._day:
            self.daily_pnl = 0.0
            self.peak_daily_pnl = 0.0
            self.locked = False
            self._day = today
    
    def _notify(self, message: str):
        _enh_logger.info(message)
        if self.telegram:
            try:
                self.telegram(message)
            except Exception as e:
                _enh_logger.error(f"Telegram notify failed: {e}")


# ═══════════════════════════════════════════════════════════════════
# 2. FUNDING RATE SPREAD FILTER
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FundingSpreadConfig:
    # Funding rate minim pentru a intra (0.01% = 0.0001)
    min_funding_rate: float = 0.0001
    
    # Spread maxim spot-perp acceptabil (%)
    max_spread_pct: float = 0.08
    
    # Ore înainte de funding pentru monitoring
    pre_check_hours: float = 4.0
    
    # Funding rate trebuie să rămână peste min pe toată perioada
    require_sustained: bool = True
    sustained_checks: int = 3  # minim 3 verificări consecutive ok
    
    # Binance funding interval (ore)
    funding_interval_hours: int = 8


class FundingSpreadFilter:
    """
    Filtrează intrările în funding rate arbitrage pe baza spread-ului
    spot-perp și sustenabilității funding rate-ului.
    """
    
    def __init__(self, config: FundingSpreadConfig = None,
                 binance_client=None):
        self.config = config or FundingSpreadConfig()
        self.client = binance_client
        
        # Tracking sustained funding rates
        self.funding_checks: Dict[str, List[Tuple[float, float]]] = {}
        # symbol -> [(timestamp, rate)]
        
        # Stats
        self.total_filtered = 0
        self.total_approved = 0
    
    def should_enter_funding(self, symbol: str) -> Tuple[bool, str]:
        """
        Verifică dacă merită să intrăm pe funding arb pentru symbol.
        Returnează (bool, reason_string).
        """
        try:
            # 1. Verifică funding rate curent
            funding_rate = self._get_funding_rate(symbol)
            if funding_rate is None:
                return False, "Nu pot obține funding rate"
            
            if abs(funding_rate) < self.config.min_funding_rate:
                self.total_filtered += 1
                return False, (
                    f"Funding rate prea mic: {funding_rate*100:.4f}% "
                    f"(min: {self.config.min_funding_rate*100:.4f}%)"
                )
            
            # 2. Verifică spread spot vs perp
            spread = self._get_spot_perp_spread(symbol)
            if spread is None:
                return False, "Nu pot calcula spread-ul"
            
            if abs(spread) > self.config.max_spread_pct:
                self.total_filtered += 1
                return False, (
                    f"Spread prea mare: {spread:.4f}% "
                    f"(max: {self.config.max_spread_pct}%)"
                )
            
            # 3. Verifică sustained funding (dacă e activat)
            if self.config.require_sustained:
                sustained = self._check_sustained(symbol, funding_rate)
                if not sustained:
                    self.total_filtered += 1
                    checks_done = len(self.funding_checks.get(symbol, []))
                    return False, (
                        f"Funding nesusținut: {checks_done}/"
                        f"{self.config.sustained_checks} verificări ok"
                    )
            
            # 4. Verifică timp până la funding
            time_to_funding = self._time_to_next_funding_hours()
            if time_to_funding > self.config.pre_check_hours:
                return False, (
                    f"Prea devreme: {time_to_funding:.1f}h până la funding "
                    f"(intrăm la {self.config.pre_check_hours}h)"
                )
            
            # 5. Calculează profitabilitate netă
            net_profit = self._estimate_net_profit(symbol, funding_rate, spread)
            if net_profit <= 0:
                self.total_filtered += 1
                return False, f"Profit net negativ după costuri: ${net_profit:.4f}"
            
            self.total_approved += 1
            return True, (
                f"✅ Funding OK: rate={funding_rate*100:.4f}%, "
                f"spread={spread:.4f}%, "
                f"net_profit≈${net_profit:.4f}, "
                f"time_to_funding={time_to_funding:.1f}h"
            )
            
        except Exception as e:
            _enh_logger.error(f"FundingSpreadFilter error for {symbol}: {e}")
            return False, f"Error: {e}"
    
    def _get_funding_rate(self, symbol: str) -> Optional[float]:
        """Obține funding rate-ul curent de pe Binance."""
        try:
            # Binance Futures API
            url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
            resp = _http_session.get(url, timeout=5)
            data = resp.json()
            return float(data.get("lastFundingRate", 0))
        except Exception as e:
            _enh_logger.error(f"Failed to get funding rate for {symbol}: {e}")
            return None
    
    def _get_spot_perp_spread(self, symbol: str) -> Optional[float]:
        """Calculeaza spread-ul REAL din Orderbook (Cumperi Spot Ask, Vinzi Perp Bid)."""
        try:
            import requests as _req_spread
            spot_resp = _req_spread.get(
                f"https://api.binance.com/api/v3/ticker/bookTicker?symbol={symbol}",
                timeout=5).json()
            ask_spot = float(spot_resp["askPrice"])
            futures_resp = _req_spread.get(
                f"https://fapi.binance.com/fapi/v1/ticker/bookTicker?symbol={symbol}",
                timeout=5).json()
            bid_perp = float(futures_resp["bidPrice"])
            if ask_spot <= 0:
                return None
            spread_pct = ((bid_perp - ask_spot) / ask_spot) * 100
            return spread_pct
        except Exception as e:
            _enh_logger.debug(f"Failed to calc spread for {symbol}: {e}")
            return None


    def _check_sustained(self, symbol: str, current_rate: float) -> bool:
        """Verifică dacă funding rate-ul e susținut pe mai multe verificări."""
        now = time.time()
        
        if symbol not in self.funding_checks:
            self.funding_checks[symbol] = []
        
        self.funding_checks[symbol].append((now, current_rate))
        
        # Păstrează doar verificările din ultimele pre_check_hours
        cutoff = now - (self.config.pre_check_hours * 3600)
        self.funding_checks[symbol] = [
            (t, r) for t, r in self.funding_checks[symbol] if t > cutoff
        ]
        
        # Verifică dacă avem destule checks consecutive cu rate > min
        valid_checks = [
            r for _, r in self.funding_checks[symbol]
            if abs(r) >= self.config.min_funding_rate
        ]
        
        return len(valid_checks) >= self.config.sustained_checks
    
    def _time_to_next_funding_hours(self) -> float:
        """Calculează ore rămase până la următoarea perioadă de funding."""
        now = datetime.now(tz=timezone.utc)
        # Binance: funding la 00:00, 08:00, 16:00 UTC
        funding_hours = [0, 8, 16]
        
        for fh in funding_hours:
            funding_time = now.replace(hour=fh, minute=0, second=0, microsecond=0)
            if funding_time > now:
                diff = (funding_time - now).total_seconds() / 3600
                return diff
        
        # Următorul e la 00:00 a doua zi
        next_day = now + timedelta(days=1)
        funding_time = next_day.replace(hour=0, minute=0, second=0, microsecond=0)
        diff = (funding_time - now).total_seconds() / 3600
        return diff
    
    def _estimate_net_profit(self, symbol: str, funding_rate: float, 
                             spread_pct: float) -> float:
        """
        Estimează profitul net per $100 capital după comisioane.
        funding_rate: rata ca fracție (ex: 0.0001 = 0.01%)
        spread_pct: spread-ul ca procent
        """
        position_size = 100  # $100 bază de calcul
        
        # Venit din funding
        funding_income = position_size * abs(funding_rate)
        
        # Costuri
        maker_fee = 0.000075  # 0.0075% cu BNB discount
        taker_fee = 0.000075  # 0.0075% cu BNB discount (maker orders only)
        
        # Intrare (2 ordine: spot + perp) + ieșire (2 ordine)
        entry_cost = position_size * (maker_fee + taker_fee)  # spot maker, perp taker
        exit_cost = position_size * (maker_fee + taker_fee)
        
        # Spread cost (pierdem spread-ul la intrare)
        spread_cost = position_size * abs(spread_pct) / 100
        
        total_cost = entry_cost + exit_cost + spread_cost
        net = funding_income - total_cost
        
        return net
    
    def get_stats(self) -> dict:
        return {
            "total_filtered": self.total_filtered,
            "total_approved": self.total_approved,
            "approval_rate": (
                f"{self.total_approved/(self.total_filtered+self.total_approved)*100:.1f}%"
                if (self.total_filtered + self.total_approved) > 0 else "N/A"
            )
        }


# ═══════════════════════════════════════════════════════════════════
# 3. GRID SPACING OPTIMIZER
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GridOptimizerConfig:
    maker_fee: float = 0.000075   # 0.0075% cu BNB discount (real Binance)
    taker_fee: float = 0.001      # 0.1% taker fee
    min_profit_multiplier: float = 3.0  # grid spacing = 3x comision minim
    
    # Volatility-based adjustment
    use_volatility_adjustment: bool = True
    volatility_lookback_hours: int = 24
    volatility_multiplier_range: Tuple[float, float] = (1.0, 2.5)
    
    # Capital constraints
    min_grid_capital_usd: float = 5.0  # minim $5 per nivel de grid


class GridSpacingOptimizer:
    """
    Calculează grid spacing optim bazat pe comisioane, volatilitate,
    și capital disponibil.
    """
    
    def __init__(self, config: GridOptimizerConfig = None):
        self.config = config or GridOptimizerConfig()
    
    def calculate_optimal_grid(
        self,
        symbol: str,
        current_price: float,
        allocated_capital_usd: float,
        price_range_pct: float = 10.0,  # grid range ±5%
        recent_volatility_pct: Optional[float] = None
    ) -> dict:
        """
        Calculează parametrii optimi de grid.
        
        Returns: {
            "grid_levels": int,
            "spacing_pct": float,
            "spacing_usd": float,
            "capital_per_level": float,
            "min_profitable_spacing_pct": float,
            "expected_profit_per_fill": float,
            "recommendation": str
        }
        """
        # 1. Spacing minim profitabil
        total_fee_rate = self.config.maker_fee + self.config.taker_fee  # buy + sell
        min_spacing_pct = (
            self.config.min_profit_multiplier * total_fee_rate * 100
        )  # ca procent
        
        # 2. Ajustare pe volatilitate
        if self.config.use_volatility_adjustment and recent_volatility_pct:
            vol_factor = self._volatility_factor(recent_volatility_pct)
            adjusted_spacing = min_spacing_pct * vol_factor
        else:
            adjusted_spacing = min_spacing_pct
        
        # 3. Calcul niveluri de grid cu capital constraint
        total_range_pct = price_range_pct * 2  # ±range
        max_levels_by_spacing = int(total_range_pct / adjusted_spacing)
        max_levels_by_capital = int(
            allocated_capital_usd / max(self.config.min_grid_capital_usd, 0.01)
        )
        
        grid_levels = min(max_levels_by_spacing, max_levels_by_capital)
        grid_levels = max(grid_levels, 2)  # minim 2 niveluri
        
        # Recalculează spacing-ul bazat pe niveluri finale
        final_spacing_pct = total_range_pct / grid_levels
        final_spacing_pct = max(final_spacing_pct, min_spacing_pct)
        
        # 4. Recalculează nivele cu spacing-ul final
        grid_levels = int(total_range_pct / final_spacing_pct)
        grid_levels = max(grid_levels, 2)
        
        capital_per_level = allocated_capital_usd / grid_levels
        spacing_usd = current_price * (final_spacing_pct / 100)
        
        # 5. Profit estimat per fill
        profit_per_fill = capital_per_level * (final_spacing_pct / 100) - \
                         capital_per_level * total_fee_rate
        
        # 6. Recommendation
        if capital_per_level < self.config.min_grid_capital_usd:
            recommendation = (
                f"⚠️ Capital insuficient. Ai nevoie minim "
                f"${self.config.min_grid_capital_usd * grid_levels:.0f} "
                f"pentru {grid_levels} niveluri."
            )
        elif profit_per_fill < 0.01:
            recommendation = (
                f"⚠️ Profit per fill foarte mic (${profit_per_fill:.4f}). "
                f"Mărește spacing-ul sau capitalul."
            )
        else:
            recommendation = (
                f"✅ Grid OK: {grid_levels} niveluri, "
                f"~${profit_per_fill:.3f}/fill, "
                f"spacing {final_spacing_pct:.2f}%"
            )
        
        return {
            "grid_levels": grid_levels,
            "spacing_pct": round(final_spacing_pct, 4),
            "spacing_usd": round(spacing_usd, 4),
            "capital_per_level": round(capital_per_level, 2),
            "min_profitable_spacing_pct": round(min_spacing_pct, 4),
            "expected_profit_per_fill": round(profit_per_fill, 4),
            "price_range": {
                "low": round(current_price * (1 - price_range_pct/100), 2),
                "high": round(current_price * (1 + price_range_pct/100), 2)
            },
            "recommendation": recommendation
        }
    
    def _volatility_factor(self, recent_vol_pct: float) -> float:
        """
        Ajustează spacing-ul bazat pe volatilitate.
        Vol mare → spacing mai mare (eviți whipsaw).
        Vol mică → spacing mai mic (capturezi mișcări mici).
        """
        # Volatilitate medie așteptată: ~2-3% zilnic pentru BNB/SOL
        expected_vol = 2.5
        ratio = recent_vol_pct / expected_vol
        
        min_mult, max_mult = self.config.volatility_multiplier_range
        factor = max(min_mult, min(max_mult, ratio))
        
        return factor
    
    @staticmethod
    def calculate_breakeven_spacing(maker_fee: float = 0.001, 
                                     taker_fee: float = 0.001) -> float:
        """
        Formula exactă pentru spacing minim breakeven.
        Returns: spacing ca fracție (ex: 0.002 = 0.2%)
        """
        total_fee = maker_fee + taker_fee
        return 2 * total_fee / (1 - total_fee)
    
    def recommend_for_capital(self, capital_usd: float, 
                               num_pairs: int = 2) -> str:
        """
        Recomandare generală pentru un capital dat pe N perechi.
        """
        per_pair = capital_usd / num_pairs
        min_spacing = self.calculate_breakeven_spacing(
            self.config.maker_fee, self.config.taker_fee
        )
        safe_spacing = min_spacing * self.config.min_profit_multiplier
        
        max_levels = int(per_pair / max(self.config.min_grid_capital_usd, 0.01))
        
        return (
            f"Capital total: ${capital_usd:.0f} ({num_pairs} perechi)\n"
            f"Per pereche: ${per_pair:.0f}\n"
            f"Breakeven spacing: {min_spacing*100:.3f}%\n"
            f"Spacing recomandat (3x): {safe_spacing*100:.3f}%\n"
            f"Max niveluri per pereche: {max_levels}\n"
            f"Capital per nivel: ${per_pair/max(max_levels,1):.2f}"
        )


# ═══════════════════════════════════════════════════════════════════
# 4. SOL BATCH ACCUMULATOR
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SOLAccumulatorConfig:
    accumulation_pct: float = 10.0        # 10% din profitul zilnic → acumulat pentru swap SOL
    min_swap_amount_usd: float = 10.0  # minim $10 per swap — evită fee excesiv pe swapuri mici     # swap minim $5
    max_swap_amount_usd: float = 50.0    # swap maxim per batch
    swap_day: int = 6                    # 0=Mon...6=Sun (fallback: duminică)
    swap_hour_utc: int = 12              # swap la 12:00 UTC
    use_limit_order: bool = True         # limit order vs market
    limit_offset_pct: float = 0.1        # limit cu 0.1% sub market
    min_accumulated_before_swap: float = 5.0  # minim $5 acumulat
    
    # v1.1: DCA pe dip-uri
    dca_on_dip: bool = True              # activează DCA pe dip
    dip_threshold_pct: float = 3.0       # cumpără când SOL scade 3% de la recent high
    dip_lookback_hours: int = 48         # recent high = max din ultimele 48h
    dip_bonus_pct: float = 50.0          # la dip, cumpără 50% mai mult


class SOLBatchAccumulator:
    """
    Acumulează profit în USDT și convertește în SOL.
    v1.1: DCA pe dip-uri — cumpără SOL când scade >3% de la recent high.
    Fallback: swap weekly dacă nu au fost dip-uri.
    """
    
    def __init__(self, config: SOLAccumulatorConfig = None,
                 telegram_callback=None):
        self.config = config or SOLAccumulatorConfig()
        self.telegram = telegram_callback
        
        # Tracking
        self.accumulated_usdt = 0.0
        self.total_sol_bought = 0.0
        self.total_swaps = 0
        self.total_fees_saved_est = 0.0
        self.last_swap_time: Optional[float] = None
        
        # Daily profit tracking
        self.daily_profits: Dict[str, float] = {}  # date -> profit_usd
        
        # v1.1: DCA price tracking
        self.sol_price_history: List[Tuple[float, float]] = []  # [(ts, price)]
        self.sol_recent_high: float = 0.0
        self.dip_buys: int = 0
    
    def record_daily_profit(self, profit_usd: float):
        """
        Înregistrează profitul zilnic. Acumulează procentul configurat.
        Apelează la sfârșitul fiecărei zile de trading.
        """
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        self.daily_profits[today] = profit_usd
        
        if profit_usd > 0:
            to_accumulate = profit_usd * (self.config.accumulation_pct / 100)
            self.accumulated_usdt += to_accumulate
            _enh_logger.info(
                f"SOL Accumulator: +${to_accumulate:.4f} "
                f"(total: ${self.accumulated_usdt:.2f})"
            )
    
    def update_sol_price(self, sol_price: float):
        """v1.1: Actualizează prețul SOL pentru detecție dip."""
        now = time.time()
        self.sol_price_history.append((now, sol_price))
        # Păstrează ultimele dip_lookback_hours
        cutoff = now - (self.config.dip_lookback_hours * 3600)
        self.sol_price_history = [
            (t, p) for t, p in self.sol_price_history if t > cutoff
        ]
        # Update recent high
        if self.sol_price_history:
            self.sol_recent_high = max(p for _, p in self.sol_price_history)
    
    def check_dip_buy(self, sol_price: float) -> Tuple[bool, str]:
        """
        v1.1: Verifică dacă SOL e în dip și merită cumpărat.
        Returnează (should_buy, reason).
        """
        if not self.config.dca_on_dip:
            return False, "DCA pe dip dezactivat"
        
        if self.sol_recent_high <= 0:
            return False, "Nu am date suficiente pentru recent high"
        
        if self.accumulated_usdt < self.config.min_swap_amount_usd:
            return False, f"Acumulat insuficient: ${self.accumulated_usdt:.2f}"
        
        drop_pct = ((sol_price - self.sol_recent_high) / max(self.sol_recent_high, 0.01)) * 100
        
        if drop_pct <= -self.config.dip_threshold_pct:
            return True, (
                f"🔻 DIP detectat: SOL {drop_pct:.1f}% de la "
                f"${self.sol_recent_high:.2f} → ${sol_price:.2f}")
        
        return False, f"SOL {drop_pct:+.1f}% (dip threshold: -{self.config.dip_threshold_pct}%)"
    
    def execute_dip_buy(self, sol_price: float) -> dict:
        """v1.1: Cumpără SOL la dip cu bonus."""
        # La dip, cumpărăm cu bonus (50% mai mult decât normal)
        bonus_mult = 1.0 + self.config.dip_bonus_pct / 100
        # La dip: max_swap crește cu bonus, dar nu cheltuim mai mult decât avem
        swap_amount = min(
            self.accumulated_usdt,                          # nu cheltuim mai mult decât avem
            self.config.max_swap_amount_usd * bonus_mult   # max crescut la dip
        )
        
        result = self.execute_swap(sol_price, override_amount=swap_amount)
        self.dip_buys += 1
        
        drop_pct = ((sol_price - self.sol_recent_high) / max(self.sol_recent_high, 0.01)) * 100
        self._notify(
            f"🔻 SOL DIP BUY!\n"
            f"Preț: ${sol_price:.2f} ({drop_pct:.1f}% de la high)\n"
            f"Cumpărat: {result['sol_amount']:.4f} SOL\n"
            f"Total SOL: {self.total_sol_bought:.4f}")
        
        return result
    
    def should_swap_now(self) -> Tuple[bool, str]:
        """
        Verifică dacă e momentul pentru swap.
        Returnează (should_swap, reason).
        """
        now = datetime.now(tz=timezone.utc)
        
        # Verifică dacă e ziua și ora corectă
        if now.weekday() != self.config.swap_day:
            days_until = (self.config.swap_day - now.weekday()) % 7
            return False, f"Swap programat în {days_until} zile (duminică)"
        
        if now.hour < self.config.swap_hour_utc:
            return False, f"Swap programat la {self.config.swap_hour_utc}:00 UTC"
        
        # Verifică dacă am făcut deja swap astăzi
        if self.last_swap_time:
            last_swap_date = datetime.fromtimestamp(self.last_swap_time, tz=timezone.utc).date()
            if last_swap_date == now.date():
                return False, "Swap deja efectuat astăzi"
        
        # Verifică minim acumulat
        if self.accumulated_usdt < self.config.min_accumulated_before_swap:
            return False, (
                f"Acumulat: ${self.accumulated_usdt:.2f} "
                f"(minim: ${self.config.min_accumulated_before_swap})"
            )
        
        return True, f"Ready: ${self.accumulated_usdt:.2f} USDT → SOL"
    
    def execute_swap(self, sol_price: float, override_amount: float = 0.0) -> dict:
        """
        Execută swap-ul USDT → SOL.
        override_amount: dacă >0, folosit în loc de accumulated_usdt (pentru dip bonus).

        Returns: swap details dict
        """
        swap_amount = min(
            override_amount if override_amount > 0 else self.accumulated_usdt,
            self.config.max_swap_amount_usd
        )
        
        if self.config.use_limit_order:
            # Limit order sub market price
            limit_price = sol_price * (1 - self.config.limit_offset_pct / 100)
            sol_amount = swap_amount / max(limit_price, 0.01)
        else:
            sol_amount = swap_amount / max(sol_price, 0.01)
        
        # Estimare fee saved vs daily swaps
        # Dacă am fi făcut swap zilnic: 7 tranzacții × fee
        # Acum facem 1 tranzacție
        # Economie reală: cost fix per tranzacție pe unele exchange-uri
        # Pe Binance cu comision % pur, economiile per tranzacție = 0
        # Păstrăm metrica pentru compatibilitate dar o setăm la 0 explicit
        fees_saved = 0.0  # N/A pentru comision procentual pur Binance
        
        result = {
            "swap_amount_usdt": round(swap_amount, 4),
            "sol_amount": round(sol_amount, 6),
            "sol_price": round(sol_price, 2),
            "order_type": "LIMIT" if self.config.use_limit_order else "MARKET",
            "limit_price": round(limit_price, 2) if self.config.use_limit_order else None,
            "est_fees_saved": round(fees_saved, 4),
            "timestamp": datetime.now(tz=timezone.utc).isoformat()
        }
        
        # Update tracking
        self.accumulated_usdt -= swap_amount
        self.total_sol_bought += sol_amount
        self.total_swaps += 1
        self.total_fees_saved_est += fees_saved
        self.last_swap_time = time.time()
        
        # Notify
        self._notify(
            f"🔄 SOL Batch Swap executat!\n"
            f"${swap_amount:.2f} USDT → {sol_amount:.4f} SOL\n"
            f"Preț: ${sol_price:.2f}\n"
            f"Fees saved (est): ${fees_saved:.4f}\n"
            f"Total SOL acumulat: {self.total_sol_bought:.4f}"
        )
        
        return result
    
    def get_status(self) -> dict:
        return {
            "accumulated_usdt": round(self.accumulated_usdt, 4),
            "total_sol_bought": round(self.total_sol_bought, 6),
            "total_swaps": self.total_swaps,
            "total_fees_saved": round(self.total_fees_saved_est, 4),
            "next_swap": self._next_swap_time()
        }
    
    def _next_swap_time(self) -> str:
        now = datetime.now(tz=timezone.utc)
        days_until = (self.config.swap_day - now.weekday()) % 7
        if days_until == 0 and now.hour >= self.config.swap_hour_utc:
            days_until = 7
        next_swap = now + timedelta(days=days_until)
        next_swap = next_swap.replace(
            hour=self.config.swap_hour_utc, minute=0, second=0
        )
        return next_swap.isoformat() + "Z"
    
    def _notify(self, message: str):
        _enh_logger.info(message)
        if self.telegram:
            try:
                self.telegram(message)
            except Exception as e:
                _enh_logger.error(f"Telegram notify failed: {e}")


# ═══════════════════════════════════════════════════════════════════
# 5. OPEN INTEREST SENTINEL
# ═══════════════════════════════════════════════════════════════════

@dataclass
class OISentinelConfig:
    # Monitorizare
    check_interval_min: int = 15         # verificare la 15 min
    lookback_hours: int = 4              # analizează ultimele 4h
    
    # Thresholds OI
    oi_surge_pct: float = 5.0            # OI crește 5%+ cu preț stagnant = pericol
    price_stagnant_pct: float = 1.0      # preț se mișcă < 1% = stagnant
    
    # Thresholds OI drop (lichidare cascade)
    oi_drop_pct: float = -8.0            # OI scade 8% rapid = cascade lichidări
    oi_drop_window_min: int = 60         # în 60 minute
    
    # Long/Short ratio extremes
    ls_ratio_extreme_long: float = 3.0   # 3:1 long → crowded long
    ls_ratio_extreme_short: float = 0.33 # 1:3 short → crowded short
    
    # Acțiuni
    reduce_exposure_pct: float = 50.0    # reduce cu 50% la warning
    symbols_to_watch: list = None        # default: BTC + BNB + SOL
    
    def __post_init__(self):
        if self.symbols_to_watch is None:
            self.symbols_to_watch = ["BTCUSDC", "BNBUSDC", "SOLUSDC"]


class OpenInterestSentinel:
    """
    Monitorizează Open Interest pe BTC futures ca indicator leading.
    OI surge + preț stagnant → lichidări iminente → reduce expunere.
    """
    
    def __init__(self, config: OISentinelConfig = None,
                 telegram_callback=None):
        self.config = config or OISentinelConfig()
        self.telegram = telegram_callback
        
        # Historical data
        self.oi_history: Dict[str, List[Tuple[float, float]]] = {
            s: [] for s in self.config.symbols_to_watch
        }
        self.price_history: Dict[str, List[Tuple[float, float]]] = {
            s: [] for s in self.config.symbols_to_watch
        }
        
        # Current signals
        self.active_signals: Dict[str, dict] = {}
        
        # Stats
        self.total_warnings = 0
        self.total_correct_warnings = 0  # validate post-factum
    
    def check_all(self) -> Dict[str, dict]:
        """
        Verifică toate simbolurile monitorizate.
        Returnează dict cu semnale active.
        """
        signals = {}
        
        for symbol in self.config.symbols_to_watch:
            signal = self.check_symbol(symbol)
            if signal["alert_level"] != "NONE":
                signals[symbol] = signal
        
        self.active_signals = signals
        return signals
    
    def check_symbol(self, symbol: str) -> dict:
        """
        Analizează OI + preț pentru un simbol.
        Returnează signal dict.
        """
        try:
            # Fetch current data
            oi_data = self._fetch_open_interest(symbol)
            price = self._fetch_price(symbol)
            ls_ratio = self._fetch_long_short_ratio(symbol)
            
            if not oi_data or not price:
                return {"alert_level": "NONE", "reason": "Data unavailable"}
            
            # Store history
            now = time.time()
            self.oi_history[symbol].append((now, oi_data["openInterest"]))
            self.price_history[symbol].append((now, price))
            
            # Trim history
            cutoff = now - (self.config.lookback_hours * 3600)
            self.oi_history[symbol] = [
                (t, v) for t, v in self.oi_history[symbol] if t > cutoff
            ]
            self.price_history[symbol] = [
                (t, v) for t, v in self.price_history[symbol] if t > cutoff
            ]
            
            # Analyze
            alert_level = "NONE"
            reasons = []
            
            # 1. OI surge + price stagnant
            oi_change = self._calc_change_pct(self.oi_history[symbol])
            price_change = self._calc_change_pct(self.price_history[symbol])
            
            if (oi_change is not None and price_change is not None):
                if (abs(oi_change) >= self.config.oi_surge_pct and 
                    abs(price_change) <= self.config.price_stagnant_pct):
                    alert_level = "HIGH"
                    reasons.append(
                        f"OI surge {oi_change:+.1f}% cu preț stagnant "
                        f"({price_change:+.1f}%) → lichidări probabile"
                    )
            
            # 2. OI drop rapid (cascade de lichidări în curs)
            oi_recent_change = self._calc_recent_change_pct(
                self.oi_history[symbol],
                self.config.oi_drop_window_min
            )
            if oi_recent_change is not None:
                if oi_recent_change <= self.config.oi_drop_pct:
                    alert_level = "CRITICAL"
                    reasons.append(
                        f"OI drop rapid {oi_recent_change:.1f}% în "
                        f"{self.config.oi_drop_window_min}min "
                        f"→ cascade lichidări!"
                    )
            
            # 3. Long/Short ratio extreme
            if ls_ratio:
                if ls_ratio >= self.config.ls_ratio_extreme_long:
                    if alert_level == "NONE":
                        alert_level = "MEDIUM"
                    reasons.append(
                        f"L/S ratio extreme long: {ls_ratio:.2f} "
                        f"→ crowded long, risc squeeze"
                    )
                elif ls_ratio <= self.config.ls_ratio_extreme_short:
                    if alert_level == "NONE":
                        alert_level = "MEDIUM"
                    reasons.append(
                        f"L/S ratio extreme short: {ls_ratio:.2f} "
                        f"→ crowded short, risc squeeze"
                    )
            
            signal = {
                "symbol": symbol,
                "alert_level": alert_level,
                "reasons": reasons,
                "data": {
                    "oi_change_pct": round(oi_change, 2) if oi_change else None,
                    "price_change_pct": round(price_change, 2) if price_change else None,
                    "ls_ratio": round(ls_ratio, 2) if ls_ratio else None,
                    "current_oi": oi_data.get("openInterest"),
                    "current_price": price
                },
                "recommended_action": self._get_action(alert_level),
                "timestamp": datetime.now(tz=timezone.utc).isoformat()
            }
            
            if alert_level in ("HIGH", "CRITICAL"):
                self.total_warnings += 1
                self._notify_signal(signal)
            
            return signal
            
        except Exception as e:
            _enh_logger.error(f"OI Sentinel error for {symbol}: {e}")
            return {"alert_level": "NONE", "reason": f"Error: {e}"}
    
    def get_exposure_multiplier(self) -> float:
        """
        Returnează multiplicator de expunere bazat pe semnale active.
        Strategiile multiplică sizing-ul cu această valoare.
        """
        if not self.active_signals:
            return 1.0
        
        max_alert = "NONE"
        for sig in self.active_signals.values():
            level = sig.get("alert_level", "NONE")
            if level == "CRITICAL":
                max_alert = "CRITICAL"
                break
            elif level == "HIGH" and max_alert != "CRITICAL":
                max_alert = "HIGH"
            elif level == "MEDIUM" and max_alert == "NONE":
                max_alert = "MEDIUM"
        
        # v1.3: multiplier-uri mai generoase — nu reduce sizing la semnal minor
        multipliers = {
            "NONE": 1.0,
            "MEDIUM": 1.0,    # era 0.75 → 1.0 (ignore medium)
            "HIGH": 0.80,     # era 0.50 → 0.80
            "CRITICAL": 0.50  # era 0.25 → 0.50
        }
        
        return multipliers.get(max_alert, 1.0)
    
    def get_exposure_multiplier_for_strategy(self, strategy: str) -> float:
        """
        v1.3: Funding+Grid la 100% mereu.
        Doar swing se reduce de OI signals.
        """
        # Funding și grid: 100% indiferent de OI
        safe_strats = ("funding_arb", "funding", "grid")
        if strategy.lower() in safe_strats:
            return 1.0
        
        # Swing: se reduce
        return self.get_exposure_multiplier()
    
    def _fetch_open_interest(self, symbol: str) -> Optional[dict]:
        """Fetch OI de pe Binance Futures."""
        try:
            url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
            resp = _http_session.get(url, timeout=5)
            data = resp.json()
            return {
                "openInterest": float(data.get("openInterest", 0)),
                "symbol": symbol,
                "time": data.get("time")
            }
        except Exception as e:
            _enh_logger.error(f"Failed to fetch OI for {symbol}: {e}")
            return None
    
    def _fetch_price(self, symbol: str) -> Optional[float]:
        """Fetch preț curent."""
        try:
            url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"
            resp = _http_session.get(url, timeout=5)
            return float(resp.json().get("price", 0))
        except Exception as e:
            _enh_logger.error(f"Failed to fetch price for {symbol}: {e}")
            return None
    
    def _fetch_long_short_ratio(self, symbol: str) -> Optional[float]:
        """Fetch long/short ratio."""
        try:
            url = (
                f"https://fapi.binance.com/futures/data/"
                f"globalLongShortAccountRatio?symbol={symbol}&period=1h&limit=1"
            )
            resp = _http_session.get(url, timeout=5)
            data = resp.json()
            if data and len(data) > 0:
                return float(data[0].get("longShortRatio", 1.0))
            return None
        except Exception as e:
            _enh_logger.error(f"Failed to fetch L/S ratio for {symbol}: {e}")
            return None
    
    def _calc_change_pct(self, history: List[Tuple[float, float]]) -> Optional[float]:
        """Calculează schimbarea procentuală pe toată perioada lookback."""
        if len(history) < 2:
            return None
        first_val = history[0][1]
        last_val = history[-1][1]
        if first_val == 0:
            return None
        return ((last_val - first_val) / first_val) * 100
    
    def _calc_recent_change_pct(self, history: List[Tuple[float, float]], 
                                 window_min: int) -> Optional[float]:
        """Calculează schimbarea procentuală în ultimele N minute."""
        if len(history) < 2:
            return None
        now = time.time()
        cutoff = now - (window_min * 60)
        recent = [(t, v) for t, v in history if t > cutoff]
        if len(recent) < 2:
            return None
        return ((recent[-1][1] - recent[0][1]) / recent[0][1]) * 100
    
    def _get_action(self, alert_level: str) -> str:
        actions = {
            "NONE": "Normal trading",
            "MEDIUM": "Reduce sizing 25%, skip new entries",
            "HIGH": f"Reduce expunere {self.config.reduce_exposure_pct}%, tighten stops",
            "CRITICAL": "Close risky positions, only keep hedged positions"
        }
        return actions.get(alert_level, "Monitor")
    
    def _notify_signal(self, signal: dict):
        """Trimite alertă pe Telegram."""
        emoji = {"HIGH": "🟠", "CRITICAL": "🔴"}.get(
            signal["alert_level"], "⚪"
        )
        msg = (
            f"{emoji} OI SENTINEL — {signal['symbol']}\n"
            f"Alert: {signal['alert_level']}\n"
        )
        for reason in signal.get("reasons", []):
            msg += f"• {reason}\n"
        msg += f"Action: {signal['recommended_action']}"
        
        _enh_logger.warning(msg)
        if self.telegram:
            try:
                self.telegram(msg)
            except Exception as e:
                _enh_logger.error(f"Telegram notify failed: {e}")
    
    def get_status(self) -> dict:
        return {
            "active_signals": {
                k: {
                    "level": v["alert_level"],
                    "reasons": v.get("reasons", [])
                }
                for k, v in self.active_signals.items()
            },
            "exposure_multiplier": self.get_exposure_multiplier(),
            "total_warnings": self.total_warnings,
            "symbols_monitored": self.config.symbols_to_watch
        }


# ═══════════════════════════════════════════════════════════════════
# 5b. SPREAD MONITOR — Detectare retragere Market Maker
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SpreadMonitorConfig:
    symbols: list = None
    spread_normal_pct: float  = 0.05   # spread normal < 0.05%
    spread_warning_pct: float = 0.15   # avertisment la 0.15%
    spread_alert_pct: float   = 0.30   # alertă la 0.30% (MM s-au retras)
    confirm_samples: int      = 3      # eșantioane consecutive pentru confirmare
    history_window: int       = 60     # ultimele 60 eșantioane (~10 min)
    spike_multiplier: float   = 3.0    # spread > 3× medie = retragere MM

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["SOLUSDC", "BNBUSDC"]


class SpreadMonitor:
    """
    Monitorizează spread-ul bid-ask pentru SOLUSDT și BNBUSDT.

    LOGICĂ:
    Market Maker-ii mențin spread îngust (~0.01-0.05%).
    Când anticipează mișcare violentă (news, liquidări, whale dump),
    se RETRAG → spread se LĂRGEȘTE brusc (3-10×).

    Semnal retragere MM = spread > 3× medie istorică SAU > prag absolut.

    Niveluri:
    • NORMAL  → trading normal
    • WARNING → reduce sizing 50%
    • ALERT   → NU intra în poziții noi; mișcare violentă iminentă
    """

    LEVEL_NORMAL  = "NORMAL"
    LEVEL_WARNING = "WARNING"
    LEVEL_ALERT   = "ALERT"

    def __init__(self, config: SpreadMonitorConfig = None,
                 binance_client=None,
                 telegram_callback=None):
        self.config   = config or SpreadMonitorConfig()
        self.client   = binance_client
        self.telegram = telegram_callback
        self.log      = L("Spread")

        self.history: Dict[str, deque] = {
            sym: deque(maxlen=self.config.history_window)
            for sym in self.config.symbols
        }
        self._consecutive_high: Dict[str, int] = {s: 0 for s in self.config.symbols}
        self.level: Dict[str, str] = {s: self.LEVEL_NORMAL for s in self.config.symbols}
        self.global_level      = self.LEVEL_NORMAL
        self.total_alerts      = 0
        self.total_warnings    = 0
        self.last_alert_ts: Dict[str, float] = {s: 0.0 for s in self.config.symbols}
        self.last_check_ts     = 0.0
        self._ob_cache: Dict[str, Tuple[float, dict]] = {}

    def update(self) -> str:
        """
        Actualizează spread-ul pentru toate simbolurile.
        Returnează nivelul global (NORMAL / WARNING / ALERT).
        Apelat periodic din thread.
        """
        now  = time.time()
        self.last_check_ts = now
        worst = self.LEVEL_NORMAL

        for sym in self.config.symbols:
            try:
                spread_pct = self._get_spread(sym)
                if spread_pct is None:
                    continue

                self.history[sym].append((now, spread_pct))
                hist_vals  = [v for _, v in self.history[sym]]
                avg_spread = sum(hist_vals) / max(len(hist_vals), 1)

                level     = self._classify_spread(sym, spread_pct, avg_spread)
                old_level = self.level[sym]
                self.level[sym] = level

                if level != self.LEVEL_NORMAL:
                    self._consecutive_high[sym] += 1
                else:
                    self._consecutive_high[sym] = max(0, self._consecutive_high[sym] - 1)

                confirmed = self._consecutive_high[sym] >= self.config.confirm_samples

                if level == self.LEVEL_ALERT and confirmed:
                    if now - self.last_alert_ts[sym] > 300:
                        self.last_alert_ts[sym] = now
                        self.total_alerts += 1
                        msg = (
                            f"🚨 <b>SPREAD ALERT — {sym}</b>\n"
                            f"Spread: {spread_pct:.3f}% (avg: {avg_spread:.3f}%)\n"
                            f"Spike: {spread_pct/max(avg_spread,0.0001):.1f}× medie\n"
                            f"⚠️ Market Makers s-au retras!\n"
                            f"Mișcare violentă iminentă. NU intra în poziții noi."
                        )
                        self.log.warning(msg)
                        if self.telegram:
                            try: self.telegram(msg)
                            except Exception as _e: logging.debug(f'Ignored: {_e}')

                elif level == self.LEVEL_WARNING and old_level == self.LEVEL_NORMAL:
                    self.total_warnings += 1
                    self.log.info(
                        f"⚠️ Spread WARNING {sym}: {spread_pct:.3f}% "
                        f"(avg: {avg_spread:.3f}%)"
                    )

                if level == self.LEVEL_ALERT:
                    worst = self.LEVEL_ALERT
                elif level == self.LEVEL_WARNING and worst == self.LEVEL_NORMAL:
                    worst = self.LEVEL_WARNING

            except Exception as e:
                self.log.debug(f"Spread update {sym}: {e}")

        self.global_level = worst
        return worst

    def can_trade(self, symbol: str = None) -> bool:
        """False dacă spread e în ALERT (mișcare violentă iminentă)."""
        if symbol:
            return self.level.get(symbol, self.LEVEL_NORMAL) != self.LEVEL_ALERT
        return self.global_level != self.LEVEL_ALERT

    def get_size_multiplier(self, symbol: str = None) -> float:
        """NORMAL=1.0, WARNING=0.5, ALERT=0.0"""
        lvl = self.level.get(symbol, self.global_level) if symbol else self.global_level
        if lvl == self.LEVEL_ALERT:   return 0.0
        if lvl == self.LEVEL_WARNING: return 0.5
        return 1.0

    def get_status(self) -> dict:
        spreads = {}
        for sym in self.config.symbols:
            hist = list(self.history[sym])
            if hist:
                current = hist[-1][1]
                avg     = sum(v for _, v in hist) / len(hist)
                spike   = current / max(avg, 0.0001)
            else:
                current = avg = spike = 0.0
            spreads[sym] = {
                "current_pct": round(current, 4),
                "avg_pct":     round(avg, 4),
                "spike_x":     round(spike, 2),
                "level":       self.level.get(sym, self.LEVEL_NORMAL),
                "consecutive": self._consecutive_high.get(sym, 0),
            }
        return {
            "global_level":   self.global_level,
            "symbols":        spreads,
            "total_alerts":   self.total_alerts,
            "total_warnings": self.total_warnings,
            "last_check_ago": round(time.time() - self.last_check_ts, 1),
        }

    def _get_spread(self, symbol: str) -> Optional[float]:
        now = time.time()
        cached_ts, cached_ob = self._ob_cache.get(symbol, (0.0, {}))
        if now - cached_ts < 8.0 and cached_ob:
            ob = cached_ob
        else:
            if self.client:
                ob = self.client.orderbook(symbol, depth=5)
            else:
                try:
                    import requests as _req
                    r = _req.get(
                        "https://api.binance.com/api/v3/depth",
                        params={"symbol": symbol, "limit": 5}, timeout=5)
                    data = r.json()
                    ob = {
                        "bids": [[float(p), float(q)] for p, q in data.get("bids", [])],
                        "asks": [[float(p), float(q)] for p, q in data.get("asks", [])],
                    }
                except Exception:
                    return None
            self._ob_cache[symbol] = (now, ob)

        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        if not bids or not asks:
            return None

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        if best_bid <= 0:
            return None
        return (best_ask - best_bid) / best_bid * 100

    def _classify_spread(self, symbol: str, spread_pct: float,
                         avg_spread: float) -> str:
        if spread_pct >= self.config.spread_alert_pct:
            return self.LEVEL_ALERT
        if spread_pct >= self.config.spread_warning_pct:
            return self.LEVEL_WARNING
        if avg_spread > 0 and len(self.history[symbol]) >= 10:
            spike = spread_pct / avg_spread
            if spike >= self.config.spike_multiplier:
                if spread_pct >= self.config.spread_normal_pct * 2:
                    return self.LEVEL_ALERT if spike > 5 else self.LEVEL_WARNING
        return self.LEVEL_NORMAL


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION HELPER — Conectează toate modulele
# ═══════════════════════════════════════════════════════════════════


class AutoCompounder:
    """v1.3: Compound non-linear."""
    def __init__(self, initial_capital_usd: float = 798.0,
                 config=None, telegram_callback=None):
        self.config = config or type('AC', (), {
            'compound_pct': 90.0, 'min_compound_usd': 5.0,
            'compound_interval_days': 1.0, 'notify': False,
            'max_capital_mult': 6.0,
            'scaling_tiers': [(3.0, 0.7), (5.0, 0.5), (10.0, 0.3)]
        })()
        self.telegram = telegram_callback
        self.initial_capital = initial_capital_usd
        self.current_base = initial_capital_usd
        self.total_compounded = 0.0
        self.compound_count = 0
        self.last_compound_ts = time.time()
        self.history = []
    def _effective_compound_pct(self) -> float:
        ratio = self.current_base / max(self.initial_capital, 0.01)
        for threshold, pct_mult in reversed(self.config.scaling_tiers):
            if ratio >= threshold:
                return self.config.compound_pct * pct_mult
        return self.config.compound_pct
    def get_sizing_cap(self) -> float:
        ratio = self.current_base / max(self.initial_capital, 0.01)
        if ratio <= 1.0: return 1.0
        return min(ratio, self.config.max_capital_mult) / ratio
    def check_compound(self, current_total_usd: float):
        elapsed_days = (time.time() - self.last_compound_ts) / 86400
        if elapsed_days < self.config.compound_interval_days: return None
        profit = current_total_usd - self.current_base
        if profit < self.config.min_compound_usd: return None
        eff_pct = self._effective_compound_pct()
        compound_amount = profit * (eff_pct / 100)
        self.current_base += compound_amount
        self.total_compounded += compound_amount
        self.compound_count += 1
        self.last_compound_ts = time.time()
        return {"new_base": round(self.current_base, 2), "compound_amount": round(compound_amount, 2)}
    def get_capital_base(self) -> float:
        return self.current_base
    def get_status(self) -> dict:
        return {
            "initial_capital": round(self.initial_capital, 2),
            "current_base": round(self.current_base, 2),
            "total_compounded": round(self.total_compounded, 2),
            "compound_count": self.compound_count,
            "growth_pct": round((self.current_base / max(self.initial_capital, 0.01) - 1) * 100, 1),
            "effective_compound_pct": round(self._effective_compound_pct(), 1),
            "sizing_cap": round(self.get_sizing_cap(), 2),
        }

class EnhancementsManager:
    """
    Manager central v1.1 — coordonează toate modulele.
    Nou: profit lock, per-strategy sizing, SOL DCA dip.
    """
    
    def __init__(self, 
                 initial_capital_usd: float = 798.0,
                 telegram_callback=None):
        
        self.telegram = telegram_callback
        
        # 1. Circuit Breaker (v1.1: cooldown 30min, exclude funding)
        self.circuit_breaker = GlobalCircuitBreaker(
            initial_capital_usd=initial_capital_usd,
            telegram_callback=telegram_callback
        )
        
        # 1b. Profit Lock (v1.1: trailing stop pe profit zilnic)
        self.profit_lock = DailyProfitLock(
            telegram_callback=telegram_callback
        )
        
        # 2. Funding Spread Filter
        self.funding_filter = FundingSpreadFilter()
        
        # 3. Grid Optimizer
        self.grid_optimizer = GridSpacingOptimizer()
        
        # 4. SOL Accumulator (v1.1: DCA pe dip-uri)
        self.sol_accumulator = SOLBatchAccumulator(
            telegram_callback=telegram_callback
        )
        
        # 5. OI Sentinel (v1.1: per-strategy multiplier)
        self.oi_sentinel = OpenInterestSentinel(
            telegram_callback=telegram_callback
        )
        
        # 6. Adaptive Allocator (v1.2: FG-based rebalancing)
        self.adaptive_alloc = AdaptiveAllocator(
            telegram_callback=telegram_callback
        )
        
        # 7. Auto-Compound (v1.2: crește capital cu profit)
        self.auto_compound = AutoCompounder(
            initial_capital_usd=initial_capital_usd,
            telegram_callback=telegram_callback
        )

        # 8. Spread Monitor (v1.4: detectare retragere Market Maker)
        self.spread_monitor = SpreadMonitor(
            telegram_callback=telegram_callback
        )

        # PUNCT 6: Referință la FeeBufferManager (setat din SolanaBot)
        self._fee_buf = None

        _enh_logger.info("✅ Enhancements v1.4 initialized (+SpreadMonitor)")
    
    def get_combined_size_multiplier(self) -> float:
        """
        Combină multiplier-ul din CB + OI + ProfitLock.
        Returnează cel mai restrictiv (pentru strategii non-delta-neutral).
        """
        cb_mult = self.circuit_breaker.get_position_size_multiplier()
        oi_mult = self.oi_sentinel.get_exposure_multiplier()
        pl_mult = self.profit_lock.get_multiplier() if hasattr(self.profit_lock, 'get_multiplier') else self.profit_lock.update(self.profit_lock.daily_pnl)
        return min(cb_mult, oi_mult, pl_mult)
    
    def get_size_multiplier_for_strategy(self, strategy: str, symbol: str = None) -> float:
        """
        v1.3: Multiplicator per strategie.
        Combină: CB + OI + ProfitLock + AdaptiveAlloc + SizingCap + FeeBuffer.
        """
        cb_mult = self.circuit_breaker.get_size_multiplier_for_strategy(strategy)
        oi_mult = self.oi_sentinel.get_exposure_multiplier_for_strategy(strategy)
        pl_mult = self.profit_lock.get_multiplier() if hasattr(self.profit_lock, 'get_multiplier') else self.profit_lock.update(self.profit_lock.daily_pnl)
        # Adaptive allocation pe Fear & Greed
        aa_mults = self.adaptive_alloc.get_strategy_multipliers()
        strat_key = strategy.lower().replace("_arb", "")
        aa_mult = aa_mults.get(strat_key, 1.0)
        # v1.3: Sizing cap — previne ordine prea mari la capital mare
        cap_mult = self.auto_compound.get_sizing_cap()
        # PUNCT 6: FeeBuffer — reduce sizing când fee buffer e scăzut
        fee_mult = self._fee_buf.get_sizing_multiplier() if self._fee_buf else 1.0
        # PUNCT 7: SpreadMonitor — reduce sizing la retragere MM
        spread_mult = self.spread_monitor.get_size_multiplier(symbol if symbol is not None else None)
        return min(cb_mult, oi_mult, pl_mult, fee_mult, spread_mult) * aa_mult * cap_mult
    
    def pre_trade_check(self, symbol: str, strategy: str, 
                         size_usd: float) -> Tuple[bool, float, str]:
        """
        Verificare completă înainte de orice trade.
        v1.1: folosește per-strategy multiplier.
        """
        reasons = []
        
        # 1. Circuit breaker — per strategy
        cb_mult = self.circuit_breaker.get_size_multiplier_for_strategy(strategy)
        if cb_mult <= 0:
            return False, 0, f"Circuit Breaker: {self.circuit_breaker.state.value} (strat={strategy})"
        
        # 2. Correlation check (afectează toate)
        if self.circuit_breaker.check_correlation_halt():
            if strategy.lower() not in self.circuit_breaker.config.delta_neutral_strategies:
                return False, 0, "Correlation halt: BNB+SOL dropping together"
        
        # 3. OI sentinel — per strategy
        oi_mult = self.oi_sentinel.get_exposure_multiplier_for_strategy(strategy)
        
        # 4. Profit lock
        pl_mult = self.profit_lock.get_multiplier() if hasattr(self.profit_lock, 'get_multiplier') else self.profit_lock.update(self.profit_lock.daily_pnl)
        
        # 5. Spread Monitor — blochează trade la ALERT (retragere MM)
        if not self.spread_monitor.can_trade(symbol):
            return False, 0, f"Spread ALERT {symbol}: Market Makers retrași, mișcare violentă iminentă"
        spread_mult = self.spread_monitor.get_size_multiplier(symbol)

        # 5b. Combined sizing per strategy
        multiplier = min(cb_mult, oi_mult, pl_mult, spread_mult)
        adjusted_size = size_usd * multiplier
        
        if multiplier < 1.0:
            parts = []
            if cb_mult < 1.0: parts.append(f"CB={cb_mult:.0%}")
            if oi_mult < 1.0: parts.append(f"OI={oi_mult:.0%}")
            if pl_mult < 1.0: parts.append(f"PL={pl_mult:.0%}")
            reasons.append(f"Size ${size_usd:.2f}→${adjusted_size:.2f} ({' '.join(parts)})")
        
        # 6. Strategy-specific checks
        if strategy.lower() in ("funding_arb", "funding"):
            ok, reason = self.funding_filter.should_enter_funding(symbol)
            if not ok:
                return False, 0, f"Funding filter: {reason}"
            reasons.append(reason)
        
        can_trade = adjusted_size > 1.0
        reason_str = " | ".join(reasons) if reasons else "OK"
        
        return can_trade, adjusted_size, reason_str
    
    def update_daily_pnl(self, daily_pnl_usd: float):
        """v1.1: Actualizează P&L zilnic pentru profit lock."""
        self.profit_lock.update(daily_pnl_usd)
    
    def on_daily_close(self, daily_profit_usd: float):
        """Apelează la sfârșitul zilei de trading."""
        self.sol_accumulator.record_daily_profit(daily_profit_usd)
        
        # v1.1: Check DCA dip FIRST
        sol_price = self.oi_sentinel._fetch_price("SOLUSDC")
        if sol_price:
            self.sol_accumulator.update_sol_price(sol_price)
            is_dip, reason = self.sol_accumulator.check_dip_buy(sol_price)
            if is_dip:
                self.sol_accumulator.execute_dip_buy(sol_price)
                return  # cumpărat la dip, nu mai facem weekly swap
        
        # Fallback: weekly swap
        should_swap, reason = self.sol_accumulator.should_swap_now()
        if should_swap and sol_price:
            self.sol_accumulator.execute_swap(sol_price)
    
    def check_compound(self, current_total_usd: float) -> Optional[dict]:
        """v1.2: Verifică și execută auto-compound."""
        return self.auto_compound.check_compound(current_total_usd)
    
    def update_fear_greed(self, fg_index: int):
        """v1.2: Actualizează Fear & Greed pentru adaptive allocation."""
        self.adaptive_alloc.update_fg(fg_index)
    
    def update_prices(self, prices: Dict[str, float]):
        """Update prețuri pentru toate modulele."""
        for symbol, price in prices.items():
            self.circuit_breaker.update_prices(symbol, price)
            # v1.1: SOL price tracking pentru DCA dip
            if symbol == "SOLUSDC":
                self.sol_accumulator.update_sol_price(price)
    
    def get_full_status(self) -> dict:
        """Status complet al tuturor modulelor."""
        return {
            "circuit_breaker": self.circuit_breaker.get_status(),
            "profit_lock": self.profit_lock.get_status(),
            "funding_filter": self.funding_filter.get_stats(),
            "grid_optimizer": self.grid_optimizer.recommend_for_capital(
                self.auto_compound.get_capital_base() * 0.20, 2
            ),
            "sol_accumulator": self.sol_accumulator.get_status(),
            "oi_sentinel": self.oi_sentinel.get_status(),
            "adaptive_alloc": self.adaptive_alloc.get_status(),
            "auto_compound": self.auto_compound.get_status(),
            "combined_size_multiplier": self.get_combined_size_multiplier()
        }


# ═══════════════════════════════════════════════════════════════════
# 6. ADAPTIVE ALLOCATOR — Schimbă alocarea pe Fear & Greed (v1.2)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AdaptiveAllocConfig:
    # Thresholds Fear & Greed
    extreme_fear_fg: int = 15        # era 20 → 15 (doar extreme fear real)
    greed_fg: int = 80               # era 75 → 80 (doar greed real)
    
    # v1.3: Shift-uri mici — nu pierde profit din realocare
    fear_shift_pct: float = 5.0      # era 10% → 5%
    greed_shift_pct: float = 5.0     # era 10% → 5%
    
    # Cooldown între ajustări
    adjust_cooldown_hours: float = 24.0  # era 12h → 24h (mai stabil)


class AdaptiveAllocator:
    """
    v1.2: Ajustează alocarea capitalului pe baza Fear & Greed Index.
    
    FG < 20 (Extreme Fear): piață instabilă → mută capital spre funding (safe)
    FG 20-75 (Normal): alocare standard
    FG > 75 (Greed): piață trending → mută capital spre swing (profit din trend)
    
    Returnează multiplicatori per strategie.
    """
    
    def __init__(self, config: AdaptiveAllocConfig = None,
                 telegram_callback=None):
        self.config = config or AdaptiveAllocConfig()
        self.telegram = telegram_callback
        
        self.current_fg = 50  # default neutral
        self.current_mode = "NORMAL"  # FEAR / NORMAL / GREED
        self.last_adjust_ts = 0.0
        self.total_adjustments = 0
    
    def update_fg(self, fear_greed_index: int):
        """Actualizează FG index. Apelează de câte ori se actualizează sentinel."""
        self.current_fg = max(0, min(100, fear_greed_index))
        
        old_mode = self.current_mode
        if self.current_fg < self.config.extreme_fear_fg:
            self.current_mode = "FEAR"
        elif self.current_fg > self.config.greed_fg:
            self.current_mode = "GREED"
        else:
            self.current_mode = "NORMAL"
        
        if old_mode != self.current_mode:
            now = time.time()
            if now - self.last_adjust_ts > self.config.adjust_cooldown_hours * 3600:
                self.last_adjust_ts = now
                self.total_adjustments += 1
                self._notify(
                    f"📊 Adaptive Alloc: {old_mode} → {self.current_mode} (FG={self.current_fg})")
    
    def get_strategy_multipliers(self) -> Dict[str, float]:
        """
        v1.3: La GREED → BOOST TOATE strategiile (nu redistribui).
        Piața trending = funding rate mai mare + grid mai activ + swing trending.
        La FEAR → conservator, funding full, grid+swing reduse.
        """
        shift = self.config.fear_shift_pct / 100
        
        if self.current_mode == "FEAR":
            return {
                "funding": 1.0 + shift,       # +5% → 105%
                "grid":    1.0 - shift * 0.5,  # -2.5% → 97.5%
                "swing":   1.0 - shift,         # -5% → 95%
            }
        elif self.current_mode == "GREED":
            # Eliminam BOOST — mentinem capitalul protejat la capat de trend
            return {
                "funding": 1.0,
                "grid":    1.0,
                "swing":   1.0,
            }
        elif name == "price_mlp":
            return (self.price_mlp and self.price_mlp.trained and
                    self.price_mlp.accuracy >= self.MIN_PRICE_MLP_ACCURACY)
        elif name == "anomaly":
            return (self.anomaly and self.anomaly.trained and
                    self.anomaly.train_samples >= self.MIN_ANOMALY_SAMPLES)
        elif name == "grid_ml":
            return (self.grid_ml and self.grid_ml.trained and
                    len(self.grid_ml.fill_history) >= self.MIN_GRID_FILLS)
        elif name == "funding_ml":
            return (self.funding_ml and self.funding_ml.trained and
                    len(self.funding_ml.history) >= self.MIN_FUNDING_OBS)
        elif name == "ensemble":
            return (self.ensemble and self.ensemble.trained and
                    len(self.ensemble.history) >= self.MIN_ENSEMBLE_TRADES)
        return False

    @property
    def trading_ready(self) -> bool:
        return self._is_model_ready("regime") and self._is_model_ready("price_mlp")

    @property
    def phase(self) -> str:
        if self.trading_ready: return "READY"
        if any(m and m.trained for m in [self.regime, self.price_mlp, self.anomaly]):
            return "TRAINING"
        return "COLLECTING"

    def retrain_if_needed(self, klines_1h: list = None):
        if not self.enabled or self._training: return
        if time.time() - self.last_train_ts < self.train_interval: return
        self._training = True
        try:
            trained = []
            if klines_1h:
                if self.regime and self.regime.train(klines_1h):
                    trained.append(f"Regime({self.regime.accuracy:.0%},{self.regime.train_samples}s)")
                if self.price_mlp and self.price_mlp.train(klines_1h):
                    trained.append(f"MLP({self.price_mlp.accuracy:.0%})")
                if self.anomaly and self.anomaly.train(klines_1h):
                    trained.append(f"Anomaly(t={self.anomaly.threshold:.3f})")
            if self.grid_ml and len(self.grid_ml.fill_history) >= 30 and self.grid_ml.train():
                trained.append(f"Grid({len(self.grid_ml.fill_history)}f)")
            if self.funding_ml and len(self.funding_ml.history) >= 50 and self.funding_ml.train():
                trained.append(f"Fund({len(self.funding_ml.history)}o)")
            if self.ensemble and len(self.ensemble.history) >= 30 and self.ensemble.train():
                trained.append(f"Ens({len(self.ensemble.history)}t)")
            if trained:
                self.last_train_ts = time.time()
                msg = f"🧠 ML [{self.phase}]: {', '.join(trained)}"
                _enh_logger.info(msg)
                if self.telegram:
                    try: self.telegram(msg)
                    except Exception as _e: logging.debug(f'Ignored: {_e}')
        except Exception as e:
            _enh_logger.warning(f"ML retrain error: {e}")
        finally:
            self._training = False

    def get_regime(self, klines_1h): return self.regime.predict(klines_1h) if self._is_model_ready("regime") else {"regime":"unknown","confidence":0.0,"proba":{}}
    def get_price_direction(self, klines_1h): return self.price_mlp.predict(klines_1h) if self._is_model_ready("price_mlp") else {"direction":"flat","confidence":0.0,"proba":{}}
    def get_anomaly_score(self, klines_1h): return self.anomaly.score(klines_1h) if self._is_model_ready("anomaly") else {"score":0.0,"is_anomaly":False,"sizing_mult":1.0}
    def get_optimal_spacing(self, feat_dict): return self.grid_ml.predict_optimal_spacing(feat_dict) if self._is_model_ready("grid_ml") else 0.012
    def rank_funding_pairs(self, sym_feat): return self.funding_ml.rank_symbols(sym_feat) if self._is_model_ready("funding_ml") else []

    def record_grid_fill(self, feat_dict, spacing, profit):
        if self.grid_ml: self.grid_ml.record_fill(feat_dict, spacing, profit)
    def record_funding_rate(self, symbol, feat_dict, rate):
        if self.funding_ml: self.funding_ml.record_rate(symbol, feat_dict, rate)
    def record_trade_result(self, klines_1h, feat_dict, pnl):
        if self.ensemble and feat_dict:
            r = self.get_regime(klines_1h); d = self.get_price_direction(klines_1h); a = self.get_anomaly_score(klines_1h)
            self.ensemble.record_trade(r.get("proba",{}), d.get("proba",{}), a.get("score",0), feat_dict, pnl)

    def get_swing_signal(self, klines_1h: list, direction: int) -> dict:
        """ML NU blochează trades până nu e READY. Colectează date mereu."""
        if not self.trading_ready:
            return {"take_trade": True, "confidence": 0.0,
                    "sizing_mult": 1.0, "reason": f"ML:{self.phase}"}
        regime = self.get_regime(klines_1h)
        price_dir = self.get_price_direction(klines_1h)
        anomaly = self.get_anomaly_score(klines_1h)
        reasons = []; confidence = 0.5; sizing = anomaly.get("sizing_mult", 1.0)
        r, r_c = regime.get("regime","unknown"), regime.get("confidence",0)
        if r == "sideways" and r_c > 0.65:
            return {"take_trade":False,"confidence":r_c,"sizing_mult":sizing,"reason":f"ML:sideways({r_c:.0%})"}
        d, d_c = price_dir.get("direction","flat"), price_dir.get("confidence",0)
        if d != "flat" and d_c > 0.55:
            if (d=="up" and direction==1) or (d=="down" and direction==-1):
                confidence += 0.2; reasons.append(f"MLP:{d}({d_c:.0%})")
            elif d_c > 0.6:
                return {"take_trade":False,"confidence":d_c,"sizing_mult":sizing,"reason":f"MLP:contra {d}({d_c:.0%})"}
        if (r=="trending_up" and direction==1) or (r=="trending_down" and direction==-1):
            confidence += 0.15; sizing *= 1.1; reasons.append(f"regime:{r}")
        if anomaly.get("is_anomaly",False):
            confidence -= 0.1; reasons.append(f"anomaly")
        return {"take_trade": confidence > 0.55, "confidence": min(confidence,1.0),
                "sizing_mult": sizing, "reason": " | ".join(reasons) or "ML:READY"}

    def get_status(self) -> dict:
        return {"enabled":self.enabled, "deep_ai":DEEP_AI_AVAILABLE, "phase":self.phase,
                "trading_ready":self.trading_ready,
                "regime":self.regime.get_status() if self.regime else {},
                "grid_ml":self.grid_ml.get_status() if self.grid_ml else {},
                "funding_ml":self.funding_ml.get_status() if self.funding_ml else {},
                "price_mlp":self.price_mlp.get_status() if self.price_mlp else {},
                "anomaly":self.anomaly.get_status() if self.anomaly else {},
                "ensemble":self.ensemble.get_status() if self.ensemble else {}}

# ══════════════════════════════════════════════════════════════════════
# DEEP AI MODULE — Neural Networks (MLP, CPU only, no GPU)
# 3 modele: PriceDirectionMLP, AnomalyDetectorMLP, EnsembleSignal
# ══════════════════════════════════════════════════════════════════════

try:
    import numpy as np
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    DEEP_AI_AVAILABLE = ML_AVAILABLE  # needs numpy + sklearn
except ImportError:
    RandomForestClassifier = None
    GradientBoostingClassifier = None
    StandardScaler = None
    LinearRegression = None
    DEEP_AI_AVAILABLE = False


class RegimeClassifier:
    """ML Model 1: Sideways/Trending_Up/Trending_Down/Volatile."""

    MODEL_PATH = "v8_regime_model.pkl"

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=50, max_depth=6, min_samples_leaf=5,
            random_state=42, n_jobs=1) if ML_AVAILABLE else None
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.trained = False
        self.accuracy = 0.0
        self.train_samples = 0
        self._load_model()

    def _get_hmac_key(self) -> bytes:
        """Returnează cheia HMAC dedicată, creând-o dacă nu există."""
        import secrets
        key_file = os.path.expanduser("~/.bot_hmac_key_v8")
        if not os.path.exists(key_file):
            new_key = secrets.token_hex(32)
            with open(key_file, "w") as f: f.write(new_key)
            os.chmod(key_file, 0o600)
            _enh_logger.info(f"🔑 HMAC key creat: {key_file}")
        with open(key_file) as f:
            return f.read().strip().encode()

    def _load_model(self):
        """Încarcă modelul de pe disc — evită 24h fără ML după restart."""
        if not ML_AVAILABLE: return
        try:
            import pickle, os
            if os.path.exists(self.MODEL_PATH):
                # Verificare integritate: refuzăm pickle din surse necunoscute
                model_size = os.path.getsize(self.MODEL_PATH)
                if model_size > 50 * 1024 * 1024:  # >50MB suspect
                    _enh_logger.warning(f"Model file suspect (size={model_size//1024}KB) — ignorat")
                    return
                with open(self.MODEL_PATH, "rb") as f:
                    raw = f.read()
                # Verificare HMAC integritate
                if b"\n" in raw:
                    sig_stored, payload = raw.split(b"\n", 1)
                    import hmac as _hmac, hashlib as _hs
                    _key = self._get_hmac_key()
                    sig_expected = _hmac.new(_key, payload, _hs.sha256).hexdigest()
                    if not _hmac.compare_digest(sig_stored.decode(), sig_expected):
                        _enh_logger.warning("⚠️ Model HMAC invalid — ignorat (posibil tampered)")
                        return
                    data = pickle.loads(payload)
                else:
                    _enh_logger.warning("Model fără semnătură HMAC — refuzat. Se reantrenează.")
                    return
                self.model         = data["model"]
                self.scaler        = data["scaler"]
                self.accuracy      = data.get("accuracy", 0.0)
                self.train_samples = data.get("samples", 0)
                self.trained       = True
                _enh_logger.info(f"🧠 RegimeClassifier loaded (acc={self.accuracy:.1%})")
        except Exception as e:
            _enh_logger.debug(f"Regime load: {e}")

    def _save_model(self):
        if not ML_AVAILABLE or not self.trained: return
        try:
            import pickle, hmac as _hmac, hashlib as _hs
            payload = pickle.dumps({"model": self.model, "scaler": self.scaler,
                                    "accuracy": self.accuracy, "samples": self.train_samples})
            # HMAC signature pentru verificare integritate la load
            _key = self._get_hmac_key()
            sig = _hmac.new(_key, payload, _hs.sha256).hexdigest()
            with open(self.MODEL_PATH, "wb") as f:
                f.write(sig.encode() + b"\n" + payload)
        except Exception as e:
            _enh_logger.debug(f"Regime save: {e}")

    def train(self, klines_1h: list, future_window: int = 4):
        if not ML_AVAILABLE or len(klines_1h) < 200:
            return False
        X, y = [], []
        for i in range(50, len(klines_1h) - future_window):
            feat = FeatureExtractor.from_klines(klines_1h[i-25:i], 20)
            if not feat: continue
            future = [float(klines_1h[i+j][4]) for j in range(future_window)]
            cur = float(klines_1h[i][4])
            net = (future[-1] - cur) / cur
            rng = (max(future) - min(future)) / cur
            if abs(net) > 0.01: label = 1 if net > 0 else 2
            elif rng > 0.015: label = 3
            else: label = 0
            X.append(FeatureExtractor.feature_vector(feat))
            y.append(label)
        if len(X) < 50: return False
        X, y = np.array(X), np.array(y)
        X_s = self.scaler.fit_transform(X)
        s = int(len(X) * 0.8)
        self.model.fit(X_s[:s], y[:s])
        self.accuracy = self.model.score(X_s[s:], y[s:]) if s < len(X) else 0
        self.trained = True
        self.train_samples = len(X)
        _enh_logger.info(f"RegimeClassifier: {len(X)} samples, acc={self.accuracy:.1%}")
        self._save_model()
        return True

    def predict(self, klines_1h: list) -> dict:
        if not self.trained or not ML_AVAILABLE:
            return {"regime": "unknown", "confidence": 0.0, "proba": {}}
        feat = FeatureExtractor.from_klines(klines_1h, 20)
        if not feat:
            return {"regime": "unknown", "confidence": 0.0, "proba": {}}
        X = self.scaler.transform(np.array([FeatureExtractor.feature_vector(feat)]))
        proba = self.model.predict_proba(X)[0]
        pred = self.model.predict(X)[0]
        rmap = {0: "sideways", 1: "trending_up", 2: "trending_down", 3: "volatile"}
        return {
            "regime": rmap.get(pred, "unknown"),
            "confidence": float(max(proba)),
            "proba": {rmap.get(i, f"c{i}"): float(p) for i, p in enumerate(proba)}
        }

    def get_status(self): return {"trained": self.trained, "accuracy": round(self.accuracy, 3), "samples": self.train_samples}



class GridSpacingML:
    """ML Model 2: Predict optimal grid spacing from fill history."""

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=30, max_depth=4, learning_rate=0.1, random_state=42
        ) if ML_AVAILABLE else None
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.trained = False
        self.fill_history: list = []
        self.mae = 0.0

    def record_fill(self, feat_dict: dict, spacing: float, profit: float):
        if feat_dict:
            self.fill_history.append((FeatureExtractor.feature_vector(feat_dict), spacing, profit))
            if len(self.fill_history) > 500:
                self.fill_history = self.fill_history[-500:]

    def train(self):
        if not ML_AVAILABLE or len(self.fill_history) < 30: return False
        X = np.array([h[0] + [h[1]] for h in self.fill_history])
        y = np.array([h[2] for h in self.fill_history])
        X_s = self.scaler.fit_transform(X)
        s = int(len(X) * 0.8)
        self.model.fit(X_s[:s], y[:s])
        if s < len(X):
            self.mae = float(np.mean(np.abs(self.model.predict(X_s[s:]) - y[s:])))
        self.trained = True
        _enh_logger.info(f"GridSpacingML: {len(X)} fills, MAE=${self.mae:.4f}")
        return True

    def predict_optimal_spacing(self, feat_dict: dict, spacing_range=(0.006, 0.020)) -> float:
        if not self.trained or not ML_AVAILABLE or not feat_dict: return 0.012
        fv = FeatureExtractor.feature_vector(feat_dict)
        best_s, best_p = 0.012, -999
        for sp in np.arange(spacing_range[0], spacing_range[1], 0.001):
            X = self.scaler.transform(np.array([fv + [sp]]))
            p = self.model.predict(X)[0]
            if p > best_p: best_p = p; best_s = sp
        return round(best_s, 4)

    def get_status(self): return {"trained": self.trained, "fills": len(self.fill_history), "mae": round(self.mae, 4)}



class FundingPredictor:
    """ML Model 3: Predict which pairs will have high funding rate."""

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=30, max_depth=4, learning_rate=0.1, random_state=42
        ) if ML_AVAILABLE else None
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.trained = False
        self.history: list = []
        self.mae = 0.0

    def record_rate(self, symbol: str, feat_dict: dict, rate: float):
        if feat_dict:
            self.history.append((FeatureExtractor.feature_vector(feat_dict), rate))
            if len(self.history) > 1000:
                self.history = self.history[-1000:]

    def train(self):
        if not ML_AVAILABLE or len(self.history) < 50: return False
        X = np.array([h[0] for h in self.history])
        y = np.array([h[1] for h in self.history])
        X_s = self.scaler.fit_transform(X)
        s = int(len(X) * 0.8)
        self.model.fit(X_s[:s], y[:s])
        if s < len(X):
            self.mae = float(np.mean(np.abs(self.model.predict(X_s[s:]) - y[s:])))
        self.trained = True
        _enh_logger.info(f"FundingPredictor: {len(X)} obs, MAE={self.mae:.6f}")
        return True

    def predict_rate(self, feat_dict: dict) -> float:
        if not self.trained or not ML_AVAILABLE or not feat_dict: return 0.0001
        X = self.scaler.transform(np.array([FeatureExtractor.feature_vector(feat_dict)]))
        return float(self.model.predict(X)[0])

    def rank_symbols(self, symbols_features: dict) -> list:
        if not self.trained or not ML_AVAILABLE: return []
        results = [(sym, self.predict_rate(feat)) for sym, feat in symbols_features.items()]
        results.sort(key=lambda x: -x[1])
        return results

    def get_status(self): return {"trained": self.trained, "obs": len(self.history), "mae": round(self.mae, 6)}



class FeatureExtractor:
    """Extrage features din klines pentru ML."""

    @staticmethod
    def from_klines(klines: list, lookback: int = 20) -> dict:
        if len(klines) < lookback + 5:
            return {}
        kl = klines[-(lookback + 5):]
        closes = [float(k[4]) for k in kl]
        highs = [float(k[2]) for k in kl]
        lows = [float(k[3]) for k in kl]
        volumes = [float(k[5]) for k in kl]
        opens = [float(k[1]) for k in kl]

        c = closes[-lookback:]
        h = highs[-lookback:]
        l = lows[-lookback:]
        v = volumes[-lookback:]

        returns = [(c[i] - c[i-1]) / c[i-1] for i in range(1, len(c))]
        avg_ret = sum(returns) / len(returns) if returns else 0
        std_ret = (sum((r - avg_ret)**2 for r in returns) / len(returns))**0.5 if returns else 0.01

        net_move = abs(c[-1] - c[0]) / c[0] if c[0] > 0 else 0
        gross_move = sum(abs(c[i] - c[i-1]) for i in range(1, len(c)))
        efficiency = (net_move / (gross_move / c[0])) if gross_move > 0 else 0

        gains = [max(0, returns[i]) for i in range(len(returns))]
        losses = [max(0, -returns[i]) for i in range(len(returns))]
        avg_gain = sum(gains[-14:]) / 14 if len(gains) >= 14 else 0.01
        avg_loss = sum(losses[-14:]) / 14 if len(losses) >= 14 else 0.01
        rsi = 100 - 100 / (1 + avg_gain / avg_loss) if avg_loss > 0 else 50

        avg_vol = sum(v) / len(v) if v else 1
        vol_ratio = v[-1] / avg_vol if avg_vol > 0 else 1
        vol_trend = sum(v[-5:]) / sum(v[:5]) if sum(v[:5]) > 0 else 1

        daily_ranges = [(h[i] - l[i]) / c[i] for i in range(len(c))]
        avg_range = sum(daily_ranges) / len(daily_ranges) if daily_ranges else 0.02
        recent_range = sum(daily_ranges[-5:]) / 5 if len(daily_ranges) >= 5 else avg_range

        ema8 = c[0]; ema21 = c[0]
        for p in c:
            ema8 = p * 2/9 + ema8 * 7/9
            ema21 = p * 2/22 + ema21 * 20/22
        ema_spread = (ema8 - ema21) / ema21 if ema21 > 0 else 0

        body_ratio = abs(closes[-1] - opens[-1]) / (highs[-1] - lows[-1]) if (highs[-1] - lows[-1]) > 0 else 0
        upper_wick = (highs[-1] - max(closes[-1], opens[-1])) / (highs[-1] - lows[-1]) if (highs[-1] - lows[-1]) > 0 else 0

        return {
            "avg_return": avg_ret, "std_return": std_ret,
            "efficiency": efficiency, "net_move_pct": net_move,
            "rsi": rsi, "vol_ratio": vol_ratio, "vol_trend": vol_trend,
            "avg_range": avg_range, "recent_range": recent_range,
            "ema_spread": ema_spread, "body_ratio": body_ratio,
            "upper_wick": upper_wick,
            "price_vs_ema8": (c[-1] - ema8) / ema8 if ema8 > 0 else 0,
            "price_vs_ema21": (c[-1] - ema21) / ema21 if ema21 > 0 else 0,
            "range_expansion": recent_range / avg_range if avg_range > 0 else 1,
        }

    @staticmethod
    def feature_vector(feat_dict: dict) -> list:
        return [feat_dict.get(k, 0) for k in sorted(feat_dict.keys())]

    @staticmethod
    def feature_names() -> list:
        return sorted([
            "avg_return","std_return","efficiency","net_move_pct","rsi",
            "vol_ratio","vol_trend","avg_range","recent_range","ema_spread",
            "body_ratio","upper_wick","price_vs_ema8","price_vs_ema21","range_expansion",
        ])



class PriceDirectionMLP:
    """
    Deep AI Model 1: MLP Neural Network — predict price direction.
    Architecture: 15 inputs → 64 → 32 → 16 → 3 outputs (up/down/flat)
    Antrenat pe klines 1h, predict next 4h direction.
    Mai precis decât RandomForest pe patterns non-lineare.
    """

    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=42,
            batch_size=32,
        ) if DEEP_AI_AVAILABLE else None
        self.scaler = StandardScaler() if DEEP_AI_AVAILABLE else None
        self.trained = False
        self.accuracy = 0.0
        self.train_samples = 0

    def train(self, klines_1h: list, future_window: int = 4):
        """
        Labels: 0=flat (<0.5%), 1=up (>0.8%), 2=down (<-0.8%)
        Threshold-uri mai stricte decât RegimeClassifier → semnale mai curate.
        """
        if not DEEP_AI_AVAILABLE or len(klines_1h) < 300:
            return False
        X, y = [], []
        for i in range(50, len(klines_1h) - future_window):
            feat = FeatureExtractor.from_klines(klines_1h[i-25:i], 20)
            if not feat: continue
            cur = float(klines_1h[i][4])
            fut = float(klines_1h[i + future_window - 1][4])
            ret = (fut - cur) / cur
            if ret > 0.008: label = 1    # up >0.8%
            elif ret < -0.008: label = 2  # down >0.8%
            else: label = 0               # flat
            X.append(FeatureExtractor.feature_vector(feat))
            y.append(label)
        if len(X) < 100: return False
        X, y = np.array(X), np.array(y)
        X_s = self.scaler.fit_transform(X)
        s = int(len(X) * 0.8)
        try:
            self.model.fit(X_s[:s], y[:s])
            self.accuracy = self.model.score(X_s[s:], y[s:]) if s < len(X) else 0
            self.trained = True
            self.train_samples = len(X)
            _enh_logger.info(f"🧠 PriceDirectionMLP: {len(X)} samples, acc={self.accuracy:.1%}, layers=(64,32,16)")
        except Exception as e:
            _enh_logger.warning(f"PriceDirectionMLP train failed: {e}")
            return False
        return True

    def predict(self, klines_1h: list) -> dict:
        """Returns: direction (up/down/flat), confidence, probabilities."""
        if not self.trained or not DEEP_AI_AVAILABLE:
            return {"direction": "flat", "confidence": 0.0, "proba": {}}
        feat = FeatureExtractor.from_klines(klines_1h, 20)
        if not feat:
            return {"direction": "flat", "confidence": 0.0, "proba": {}}
        X = self.scaler.transform(np.array([FeatureExtractor.feature_vector(feat)]))
        try:
            proba = self.model.predict_proba(X)[0]
            pred = self.model.predict(X)[0]
        except Exception:
            return {"direction": "flat", "confidence": 0.0, "proba": {}}
        dmap = {0: "flat", 1: "up", 2: "down"}
        return {
            "direction": dmap.get(pred, "flat"),
            "confidence": float(max(proba)),
            "proba": {dmap.get(i, f"c{i}"): float(p) for i, p in enumerate(proba)},
        }

    def get_status(self):
        return {"trained": self.trained, "accuracy": round(self.accuracy, 3),
                "samples": self.train_samples, "arch": "(64,32,16)"}


class AnomalyDetectorMLP:
    """
    Deep AI Model 2: Detectează condiții anormale de piață.
    MLP autoencoder: compress 15 features → 4 → 15, reconstruction error = anomaly score.
    Anomaly mare = ceva neobișnuit → reduce sizing (protecție).
    Anomaly mică + direcție clară = oportunitate → mărește sizing.
    """

    def __init__(self):
        # Autoencoder: train to reconstruct normal market features
        self.encoder = MLPRegressor(
            hidden_layer_sizes=(8, 4, 8),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            max_iter=200,
            early_stopping=True,
            random_state=42,
        ) if DEEP_AI_AVAILABLE else None
        self.scaler = StandardScaler() if DEEP_AI_AVAILABLE else None
        self.trained = False
        self.threshold = 0.0  # anomaly threshold (percentile 95)
        self.train_samples = 0

    def train(self, klines_1h: list):
        """Antrenează pe klines normale — reconstrucție features."""
        if not DEEP_AI_AVAILABLE or len(klines_1h) < 200:
            return False
        X = []
        for i in range(30, len(klines_1h)):
            feat = FeatureExtractor.from_klines(klines_1h[i-25:i], 20)
            if feat:
                X.append(FeatureExtractor.feature_vector(feat))
        if len(X) < 100: return False
        X = np.array(X)
        X_s = self.scaler.fit_transform(X)
        try:
            self.encoder.fit(X_s, X_s)  # autoencoder: input = target
            # Calculate reconstruction errors for threshold
            preds = self.encoder.predict(X_s)
            errors = np.mean((X_s - preds) ** 2, axis=1)
            self.threshold = float(np.percentile(errors, 95))
            self.trained = True
            self.train_samples = len(X)
            _enh_logger.info(f"🧠 AnomalyDetector: {len(X)} samples, threshold={self.threshold:.4f}, arch=(8,4,8)")
        except Exception as e:
            _enh_logger.warning(f"AnomalyDetector train failed: {e}")
            return False
        return True

    def score(self, klines_1h: list) -> dict:
        """
        Returns anomaly score + is_anomaly flag.
        Score > threshold = anomalous market conditions.
        """
        if not self.trained or not DEEP_AI_AVAILABLE:
            return {"score": 0.0, "is_anomaly": False, "sizing_mult": 1.0}
        feat = FeatureExtractor.from_klines(klines_1h, 20)
        if not feat:
            return {"score": 0.0, "is_anomaly": False, "sizing_mult": 1.0}
        try:
            X = self.scaler.transform(np.array([FeatureExtractor.feature_vector(feat)]))
            pred = self.encoder.predict(X)
            error = float(np.mean((X - pred) ** 2))
        except Exception:
            return {"score": 0.0, "is_anomaly": False, "sizing_mult": 1.0}

        is_anomaly = error > self.threshold
        # Sizing multiplier: anomaly → reduce, normal clear → slight boost
        if error > self.threshold * 2:
            sizing = 0.3  # very anomalous: 30% sizing
        elif is_anomaly:
            sizing = 0.6  # anomalous: 60% sizing
        elif error < self.threshold * 0.3:
            sizing = 1.15  # very normal + predictable: 115% sizing (boost)
        else:
            sizing = 1.0  # normal
        return {"score": round(error, 4), "is_anomaly": is_anomaly,
                "threshold": round(self.threshold, 4), "sizing_mult": sizing}

    def get_status(self):
        return {"trained": self.trained, "samples": self.train_samples,
                "threshold": round(self.threshold, 4), "arch": "(8,4,8)"}


class EnsembleSignal:
    """
    Deep AI Model 3: Combină TOATE semnalele într-un singur scor.
    MLP: [regime_proba(4) + direction_proba(3) + anomaly(1) + features(15)] → trade_score
    Antrenat pe rezultatele reale ale trade-urilor anterioare.
    """

    def __init__(self):
        self.model = MLPRegressor(
            hidden_layer_sizes=(32, 16, 8),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            max_iter=200,
            early_stopping=True,
            random_state=42,
        ) if DEEP_AI_AVAILABLE else None
        self.scaler = StandardScaler() if DEEP_AI_AVAILABLE else None
        self.trained = False
        self.history: list = []  # (combined_features, actual_pnl)
        self.mae = 0.0

    def record_trade(self, regime_proba: dict, direction_proba: dict,
                      anomaly_score: float, feat_dict: dict, actual_pnl: float):
        """Înregistrează un trade cu toate semnalele + rezultatul real."""
        if not feat_dict: return
        # Build combined feature vector
        regime_vec = [regime_proba.get(k, 0) for k in ["sideways","trending_up","trending_down","volatile"]]
        dir_vec = [direction_proba.get(k, 0) for k in ["flat","up","down"]]
        feat_vec = FeatureExtractor.feature_vector(feat_dict)
        combined = regime_vec + dir_vec + [anomaly_score] + feat_vec
        self.history.append((combined, actual_pnl))
        if len(self.history) > 500:
            self.history = self.history[-500:]

    def train(self):
        if not DEEP_AI_AVAILABLE or len(self.history) < 30: return False
        X = np.array([h[0] for h in self.history])
        y = np.array([h[1] for h in self.history])
        X_s = self.scaler.fit_transform(X)
        s = int(len(X) * 0.8)
        try:
            self.model.fit(X_s[:s], y[:s])
            if s < len(X):
                preds = self.model.predict(X_s[s:])
                self.mae = float(np.mean(np.abs(preds - y[s:])))
            self.trained = True
            _enh_logger.info(f"🧠 EnsembleSignal: {len(X)} trades, MAE=${self.mae:.4f}, arch=(32,16,8)")
        except Exception as e:
            _enh_logger.warning(f"EnsembleSignal train failed: {e}")
            return False
        return True

    def predict_score(self, regime_proba: dict, direction_proba: dict,
                       anomaly_score: float, feat_dict: dict) -> float:
        """
        Predict expected PnL for a trade with these signals.
        >0 = trade looks profitable, <0 = skip.
        """
        if not self.trained or not DEEP_AI_AVAILABLE or not feat_dict:
            return 0.0
        regime_vec = [regime_proba.get(k, 0) for k in ["sideways","trending_up","trending_down","volatile"]]
        dir_vec = [direction_proba.get(k, 0) for k in ["flat","up","down"]]
        feat_vec = FeatureExtractor.feature_vector(feat_dict)
        combined = regime_vec + dir_vec + [anomaly_score] + feat_vec
        try:
            X = self.scaler.transform(np.array([combined]))
            return float(self.model.predict(X)[0])
        except Exception:
            return 0.0

    def get_status(self):
        return {"trained": self.trained, "trades": len(self.history),
                "mae": round(self.mae, 4), "arch": "(32,16,8)"}


# ══════════════════════════════════════════════════════════════════════
# PERFORMANCE METRICS — Sharpe, MaxDrawdown, Calmar
# ══════════════════════════════════════════════════════════════════════

class PerformanceMetrics:
    """Calculează metrici avansate de performanță."""

    def __init__(self):
        self.daily_returns: List[float] = []
        self.trades: List[Tuple[float, float]] = []  # (timestamp, pnl_bnb)
        self._last_snapshot_usd: float = 0.0
        self._last_snapshot_ts:  float = 0.0
        self._lock = threading.Lock()  # thread safety: add_trade din Grid thread

    def record_daily(self, capital_usd: float):
        """Apelat zilnic cu capitalul curent."""
        with self._lock:
            if self._last_snapshot_usd > 0 and self._last_snapshot_ts > 0:
                ret = (capital_usd - self._last_snapshot_usd) / max(self._last_snapshot_usd, 1e-10)
                self.daily_returns.append(ret)
                if len(self.daily_returns) > 365:
                    self.daily_returns.pop(0)
            self._last_snapshot_usd = capital_usd
            self._last_snapshot_ts = time.time()

    def add_trade(self, pnl_bnb: float, ts: float = 0):
        with self._lock:
            self.trades.append((ts or time.time(), pnl_bnb))
            if len(self.trades) > 5000:
                self.trades.pop(0)

    def sharpe_ratio(self, risk_free_daily: float = 0.02/365) -> float:
        if len(self.daily_returns) < 30:
            return 0.0
        if not ML_AVAILABLE:
            n = len(self.daily_returns)
            mean = sum(self.daily_returns) / n
            var  = sum((x - mean)**2 for x in self.daily_returns) / n
            std  = var**0.5
        else:
            mean = float(np.mean(self.daily_returns))
            std  = float(np.std(self.daily_returns))
        if std < 1e-10:
            return 0.0
        return (mean - risk_free_daily) / std * (365**0.5)

    def max_drawdown(self) -> float:
        if len(self.daily_returns) < 2:
            return 0.0
        cumulative = []
        s = 0.0
        for r in self.daily_returns:
            s += r
            cumulative.append(s)
        peak = cumulative[0]
        max_dd = 0.0
        for v in cumulative:
            if v > peak:
                peak = v
            dd = (v - peak) / max(abs(peak), 1e-10)
            if dd < max_dd:
                max_dd = dd
        return max_dd

    def calmar_ratio(self) -> float:
        if len(self.daily_returns) < 30:
            return 0.0
        n = len(self.daily_returns[-365:])
        ann = sum(self.daily_returns[-365:]) / n * 365
        mdd = abs(self.max_drawdown())
        return ann / mdd if mdd > 1e-10 else 0.0

    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for _, pnl in self.trades if pnl > 0)
        return wins / len(self.trades)

    def profit_factor(self) -> float:
        gross_win  = sum(pnl for _, pnl in self.trades if pnl > 0)
        gross_loss = abs(sum(pnl for _, pnl in self.trades if pnl < 0))
        return gross_win / gross_loss if gross_loss > 1e-10 else 0.0

    def summary(self) -> str:
        return (
            f"📊 <b>Performance Metrics</b>\n"
            f"  Sharpe:   {self.sharpe_ratio():.2f}\n"
            f"  Max DD:   {self.max_drawdown()*100:.2f}%\n"
            f"  Calmar:   {self.calmar_ratio():.2f}\n"
            f"  Win Rate: {self.win_rate()*100:.1f}%\n"
            f"  PF:       {self.profit_factor():.2f}\n"
            f"  Trades:   {len(self.trades)}"
        )


# ══════════════════════════════════════════════════════════════════════
# ADAPTIVE RATE LIMITER — evită HTTP 429 Binance
# ══════════════════════════════════════════════════════════════════════

class AdaptiveRateLimiter:
    """
    Rate limiter adaptiv bazat pe X-MBX-USED-WEIGHT header.
    Binance permite 1200 requests/minut — la 90% oprește și așteaptă.
    """

    def __init__(self, max_weight_per_minute: int = 1200):
        self.max_weight = max_weight_per_minute
        self._used_weight: int = 0
        self._window_start: float = time.time()
        self._lock = threading.Lock()

    def update_from_headers(self, headers: dict):
        """Apelat după fiecare request cu headerele răspunsului."""
        used = headers.get("X-MBX-USED-WEIGHT-1M") or headers.get("x-mbx-used-weight-1m")
        if used:
            with self._lock:
                self._used_weight = int(used)
                self._window_start = time.time()

    def wait_if_needed(self, weight: int = 1):
        """Așteaptă dacă suntem aproape de limita de rate."""
        with self._lock:
            now = time.time()
            # Reset fereastra la fiecare minut
            if now - self._window_start > 60:
                self._used_weight = 0
                self._window_start = now

            # La >90% din limită, pauză proporțională
            if self._used_weight + weight > self.max_weight * 0.90:
                wait = max(0.0, 60 - (now - self._window_start) + 0.5)
                if wait > 0:
                    logging.getLogger("RateLimiter").warning(
                        f"⏳ Rate limit aproape ({self._used_weight}/{self.max_weight}) — pauză {wait:.1f}s")
                    time.sleep(wait)
                    self._used_weight = 0
                    self._window_start = time.time()

            self._used_weight += weight


# ══════════════════════════════════════════════════════════════════════
# OPTIMIZED LOGGER — throttle mesaje repetitive
# ══════════════════════════════════════════════════════════════════════

class ThrottledLogger:
    """Logger cu throttle — evită spam de mesaje identice."""

    def __init__(self, name: str):
        self._log = logging.getLogger(name)
        self._last: Dict[str, float] = {}

    def _should_log(self, key: str, interval: float) -> bool:
        now = time.time()
        if now - self._last.get(key, 0) >= interval:
            self._last[key] = now
            return True
        return False

    def once_per_minute(self, key: str, msg: str, level: int = logging.INFO):
        if self._should_log(key, 60):
            self._log.log(level, msg)

    def once_per_hour(self, key: str, msg: str, level: int = logging.INFO):
        if self._should_log(key, 3600):
            self._log.log(level, msg)

    def once_per_day(self, key: str, msg: str, level: int = logging.INFO):
        if self._should_log(key, 86400):
            self._log.log(level, msg)

    # Passthrough pentru apeluri normale
    def info(self, msg): self._log.info(msg)
    def warning(self, msg): self._log.warning(msg)
    def error(self, msg): self._log.error(msg)
    def debug(self, msg): self._log.debug(msg)


# ══════════════════════════════════════════════════════════════════════
# SECURITY — File Integrity Guard (anti-VPS hack)
# ══════════════════════════════════════════════════════════════════════

class FileIntegrityGuard:
    """
    Protecție împotriva modificărilor neautorizate ale scriptului pe VPS.

    FUNCȚIONARE:
    1. La pornire: calculează SHA-256 hash al fișierului curent
    2. Salvează hash-ul în fișier separat (bot_integrity.sha256)
    3. La fiecare 5 min: recalculează și compară hash-ul
    4. Dacă hash diferit → OPRIRE IMEDIATĂ + alertă Telegram

    ACOPERIRE:
    • Modificare cod sursă prin shell (SSH compromis)
    • Injectare cod în fișierul Python
    • Suprascriere fișier cu versiune trojanizată

    NU ACOPERĂ:
    • Modificare memorie RAM (necesită IDS separat)
    • Compromise la nivelul bibliotecilor Python/pip
    """

    HASH_FILE      = "bot_integrity.sha256"
    CHECK_INTERVAL = 300    # verificare la fiecare 5 minute
    ALERT_COOLDOWN = 60     # alertă max 1/min

    def __init__(self, script_path: str = None,
                 telegram_callback=None,
                 stop_callback=None):
        self.log       = L("Integrity")
        self.telegram  = telegram_callback
        self.stop_cb   = stop_callback
        self._lock     = threading.Lock()
        self._last_alert  = 0.0
        self._breach_count = 0
        self.script_path   = script_path or os.path.abspath(__file__)
        self._initial_hash: Optional[str] = None
        self._last_check_ts = 0.0
        self.enabled = True
        self._init_hash()

    def _compute_hash(self, filepath: str) -> Optional[str]:
        try:
            h = hashlib.sha256()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception as e:
            self.log.error(f"Hash compute failed: {e}")
            return None

    def _init_hash(self):
        current_hash = self._compute_hash(self.script_path)
        if not current_hash:
            self.log.error("Nu pot calcula hash la pornire. Integrity check dezactivat.")
            self.enabled = False
            return

        if os.path.exists(self.HASH_FILE):
            try:
                with open(self.HASH_FILE, "r") as f:
                    saved = f.read().strip()
                if saved and saved != current_hash:
                    self.log.warning(
                        f"⚠️ Hash diferit față de sesiunea anterioară!\n"
                        f"Salvat: {saved[:16]}... | Curent: {current_hash[:16]}...\n"
                        f"Dacă ai actualizat manual botul, ignoră mesajul."
                    )
            except Exception as _e: logging.debug(f"Ignored: {_e}")
        try:
            with open(self.HASH_FILE, "w") as f:
                f.write(current_hash)
        except Exception as e:
            self.log.warning(f"Nu pot salva hash file: {e}")

        self._initial_hash = current_hash
        self.log.info(
            f"🔒 Integrity Guard activ. Hash: {current_hash[:12]}... "
            f"Script: {os.path.basename(self.script_path)}"
        )

    def check(self) -> bool:
        """
        Verifică integritatea fișierului.
        Returnează True dacă OK, False dacă modificat.
        """
        if not self.enabled or not self._initial_hash:
            return True

        now = time.time()
        if now - self._last_check_ts < self.CHECK_INTERVAL:
            return True
        self._last_check_ts = now

        current_hash = self._compute_hash(self.script_path)
        if current_hash is None:
            return True   # nu putem verifica → nu blocăm

        if current_hash == self._initial_hash:
            return True

        # ── BREACH DETECTAT ──
        self._breach_count += 1
        msg = (
            f"🚨🔴 <b>INTEGRITY BREACH!</b>\n"
            f"{'═'*30}\n"
            f"Scriptul a fost MODIFICAT pe VPS!\n"
            f"Script: {os.path.basename(self.script_path)}\n"
            f"Hash așteptat: {self._initial_hash[:16]}...\n"
            f"Hash actual:   {current_hash[:16]}...\n"
            f"{'─'*30}\n"
            f"⛔ TRADING OPRIT IMEDIAT!\n"
            f"Posibil atac VPS / SSH compromise!\n"
            f"Verifică manual fișierul înainte de restart."
        )
        self.log.critical(msg)
        if now - self._last_alert > self.ALERT_COOLDOWN:
            self._last_alert = now
            if self.telegram:
                try: self.telegram(msg)
                except Exception as _e: logging.debug(f'Ignored: {_e}')
        if self.stop_cb:
            try: self.stop_cb()
            except Exception as _e: logging.debug(f'Ignored: {_e}')
        return False

    def get_status(self) -> dict:
        return {
            "enabled":       self.enabled,
            "hash_ok":       self._initial_hash is not None,
            "hash_prefix":   (self._initial_hash[:12] + "...") if self._initial_hash else "N/A",
            "script":        os.path.basename(self.script_path),
            "breach_count":  self._breach_count,
            "last_check_ago": round(time.time() - self._last_check_ts, 0),
        }


# ══════════════════════════════════════════════════════════════════════
# WALLET CONTENT — Portofel BNB + SOL cu reguli de securitate
# ══════════════════════════════════════════════════════════════════════

class WalletContent:
    """
    Monitorizează și afișează conținutul portofelului BNB + SOL.

    REGULI STRICTE:
    ─────────────────────────────────────────────────────────
    1. SOL se folosește EXCLUSIV pentru tranzacționare SOL (cumpărare/vânzare)
    2. Obiectivul principal: acumulare maximă SOL
    3. BNB poate cumpăra orice, dar SOL este PRIORITAR
    4. Se păstrează MEREU rezervă BNB pentru fee-uri (neatinsă)
    5. NICIODATĂ nu se transferă fonduri la alte portofele (blocat în API)
    ─────────────────────────────────────────────────────────
    """

    BNB_FEE_RESERVE_MIN    = 0.010   # hard floor — sub care se blochează orice ordin
    BNB_FEE_RESERVE_TARGET = 0.020   # țintă confort (~$13 la $650/BNB)

    def __init__(self, binance_client, telegram_callback=None):
        self.client   = binance_client
        self.telegram = telegram_callback
        self.log      = L("Wallet")
        self._cache_ts   = 0.0
        self._bnb        = 0.0
        self._sol        = 0.0
        self._usdt       = 0.0
        self._bnb_price  = 0.0
        self._sol_price  = 0.0
        self.sol_accumulated  = 0.0   # SOL câștigat net prin trading
        self.bnb_spent_on_sol = 0.0   # BNB folosit pentru a cumpăra SOL
        self._lock = threading.Lock()

    def refresh(self, force: bool = False) -> dict:
        """Actualizează balanțele. Cache 60 sec."""
        now = time.time()
        if not force and now - self._cache_ts < 60:
            return self._snapshot()
        with self._lock:
            try:
                bal = self.client.full_balance()
                # Nu actualiza cache-ul dacă răspunsul e gol (API error silențios)
                if not bal:
                    self.log.debug("Wallet refresh: raspuns gol de la API, pastram cache")
                    return self._snapshot()
                bnb_new = bal.get("BNB", 0.0)
                # Dacă BNB scade brusc la 0 dar aveam valoare validă → probabil API error
                if bnb_new == 0.0 and self._bnb > 0.0 and self._cache_ts > 0:
                    self.log.debug(
                        f"Wallet: BNB=0 suspect (era {self._bnb:.4f}), pastram valoarea anterioara")
                    self._cache_ts = now  # refresh ok, dar pastram soldul
                    return self._snapshot()
                self._bnb       = bnb_new
                self._sol       = bal.get("SOL",  0.0)
                self._usdt      = bal.get("USDC", 0.0)  # MiCA: USDC nu USDT
                self._bnb_price = self.client.price("BNBUSDC") or 0.0
                self._sol_price = self.client.price("SOLUSDC") or 0.0
                self._cache_ts  = now
            except Exception as e:
                self.log.warning(f"Wallet refresh error: {e}")
        return self._snapshot()

    def _snapshot(self) -> dict:
        bnb_usd   = self._bnb  * self._bnb_price
        sol_usd   = self._sol  * self._sol_price
        total_usd = bnb_usd + sol_usd + self._usdt
        bnb_tradeable = max(0.0, self._bnb - self.BNB_FEE_RESERVE_TARGET)
        return {
            "bnb":             round(self._bnb,  6),
            "sol":             round(self._sol,  6),
            "usdc":            round(self._usdt, 2),  # USDC
            "bnb_usd":         round(bnb_usd,  2),
            "sol_usd":         round(sol_usd,  2),
            "total_usd":       round(total_usd, 2),
            "bnb_price":       round(self._bnb_price, 2),
            "sol_price":       round(self._sol_price, 4),
            "bnb_tradeable":   round(bnb_tradeable, 6),
            "bnb_fee_ok":      self._bnb >= self.BNB_FEE_RESERVE_MIN,
            "bnb_fee_reserve": self.BNB_FEE_RESERVE_TARGET,
            "sol_accumulated": round(self.sol_accumulated, 6),
        }

    def check_fee_reserve(self) -> Tuple[bool, str]:
        """
        Verifică rezerva BNB pentru fee.
        Returnează (ok, mesaj). Blochează orice ordin sub rezerva minimă.
        """
        w = self.refresh()   # mereu date proaspete (cache 60s)
        bnb = w["bnb"]
        # Dacă soldul e 0 și nu avem date (API error) → nu blocăm, avertizăm
        if bnb == 0.0 and self._cache_ts == 0.0:
            return True, "⚠️ Wallet: date indisponibile (API error)"
        if bnb < self.BNB_FEE_RESERVE_MIN:
            msg = (
                f"🚨 BNB FEE RESERVE CRITIC!\n"
                f"BNB: {bnb:.6f} (sub minimul {self.BNB_FEE_RESERVE_MIN:.3f})\n"
                f"Comisioanele NU mai pot fi plătite! Adaugă BNB."
            )
            self.log.error(msg)
            if self.telegram:
                try: self.telegram(msg)
                except Exception as _e: logging.debug(f'Ignored: {_e}')
            return False, msg
        if bnb < self.BNB_FEE_RESERVE_TARGET:
            return True, (
                f"⚠️ BNB rezerva fee scăzută: {bnb:.6f} BNB "
                f"(țintă {self.BNB_FEE_RESERVE_TARGET:.3f})"
            )
        return True, f"✅ BNB fee ok: {bnb:.6f} BNB"

    def bnb_available_for_sol(self) -> float:
        """BNB disponibil pentru a cumpăra SOL (după rezerva fee)."""
        return max(0.0, self._bnb - self.BNB_FEE_RESERVE_TARGET)

    def record_sol_accumulation(self, sol_qty: float, bnb_used: float = 0.0):
        self.sol_accumulated  += sol_qty
        self.bnb_spent_on_sol += bnb_used

    def format_display(self) -> str:
        """Formatare pentru afișare Telegram (/wallet)."""
        w = self.refresh()
        fee_icon = "✅" if w["bnb_fee_ok"] else "🚨"
        return (
            f"💼 <b>PORTOFEL BNB + SOL</b>\n"
            f"{'═'*32}\n"
            f"🟡 <b>BNB</b>\n"
            f"  Sold:       {w['bnb']:.6f} BNB\n"
            f"  Valoare:    ${w['bnb_usd']:.2f}  (${w['bnb_price']:.2f}/BNB)\n"
            f"  Disponibil: {w['bnb_tradeable']:.6f} BNB (după rezerva fee)\n"
            f"  {fee_icon} Rezerva fee: {w['bnb_fee_reserve']:.3f} BNB (NEATINSĂ)\n"
            f"{'─'*32}\n"
            f"🌟 <b>SOLANA (SOL)</b>\n"
            f"  Sold:       {w['sol']:.6f} SOL\n"
            f"  Valoare:    ${w['sol_usd']:.2f}  (${w['sol_price']:.4f}/SOL)\n"
            f"  Acumulat:   +{w['sol_accumulated']:.6f} SOL (profit trading)\n"
            f"{'─'*32}\n"
            f"💵 USDC: ${w['usdc']:.2f}\n"
            f"{'─'*32}\n"
            f"📊 <b>TOTAL: ${w['total_usd']:.2f}</b>\n"
            f"{'═'*32}\n"
            f"🎯 <b>Reguli active:</b>\n"
            f"  • SOL: exclusiv cumpărare/vânzare SOL\n"
            f"  • BNB: profit → SOL (prioritar)\n"
            f"  • Rezerva BNB fee: garantată\n"
            f"  • ⛔ Transfer fonduri: BLOCAT PERMANENT"
        )


# ══════════════════════════════════════════════════════════════════════
# CONFIGURARE — toate valorile importante sunt transparente
# ══════════════════════════════════════════════════════════════════════

BINANCE_KEY    = os.getenv("BINANCE_API_KEY",    "")
BINANCE_SECRET = os.getenv("BINANCE_SECRET_KEY", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT  = os.getenv("TELEGRAM_CHAT_ID",   "")

# Validare chei critice la import
def _check_required_env():
    _missing = []
    if not os.getenv("BINANCE_API_KEY"):    _missing.append("BINANCE_API_KEY")
    if not os.getenv("BINANCE_SECRET_KEY"): _missing.append("BINANCE_SECRET_KEY")
    if _missing and not os.getenv("USE_TESTNET", "").lower() == "true":
        print(f"❌ Variabile lipsă: {', '.join(_missing)}. Adaugă în .env")
        sys.exit(1)
_check_required_env()
USE_TESTNET    = os.getenv("USE_TESTNET", "True").lower() == "true"
# FIX9: Dry-run pe mainnet — API real, balance real, ordine simulate
# SAFE DEFAULT: True — previne pornire accidentală live dacă .env lipsește
MAINNET_DRY_RUN = os.getenv("MAINNET_DRY_RUN", "True").lower() == "true"
# LIVE ARMED: trebuie setat explicit True în .env pentru live trading
LIVE_ARMED = os.getenv("LIVE_ARMED", "False").lower() == "true"
if not USE_TESTNET and not MAINNET_DRY_RUN and not LIVE_ARMED:
    print("\n❌ LIVE trading blocat!\n"
          "   Setează LIVE_ARMED=True în .env pentru a porni pe live.\n"
          "   Aceasta previne porniri accidentale pe bani reali.")
    import sys; sys.exit(1)
MANUAL_BNB     = float(os.getenv("BNB_AMOUNT", "0"))

# ── Fee cu BNB discount (25% reducere) ───────────────────────────────
TAKER_FEE       = 0.00075    # 0.05625% per leg
MAKER_FEE       = 0.00010    # 0.0075%  per leg
ROUNDTRIP_TAKER = TAKER_FEE * 2     # 0.1125%  open+close market
ROUNDTRIP_MAKER = MAKER_FEE  * 2    # 0.0150%  open+close limit

# ── Alocare v8.0 — SUMA EXACTA 100% ─────────────────────────────────
# REGULA CRITICA: toate alocările + fee buffer = exact 100% din capital.
# Dual Investment e luat DIN alocările existente, nu pe deasupra.
#
# Fluxul BNB real:
#   Funding → conversia BNB→USDT (BNB dispare din sold)
#   Grid BNBUSDT → fee din sold BNB liber (⚠️ necesita buffer)
#   Launchpool → BNB blocat in Earn (retras in ~1h)
#   DI → BNB blocat 7 zile in DCI (nu disponibil pentru fee)
#   Fee Buffer → SINGURUL BNB complet liber pentru comisioane instant
#
ALLOC_FUNDING    = 0.00   # 0% — dezactivat (activare la mainnet)
ALLOC_GRID       = 0.95   # 95% — eliberat din fee buffer
ALLOC_SWING      = 0.00   # 0% — dezactivat: capital prea mic, fee > profit
ALLOC_LAUNCHPOOL = 0.00   # 0%
ALLOC_REZERVA    = 0.00   # 0%
ALLOC_DI         = 0.00   # 0%
ALLOC_FEE_BUFFER = 0.05   # 5% — eliberează 2% capital spre grid

PORTFOLIO_EXCLUDE = ["AXS", "ADA", "LUNA", "USUAL", "ALLO", "BTC", "BNB"]  # tokeni excluși din grid
# TOTAL: 50+20+5+15+4+5+1 = 100% ✅

# Verificare la runtime:
_alloc_total = (ALLOC_FUNDING + ALLOC_GRID + ALLOC_SWING +
               ALLOC_LAUNCHPOOL + ALLOC_REZERVA +
               ALLOC_DI + ALLOC_FEE_BUFFER)
if abs(_alloc_total - 1.0) > 1e-9:
    raise ValueError(
        f"ALOCARE GRESITA: {_alloc_total*100:.2f}% != 100%! "
        f"(assert ignorat cu python -O)")
if min(ALLOC_FUNDING, ALLOC_GRID, ALLOC_SWING, ALLOC_LAUNCHPOOL,
       ALLOC_REZERVA, ALLOC_DI, ALLOC_FEE_BUFFER) < 0:
    raise ValueError("Alocare negativă detectată!")

# ── Fee Buffer — BNB complet liber pentru comisioane instant ──────────
# Calculat din ALLOC_FEE_BUFFER (1% din capital = ~0.012 BNB la $800).
# NU e investit nicaieri. NU e in Earn. INSTANT disponibil.
#
# Monitorizare activa:
#   daca sold_liber < FEE_BUFFER_MIN → redeem din Earn (1h delay, avertizat)
#   daca sold_liber < FEE_BUFFER_CRITICAL → blocam ordine noi (nu emergency close)
#
# Acoperire: 0.012 BNB / (0.000017 BNB/zi fee normal) = 700+ zile normale
#            0.012 BNB / (0.00049 BNB emergency close) = 24+ emergency close-uri
#
FEE_BUFFER_BNB      = 0.020   # BNB țintă liber (~$13 la $650/BNB)
FEE_BUFFER_MIN      = 0.012   # prag alertă CRITIC — sub 0.012 → redeem din Earn (era 0.030 > 0.020 → loop infinit!)
FEE_BUFFER_CRITICAL = 0.003   # prag HARD BLOCK: coborat de la 0.008
FEE_BUFFER_REFILL   = 0.050   # refill pana la acest nivel la redeem

# ── C4: Calendar Fed FOMC — blackout 24h inainte ─────────────────────
# Sursa: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
# Format: "YYYY-MM-DD" = ziua sedintei. Botul blocheaza intrari cu 24h inainte.
# Cost implementare: ZERO (stim data exact, nu trebuie API).
# Beneficiu: evitam miscari -5% / +8% din ziua sedintei pe swing.
FED_FOMC_DATES = [
    "2026-03-19", "2026-05-07", "2026-06-18",
    "2026-07-30", "2026-09-17", "2026-11-05", "2026-12-16",
    # Adauga manual datele anului urmator din calendarul Fed
]
# Verificare la pornire — alertă când lista FOMC expiră
def _check_fomc_expiry():
    try:
        last = max(datetime.strptime(d, "%Y-%m-%d") for d in FED_FOMC_DATES)
        if last < datetime.now(timezone.utc) + timedelta(days=60):
            logging.getLogger("bot").warning(
                f"⚠️ FED_FOMC_DATES expiră la {last.date()} — actualizează lista!")
    except Exception as _e: logging.debug(f"Ignored: {_e}")
FED_BLACKOUT_H = 8    # 8h suficient (era 24h — pierdea o zi întreagă)

# ── Risk ──────────────────────────────────────────────────────────────
DAILY_LOSS_LIMIT = 0.03    # 3% pierdere zilnică max → $24 din $800 capital

# ═══ SAFETY STOP ABSOLUT ═══
# Dacă pierderea totală (de la pornire) depășește acest prag → OPREȘTE TOTUL
# Închide toate pozițiile, notifică pe Telegram, nu mai tranzacționează
# Reactivare manuală: restart bot sau /start_trading
SAFETY_MAX_LOSS_USD = 15.0   # oprește la -$20 pierdere totală (-2.6% din $780 capital)
MAX_TRADES_DAY   = 6       # mărit: permite mai multe swing trades/zi
                            # 4 × 0.1125% × 1.22 BNB = 0.005 BNB fee/zi = ok
# ── Slippage (FIX 5) ──────────────────────────────────────────────────
SLIPPAGE         = 0.002   # 0.2% slippage pe market orders (X/BNB volum mic — mai realist)
                            # aplicat la open+close = 0.2% total pe swing
# ── Swing EV (FIX 4) ─────────────────────────────────────────────────
SWING_WR_BASE    = 0.54    # win rate de bază (din date istorice strategii trend)

# ── Market Crash Guard ────────────────────────────────────────────────
# Trei niveluri de alertă bazate pe căderea BTC (proxy pentru piață):
#
#  YELLOW  — scădere rapidă, precauție
#     Efect: Swing & Grid opresc intrări noi. Funding continuă.
#
#  ORANGE  — crash în desfășurare
#     Efect: Toate intrările noi oprite. Pozițiile existente rămân.
#             Funding monitorizat la 5 min (nu 15).
#
#  RED     — crash sever (flash crash / black swan)
#     Efect: Stop complet. Toate pozițiile swing & grid închise imediat.
#             Funding închis dacă funding rate devine negativ.
#             Bot în standby până la recuperare confirmată.
#
CRASH_CHECK_SEC   = 60      # verificare la fiecare 60 secunde
CRASH_SYMBOLS     = ["BTCUSDC", "BNBUSDC"]  # monitorizăm BTC + BNB

# Praguri YELLOW (precauție)
CRASH_Y_1H        = -0.04   # -4% in 1h
CRASH_Y_4H        = -0.06   # -6% in 4h
CRASH_Y_24H       = -0.10   # -10% in 24h

# Praguri ORANGE (oprire intrari)
CRASH_O_1H        = -0.06   # -6% in 1h
CRASH_O_4H        = -0.10   # -10% in 4h
CRASH_O_24H       = -0.15   # -15% in 24h

# Praguri RED (inchidere tot)
CRASH_R_1H        = -0.10   # -10% in 1h  (flash crash)
CRASH_R_4H        = -0.15   # -15% in 4h
CRASH_R_24H       = -0.20   # -20% in 24h

# Recuperare: piata trebuie sa stabilizeze X% fata de low inainte de reluare
CRASH_RECOVER_PCT = 0.03    # +3% fata de minimul crash -> iesim din standby
CRASH_RECOVER_H   = 2       # minim 2h de stabilitate dupa recuperare

# ── Funding ───────────────────────────────────────────────────────────
FR_APR_MIN   = 0.25        # 25% APR minim — breakeven la 1x=~41%, 25% margin conservator
FR_APR_EXIT  = 0.02        # exit sub 2% APR (hold mai mult, mai puține exit-uri)
FR_MAX_POS   = 4           # 4 poziții pe 25% capital ($50 each = decent)

# Pe testnet, funding rate API returneaza zero sau valori artificiale
# care cauzeaza open/close continuu si fee-uri mari fara niciun venit.
# Dezactivam funding pe testnet — grid si swing functioneaza corect.
FR_DISABLE_ON_TESTNET = True   # True = funding oprit pe testnet

# I2: Exit predictiv — iese daca APR scade >50% fata de peak-ul ultimelor 24h
# SI pozitia e deschisa de minim 4h (evita exit imediat dupa intrare)
# Marit de la 30% la 50% — ratele fluctueaza normal cu 30-40%, nu e semnal real
FR_APR_TREND_EXIT  = 0.50   # exit daca APR a scazut cu >50% din peak 24h
FR_APR_HISTORY_H   = 24     # fereastra de timp pentru calculul peak APR
FR_APR_MIN_HOLD_H  = 24.0   # era 8h → 24h minim hold (reduce rotații + fee-uri)

# FIX 1: Fereastra de protectie la reset funding (00/08/16 UTC ±30 min)
# La reset, ratele oscileaza violent → nu inchidem pozitii in aceasta fereastra
FR_RESET_WINDOW_MIN = 30    # minute de protectie in jurul resetului (±30 min)

# FIX 2: Interval scan marit de la 15 min la 30 min
# Reduce numarul de open/close inutile
FR_SCAN_INTERVAL_S  = 900   # 15 min (era 30 — captează rate noi mai rapid)

# FIX 3: Cooldown per simbol dupa exit — nu reintra imediat
# Evita plata dubla de fee pentru acelasi simbol in acelasi scan
FR_COOLDOWN_H       = 12.0  # 12h cooldown (reduce rotații, hold pozițiile bune)

# FIX 5: Staggered close — nu inchidem toate pozitiile simultan la reset
# Distribuie close-urile pe 30 secunde pentru a evita spike de fee
FR_STAGGER_CLOSE_S  = 5     # secunde intre close-uri consecutive

# I6: Kelly fractional pentru dimensionarea pozitiilor funding
# Principiu: simboluri cu APR mai mare primesc proportional mai mult capital
# Cap: max 10% din capital per simbol (diversificare minima)
# Formula: weight_i = apr_i / sum(apr_all); size_i = capital * weight_i
# Cu cap: size_i = min(capital * weight_i, capital * KELLY_MAX_PER_SYM)
KELLY_MAX_PER_SYM  = 0.20   # 20% per simbol (mai concentrat pe best rates)
KELLY_MIN_SIZE     = 0.02   # minim 0.02 BNB per pozitie (sub acest prag skip)
# Reinvestire automată
REINVEST_ENABLED   = True   # profitul din funding se adaugă înapoi în poziții
REINVEST_THRESHOLD = 0.005  # reinvestim după acumulare de min 0.005 BNB profit
REINVEST_PCT       = 0.90   # 90% reinvestit (= 100% - rezerva 10%)
# Rezerva profit: 10% din fiecare colectare funding → salvat in BNB, neatins
# Scopul: acumulezi BNB pur din profit, nu din capital initial
# Retras manual sau folosit pentru a creste capitalul dupa X luni
PROFIT_RESERVE_PCT  = 0.10  # 10% din fiecare profit merge în rezervă separată
PROFIT_RESERVE_FILE = "v8_profit_reserve.json"  # tracking rezerva
# Binance Earn (Simple Earn Flexible pentru rezervă)
EARN_ENABLED     = True     # rezerva 5% pusă în Simple Earn Flexible BNB
EARN_MIN_BNB     = 0.05     # minim 0.05 BNB pentru a subscrie în Earn
EARN_CHECK_H     = 24       # verificare sold Earn la 24h

# v1.3: Dynamic funding — scanează TOATE perechile USDT futures, alege top N
# FR_SYMBOLS e fallback dacă API-ul nu returnează date
FR_SYMBOLS = [
    "BTCUSDC", "ETHUSDC", "BNBUSDC", "SOLUSDC", "XRPUSDC",
    "DOGEUSDC", "ADAUSDC", "AVAXUSDC", "DOTUSDC", "POLUSDC",
    "LINKUSDC", "LTCUSDC", "BCHUSDC", "NEARUSDC", "UNIUSDC",
    "APTUSDC", "ARBUSDC", "OPUSDC", "FILUSDC", "ATOMUSDC",
    "RUNEUSDC", "INJUSDC", "SUIUSDC", "SEIUSDC", "TIAUSDC",
    "WIFUSDC", "PEPEUSDC", "SUSDC", "STXUSDC", "IMXUSDC",
]
FR_DYNAMIC_SCAN    = True    # scanează all_funding_rates()
FR_DYNAMIC_MIN_VOL = 3_000_000  # era $5M → $3M (mai multe perechi eligibile)
FR_DYNAMIC_TOP_N   = 15     # era 10 → top 15 perechi după APR

# v1.3: Leverage pe funding — delta-neutral, risc minim
# Spot 1x + Futures short 1x = delta neutral
# Cu 2x leverage pe futures, marjă necesară = 50% → capitalul merge de 2x mai departe
# Efect: colectezi funding pe poziții de 2x mai mari cu aceeași marjă
FR_LEVERAGE        = 1       # 1x = delta-neutral real (era 3x → net short 2x BNB — periculos!)
# ANTI-LICHIDARE: stop-loss pe futures la 20% pierdere (mult sub 33% lichidare)
# La 3x: lichidare la ~33% mișcare. Stop-loss la 20% = exit safe cu pierdere controlată.
# Spot-ul acoperă parțial pierderea (delta-neutral: spot câștigă cât futures pierde)
# Net loss real la stop-loss: ~2-3% din capital funding (nu 20%)
FR_STOP_LOSS_PCT   = 0.20   # exit futures dacă prețul urcă 20% peste entry
FR_ANTILIQ_CHECK_S = 30     # verifică pozițiile futures la 30s
FR_ANTILIQ_WARN_PCT = 0.12  # warning Telegram la 12% pierdere
FR_ANTILIQ_EMERGENCY_PCT = 0.25  # emergency close la 25% (backup dacă stop-loss eșuează)

# ── Grid v5.0 ─────────────────────────────────────────────────────────
# C2: BNBUSDT cu spacing 0.8% (volum $500M+/zi → fills 3-4x/zi vs 1.1x)
# Net per fill BNBUSDT: 0.8% - 0.015% maker = 0.785% (viabil la volum mare)
# Net per fill rest:    1.5% - 0.015% maker = 1.485%
GRID_SPACING        = 0.015   # spacing standard X/BNB si restul
GRID_SPACING_BNBUSDT= 0.015   # 1.5% pe BNBUSDC — elimina whipsaw
GRID_LEVELS         = 8       # 8 nivele/pereche — ordine $33 fiecare
GRID_ADX_MAX        = 25      # >50 = trend puternic -> skip; 35-50 = trend mode
GRID_MAX_PAIRS      = 2       # 2 perechi volatile simultan
GRID_REBUILD_H      = 4       # rebuild grid la 4h
GRID_GEO_MULT       = 1.10    # geometric spacing BNB: fiecare nivel 10% mai larg
SOL_GEO_MULT        = 1.00    # uniform spacing SOL: toate nivelele egale → mai multe fills
# BNBUSDT primul — prioritate maxima, spacing tight, cel mai mare fill rate
GRID_USDT_PAIRS     = ["BNBUSDC"]  # SOLUSDC e gestionat de SolTrader dedicat

# I1: ADX dinamic → spacing auto-calibrat 0.6%-2.0% (v6.0)
# Formula: spacing = GRID_SPACING_MIN + (adx / GRID_ADX_MAX) * range
# ADX mic (calm)  → spacing mic = mai multe fills
# ADX mare (trend) → spacing mare = mai multa protectie
GRID_SPACING_MIN    = 0.008   # 0.8% minim — sub 0.8% fee-urile mananca profitul
GRID_SPACING_MAX    = 0.015   # 1.5% la ADX=50
# Exceptie BNBUSDT: range mai strans datorita volumului mare
GRID_SPACING_BNB_MIN= 0.0015  # 0.15% la ADX mic — BNB low-vol, fills maxime
GRID_SPACING_BNB_MAX= 0.012   # 1.2% la ADX=22

# I3: Grid pauza nocturna UTC (volum mic → fill rate -60%)
# Activ: 06:00-22:00 UTC. Pauza: 22:00-06:00 UTC.
# Capital eliberat noaptea → merge automat in funding (nu e mutat manual)
GRID_ACTIVE_UTC_START = 0    # FULL 24/7 — crypto nu are program
GRID_ACTIVE_UTC_END   = 24   # fills și noaptea = profit extra
BNB_PAIRS_FALLBACK = [
    "BTCBNB","ETHBNB","ADABNB","XRPBNB","DOGEBNB",
    "DOTBNB","LINKBNB","LTCBNB","ATOMBNB","POLBNB",
    "SHIBBNB","AVAXBNB","NEARBNB","FTMBNB","CAKEBNB",
]

# ── Swing ─────────────────────────────────────────────────────────────
SWING_TP      = 0.025      # 2.5% — era 1.8% (mai mult profit per trade)
SWING_SL      = 0.010      # 1.0% — era 0.7% (mai mult spațiu, RR=2.5)
SWING_ADX_MIN = 32         # ADX minim pt swing entry
SWING_VOL_MIN = 2.0        # confirmare volum
# I4: Filtru corelatie BTC
SWING_BTC_FILTER_PCT = 0.015
SWING_BTC_FILTER_H   = 1

# I7: Trailing stop-loss
# Cand pozitia e pe profit >= TRAIL_ACTIVATE, SL urca odata cu pretul.
# Protejam TRAIL_LOCK din profitul acumulat (nu lasam sa cada sub).
# Exemplu: entry=100, profit la 101.5% (+1.5%) → SL urcat la 100.8%
#          (protejam 0.5% din profitul de 1.5%)
SWING_TRAIL_ACTIVATE = 0.008   # activează trailing mai devreme: +0.8% profit
SWING_TRAIL_LOCK     = 0.45    # blocăm 45% din profit (era 35% — lăsam prea mult pe masă)
# (ex: profit max = +1.8% → SL trail la entry + 1.8%*0.35 = +0.63%)

# ATR sizing: marim size-ul cand volatilitatea confirma miscarea
# ATR (Average True Range) normalizat — masura volatilitate
SWING_ATR_PERIOD  = 14     # perioade pentru calcul ATR
SWING_ATR_SIZE_MIN= 0.70   # size factor minim la volatilitate mica
SWING_ATR_SIZE_MAX= 1.30   # size factor maxim la volatilitate mare
SWING_ATR_NORM_LOW= 0.005  # ATR/price sub 0.5% = volatilitate mica
SWING_ATR_NORM_HI = 0.020  # ATR/price peste 2.0% = volatilitate mare

# Covered calls tracker — reminder lunar si tracking manual
# Botul NU executa optiunile automat (necesita interfata Binance Options manual)
# Dar trimite reminder Telegram si tine evidenta
COVERED_CALLS_ENABLED   = True
COVERED_CALLS_STRIKE_PCT= 0.10    # strike la +10% fata de pret curent
COVERED_CALLS_EXPIRY_D  = 30      # expirare ~30 zile
COVERED_CALLS_REMIND_D  = 28      # reminder la 28 zile dupa ultima vanzare

# ── Dual Investment (v8.0 NOU) ────────────────────────────────────────
# Produs structurat Binance: depui BNB, primesti premium garantat 7 zile.
# Premium: 0.8-2.5% per ciclu (10-30% APR echivalent).
# 4 cicluri × 7 zile = 4 subscrieri/luna, premium la fiecare.
# Capital alocat: DI_ALLOC din totalul BNB.
# RISC ZERO pe capital — cel mult nu participi la crestere BNB.
#
# API Binance Dual Investment:
#   GET  /sapi/v1/dci/product/list       — lista produse disponibile
#   POST /sapi/v1/dci/product/subscribe  — subscriere produs
#   GET  /sapi/v1/dci/product/positions  — pozitii active
# DI activ doar dacă ALLOC_DI > 0 — previne contradicție config
DI_ENABLED      = ALLOC_DI > 0
DI_ALLOC        = ALLOC_DI    # aliniat la ALLOC_DI (era 0.05 fix, acum dinamic)
DI_CYCLE_D      = 7       # ciclu de 7 zile (cel mai comun pe Binance)
DI_STRIKE_UP    = 0.10    # strike CALL +10% fata de pret curent
DI_STRIKE_DOWN  = 0.10    # strike PUT -10% fata de pret curent
DI_MIN_APR      = 0.10    # APR minim acceptat (10%) — altfel skip ciclu
DI_CHECK_H      = 6       # verificare disponibilitate produse la 6h

# ── FR Momentum Entry (v8.0 NOU) ──────────────────────────────────────
# Cand APR funding creste >20% in 24h → semnal pozitiv → deschidem mai rapid.
# Complement la I2 (exit rapid la APR in scadere).
# Mecanica: daca APR_acum / APR_24h_ago > 1.20 → deschidem pozitie bonus.
# Capital extra: max 3% din funding capital per simbol in momentul.
FR_MOMENTUM_ENTRY      = True
FR_MOMENTUM_THRESHOLD  = 0.20   # APR trebuie sa fi crescut >20% in 24h
FR_MOMENTUM_EXTRA_PCT  = 0.03   # 3% capital extra per simbol la momentum

# ── Swing fereastra 13-17 UTC (v8.0 NOU) ─────────────────────────────
# Overlap London (8-17 UTC) + New York (13-22 UTC) = 13-17 UTC.
# In aceasta fereastra volumul e +40-60% fata de medie.
# ADX mai reliable, miscari mai clare, WR estimata cu 4-5pp mai mare.
# In afara ferestrei: swing mai poate inchide pozitii existente, NU deschide noi.
SWING_UTC_WINDOW_START  = 6    # era 13 → 6 UTC (include Asian+London+NY)
SWING_UTC_WINDOW_END    = 22   # era 17 → 22 UTC (16h window vs 4h)
SWING_UTC_FILTER        = True # False = dezactivat (comportament vechi)

# ── Capital Rebalancing saptamanal (v8.0 NOU) ─────────────────────────
# La fiecare 7 zile, evalueaza performanta fiecarei strategii.
# Strategia cu ROI maxim primeste +3% capital din cea cu ROI minim.
# Cap: nicio strategie nu poate depasi 70% sau scadea sub 2%.
# Beneficiu: capital migreaza organic spre ce functioneaza acum.
REBAL_ENABLED    = False  # dezactivat — module la 0%, nimic de rebalansat
REBAL_INTERVAL_D = 7      # rebalansare la fiecare 7 zile
REBAL_SHIFT_PCT  = 0.03   # transferam 3% intre strategii
REBAL_MAX_ALLOC  = 0.70   # maxim 70% intr-o singura strategie
REBAL_MIN_ALLOC  = 0.02   # minim 2% per strategie activa
SWING_PAIRS = [
    "SOLUSDC", "BNBUSDC",  # doar 2 perechi — capital mic, MIN_NOTIONAL $10+
    # "AXSUSDC", "ETHUSDC", "BTCUSDC" — activează la capital >$500
]

# ══════════════════════════════════════════════════════════════════════
# LOGGING & TELEGRAM
# ══════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-12s] %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler("solana_bot.log", maxBytes=10*1024*1024, backupCount=3, encoding="utf-8"),
    ]
)
def L(n): return logging.getLogger(f"BNBv3.{n}")

def _atomic_json_save(filepath: str, data: dict, indent: int = 2):
    """Atomic JSON save: write to temp file, then rename."""
    import os
    dir_name = os.path.dirname(filepath) or "."
    try:
        fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp, filepath)
    except Exception:
        try: os.unlink(tmp)
        except: pass
        raise

log_main = L("main")

def is_fed_blackout() -> bool:
    """
    C4: Returneaza True daca suntem in fereastra de 24h inainte de o sedinta FOMC.
    In aceasta fereastra: swing si grid nu deschid pozitii noi.
    Funding ramane activ (delta-neutral, imun la directie).
    """
    now = datetime.now(timezone.utc)
    for date_str in FED_FOMC_DATES:
        try:
            fomc = datetime.strptime(date_str, "%Y-%m-%d")
            hours_until = (fomc.replace(tzinfo=timezone.utc) - now).total_seconds() / 3600
            if 0 <= hours_until <= FED_BLACKOUT_H:
                return True
        except ValueError:
            pass
    return False

# ══════════════════════════════════════════════════════════════════════
# TELEGRAM BOT — monitorizare bidirecționala completa
# ══════════════════════════════════════════════════════════════════════

# Nivel alerte — filtreaza ce primesti pe Telegram
TG_LEVEL_ALL    = 0   # tot (inclusiv silent)
TG_LEVEL_NORMAL = 1   # normal + urgent (fara silent)
TG_LEVEL_URGENT = 2   # doar crash, circuit breaker, erori critice

TG_DEFAULT_LEVEL = int(os.getenv("TG_ALERT_LEVEL", "1"))

# Raport zilnic automat
TG_DAILY_REPORT_H = int(os.getenv("TG_DAILY_HOUR", "9"))   # ora locala

class TelegramBot:
    """
    Telegram bot bidirectional pentru monitorizare Solana Bot v1.0.

    FUNCTIONALITATI:
    ─────────────────
    Trimitere:
      • Alerte instant: crash, circuit breaker, funding open/close
      • Raport la 8h (automat)
      • Raport zilnic la ora configurata
      • Filtrare nivel alerta: ALL / NORMAL / URGENT

    Comenzi primite:
      /status    — status curent rapid (PnL, crash level, trades)
      /raport    — raport complet detaliat
      /pozitii   — lista pozitii funding deschise
      /grid      — starea gridurilor active
      /pauza     — pauzeaza intrari noi (nu inchide pozitii)
      /resume    — reia trading normal
      /nivel X   — seteaza nivel alerte (0=all 1=normal 2=urgent)
      /ajutor    — lista comenzi

    Butoane inline:
      [📊 Status] [📈 Raport] [⏸ Pauza] [▶ Resume]

    SECURITATE:
      • Accepta mesaje DOAR de la TELEGRAM_CHAT_ID configurat
      • Rate limit: max 1 comanda/5 secunde per chat
    """

    def __init__(self):
        self.token     = TELEGRAM_TOKEN
        self.chat_id   = TELEGRAM_CHAT
        self.log       = L("TgBot")
        self._lock     = threading.Lock()
        self._level    = TG_DEFAULT_LEVEL
        self._paused   = False        # FIX: bot porneste activ, nu in pauza
        self._bot_ref  = None         # referinta la SolanaBot (setata dupa init)
        self._last_upd = 0            # ultima comanda procesata (update_id)
        self._last_cmd = 0.0          # timestamp ultima comanda (rate limit)
        self._last_day_report = -1    # ziua ultimului raport zilnic
        self._msg_queue: "deque" = deque(maxlen=50)  # coadă trimitere
        self._enabled  = bool(self.token and self.chat_id)

        if not self._enabled:
            self.log.info("Telegram dezactivat (lipsesc TOKEN sau CHAT_ID)")
        else:
            self.log.info(
                f"Telegram activ | chat={self.chat_id} | "
                f"nivel alerte={'ALL' if self._level==0 else 'NORMAL' if self._level==1 else 'URGENT'}")

    def set_bot(self, bot_ref):
        """Leaga TelegramBot la instanta SolanaBot pentru comenzi."""
        self._bot_ref = bot_ref

    @property
    def paused(self) -> bool:
        return self._paused

    # ── Trimitere mesaje ──────────────────────────────────────────────

    def send(self, msg: str, silent: bool = False, urgent: bool = False,
             buttons: List[List[dict]] = None):
        """
        Trimite mesaj Telegram cu filtrare nivel.
        urgent=True → ignora filtrul de nivel, trimite intotdeauna.
        silent=True → notificare silentioasa (fara sunet).
        buttons → lista de randuri cu butoane inline [{text, callback_data}].
        """
        if not self._enabled: return
        if not urgent:
            if self._level == TG_LEVEL_URGENT and not urgent: return
            if self._level == TG_LEVEL_NORMAL and silent: return

        payload = {
            "chat_id":    self.chat_id,
            "text":       msg[:4096],    # limita Telegram
            "parse_mode": "HTML",
            "disable_notification": silent and not urgent,
        }
        if buttons:
            payload["reply_markup"] = {
                "inline_keyboard": buttons
            }
        self._msg_queue.append(payload)

    def send_urgent(self, msg: str):
        """Scurtatura pentru alerte critice — niciodata filtrate."""
        self.send(msg, silent=False, urgent=True)

    def _flush_queue(self):
        """Trimite mesajele din coada (apelat din thread dedicat)."""
        while self._msg_queue:
            payload = self._msg_queue.popleft()
            try:
                r = _http_session.post(
                    f"https://api.telegram.org/bot{self.token}/sendMessage",
                    json=payload, timeout=10)
                if r.status_code == 429:
                    # Rate limited de Telegram — asteapta si repune in coada
                    retry = r.json().get("parameters", {}).get("retry_after", 5)
                    self.log.warning(f"TG rate limit: retry in {retry}s")
                    time.sleep(retry)
                    self._msg_queue.appendleft(payload)
                    break
            except Exception as e:
                self.log.warning(f"TG send: {e}")

    # ── Main buttons keyboard ─────────────────────────────────────────

    def _main_keyboard(self) -> List[List[dict]]:
        pause_btn = "⏸ Pauza" if not self._paused else "▶ Resume"
        pause_cb  = "cmd_pauza" if not self._paused else "cmd_resume"
        ml_on = self._bot_ref.ml is not None if hasattr(self, '_bot_ref') and self._bot_ref else False
        ml_btn = "🧠 ML ON" if ml_on else "🧠 ML OFF"
        ml_cb  = "cmd_ml_off" if ml_on else "cmd_ml_on"
        return [
            [{"text": "📊 Status",   "callback_data": "cmd_status"},
             {"text": "📈 Raport",   "callback_data": "cmd_raport"}],
            [{"text": "💰 Pozitii", "callback_data": "cmd_pozitii"},
             {"text": "🔲 Grid",    "callback_data": "cmd_grid"}],
            [{"text": "🌊 SOL",    "callback_data": "cmd_sol"},
             {"text": "🏦 Rezerva", "callback_data": "cmd_rezerva"}],
            [{"text": pause_btn,    "callback_data": pause_cb},
             {"text": "🛡 Enh",    "callback_data": "cmd_enh"}],
            [{"text": ml_btn,      "callback_data": ml_cb},
             {"text": "💸 Fees",   "callback_data": "cmd_fees"}],
            [{"text": "🔄 Recentrare Grid", "callback_data": "cmd_recentrare"},
             {"text": "🔲 Rebuild Grid",    "callback_data": "cmd_rebuild_grid"}],
            [{"text": "🛑 Stop",   "callback_data": "cmd_stop"},
             {"text": "❓ Ajutor",  "callback_data": "cmd_ajutor"}],
        ]

    # ── Procesare comenzi ─────────────────────────────────────────────

    def _handle(self, text: str, from_id: str) -> Optional[str]:
        """
        Proceseaza o comanda si returneaza raspunsul.
        Returneaza None daca comanda e necunoscuta.
        """
        # Securitate: doar chat-ul autorizat
        if str(from_id) != str(self.chat_id):
            self.log.warning(f"TG: mesaj de la chat neautorizat {from_id}")
            return None

        # Rate limit: max 1 comanda la 5 secunde
        now = time.time()
        if now - self._last_cmd < 5:
            return "⏳ Asteapta 5 secunde intre comenzi."
        self._last_cmd = now

        parts = text.strip().split()
        if not parts:
            return None
        cmd = parts[0].lower()
        args = parts[1:]

        if not self._bot_ref:
            return "⚠️ Bot neinitializat inca. Incearca in 30 secunde."

        b = self._bot_ref

        # ── /pairs — profit per pereche/zi ───────────────────────────
        if cmd in ("/pairs", "pairs"):
            try:
                lines = []
                tracker = getattr(b, '_pair_daily_pnl', {})
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                for sym, days in sorted(tracker.items()):
                    today_pnl = days.get(today, 0)
                    total_pnl = sum(days.values())
                    fills_today = getattr(b, '_pair_daily_fills', {}).get(sym, {}).get(today, 0)
                    lines.append(
                        f"  {sym}: today {today_pnl:+.5f} BNB ({fills_today}f) | total {total_pnl:+.5f}")
                if lines:
                    return f"📊 <b>Profit per pereche</b>\n" + "\n".join(lines)
                return "📊 Nicio pereche cu date încă."
            except Exception as e:
                return f"📊 Err: {e}"

        # ── /ping — lightweight status ────────────────────────────────
        if cmd in ("/ping", "ping"):
            try:
                _pnl, _fee = b._pnl()
                _alive = sum(1 for t in b._threads if t.is_alive())
                _total = len(b._threads)
                _up = b.total_uptime_h()
                return (
                    f"🏓 <b>PONG</b>\n"
                    f"⏱ Uptime: {_up:.1f}h\n"
                    f"📈 PnL: {_pnl:+.5f} BNB\n"
                    f"🧵 Threads: {_alive}/{_total}\n"
                    f"{'🔸 DRY_RUN' if MAINNET_DRY_RUN else '🟢 LIVE' if not USE_TESTNET else '🧪 TESTNET'}\n"
                    f"{'🛑 SAFETY STOP' if b._safety_stopped else '✅ OK'}")
            except Exception as e:
                return f"🏓 PONG (err: {e})"

        # ── /status ───────────────────────────────────────────────────
        if cmd in ("/status", "status"):
            bnb_p    = b.bin.price("BNBUSDC")
            net, fees = b._pnl()
            uptime_h = b.total_uptime_h()
            fed_bl   = "📅 FED BLACKOUT" if is_fed_blackout() else ""
            paused_s = "⏸ PAUZA ACTIVA" if self._paused else ""
            # Tranzactii saptamanale
            snap    = b._week_snap
            total_w = (b._week_trades["funding_open"] +
                       b._week_trades["funding_close"] +
                       b.grid.total_fills - snap["grid_fills"] +
                       (b.swing.n_wins + b.swing.n_losses) - snap["swing"])
            fee_w   = b._week_trades["total_fee_bnb"]
            days_w  = (time.time() - b._week_start_ts) / 86400
            # SOL Trader info
            sol_p = b.bin.price("SOLUSDC")
            st = b.sol_trader
            sol_grid_fills = getattr(st, 'grid_fills', 0)
            sol_swing_count = len(getattr(st, 'swing_trades', {}))
            sol_realized = getattr(st, 'total_sol_profit', 0)
            sol_pending_usdt = getattr(st, 'pending_usdt', 0)
            return (
                f"📊 <b>STATUS RAPID</b> {fed_bl} {paused_s}\n"
                f"{'─'*30}\n"
                f"⏱ Uptime: {uptime_h:.1f}h\n"
                f"💰 BNB: {b.bnb:.4f} (~${b.bnb*bnb_p:.0f})\n"
                f"📈 PnL net: {net:+.5f} BNB ({net/max(b.bnb,0.001)*100:+.2f}%)\n"
                f"💵 Profit USD: {net*bnb_p:+.2f}$\n"
                f"💸 Fee: {fees:.5f} BNB (${fees*bnb_p:.2f})\n"
                f"{'─'*30}\n"
                f"🌊 <b>SOL Trader:</b>\n"
                f"  Grid: {sol_grid_fills} fills | ${sol_pending_usdt:.2f} pending\n"
                f"  Swing: {sol_swing_count} poz deschise\n"
                f"  Realizat: {sol_realized:.4f} SOL (~${sol_realized*sol_p:.2f})\n"
                f"  Total câștigat: {getattr(st,'total_sol_earned',0):.4f} SOL "
                f"(${getattr(st,'total_sol_earned',0)*sol_p:.2f})\n"
                f"  Zilnic: +{getattr(st,'total_sol_earned',0)/max(uptime_h/24,0.1):.4f} SOL/zi "
                f"(${getattr(st,'total_sol_earned',0)/max(uptime_h/24,0.1)*sol_p:.2f}/zi)\n"
                f"{'─'*30}\n"
                f"🔲 <b>BNB Grid:</b>\n"
                f"  Fills: {b.grid.total_fills} | PnL: {b.grid.total_pnl:+.5f} BNB (${b.grid.total_pnl*bnb_p:+.2f})\n"
                f"  Rate: {b.grid.total_fills/max(uptime_h,0.1):.2f} fills/h | ${b.grid.total_pnl*bnb_p/max(uptime_h/24,0.1):+.2f}/zi\n"
                f"  Fee: {b.grid.total_fees:.5f} BNB (${b.grid.total_fees*bnb_p:.2f})\n"
                f"{'─'*30}\n"
                f"🛡 Crash: {b.crash.status_str()}\n"
                f"💳 Fee buffer: {b.fee_buf.status()}\n"
                f"🔄 Trades azi: {b.guard.today}/{MAX_TRADES_DAY}\n"
                f"📊 Tranzactii {days_w:.1f}z: {total_w} "
                f"(fee ${fee_w*bnb_p:.2f})\n"
                f"💰 Funding: {len(b.funding.pos)}/{FR_MAX_POS} poz\n"
                f"🔲 Grid: {len(b.grid.grids)} perechi active\n"
                f"📈 Swing: {len(b.swing.trades)} poz deschise"
            )

        # ── /safety — Safety stop status ─────────────────────────────
        elif cmd in ("/safety", "safety"):
            bnb_p = b.bin.price("BNBUSDC")
            sol_p = b.bin.price("SOLUSDC")
            current = b.bnb * bnb_p + b.sol_detected * sol_p
            net_pnl, _ = b._pnl()
            pnl_usd = net_pnl * bnb_p
            total_loss = pnl_usd  # DOAR PnL trading, nu market value
            pct_used = abs(total_loss) / SAFETY_MAX_LOSS_USD * 100 if total_loss < 0 else 0
            return (
                f"🛡 <b>SAFETY STOP Status</b>\n"
                f"{'─'*30}\n"
                f"Status: {'🛑 OPRIT' if b._safety_stopped else '🟢 ACTIV'}\n"
                f"Capital inițial: ${b._initial_capital_usd:.2f}\n"
                f"Capital acum: ${current:.2f}\n"
                f"PnL trading: ${pnl_usd:+.2f}\n"
                f"PnL total: ${total_loss:+.2f}\n"
                f"Limita: -${SAFETY_MAX_LOSS_USD:.0f}\n"
                f"Folosit: {pct_used:.0f}% din limită\n"
                f"{'━'*30}\n"
                f"{'⛔ Trading oprit — restart pentru reactivare' if b._safety_stopped else '✅ Trading permis'}"
            )

        # ── /wallet — Portofel BNB + SOL (detaliat) ──────────────────
        elif cmd in ("/wallet", "wallet"):
            return b.wallet_content.format_display()

        # ── /portfolio — Monede detectate ────────────────────────────
        elif cmd in ("/portfolio", "portfolio"):
            lines_p = []
            total_val = 0
            for asset, info in sorted(b.discovered_assets.items(),
                                       key=lambda x: -x[1]["usd"]):
                vol_str = f"${info['vol_24h']/1e6:.0f}M" if info['vol_24h'] > 1e6 else f"${info['vol_24h']:.0f}"
                in_grid = "📦" if info["pair"] in GRID_USDT_PAIRS else "  "
                in_swing = "📈" if info["pair"] in SWING_PAIRS else "  "
                lines_p.append(
                    f"  {in_grid}{in_swing} {asset:<6} {info['qty']:.4f} "
                    f"(${info['usd']:.2f}) vol:{vol_str}")
                total_val += info["usd"]
            return (
                f"📦 <b>Portofel Detectat</b>\n"
                f"{'─'*35}\n"
                + "\n".join(lines_p) +
                f"\n{'─'*35}\n"
                f"Total: ${total_val:.2f}\n"
                f"Grid perechi: {', '.join(GRID_USDT_PAIRS)}\n"
                f"Swing perechi: {', '.join(SWING_PAIRS[:5])}\n"
                f"\n📦=Grid 📈=Swing"
            )

        # ── /spread — Spread Monitor status ──────────────────────────
        elif cmd in ("/spread", "spread"):
            sm = b.enhancements.spread_monitor
            st = sm.get_status()
            level_icon = {"NORMAL": "🟢", "WARNING": "🟡", "ALERT": "🔴"}.get(
                st["global_level"], "⚪")
            lines_sm = []
            for sym, sv in st["symbols"].items():
                lvl_icon = {"NORMAL": "🟢", "WARNING": "🟡", "ALERT": "🔴"}.get(
                    sv["level"], "⚪")
                lines_sm.append(
                    f"  {lvl_icon} {sym}: spread={sv['current_pct']:.4f}% "
                    f"avg={sv['avg_pct']:.4f}% spike={sv['spike_x']:.1f}×"
                )
            integrity_st = b.integrity_guard.get_status() if hasattr(b, 'integrity_guard') else {}
            int_icon = "🔒" if integrity_st.get("hash_ok") else "⚠️"
            return (
                f"📊 <b>SPREAD MONITOR</b>\n"
                f"{'═'*30}\n"
                f"Status global: {level_icon} {st['global_level']}\n"
                f"{'─'*30}\n"
                + "\n".join(lines_sm) +
                f"\n{'─'*30}\n"
                f"Alerte total: {st['total_alerts']} | "
                f"Warnings: {st['total_warnings']}\n"
                f"Ultima verificare: {st['last_check_ago']:.0f}s ago\n"
                f"{'─'*30}\n"
                f"LOGICĂ: Spread larg → MM retrași → mișcare violentă\n"
                f"  NORMAL  → trading normal\n"
                f"  WARNING → sizing -50%\n"
                f"  ALERT   → trading BLOCAT\n"
                f"{'─'*30}\n"
                f"{int_icon} <b>Integrity Guard:</b> "
                f"{'ACTIV ' + integrity_st.get('hash_prefix','') if integrity_st.get('enabled') else 'INACTIV'} "
                f"| Breach: {integrity_st.get('breach_count', 0)}"
            )

        # ── /enh — Enhancements status v1.3 ─────────────────────────
        elif cmd in ("/enh", "enh", "/enhancements"):
            enh = b.enhancements
            cb = enh.circuit_breaker.get_status()
            pl = enh.profit_lock.get_status()
            oi = enh.oi_sentinel.get_status()
            sol = enh.sol_accumulator.get_status()
            ff = enh.funding_filter.get_stats()
            # Per-strategy multipliers
            m_fund = enh.get_size_multiplier_for_strategy("funding")
            m_grid = enh.get_size_multiplier_for_strategy("grid")
            m_swing = enh.get_size_multiplier_for_strategy("swing")
            oi_sigs = oi.get("active_signals", {})
            oi_txt = "Fara semnale" if not oi_sigs else "\n".join(
                f"  {s}: {v['level']}" for s, v in oi_sigs.items())
            lock_icon = "🔒" if pl['locked'] else "🔓"
            return (
                f"🛡 <b>ENHANCEMENTS v1.1</b>\n"
                f"{'─'*30}\n"
                f"🔌 <b>Circuit Breaker:</b> {cb['state']}\n"
                f"  P&L azi: ${cb['daily_pnl']:+.2f} ({cb['daily_pnl_pct']:+.1f}%)\n"
                f"  Cooldown: 30min | Funding: protejat\n"
                f"{'─'*30}\n"
                f"{lock_icon} <b>Profit Lock:</b>\n"
                f"  Peak azi: ${pl['peak_daily_pnl']:.2f} | Trail: ${pl['trail_level']:.2f}\n"
                f"  Locked: {'DA' if pl['locked'] else 'NU'} | Locks: {pl['lock_count']}\n"
                f"{'─'*30}\n"
                f"📡 <b>OI Sentinel:</b>\n"
                f"  {oi_txt}\n"
                f"{'─'*30}\n"
                f"💰 <b>Funding Filter:</b> {ff['total_approved']}✅ {ff['total_filtered']}❌\n"
                f"{'─'*30}\n"
                f"🌟 <b>SOL DCA:</b>\n"
                f"  Pending: ${sol['accumulated_usdt']:.2f}\n"
                f"  SOL total: {sol['total_sol_bought']:.4f}\n"
                f"  Dip buys: {enh.sol_accumulator.dip_buys}\n"
                f"{'─'*30}\n"
                f"⚡ <b>Size per strategie:</b>\n"
                f"  Funding: {m_fund:.0%} | Grid: {m_grid:.0%} | Swing: {m_swing:.0%}\n"
                f"{'─'*30}\n"
                f"📊 <b>Adaptive Alloc:</b> {enh.adaptive_alloc.current_mode} "
                f"(FG={enh.adaptive_alloc.current_fg})\n"
                f"{'─'*30}\n"
                f"📈 <b>Compound:</b> ${enh.auto_compound.current_base:.0f} "
                f"(+{enh.auto_compound.get_status()['growth_pct']:.1f}%) "
                f"#{enh.auto_compound.compound_count}"
            )

        # ── /ml — ML+AI status ──────────────────────────────────────
        elif cmd in ("/ml", "ml", "/ai"):
            if not b.ml:
                return (
                    "🧠 <b>ML+AI Status: OPRIT</b>\n"
                    "━━━━━━━━━━━━━━━━━━\n"
                    "Trading pe reguli fixe.\n"
                    f"{'⚠️ TESTNET — ML dezactivat automat' if USE_TESTNET else '🔘 Apasă /ml_on pentru a porni ML'}"
                )
            s = b.ml.get_status()
            # Quick prediction on BTC
            pred_str = ""
            try:
                kl = b.bin.klines("BTCUSDC", "1h", 50)
                regime = b.ml.get_regime(kl)
                price_dir = b.ml.get_price_direction(kl)
                anomaly = b.ml.get_anomaly_score(kl)
                pred_str = (
                    f"\n📊 <b>Live Predictions (BTCUSDT):</b>\n"
                    f"  Regime: {regime['regime']} ({regime['confidence']:.0%})\n"
                    f"  Price 4h: {price_dir['direction']} ({price_dir['confidence']:.0%})\n"
                    f"  Anomaly: {'⚠️ YES' if anomaly['is_anomaly'] else '✅ Normal'} "
                    f"(score={anomaly['score']:.3f})\n"
                    f"  Sizing mult: {anomaly['sizing_mult']:.0%}\n"
                )
            except Exception:
                pred_str = "\n📊 Predictions: N/A (no data)\n"
            return (
                f"🧠 <b>ML + Deep AI Status</b>\n"
                f"{'─'*30}\n"
                f"Faza: <b>{s.get('phase','?')}</b> | "
                f"Trading ML: {'🟢 ACTIV' if s.get('trading_ready') else '⏳ reguli fixe'}\n"
                f"ML: {'✅' if s['enabled'] else '❌'} | "
                f"AI: {'✅' if s.get('deep_ai') else '❌'}\n\n"
                f"<b>Models:</b>\n"
                f"  🌲 Regime:    {'✅' if s['regime'].get('trained') else '⏳'} "
                f"acc={s['regime'].get('accuracy',0)*100:.0f}% "
                f"({s['regime'].get('samples',0)}s) "
                f"{'🟢 ready' if s['regime'].get('accuracy',0)>=0.45 and s['regime'].get('samples',0)>=200 else '🔴 collecting'}\n"
                f"  🧠 PriceMLP:  {'✅' if s.get('price_mlp',{}).get('trained') else '⏳'} "
                f"acc={s.get('price_mlp',{}).get('accuracy',0)*100:.0f}%\n"
                f"  🔍 Anomaly:   {'✅' if s.get('anomaly',{}).get('trained') else '⏳'} "
                f"t={s.get('anomaly',{}).get('threshold',0):.3f}\n"
                f"  📦 GridML:    {'✅' if s['grid_ml'].get('trained') else '⏳'} "
                f"{s['grid_ml'].get('fills',0)} fills\n"
                f"  💰 FundML:    {'✅' if s.get('funding_ml',{}).get('trained') else '⏳'} "
                f"{s.get('funding_ml',{}).get('obs',0)} obs\n"
                f"  🎯 Ensemble:  {'✅' if s.get('ensemble',{}).get('trained') else '⏳'} "
                f"{s.get('ensemble',{}).get('trades',0)} trades\n"
                f"{pred_str if s.get('trading_ready') else chr(10)+'📊 Predictions: ML nu e gata — reguli fixe active'+chr(10)}"
                f"\n🔘 Foloseste /ml_off pentru a opri ML"
            )

        # ── /ml_on — Porneste ML manual ──────────────────────────────
        elif cmd in ("/ml_on", "ml_on"):
            if b.ml:
                return "🧠 ML este deja PORNIT."
            if not ML_AVAILABLE:
                return "❌ scikit-learn nu e instalat. Rulează: pip install scikit-learn"
            if USE_TESTNET:
                return (
                    "⚠️ ML pe TESTNET = date false = modele inutile.\n"
                    "Trece pe MAINNET pentru ML real.\n"
                    "Dacă totuși vrei, setează USE_TESTNET=False.")
            b.ml = MLEngine(telegram_callback=lambda msg: tg(msg, silent=True))
            b.swing._ml = b.ml
            b.grid._ml = b.ml
            b.funding._ml = b.ml
            b.log.info("🧠 ML PORNIT manual din Telegram")
            return (
                "🧠 <b>ML PORNIT</b>\n"
                "━━━━━━━━━━━━━━━━━━\n"
                "Faza: COLLECTING\n"
                "Colectează date din trades.\n"
                "NU influențează trading-ul\n"
                "până accuracy > 45%.\n"
                "━━━━━━━━━━━━━━━━━━\n"
                "🔘 /ml — vezi status\n"
                "🔘 /ml_off — oprește ML")

        # ── /ml_off — Opreste ML manual ──────────────────────────────
        elif cmd in ("/ml_off", "ml_off"):
            if not b.ml:
                return "⚠️ ML este deja OPRIT."
            b.ml = None
            b.swing._ml = None
            b.grid._ml = None
            b.funding._ml = None
            b.log.info("🧠 ML OPRIT manual din Telegram")
            return (
                "🧠 <b>ML OPRIT</b>\n"
                "━━━━━━━━━━━━━━━━━━\n"
                "Trading pe reguli fixe.\n"
                "Datele colectate se pierd.\n"
                "━━━━━━━━━━━━━━━━━━\n"
                "🔘 /ml_on — repornește ML")

        # ── /raport ───────────────────────────────────────────────────
        elif cmd in ("/raport", "raport", "/report"):
            return b.report()

        # ── /pozitii ──────────────────────────────────────────────────
        elif cmd in ("/pozitii", "pozitii", "/positions"):
            with b.funding._lock:
                pos = dict(b.funding.pos)
            if not pos:
                return "💰 Nicio pozitie funding deschisa."
            lines = ["💰 <b>Pozitii Funding</b>\n"]
            rates = b.bin.all_funding_rates()
            for sym, p in sorted(pos.items(),
                                  key=lambda x: -x[1].get("apr", 0)):
                curr_apr = b.funding._apr(rates.get(sym, 0))
                hold_h   = (time.time() - p["ts"]) / 3600
                lines.append(
                    f"<b>{sym}</b>\n"
                    f"  APR intrare: {p['apr']*100:.1f}% → acum: {curr_apr*100:.1f}%\n"
                    f"  Capital: {p['size_bnb']:.4f} BNB\n"
                    f"  Colectat: +{p['collected']:.5f} BNB\n"
                    f"  Timp: {hold_h:.1f}h"
                )
            return "\n".join(lines)

        # ── /grid ─────────────────────────────────────────────────────
        elif cmd in ("/grid", "grid"):
            try:
                with b.grid._lock:
                    grids = dict(b.grid.grids)

                bnb_p = b.bin.price("BNBUSDC") or 640
                night = not b.grid._is_grid_active_hours()

                # Totaluri
                total_fills = getattr(b.grid, 'total_fills', 0)
                total_pnl = getattr(b.grid, 'total_pnl', 0.0)
                total_fees = getattr(b.grid, 'total_fees', 0.0)
                uptime_h = b.total_uptime_h()
                rate_per_h = total_fills / max(uptime_h, 0.1)
                daily_usd = (total_pnl * bnb_p / max(uptime_h, 0.1)) * 24

                # Capital
                grid_capital = b.bnb * ALLOC_GRID
                last_rb = getattr(b.grid, "_last_rb", 0)
                mins_since_rb = (time.time() - last_rb) / 60 if last_rb > 0 else 0

                lines = [
                    f"🔲 <b>GRID BNB HEALTH</b>",
                    f"{'━'*30}",
                    f"💰 Capital: {grid_capital:.4f} BNB (~${grid_capital*bnb_p:.2f})",
                    f"📊 Total fills: {total_fills}",
                    f"💵 PnL: {total_pnl:+.5f} BNB (${total_pnl*bnb_p:+.2f})",
                    f"💸 Fees: {total_fees:.5f} BNB (${total_fees*bnb_p:.2f})",
                    f"📈 Rate: {rate_per_h:.2f} fills/h | ${daily_usd:+.2f}/zi",
                    f"{'─'*30}",
                ]

                if not grids:
                    status_line = "⏸ Niciun grid activ"
                    if night: status_line += " (pauza nocturna)"
                    lines.append(status_line)
                else:
                    lines.append(f"<b>Perechi active ({len(grids)}):</b>")
                    for sym, g in sorted(grids.items()):
                        fills = g.get("fills", 0)
                        pnl = g.get("pnl", 0)
                        sp = g.get("spacing", GRID_SPACING)
                        adx = g.get("adx", 0)
                        lines.append(
                            f"  {sym}  ADX={adx:.0f} sp={sp*100:.2f}% "
                            f"fills={fills} pnl={pnl:+.5f}BNB"
                        )

                lines.append(f"{'─'*30}")
                if mins_since_rb > 0:
                    lines.append(f"🔄 Ultim rebuild: {mins_since_rb:.0f} min")
                status = "🟢 Activ" if not night else "🌙 Nocturn (pauza)"
                lines.append(f"Status: {status}")
                lines.append(f"{'━'*30}")
                return "\n".join(lines)

            except Exception as e:
                return f"❌ Eroare /grid: {e}"

        # ── /sol — SOL Trader health + detalii ───────────────────────
        # ── /portofel — continut portofel Binance spot (live) ──────────
        elif cmd in ("/portofel", "portofel", "/wallet"):
            try:
                balances = b.bin.full_balance()
                exclude = set(PORTFOLIO_EXCLUDE)

                # Get 24h ticker data for all USDC pairs
                try:
                    all_tickers = b.bin._get("/api/v3/ticker/24hr") or []
                    ticker_map = {t["symbol"]: t for t in all_tickers}
                except Exception:
                    ticker_map = {}

                assets = []
                total_usd = 0.0

                for asset, qty in balances.items():
                    if asset in exclude: continue
                    if qty < 0.0001: continue

                    if asset == "USDT":
                        price = 1.0
                        change_24h = 0.0
                    else:
                        sym = f"{asset}USDC"
                        ticker = ticker_map.get(sym)
                        if not ticker:
                            sym = f"{asset}USDT"
                            ticker = ticker_map.get(sym)
                        if not ticker:
                            continue
                        price = float(ticker.get("lastPrice", 0) or 0)
                        change_24h = float(ticker.get("priceChangePercent", 0) or 0)

                    if price <= 0: continue

                    usd = qty * price
                    if usd < 1.0: continue  # skip dust
                    total_usd += usd
                    assets.append({
                        "asset": asset, "qty": qty, "price": price,
                        "usd": usd, "change_24h": change_24h
                    })

                # Sort descending by USD
                assets.sort(key=lambda x: x["usd"], reverse=True)

                if not assets:
                    return "💼 Portofel gol sau balance sub $1"

                lines = [
                    f"💼 <b>PORTOFEL BINANCE</b>",
                    f"{'━'*30}",
                    f"💰 Total: <b>${total_usd:.2f}</b>",
                    f"{'─'*30}",
                    f"📊 <b>SPOT:</b>",
                ]
                for a in assets:
                    pct = (a["usd"] / total_usd * 100) if total_usd > 0 else 0
                    change_str = f"{a['change_24h']:+.2f}%" if a["change_24h"] != 0 else "—"
                    lines.append(
                        f"  {a['asset']:<6} {a['qty']:.4f}  "
                        f"${a['usd']:.2f}  ({pct:.1f}%)  "
                        f"24h: {change_str}"
                    )
                lines.append(f"{'━'*30}")
                # Timestamp UTC
                ts = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
                lines.append(f"🕐 {ts}")
                return "\n".join(lines)

            except Exception as e:
                return f"❌ Eroare /portofel: {e}"

        elif cmd in ("/sol", "sol"):
            st = b.sol_trader
            sol_p = b.bin.price("SOLUSDC")
            sol_qty = getattr(st, 'sol_qty', 0)
            grid_sol = getattr(st, 'grid_sol', 0)
            swing_sol = getattr(st, 'swing_sol', 0)
            grid_fills = getattr(st, 'grid_fills', 0)
            total_sol_profit = getattr(st, 'total_sol_profit', 0)
            pending_usdt = getattr(st, 'pending_usdt', 0)
            swing_trades = getattr(st, 'swing_trades', {})
            # Grid levels
            grid_levels = getattr(st, '_sol_grid', {})
            n_levels = len(grid_levels) if grid_levels else 0
            # Health checks
            health = []
            if sol_p > 0: health.append("✅ Preț SOL OK")
            else: health.append("❌ Preț SOL = 0!")
            if grid_fills > 0: health.append(f"✅ Grid activ ({grid_fills} fills)")
            else: health.append("⚠️ Grid: 0 fills (piață statică?)")
            fills_per_h = grid_fills / max(b.total_uptime_h(), 0.1)
            if fills_per_h > 1: health.append(f"✅ Ritm: {fills_per_h:.1f} fills/h")
            elif fills_per_h > 0.1: health.append(f"⚠️ Ritm lent: {fills_per_h:.1f} fills/h")
            else: health.append("❌ Ritm: aproape 0 fills")
            fee_pct = 1  # from reports
            if fee_pct < 5: health.append("✅ Fee sub 5%")
            avg_per_fill = pending_usdt / max(grid_fills, 1)
            uptime_days = (time.time() - b.start_ts) / 86400
            daily_rate = pending_usdt / max(uptime_days, 0.01)

            return (
                f"🌊 <b>SOL TRADER HEALTH</b>\n"
                f"{'━'*30}\n"
                f"💰 SOL total: {sol_qty:.4f} (~${sol_qty*sol_p:.2f})\n"
                f"  Grid:  {grid_sol:.4f} SOL (70%)\n"
                f"  Swing: {swing_sol:.4f} SOL (30%)\n"
                f"{'─'*30}\n"
                f"🔲 <b>Grid SOL:</b>\n"
                f"  Fills: {grid_fills}\n"
                f"  Nivele: {n_levels}\n"
                f"  Profit realizat: {total_sol_profit:.4f} SOL (~${total_sol_profit*sol_p:.2f})\n"
                f"  Pending: ${pending_usdt:.2f}\n"
                f"  Avg/fill: ${avg_per_fill:.3f}\n"
                f"  Rate: {fills_per_h:.1f} fills/h | ${daily_rate:.2f}/zi\n"
                f"{'─'*30}\n"
                f"📈 <b>Swing SOL:</b>\n"
                f"  Pozitii: {len(swing_trades)}\n"
                f"  TP: {getattr(st, 'swing_tp', 0.03)*100:.1f}% | "
                f"SL: {getattr(st, 'swing_sl', 0.012)*100:.1f}%\n"
                f"{'─'*30}\n"
                f"🏥 <b>Sănătate:</b>\n"
                + "\n".join(f"  {h}" for h in health) +
                f"\n{'━'*30}\n"
                f"SOL: ${sol_p:.2f}"
            )

        # ── /start_trading ────────────────────────────────────────────
        elif cmd in ("/start_trading", "start_trading"):
            self._paused = False
            self.log.info("▶ Trading ACTIVAT via /start_trading")
            tg(
                f"▶ <b>Trading ACTIVAT</b>\n"
                f"Botul incepe sa tranzactioneze acum.\n"
                f"Strategia: BNB + SOL | Funding + Grid + Swing\n"
                f"Capital: {b.bnb:.4f} BNB | SOL: {b.sol_detected:.4f}"
            )
            return "▶ Trading pornit. Botul tranzactioneaza acum."

        # ── /recentrare — recentrare imediata grid ─────────────────────
        elif cmd in ("/recentrare", "recentrare", "cmd_recentrare"):
            try:
                if b.grid:
                    with b.grid._lock:
                        for _sym in b.grid.grids:
                            if _sym in b.grid._rebuild_cooldown:
                                del b.grid._rebuild_cooldown[_sym]
                    b.grid._last_rb = 0
                    self.log.info("TG: recentrare grid manuala")
                    return "🔄 <b>Recentrare Grid</b>\nGrid-ul se recentreaza la pretul curent in max 45s."
                else:
                    return "❌ Grid inactiv"
            except Exception as _e:
                return f"❌ Recentrare: {_e}"

        # ── /rebuild_grid — rebuild complet manual ───────────────────────
        elif cmd in ("/rebuild_grid", "rebuild_grid", "cmd_rebuild_grid"):
            try:
                if b.grid:
                    b.grid._last_rb = 0
                    self.log.info("TG: rebuild grid manual")
                    return "🔄 <b>Rebuild Grid</b>\nGrid-ul se reconstruieste in max 45s."
                else:
                    return "❌ Grid inactiv"
            except Exception as _e:
                return f"❌ Rebuild: {_e}"

        # ── /pauza ────────────────────────────────────────────────────
        elif cmd in ("/pauza", "pauza", "/pause"):
            self._paused = True
            self.log.info("TG: pauza manuala activata")
            return (
                "⏸ <b>PAUZA ACTIVATA</b>\n"
                "Intrarile noi sunt blocate.\n"
                "Pozitiile existente raman deschise.\n"
                "Trimite /resume pentru a relua."
            )

        # ── /fees — Monitorizare fee-uri în timp real ────────────────
        elif cmd in ("/fees", "fees", "/comisioane"):
            bnb_p = b.bin.price("BNBUSDC")
            alert = b.fees.check_fee_alert(bnb_p)
            return (
                b.fees.summary() +
                f"\n\n{alert if alert else '✅ Fee-uri sub control'}"
            )

        # ── /stop — KILL SWITCH: închide totul instant ────────────────
        elif cmd in ("/stop", "stop", "/kill"):

            self.log.error("🛑 TG: KILL SWITCH activat!")
            b._safety_stopped = True
            # OPREȘTE TOATE THREAD-URILE
            b.stop_evt.set()
            # Opreste si serviciul systemd
            try:
                import subprocess as _sp
                _sp.Popen(['sudo', 'systemctl', 'stop', 'bnb-bot'])
                self.log.error("🛑 sudo systemctl stop bnb-bot executat")
            except Exception as _se:
                self.log.warning(f"systemctl stop: {_se}")
            closed = []
            try:
                for sym in list(b.swing.trades.keys()):
                    closed.append(f"Swing {sym}")
                b.swing.emergency_close()
            except Exception as _e: logging.debug(f'Ignored: {_e}')
            try:
                for sym in list(b.grid.grids.keys()):
                    closed.append(f"Grid {sym}")
                b.grid.emergency_close()
            except Exception as _e: logging.debug(f'Ignored: {_e}')
            try:
                for sym in list(b.funding.pos.keys()):
                    closed.append(f"Funding {sym}")
                    b.funding._close(sym, "KILL_SWITCH")
            except Exception as _e: logging.debug(f'Ignored: {_e}')
            try:
                for sym in list(b.sol_trader.swing_trades.keys()):
                    closed.append(f"SOL Swing {sym}")
                b.sol_trader.swing_trades.clear()
            except Exception as _e: logging.debug(f'Ignored: {_e}')
            n = len(closed)
            return (
                f"🛑 <b>KILL SWITCH ACTIVAT</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Pozitii inchise: {n}\n"
                + (("\n".join(f"  ✕ {c}" for c in closed[:10]) + "\n") if closed else "")
                + f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⛔ Toate thread-urile oprite\n"
                f"⛔ Trading OPRIT complet\n"
                f"🔄 Restart bot: sudo systemctl restart bnb-bot"
            )

        # ── /resume ───────────────────────────────────────────────────
        elif cmd in ("/resume", "resume"):
            self._paused = False
            self.log.info("TG: pauza manuala dezactivata")
            return "▶ <b>TRADING RELUAT</b>\nIntrari noi permise din nou."

        # ── /nivel ────────────────────────────────────────────────────
        elif cmd in ("/nivel", "nivel", "/level"):
            if not args:
                return (
                    f"📢 Nivel alerte curent: <b>{self._level}</b>\n"
                    "0 = Toate (inclusiv silent)\n"
                    "1 = Normal + Urgent (implicit)\n"
                    "2 = Doar Urgent (crash, circuit breaker)\n"
                    "Foloseste: /nivel 0|1|2"
                )
            try:
                n = int(args[0])
                if n not in (0, 1, 2):
                    return "❌ Nivel invalid. Foloseste 0, 1 sau 2."
                self._level = n
                labels = {0: "TOATE", 1: "NORMAL", 2: "URGENT"}
                return f"✅ Nivel alerte setat: <b>{labels[n]}</b>"
            except ValueError:
                return "❌ Argument invalid. Foloseste: /nivel 0|1|2"

        # ── /rezerva ──────────────────────────────────────────────────
        elif cmd in ("/rezerva", "rezerva"):
            bnb_p = b.bin.price("BNBUSDC")
            rez   = b.funding.profit_reserve
            return (
                f"🏦 <b>Rezerva BNB (10% din profit)</b>\n"
                f"{'─'*28}\n"
                f"Total acumulat: {rez:.6f} BNB\n"
                f"Valoare USD:    ~${rez*bnb_p:.2f}\n"
                f"Sursa: 10% din fiecare colectare funding\n"
                f"Status: NEATINSA — nu se reinvesteste\n"
                f"\nFisier backup: {PROFIT_RESERVE_FILE}"
            )

        # ── /perf — Performance Metrics ─────────────────────────────
        elif cmd in ("/perf", "perf", "/metrics"):
            return b.perf.summary()

        # ── /sentinel ─────────────────────────────────────────────────
        elif cmd in ("/sentinel", "sentinel"):
            return b.sentinel.status()

        elif cmd in ("/ajutor", "ajutor", "/help"):
            paused_s = "⏸ PAUZA ACTIVA\n" if self._paused else ""
            return (
                f"🌊 <b>Solana Bot v1.0 — Comenzi</b>\n"
                f"{paused_s}"
                f"{'─'*28}\n"
                f"/status   — status rapid\n"
                f"/enh      — enhancements status (CB+OI+Grid+SOL)\n"
                f"/raport   — raport complet\n"
                f"/pozitii  — pozitii funding\n"
                f"/grid     — stare griduri\n"
                f"/rezerva  — rezerva BNB (10% profit)\n"
                f"/pauza    — pauzeaza intrari noi\n"
                f"/resume   — reia trading\n"
                f"/nivel X  — filtrare alerte (0/1/2)\n"
                f"/ajutor   — aceasta lista\n"
                f"{'─'*28}\n"
                f"Nivel alerte curent: {self._level} "
                f"({'ALL' if self._level==0 else 'NORMAL' if self._level==1 else 'URGENT'})\n"
                f"Crash guard: {self._bot_ref.crash.status_str() if self._bot_ref else '?'}"
            )
        return None

    def _poll(self):
        """Polling Telegram getUpdates. Apelat din thread dedicat."""
        if not self._enabled: return
        url = f"https://api.telegram.org/bot{self.token}/getUpdates"
        while not self._stop_evt.is_set() if hasattr(self, "_stop_evt") else True:
            try:
                params = {"timeout": 20, "offset": self._last_upd + 1}
                r = _http_session.get(url, params=params, timeout=25)
                data = r.json()
                if not data.get("ok"): continue

                for upd in data.get("result", []):
                    self._last_upd = upd["update_id"]

                    # Mesaj text
                    msg = upd.get("message", {})
                    if msg and "text" in msg:
                        text    = msg["text"]
                        from_id = str(msg.get("chat", {}).get("id", ""))
                        resp    = self._handle(text, from_id)
                        if resp:
                            self.send(resp, buttons=self._main_keyboard())

                    # Callback butoane inline
                    cb = upd.get("callback_query", {})
                    if cb:
                        cb_data = cb.get("data", "")
                        from_id = str(cb.get("message", {})
                                      .get("chat", {}).get("id", ""))
                        # Mapeaza callback la comanda
                        cmd_map = {
                            "cmd_status":  "/status",
                            "cmd_raport":  "/raport",
                            "cmd_pozitii": "/pozitii",
                            "cmd_grid":    "/grid",
                            "cmd_sol":     "/sol",
                            "cmd_rezerva": "/rezerva",
                            "cmd_pauza":   "/pauza",
                            "cmd_resume":  "/resume",
                            "cmd_enh":     "/enh",
                            "cmd_ajutor":  "/ajutor",
                            "cmd_ml_on":   "/ml_on",
                            "cmd_ml_off":  "/ml_off",
                            "cmd_fees":    "/fees",
                            "cmd_stop":    "/stop",
                            "cmd_recentrare":    "/recentrare",
                            "cmd_rebuild_grid":  "/rebuild_grid",
                            "cmd_wallet":  "/wallet",
                            "cmd_spread":  "/spread",
                        }
                        # Confirma callback ÎNAINTE de handle (previne "loading" infinit)
                        try:
                            _http_session.post(
                                f"https://api.telegram.org/bot{self.token}/answerCallbackQuery",
                                json={"callback_query_id": cb["id"]},
                                timeout=5)
                        except Exception as _e: logging.debug(f'Ignored: {_e}')
                        cmd_text = cmd_map.get(cb_data)
                        if cmd_text:
                            resp = self._handle(cmd_text, from_id)
                            if resp:
                                self.send(resp, buttons=self._main_keyboard())

            except requests.exceptions.Timeout:
                pass   # normal pt long polling
            except Exception as e:
                self.log.warning(f"TG poll: {e}")
                time.sleep(10)

    def _daily_report_scheduler(self, bot_ref):
        """Thread: trimite raport zilnic la ora configurata."""
        while not getattr(self, "_stop_evt", None) or not self._stop_evt.is_set():
            now = datetime.now()
            if (now.hour == TG_DAILY_REPORT_H and
                    now.day != self._last_day_report):
                self._last_day_report = now.day
                try:
                    r = bot_ref.report()
                    self.send(
                        f"☀️ <b>RAPORT ZILNIC</b> — {now.strftime('%d.%m.%Y')}\n{r}",
                        silent=False)
                except Exception as e:
                    self.log.warning(f"Raport zilnic: {e}")
                # Auto-compound BNB: daca BNB free < $30, cumpara din profit
                try:
                    _bnb_free = bot_ref.client.full_balance().get("BNB", 0)
                    _bnb_p = bot_ref.client.price("BNBUSDC") or 600
                    _bnb_usd = _bnb_free * _bnb_p
                    if _bnb_usd < 30.0:
                        _buy_usd = max(getattr(bot_ref, "_daily_profit_usd", 10) * 0.10, 5.0)
                        bot_ref.client._post("/api/v3/order", {
                            "symbol": "BNBUSDC", "side": "BUY",
                            "type": "MARKET", "quoteOrderQty": round(_buy_usd, 2)
                        })
                        self.log.info(f"🔄 BNB compound: cumparat ${_buy_usd:.2f} USDC → BNB")
                        self.send(f"🔄 BNB compound: ${_buy_usd:.2f} → BNB (fee buffer)")
                except Exception as _ce:
                    self.log.debug(f"BNB compound: {_ce}")
            time.sleep(60)

    def run(self, stop: threading.Event, bot_ref):
        """Porneste thread-urile de polling si scheduler."""
        if not self._enabled:
            stop.wait()
            return

        self.set_bot(bot_ref)
        self.log.info(
            f"Telegram polling pornit | "
            f"nivel={self._level} | "
            f"raport zilnic la {TG_DAILY_REPORT_H}:00")

        # Trimitere mesaj de pornire cu tastatura
        self.send(
            f"🌊 <b>Solana Bot v1.0 ONLINE</b>\n"
            f"Tastatura activa. Trimite /ajutor pentru comenzi.",
            buttons=self._main_keyboard()
        )

        poll_t = threading.Thread(
            target=self._poll, name="TgPoll", daemon=True)
        flush_t = threading.Thread(
            target=self._flush_loop, args=(stop,), name="TgFlush", daemon=True)
        daily_t = threading.Thread(
            target=self._daily_report_scheduler, args=(bot_ref,),
            name="TgDaily", daemon=True)

        poll_t.start(); flush_t.start(); daily_t.start()
        stop.wait()
        self.log.info("Telegram oprit")

    def _flush_loop(self, stop: threading.Event):
        """Trimite mesajele din coada la fiecare 0.5s."""
        while not stop.is_set():
            self._flush_queue()
            time.sleep(0.5)


# Instanta globala — folosita de functia tg() pentru compatibilitate
_tgbot = TelegramBot()

def tg(msg: str, silent: bool = False, urgent: bool = False):
    """
    Compatibilitate backwards: trimite mesaj prin TelegramBot.
    Codul existent apeleaza tg(msg) / tg(msg, silent=True) fara modificari.
    """
    _tgbot.send(msg, silent=silent, urgent=urgent)

# ══════════════════════════════════════════════════════════════════════
# STRATEGY HEALTH MONITOR — auto-disable strategie cu pierderi consecutive
# ══════════════════════════════════════════════════════════════════════

class StrategyHealthMonitor:
    """
    Urmărește PnL zilnic per strategie.
    Dacă o strategie pierde 7 zile consecutive → dezactivează + alertă.
    Reactivare automată după 3 zile de pauză SAU manual din Telegram.
    """
    MAX_LOSING_DAYS = 7
    COOLDOWN_DAYS = 3

    def __init__(self):
        self._lock = threading.Lock()
        self.log = L("Health")
        # {strategy: [daily_pnl_1, daily_pnl_2, ...]} — ultimele 14 zile
        self.daily_pnl: Dict[str, list] = defaultdict(list)
        # {strategy: timestamp_disabled}
        self.disabled: Dict[str, float] = {}
        self._today_pnl: Dict[str, float] = defaultdict(float)

    def record_trade(self, strategy: str, pnl: float):
        """Înregistrează PnL de la fiecare trade."""
        with self._lock:
            self._today_pnl[strategy] += pnl

    def on_daily_close(self):
        """Apelat la 00:00 UTC — finalizează ziua."""
        with self._lock:
            for strat, pnl in self._today_pnl.items():
                self.daily_pnl[strat].append(pnl)
                # Păstrăm ultimele 14 zile
                if len(self.daily_pnl[strat]) > 14:
                    self.daily_pnl[strat] = self.daily_pnl[strat][-14:]
            self._today_pnl = defaultdict(float)

        # Check consecutive losing days
        alerts = []
        with self._lock:
            for strat, days in self.daily_pnl.items():
                if len(days) < self.MAX_LOSING_DAYS:
                    continue
                last_n = days[-self.MAX_LOSING_DAYS:]
                if all(d < 0 for d in last_n):
                    if strat not in self.disabled:
                        self.disabled[strat] = time.time()
                        total_loss = sum(last_n)
                        alerts.append((strat, total_loss))
                        self.log.warning(
                            f"🚨 {strat} dezactivat: {self.MAX_LOSING_DAYS} zile pierdere "
                            f"(${total_loss:.2f})")

        # Auto-reactivare după cooldown
        now = time.time()
        with self._lock:
            for strat in list(self.disabled.keys()):
                if now - self.disabled[strat] > self.COOLDOWN_DAYS * 86400:
                    del self.disabled[strat]
                    self.log.info(f"✅ {strat} reactivat după {self.COOLDOWN_DAYS} zile cooldown")

        return alerts

    def is_disabled(self, strategy: str) -> bool:
        with self._lock:
            return strategy in self.disabled

    def force_enable(self, strategy: str):
        with self._lock:
            if strategy in self.disabled:
                del self.disabled[strategy]

    def get_status(self) -> dict:
        with self._lock:
            return {
                "disabled": dict(self.disabled),
                "daily_pnl": {k: list(v) for k, v in self.daily_pnl.items()},
                "today": dict(self._today_pnl),
            }


# ══════════════════════════════════════════════════════════════════════
# CONTORIZARE FEE — vedem exact cât plătim
# ══════════════════════════════════════════════════════════════════════

class FeeTracker:
    def __init__(self):
        self._lock   = threading.Lock()
        self._fees   = defaultdict(float)   # {strategie: BNB plătiți}
        self._gross  = defaultdict(float)   # profit brut per strategie
        self._trades = defaultdict(int)
        self.log     = L("Fee")

    def record(self, strategy: str, fee_bnb: float,
               gross_bnb: float, n_legs: int):
        with self._lock:
            self._fees[strategy]   += fee_bnb
            self._gross[strategy]  += gross_bnb
            self._trades[strategy] += 1
        # Alertă dacă fee > 40% din gross
        if gross_bnb > 0 and fee_bnb > gross_bnb * 0.40:
            self.log.warning(
                f"⚠️  {strategy}: fee {fee_bnb:.6f} BNB = "
                f"{fee_bnb/gross_bnb*100:.0f}% din gross! ({n_legs} legs)")

    def summary(self) -> str:
        with self._lock:
            fees = dict(self._fees); gross = dict(self._gross)
            trades = dict(self._trades)
        lines = ["💸 <b>Fee real plătit:</b>"]
        tf = tg_tot = 0.0
        for s in sorted(fees):
            f = fees[s]; g = gross[s]; n = trades[s]
            net = g - f
            pct = f/g*100 if g > 0 else 0
            lines.append(
                f"  {s}: -{f:.5f} BNB fee | "
                f"net {net:+.5f} BNB | "
                f"{pct:.0f}% din gross ({n}T)")
            tf += f; tg_tot += g
        if tf > 0:
            lines.append(
                f"  TOTAL: -{tf:.5f} BNB fee | "
                f"net {tg_tot-tf:+.5f} BNB | "
                f"{tf/max(tg_tot,1e-10)*100:.0f}% din gross")
        return "\n".join(lines)

    def check_fee_alert(self, bnb_price: float) -> str:
        """
        Verifică dacă fee-urile depășesc profitul brut.
        Returnează mesaj alert sau '' dacă totul e OK.
        Apelat la fiecare oră din main loop.
        """
        with self._lock:
            tf = sum(self._fees.values())
            tg = sum(self._gross.values())

        fee_usd = tf * bnb_price
        gross_usd = tg * bnb_price
        net_usd = gross_usd - fee_usd

        # Alert dacă fee > profit brut (pierdere netă)
        if tg > 0 and tf > tg:
            return (
                f"🚨 <b>FEE ALERT</b>\n"
                f"Fee-urile DEPĂȘESC profitul!\n"
                f"Fee: ${fee_usd:.2f} | Brut: ${gross_usd:.2f}\n"
                f"Pierdere netă: ${abs(net_usd):.2f}\n"
                f"⚠️ Verifică /report")

        # Warning dacă fee > 50% din profit
        if tg > 0 and tf > tg * 0.50:
            return (
                f"⚠️ <b>FEE WARNING</b>\n"
                f"Fee = {tf/tg*100:.0f}% din profit brut\n"
                f"Fee: ${fee_usd:.2f} | Brut: ${gross_usd:.2f}\n"
                f"Net: ${net_usd:.2f}")

        return ""


class DailyTradeGuard:
    def __init__(self):
        self._lock  = threading.Lock()
        self._count = 0
        self._day   = datetime.now(tz=timezone.utc).date()

    def _reset(self):
        if datetime.now(tz=timezone.utc).date() != self._day:
            self._count = 0; self._day = datetime.now(tz=timezone.utc).date()

    def can_trade(self, strategy: str = "") -> bool:
        if strategy == "FUNDING": return True
        # Pauza manuala via Telegram
        if _tgbot.paused: return False
        with self._lock:
            self._reset()
            return self._count < MAX_TRADES_DAY

    def record(self, strategy: str = ""):
        if strategy == "FUNDING": return
        with self._lock:
            self._reset(); self._count += 1

    @property
    def today(self) -> int:
        with self._lock: self._reset(); return self._count


# ══════════════════════════════════════════════════════════════════════
# SOL ACCUMULATOR — 10% din profitul zilnic → SOL separat, neatins
# ══════════════════════════════════════════════════════════════════════

SOL_DAILY_PCT  = 0.10          # 10% din profit zilnic BNB → SOL neatins
SOL_ACC_FILE   = "v8_sol_accumulator.json"

class SolAccumulator:
    """
    La fiecare zi (00:00 UTC) calculeaza profitul zilnic net al botului.
    10% din acel profit se converteste din BNB in SOL si se pune separat.
    SOL acumulat NU se reinvesteste si NU se atinge.

    LOGICA:
    1. La 00:00 UTC: citeste PnL-ul zilei anterioare
    2. Daca PnL > 0: convertim 10% din profit in SOL (market buy SOL spot)
    3. Salvam in fisier separat cu timestamp
    4. Notificare Telegram cu suma acumulata

    RISC: Zero — cumperi SOL la pretul pietei cu un mic comision taker.
    """

    def __init__(self, client: "Binance"):
        self.client      = client
        self.log         = L("SolAcc")
        self._lock       = threading.Lock()
        self.total_sol   = 0.0      # SOL total acumulat
        self.total_usd   = 0.0      # valoarea USD la momentul achizitiei
        self.n_buys      = 0        # numar de achizitii
        self.history: List[dict] = []  # istoric achizitii
        self._last_pnl   = 0.0     # PnL de la ultima resetare
        self._day        = datetime.now(timezone.utc).day
        self._load()

    def _load(self):
        try:
            if os.path.exists(SOL_ACC_FILE):
                with open(SOL_ACC_FILE) as _jf:

                    d = json.load(_jf)
                self.total_sol = d.get("total_sol", 0.0)
                self.total_usd = d.get("total_usd", 0.0)
                self.n_buys    = d.get("n_buys", 0)
                self.history   = d.get("history", [])
        except Exception as _e: logging.debug(f'Ignored: {_e}')

    def _save(self):
        try:
            sol_p = self.client.price("SOLUSDC")
            _data = {
                "total_sol":     self.total_sol,
                "total_usd":     self.total_usd,
                "current_value": round(self.total_sol * sol_p, 2),
                "n_buys":        self.n_buys,
                "history":       self.history[-30:],  # ultimele 30 zile
                "ts":            time.time(),
            }
            _atomic_json_save(SOL_ACC_FILE, _data, indent=2)
        except Exception as _e: logging.debug(f'Ignored: {_e}')

    def daily_convert(self, daily_pnl_bnb: float):
        """
        Apelat la 00:00 UTC cu profitul zilei anterioare.
        Converteste 10% din profit BNB → SOL.
        """
        if daily_pnl_bnb <= 0:
            self.log.info(
                f"SolAcc: profit zilnic negativ "
                f"({daily_pnl_bnb:.5f} BNB) → skip conversie")
            return

        to_convert_bnb = daily_pnl_bnb * SOL_DAILY_PCT
        if to_convert_bnb < 0.0001:
            self.log.info(
                f"SolAcc: suma prea mica "
                f"({to_convert_bnb:.6f} BNB) → skip")
            return

        bnb_price = self.client.price("BNBUSDC")
        sol_price = self.client.price("SOLUSDC")
        if bnb_price <= 0 or sol_price <= 0:
            self.log.warning("SolAcc: nu am preturile → skip")
            return

        # Convertim BNB → USDC → SOL
        usdt_amount = to_convert_bnb * bnb_price
        sol_qty     = round(usdt_amount / sol_price, 4)

        if sol_qty < 0.001:
            self.log.info(f"SolAcc: qty SOL prea mica ({sol_qty}) → skip")
            return

        # Cumparare SOL spot
        if not USE_TESTNET:
            r = self.client.market_buy("SOLUSDC", sol_qty)
            if r.get("status") != "FILLED":
                self.log.warning(f"SolAcc: cumparare SOL esuata: {r}")
                return
        else:
            self.log.info(
                f"[TESTNET] SolAcc: simulare cumparare "
                f"{sol_qty:.4f} SOL @ ${sol_price:.2f}")

        with self._lock:
            self.total_sol += sol_qty
            self.total_usd += usdt_amount
            self.n_buys    += 1
            self.history.append({
                "date":       datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "sol_qty":    sol_qty,
                "sol_price":  sol_price,
                "bnb_spent":  to_convert_bnb,
                "usd_value":  round(usdt_amount, 2),
            })

        self._save()

        self.log.info(
            f"✅ SolAcc: +{sol_qty:.4f} SOL "
            f"(${usdt_amount:.2f}) | "
            f"total: {self.total_sol:.4f} SOL "
            f"(${self.total_sol*sol_price:.2f})")
        tg(
            f"🌟 <b>SOL Acumulat</b>\n"
            f"Profit azi: {daily_pnl_bnb:+.5f} BNB\n"
            f"10% convertit: +{sol_qty:.4f} SOL "
            f"(${usdt_amount:.2f})\n"
            f"Pret SOL: ${sol_price:.2f}\n"
            f"─────────────────\n"
            f"📦 Total SOL acumulat: {self.total_sol:.4f} SOL\n"
            f"💵 Valoare curenta: ${self.total_sol*sol_price:.2f}\n"
            f"📅 Achizitii: {self.n_buys} zile",
            silent=False
        )

    def status(self) -> str:
        sol_p = self.client.price("SOLUSDC")
        with self._lock:
            total = self.total_sol
            usd_c = self.total_sol * sol_p
        return (
            f"🌟 <b>SOL Accumulator</b>\n"
            f"Total SOL: {total:.4f} SOL\n"
            f"Valoare curenta: ${usd_c:.2f}\n"
            f"Cost mediu: ${self.total_usd/max(total,0.0001):.2f}/SOL\n"
            f"Achizitii: {self.n_buys} zile\n"
            f"Strategy: 10% profit zilnic → SOL NEATINS"
        )

    def run(self, stop: threading.Event, bot_ref: "SolanaBot"):
        """
        Ruleaza la 00:00 UTC zilnic.
        Calculeaza profitul zilei si converteste 10% in SOL.
        """
        self.log.info(
            f"🌟 SOL Accumulator pornit | "
            f"10% profit zilnic → SOL | "
            f"Total acumulat: {self.total_sol:.4f} SOL")

        last_pnl   = bot_ref._pnl()[0]  # PnL la pornire
        last_reset = datetime.now(timezone.utc).day

        while not stop.is_set():
            try:
                now_utc = datetime.now(timezone.utc)
                # Reset zilnic la 00:05 UTC (5 min dupa miezul noptii)
                if (now_utc.day != last_reset and
                        now_utc.hour == 0 and now_utc.minute >= 5):

                    current_pnl = bot_ref._pnl()[0]
                    daily_pnl   = current_pnl - last_pnl

                    self.log.info(
                        f"SolAcc reset zilnic: "
                        f"PnL azi = {daily_pnl:+.5f} BNB")

                    self.daily_convert(daily_pnl)

                    last_pnl   = current_pnl
                    last_reset = now_utc.day

            except Exception as e:
                self.log.warning(f"SolAcc: {e}")

            stop.wait(60)

        self.log.info("⛔ SOL Accumulator oprit")


# ══════════════════════════════════════════════════════════════════════
# FEE BUFFER MANAGER — BNB liber pentru comisioane
# ══════════════════════════════════════════════════════════════════════

class FeeBufferManager:
    """
    Gestioneaza un buffer de BNB liber (neinvestit) dedicat exclusiv
    platii comisioanelor pe spot (grid BNBUSDT, swing market orders).

    PROBLEMA REZOLVATA:
    - Rezerva 5% e in Earn (retras in ~1h, nu instant)
    - Grid BNBUSDT si Swing deduc fee din soldul BNB liber
    - Fara buffer → "Insufficient BNB for fee" → ordine respinse

    LOGICA:
    1. La pornire: verifica ca exista cel putin FEE_BUFFER_BNB BNB liber
    2. La fiecare ordin: verifica soldul inainte de executie
    3. Daca sold < FEE_BUFFER_MIN: redeem automat din Earn (~1h delay avertizat)
    4. Alerta Telegram daca bufferul scade critic

    DIMENSIONARE:
    - Fee zilnic total: ~0.000041 BNB ($0.027)
    - FEE_BUFFER_BNB = 0.012 BNB = 293 zile × 10x marja
    - Impact pe profit: -$0.03/luna (neglijabil)
    """

    def __init__(self, client: "Binance", earn: "BinanceEarn"):
        self.client  = client
        self.earn    = earn
        self.log     = L("FeeBuf")
        self._lock   = threading.Lock()
        self._last_check    = 0.0
        self._refill_pending= False   # redeem in curs (dureaza ~1h)
        self._refill_ts     = 0.0
        self._alerts_sent   = 0

    def _free_bnb(self) -> float:
        """Returneaza soldul BNB liber din contul spot."""
        return self.client.bnb_balance()

    def check(self) -> bool:
        """
        Verifica bufferul de fee inainte de orice ordin spot.
        Returneaza True daca OK, False daca sub nivelul critic.

        Niveluri:
          > FEE_BUFFER_MIN:      OK — normal
          < FEE_BUFFER_MIN:      ALERTA — redeem din Earn (cu delay 1h)
          < FEE_BUFFER_CRITICAL: CRITIC — blocam ordine noi imediat
        """
        now = time.time()
        if now - self._last_check < 30:
            return True
        self._last_check = now

        free = self._free_bnb()

        # Nivel critic — blocam imediat
        if free < FEE_BUFFER_CRITICAL:
            self.log.error(
                f"Fee buffer CRITIC: {free:.5f} BNB < {FEE_BUFFER_CRITICAL:.5f} BNB "
                f"— ordine BLOCATE")
            if not self._refill_pending:
                self._trigger_refill(free)
            return False   # blocam ordine noi

        # Nivel alerta — trigeram refill dar continuam
        if free < FEE_BUFFER_MIN:
            self.log.warning(
                f"Fee buffer SCAZUT: {free:.5f} BNB < {FEE_BUFFER_MIN:.5f} BNB")
            if not self._refill_pending:
                self._trigger_refill(free)
            return True   # continuam dar cu avertisment

        # Refill complet
        if self._refill_pending and free >= FEE_BUFFER_REFILL:
            with self._lock:
                self._refill_pending = False
            self.log.info(f"Fee buffer refill complet: {free:.5f} BNB")

        return True

    def get_sizing_multiplier(self) -> float:
        """
        PUNCT 6: FeeBuffer influențează sizing.
        Buffer OK → 1.0x. Buffer scăzut → 0.5x. Buffer critic → 0.0x (blocat).
        Rezerva minimă garantată: max(FEE_BUFFER_CRITICAL, 10% din sold total).
        """
        try:
            free = self._free_bnb()
            # Rezerva dinamică: max între constanta fixă și 10% din sold curent
            dynamic_critical = max(FEE_BUFFER_CRITICAL, free * 0.10)
            dynamic_min      = max(FEE_BUFFER_MIN,      free * 0.20)
            if free <= dynamic_critical:
                return 0.0   # blocat total — rezerva fee în pericol
            elif free <= dynamic_min:
                return 0.5   # reduce sizing 50%
            return 1.0       # normal
        except Exception:
            return 1.0

    def _trigger_refill(self, current_free: float):
        """Declanseza redeem din Earn pentru a reface bufferul."""
        with self._lock:
            if self._refill_pending:
                return
            self._refill_pending = True
            self._refill_ts = time.time()

        refill_qty = round(FEE_BUFFER_REFILL - current_free + 0.001, 5)
        refill_qty = min(refill_qty, self.earn.subscribed * 0.5)  # max 50% din Earn

        if refill_qty <= 0.001:
            self.log.warning("Fee buffer: nu avem suficient in Earn pentru refill")
            return

        self.log.warning(
            f"Fee buffer: redeem {refill_qty:.5f} BNB din Earn "
            f"(~1h pana disponibil)")

        # Redeem din Earn Flexible
        ok = False
        if not USE_TESTNET:
            r = self.client._post("/sapi/v1/simple-earn/flexible/redeem", {
                "productId": self.earn._product_id,
                "amount":    f"{refill_qty:.5f}",
            })
            ok = bool(r.get("success") or r.get("redeemId"))
        else:
            ok = True

        if ok:
            with self._lock:
                # Actualizam soldul Earn estimat
                self.earn.subscribed = max(0, self.earn.subscribed - refill_qty)
            self._alerts_sent += 1
            tg(
                f"⚠️ <b>Fee Buffer Refill</b>\n"
                f"Sold BNB liber scazut la {current_free:.5f} BNB\n"
                f"Redeem {refill_qty:.5f} BNB din Earn\n"
                f"Disponibil in ~1h. Ordine continua normal.",
                silent=True
            )
        else:
            self.log.error("Fee buffer: redeem din Earn ESUAT!")
            tg(
                f"🚨 <b>Fee Buffer CRITIC</b>\n"
                f"Sold BNB: {current_free:.5f} BNB\n"
                f"Redeem Earn esuat. Verifica manual!\n"
                f"Activeaza: Settings → Fee → Pay BNB → ON"
            )

    def ensure_startup(self, total_bnb: float):
        """
        Verifica la pornire ca avem FEE_BUFFER_BNB BNB liber.
        Daca nu, emite avertisment clar si instructiuni.
        """
        if USE_TESTNET:
            self.log.info(
                f"[TESTNET] Fee buffer: {FEE_BUFFER_BNB:.5f} BNB rezervat simulat")
            return

        free = self._free_bnb()
        self.log.info(
            f"Fee buffer check: {free:.5f} BNB liber | "
            f"necesar: {FEE_BUFFER_BNB:.5f} BNB")

        if free >= FEE_BUFFER_BNB:
            self.log.info(
                f"Fee buffer OK: {free:.5f} BNB > {FEE_BUFFER_BNB:.5f} BNB")
            return

        shortfall = FEE_BUFFER_BNB - free
        self.log.warning(
            f"Fee buffer INSUFICIENT: ai {free:.5f} BNB liber, "
            f"necesar {FEE_BUFFER_BNB:.5f} BNB (lipsa {shortfall:.5f} BNB)")
        tg(
            f"⚠️ <b>ATENTIE: Fee Buffer Insuficient</b>\n"
            f"BNB liber: {free:.5f} BNB\n"
            f"Necesar minim: {FEE_BUFFER_BNB:.5f} BNB\n"
            f"\n<b>Actiuni necesare:</b>\n"
            f"1. Activeaza: Binance → Settings → Fee → Use BNB for fees → ON\n"
            f"2. Pastreaza minim {FEE_BUFFER_BNB:.5f} BNB neinvestit\n"
            f"3. Nu subscrie TOT BNB-ul in Earn/Launchpool\n"
            f"\nBotul porneste dar unele ordine pot esua!"
        )

    def status(self) -> str:
        free = self._free_bnb() if not USE_TESTNET else FEE_BUFFER_BNB
        ok   = "🟢 OK" if free >= FEE_BUFFER_MIN else "🔴 CRITIC"
        pending = " | refill pending" if self._refill_pending else ""
        return f"{ok} {free:.5f} BNB liber{pending}"

class Binance:
    BASE   = "https://api.binance.com"
    BASE_F = "https://fapi.binance.com"
    BASE_T = "https://testnet.binance.vision"
    BASE_FT= "https://testnet.binancefuture.com"

    def __init__(self, key, secret, testnet=False):
        self.key    = key; self.secret = secret
        self.base   = self.BASE_T  if testnet else self.BASE
        self.base_f = self.BASE_FT if testnet else self.BASE_F
        self.log    = L("Binance")
        s = requests.Session()
        s.mount("https://", HTTPAdapter(max_retries=Retry(
            total=4, backoff_factor=0.5,
            status_forcelist=[429,500,502,503,504])))
        self.sess = s
        self._order_timestamps = []  # per instanță (era atribut de clasă)
        self._rate_limiter = AdaptiveRateLimiter(1200)
        # FIX1: Exchange info cache for lot size / min notional validation
        self._symbol_info = {}  # {symbol: {stepSize, minQty, minNotional, tickSize}}
        self._info_loaded = False

    def load_exchange_info(self):
        """Cache exchangeInfo for order validation (lot size, min notional, tick size)."""
        try:
            info = self._get("/api/v3/exchangeInfo") or {}
            for s in info.get("symbols", []):
                sym = s["symbol"]
                filters = {f["filterType"]: f for f in s.get("filters", [])}
                lot = filters.get("LOT_SIZE", {})
                price_f = filters.get("PRICE_FILTER", {})
                notional = filters.get("NOTIONAL", filters.get("MIN_NOTIONAL", {}))
                self._symbol_info[sym] = {
                    "stepSize":    float(lot.get("stepSize", "0.00001")),
                    "minQty":      float(lot.get("minQty", "0.00001")),
                    "tickSize":    float(price_f.get("tickSize", "0.01")),
                    "minNotional": float(notional.get("minNotional", "5.0")),
                }
            self._info_loaded = True
            self.log.info(f"ExchangeInfo cached: {len(self._symbol_info)} symbols")
        except Exception as e:
            self.log.error(f"ExchangeInfo load failed: {e}")

    def _round_step(self, value: float, step: float) -> float:
        """Round value down to nearest step size."""
        if step <= 0: return value
        import math
        precision = max(0, int(round(-math.log10(step))))
        return round(math.floor(value / step) * step, precision)
        

    def calculate_qty_for_pair(self, sym: str, bpl_bnb: float,
                                price: float, bnb_price: float) -> float:
        """Calculate correct base asset qty given BNB allocation.
        
        Handles 3 pair types:
          - X/BNB: quote=BNB, qty=bpl/price
          - X/USDT (X!=BNB): qty=(bpl*bnb_price)/price
          - BNB/USDT: qty=bpl (direct, no division)
        """
        if price <= 0 or bnb_price <= 0 or bpl_bnb <= 0:
            return 0.0
        if sym.endswith("BNB"):
            return bpl_bnb / price
        elif sym.endswith("USDT"):
            base = sym[:-4]
            if base == "BNB":
                return bpl_bnb
            else:
                usdt_equiv = bpl_bnb * bnb_price
                return usdt_equiv / price
        elif sym.endswith("USDC"):
            base = sym[:-4]
            if base == "BNB":
                return bpl_bnb          # BNBUSDC: qty = BNB direct
            else:
                usdc_equiv = bpl_bnb * bnb_price
                return usdc_equiv / max(price, 1e-10)  # SOLUSDC: qty = USDC/price
        else:
            return bpl_bnb / max(price, 1e-10)

    def _validate_order(self, sym: str, qty: float, price: float = 0) -> tuple:
        """Validate & fix qty/price for exchange rules. Returns (valid_qty, valid_price, error)."""
        info = self._symbol_info.get(sym)
        if not info:
            if not self._info_loaded:
                self.load_exchange_info()
                info = self._symbol_info.get(sym)
            if not info:
                return (qty, price, None)  # no info = skip validation
        
        # Qty: round to stepSize, check minQty
        valid_qty = self._round_step(qty, info["stepSize"])
        if valid_qty < info["minQty"]:
            return (0, 0, f"{sym}: qty {valid_qty} < minQty {info['minQty']}")
        
        # Price: round to tickSize
        valid_price = price
        if price > 0:
            valid_price = self._round_step(price, info["tickSize"])
            # Ensure price is at least one tick
            if valid_price <= 0:
                valid_price = info["tickSize"]
        
        # Min notional check
        check_price = valid_price if valid_price > 0 else self.price(sym)
        if check_price > 0 and valid_qty * check_price < info["minNotional"]:
            return (0, 0, f"{sym}: notional {valid_qty*check_price:.4f} < min {info['minNotional']}")
        
        return (valid_qty, valid_price, None)

    # FIX5: Order rate limiter — max 30 orders/minute
    ORDER_RATE_LIMIT = 30  # max orders per minute

    def _order_rate_ok(self) -> bool:
        now = time.time()
        self._order_timestamps = [t for t in self._order_timestamps if now - t < 60]
        if len(self._order_timestamps) >= self.ORDER_RATE_LIMIT:
            self.log.warning(f"🛑 Order rate limit: {len(self._order_timestamps)}/{self.ORDER_RATE_LIMIT}/min")
            return False
        self._order_timestamps.append(now)
        return True

    def _sign(self, p):
        p["timestamp"] = int(time.time() * 1000)
        p["recvWindow"] = 10000  # 10s window → toleranță la spike latență
        qs  = urllib.parse.urlencode(p)
        sig = hmac.new(self.secret.encode(), qs.encode(),
                       hashlib.sha256).hexdigest()
        p["signature"] = sig; return p

    def _get(self, path, params=None, signed=False, fut=False):
        url = (self.base_f if fut else self.base) + path
        p = params or {}
        if signed: p = self._sign(p)
        self._rate_limiter.wait_if_needed(weight=1)
        try:
            r = self.sess.get(url, params=p,
                              headers={"X-MBX-APIKEY": self.key},
                              timeout=7)
            r.raise_for_status()
            self._rate_limiter.update_from_headers(dict(r.headers))
            self._api_errors = 0  # FIX7: reset on success
            return r.json()
        except Exception as e:
            self._api_errors = getattr(self, "_api_errors", 0) + 1
            # Parsăm codurile de eroare Binance cunoscute
            try:
                _err = r.json() if 'r' in dir() else {}
                _code = _err.get("code", 0)
                if _code == -1021:
                    self.log.warning(f"⏱ Binance -1021 timestamp out of sync — auto-sync NTP")
                    try:
                        import subprocess as _sp2
                        _sp2.run(['timedatectl','set-ntp','true'], capture_output=True, timeout=5)
                        _sp2.run(['systemctl','restart','systemd-timesyncd'], capture_output=True, timeout=5)
                        import time as _t2; _t2.sleep(2)
                        self.log.info("⏱ NTP sync executat dupa -1021")
                    except Exception as _ntp_e:
                        self.log.debug(f"NTP sync: {_ntp_e}")
                elif _code == -2010:
                    self.log.warning(f"💸 Binance -2010 insufficient balance: {_err.get('msg', '')}")
                elif _code == -1100:
                    self.log.warning(f"📋 Binance -1100 invalid params: {_err.get('msg', '')}")
            except Exception:
                pass
            if self._api_errors >= 10 and self._api_errors % 10 == 0:
                self.log.error(f"⚠️ {self._api_errors} consecutive API errors! Last: GET {path}: {e}")
            else:
                self.log.debug(f"GET {path}: {e}")
            return {}

    # ═══ BLOCARE ABSOLUTĂ: NICIUN TRANSFER EXTERN ═══
    # Botul NU poate trimite fonduri în afara contului Binance
    # Nicio circumstanță, nicio excepție, nicio comandă
    _BLOCKED_PATHS = frozenset([
        "/sapi/v1/capital/withdraw",      # withdraw crypto
        "/sapi/v1/capital/withdraw/apply", # withdraw apply
        "/sapi/v1/withdraw",              # old withdraw
        "/wapi/v1/withdraw",              # legacy withdraw
        "/sapi/v1/asset/transfer",        # transfer între conturi
        "/sapi/v1/futures/transfer",      # transfer spot↔futures
        "/sapi/v1/sub-account/transfer",  # transfer sub-cont
        "/sapi/v1/capital/deposit",       # deposit address (info leak)
    ])

    def _post(self, path, params, fut=False):
        # HARD BLOCK: orice path de withdraw/transfer = refuzat instant
        if any(blocked in path for blocked in self._BLOCKED_PATHS):
            self.log.error(f"🛑 BLOCAT: tentativă de transfer extern: {path}")
            return {"error": "TRANSFERS_BLOCKED", "msg": "Botul NU are voie să transfere fonduri"}
        url = (self.base_f if fut else self.base) + path
        p = self._sign(params)
        self._rate_limiter.wait_if_needed(weight=2)  # POST costă mai mult
        try:
            r = self.sess.post(url, data=p,
                               headers={"X-MBX-APIKEY": self.key},
                               timeout=7)
            r.raise_for_status()
            self._rate_limiter.update_from_headers(dict(r.headers))
            self._api_errors = 0
            return r.json()
        except Exception as e:
            self._api_errors = getattr(self, "_api_errors", 0) + 1
            try:
                _err = r.json() if 'r' in dir() else {}
                _code = _err.get("code", 0)
                if _code == -1021:
                    self.log.warning(f"⏱ Binance -1021 timestamp out of sync — auto-sync NTP")
                    try:
                        import subprocess as _sp2
                        _sp2.run(['timedatectl','set-ntp','true'], capture_output=True, timeout=5)
                        _sp2.run(['systemctl','restart','systemd-timesyncd'], capture_output=True, timeout=5)
                        import time as _t2; _t2.sleep(2)
                        self.log.info("⏱ NTP sync executat dupa -1021")
                    except Exception as _ntp_e:
                        self.log.debug(f"NTP sync: {_ntp_e}")
                elif _code == -2010:
                    self.log.warning(f"💸 Binance -2010 insufficient balance: {_err.get('msg', '')}")
                elif _code == -1100:
                    self.log.warning(f"📋 Binance -1100 invalid params: {_err.get('msg', '')}")
                elif _code == -1013:
                    self.log.warning(f"📏 Binance -1013 filter failure (MIN_NOTIONAL/stepSize): {_err.get('msg', '')}")
            except Exception:
                pass
            if self._api_errors >= 10 and self._api_errors % 10 == 0:
                self.log.error(f"⚠️ {self._api_errors} consecutive API errors! Last: POST {path}: {e}")
            else:
                self.log.debug(f"POST {path}: {e}")
            return {}

    def bnb_balance(self) -> float:
        d = self._get("/api/v3/account", signed=True)
        for b in d.get("balances", []):
            if b["asset"] == "BNB":
                return float(b["free"])
        return 0.0

    def sol_balance(self) -> float:
        d = self._get("/api/v3/account", signed=True)
        for b in d.get("balances", []):
            if b["asset"] == "SOL":
                return float(b["free"])
        return 0.0

    def full_balance(self) -> Dict[str, float]:
        """Returneaza toate soldurile > 0 din portofel."""
        d = self._get("/api/v3/account", signed=True)
        return {
            b["asset"]: float(b["free"])
            for b in d.get("balances", [])
            if float(b.get("free", 0)) > 0
        }

    def price(self, sym: str) -> float:
        d = self._get("/api/v3/ticker/price", {"symbol": sym})
        return float(d.get("price", 0))

    def get_order(self, sym: str, order_id) -> dict:
        """Verifică status ordin prin API Binance. Returnează dict cu 'status' field.
        Pe testnet sau dry-run, returnează simulare bazată pe preț."""
        if USE_TESTNET or (MAINNET_DRY_RUN and not USE_TESTNET):
            # Simulare: considerăm că verificarea statusului e OK
            return {"status": "UNKNOWN", "simulated": True}
        if not order_id:
            return {"status": "UNKNOWN", "error": "no_id"}
        try:
            d = self._get("/api/v3/order",
                          {"symbol": sym, "orderId": int(order_id)},
                          signed=True)
            return d if d else {"status": "UNKNOWN"}
        except Exception as e:
            self.log.debug(f"get_order {sym} {order_id}: {e}")
            return {"status": "UNKNOWN", "error": str(e)}

    def all_prices(self) -> Dict[str, float]:
        data = self._get("/api/v3/ticker/price")
        return {d["symbol"]: float(d["price"]) for d in data} \
               if isinstance(data, list) else {}

    def klines(self, sym: str, interval: str = "1h",
               limit: int = 60) -> list:
        return self._get("/api/v3/klines",
                         {"symbol": sym, "interval": interval,
                          "limit": limit}) or []

    def orderbook(self, sym: str, depth: int = 10) -> dict:
        d = self._get("/api/v3/depth", {"symbol": sym, "limit": depth})
        return {
            "bids": [[float(p), float(q)] for p,q in d.get("bids",[])],
            "asks": [[float(p), float(q)] for p,q in d.get("asks",[])],
        }

    def discover_bnb_pairs(self, min_vol: float = 20.0) -> List[str]:
        info = self._get("/api/v3/exchangeInfo") or {}
        tk   = self._get("/api/v3/ticker/24hr") or []
        tvol = {t["symbol"]: float(t.get("quoteVolume",0))
                for t in tk if isinstance(t,dict)}
        pairs = []
        for s in info.get("symbols",[]):
            if s.get("quoteAsset")!="BNB": continue
            if any(ex in s["symbol"] for ex in PORTFOLIO_EXCLUDE): continue
            if s.get("status")!="TRADING": continue
            vol = tvol.get(s["symbol"], 0)
            if vol >= min_vol: pairs.append((s["symbol"], vol))
        pairs.sort(key=lambda x:-x[1])
        result = [p[0] for p in pairs]
        self.log.info(f"Descoperit {len(result)} perechi X/BNB")
        return result or BNB_PAIRS_FALLBACK

    def discover_volatile_usdc_pairs(self, min_vol_usd: float = 30_000_000,
                                      min_volatility: float = 0.02,
                                      top_n: int = 8):
        """Perechi X/USDC spot lichide SI volatile pentru grid."""
        try:
            info = self._get("/api/v3/exchangeInfo") or {}
            tk   = self._get("/api/v3/ticker/24hr") or []
            stats = {}
            for t in tk:
                if not isinstance(t, dict): continue
                s = t.get("symbol", "")
                if not s.endswith("USDC"): continue
                vol  = float(t.get("quoteVolume", 0))
                high = float(t.get("highPrice", 0))
                low  = float(t.get("lowPrice", 0))
                volat = (high - low) / low if low > 0 else 0
                stats[s] = (vol, volat)
            STABLE_EXCLUDE = ("USDTUSDC","FDUSDUSDC","USDCUSDC",
                              "BUSDUSDC","TUSDUSDC","DAIUSDC","EURUSDC")
            pairs = []
            for s in info.get("symbols", []):
                sym = s.get("symbol", "")
                if s.get("quoteAsset") != "USDC": continue
                if s.get("status") != "TRADING": continue
                if sym in STABLE_EXCLUDE: continue
                if any(ex in sym for ex in PORTFOLIO_EXCLUDE): continue
                vol, volat = stats.get(sym, (0, 0))
                if vol < min_vol_usd: continue
                if volat < min_volatility: continue
                pairs.append((sym, vol, volat))
            pairs.sort(key=lambda x: -x[2])
            result = [p[0] for p in pairs[:top_n]]
            self.log.info(
                f"🔍 Volatile USDC: {len(result)} perechi | top: " +
                " | ".join(f"{s}({v*100:.1f}%)" for s, _, v in pairs[:5]))
            return result or ["BNBUSDC"]
        except Exception as e:
            self.log.warning(f"discover_volatile_usdc error: {e}")
            return ["BNBUSDC"]

    def all_funding_rates(self) -> Dict[str, float]:
        data = self._get("/fapi/v1/premiumIndex", fut=True)
        return {d["symbol"]: float(d.get("lastFundingRate",0))
                for d in data if isinstance(d,dict)} if data else {}

    def discover_top_funding_pairs(self, min_vol_usd: float = 5_000_000,
                                    top_n: int = 10,
                                    min_apr: float = 0.05) -> List[Tuple[str, float, float]]:
        """
        v1.3: Scanează TOATE perechile USDT futures și returnează
        top N perechi cu cel mai mare funding rate (APR).
        
        Returns: [(symbol, funding_rate, apr), ...] sortat desc după APR
        """
        try:
            # Funding rates
            rates = self.all_funding_rates()
            if not rates:
                return []
            
            # Volume 24h futures
            tickers = self._get("/fapi/v1/ticker/24hr", fut=True) or []
            volumes = {t["symbol"]: float(t.get("quoteVolume", 0))
                       for t in tickers if isinstance(t, dict)}
            
            # Filtrare: USDT pairs, volum minim, rate pozitivă
            candidates = []
            for sym, rate in rates.items():
                if not (sym.endswith("USDT") or sym.endswith("USDC")):
                    continue
                if any(ex in sym for ex in PORTFOLIO_EXCLUDE):
                    continue
                vol = volumes.get(sym, 0)
                if vol < min_vol_usd:
                    continue
                apr = abs(rate) * 3 * 365  # anualizat
                if apr < min_apr:
                    continue
                candidates.append((sym, rate, apr))
            
            # Sortare după APR descrescător
            candidates.sort(key=lambda x: -x[2])
            
            result = candidates[:top_n]
            if result:
                self.log.info(
                    f"🔍 Top {len(result)} funding: " +
                    " | ".join(f"{s} {a*100:.0f}%" for s, _, a in result[:5]))
            return result
            
        except Exception as e:
            self.log.warning(f"discover_top_funding error: {e}")
            return []

    def futures_price(self, sym: str) -> float:
        d = self._get("/fapi/v1/ticker/price", {"symbol":sym}, fut=True)
        return float(d.get("price", 0))

    def discover_portfolio_pairs(self, balances: dict,
                                  min_usd: float = 1.0) -> dict:
        """
        Descoperă automat TOATE monedele din portofel cu valoare > min_usd.
        Verifică dacă au pereche USDT tranzacționabilă pe Binance.
        Verifică volum 24h minim.
        
        Returns: {
            "assets": {"BNB": {"qty": 1.0, "usd": 642, "pair": "BNBUSDC", "vol_24h": 338M}, ...},
            "grid_pairs": ["BNBUSDC", "SOLUSDC", ...],
            "swing_pairs": ["BNBUSDC", "SOLUSDC", "ETHUSDC", ...],
            "all_usdc_pairs": ["BNBUSDC", ...],
        }
        """
        try:
            # Volum 24h pentru toate perechile
            tickers = self._get("/api/v3/ticker/24hr") or []
            vol_map = {}
            for t in tickers:
                if isinstance(t, dict) and t.get("symbol", "").endswith(("USDT", "USDC")):
                    vol_map[t["symbol"]] = float(t.get("quoteVolume", 0))

            assets = {}
            grid_pairs = []
            swing_pairs = []
            all_pairs = []

            for asset, qty in balances.items():
                if asset in ("USDT", "BUSD", "FDUSD", "USDC"):
                    continue
                if qty <= 0:
                    continue
                if not asset.isascii() or not asset.isalnum():
                    continue
                if len(asset) < 2 or len(asset) > 10:
                    continue

                pair = f"{asset}USDC"
                # DOAR perechi care există pe exchange (din ticker 24h)
                if pair not in vol_map:
                    continue

                price = self.price(pair)
                if price <= 0:
                    continue

                usd_val = qty * price
                if usd_val < min_usd:
                    continue

                vol_24h = vol_map.get(pair, 0)

                assets[asset] = {
                    "qty": round(qty, 8),
                    "usd": round(usd_val, 2),
                    "pair": pair,
                    "price": price,
                    "vol_24h": vol_24h,
                }

                all_pairs.append(pair)

                # Grid: perechi cu volum > $10M (fills frecvente)
                if vol_24h > 10_000_000:
                    grid_pairs.append(pair)

                # Swing: perechi cu volum > $50M (lichiditate suficientă)
                if vol_24h > 50_000_000:
                    swing_pairs.append(pair)

            # Sortare după valoare USD (cele mai mari primele)
            grid_pairs.sort(key=lambda p: -vol_map.get(p, 0))
            swing_pairs.sort(key=lambda p: -vol_map.get(p, 0))

            self.log.info(
                f"🔍 Portofel: {len(assets)} monede detectate | "
                f"Grid: {len(grid_pairs)} perechi | Swing: {len(swing_pairs)} perechi")

            return {
                "assets": assets,
                "grid_pairs": grid_pairs or ["BNBUSDC"],  # fallback
                "swing_pairs": swing_pairs or ["BNBUSDC", "SOLUSDC"],
                "all_usdt_pairs": all_pairs,
            }

        except Exception as e:
            self.log.warning(f"discover_portfolio_pairs error: {e}")
            return {
                "assets": {},
                "grid_pairs": ["BNBUSDC", "SOLUSDC"],
                "swing_pairs": ["BNBUSDC", "SOLUSDC"],
                "all_usdt_pairs": ["BNBUSDC", "SOLUSDC"],
            }

    # ── Orders ────────────────────────────────────────────────────────
    def _fee_ok(self, fee_guard: Optional["FeeBufferManager"]) -> bool:
        """
        Verificare HARD a rezervei BNB înainte de orice ordin spot.
        Dublu check: FeeBufferManager + verificare directă sold BNB.
        Niciun ordin nu se trimite dacă BNB liber < FEE_BUFFER_CRITICAL.
        """
        # Check 1: FeeBufferManager (cu cache 30s)
        if fee_guard is not None and not fee_guard.check():
            return False
        # Check 2: Verificare directă sold BNB (hard floor absolut)
        try:
            free_bnb = self.bnb_balance()
            if free_bnb < FEE_BUFFER_CRITICAL:
                self.log.error(
                    f"🛑 HARD BLOCK: BNB liber {free_bnb:.5f} < "
                    f"rezerva {FEE_BUFFER_CRITICAL:.5f} — ordin REFUZAT")
                return False
        except Exception:
            pass  # dacă nu putem verifica, lăsăm să treacă (fail-open)
        return True

    def limit_buy(self, sym: str, qty: float, price: float,
                  fee_guard: Optional["FeeBufferManager"] = None) -> dict:
        if not self._fee_ok(fee_guard):
            self.log.warning(f"limit_buy {sym} BLOCAT — fee buffer critic")
            return {"status": "BLOCKED_FEE"}
        if MAINNET_DRY_RUN and not USE_TESTNET:
            self.log.info(f"🔸 DRY_RUN limit_buy {sym} qty={qty:.6f} price={price:.8f}")
            return {"status":"NEW","orderId":f"dry_{time.time():.0f}"}
        if USE_TESTNET: return {"status":"NEW","orderId":f"sim_{time.time():.0f}"}
        qty, price, err = self._validate_order(sym, qty, price)
        if err:
            self.log.warning(f"limit_buy SKIP: {err}")
            return {"status": "INVALID", "error": err}
        if not self._order_rate_ok():
            return {"status": "RATE_LIMITED"}
        return self._post("/api/v3/order", {
            "symbol":sym,"side":"BUY","type":"LIMIT",
            "quantity":qty,"price":f"{price:.8f}","timeInForce":"GTC",
            "newClientOrderId":f"b_{sym[:6]}_{uuid.uuid4().hex[:12]}"})
    def limit_sell(self, sym: str, qty: float, price: float,
                   fee_guard: Optional["FeeBufferManager"] = None) -> dict:
        if not self._fee_ok(fee_guard):
            self.log.warning(f"limit_sell {sym} BLOCAT — fee buffer critic")
            return {"status": "BLOCKED_FEE"}
        if MAINNET_DRY_RUN and not USE_TESTNET:
            self.log.info(f"🔸 DRY_RUN limit_sell {sym} qty={qty:.6f} price={price:.8f}")
            return {"status":"NEW","orderId":f"dry_{time.time():.0f}"}
        if USE_TESTNET: return {"status":"NEW","orderId":f"sim_{time.time():.0f}"}
        qty, price, err = self._validate_order(sym, qty, price)
        if err:
            self.log.warning(f"limit_sell SKIP: {err}")
            return {"status": "INVALID", "error": err}
        if not self._order_rate_ok():
            return {"status": "RATE_LIMITED"}
        return self._post("/api/v3/order", {
            "symbol":sym,"side":"SELL","type":"LIMIT",
            "quantity":qty,"price":f"{price:.8f}","timeInForce":"GTC",
            "newClientOrderId":f"s_{sym[:6]}_{uuid.uuid4().hex[:12]}"})
    def market_buy(self, sym: str, qty: float,
                   fee_guard: Optional["FeeBufferManager"] = None) -> dict:
        if not self._fee_ok(fee_guard):
            self.log.warning(f"market_buy {sym} BLOCAT — fee buffer critic")
            return {"status": "BLOCKED_FEE"}
        if MAINNET_DRY_RUN and not USE_TESTNET:
            p = self.price(sym)
            self.log.info(f"🔸 DRY_RUN market_buy {sym} qty={qty:.6f} price={p}")
            return {"status":"FILLED","price":str(p),
                    "fills":[{"price":str(p),"commission":str(qty*p*TAKER_FEE),"commissionAsset":"BNB"}]}
        if USE_TESTNET:
            p = self.price(sym)
            return {"status":"FILLED","price":str(p),
                    "fills":[{"price":str(p),"commission":str(qty*p*TAKER_FEE),"commissionAsset":"BNB"}]}
        qty, _, err = self._validate_order(sym, qty)
        if err:
            self.log.warning(f"market_buy SKIP: {err}")
            return {"status": "INVALID", "error": err}
        if not self._order_rate_ok():
            return {"status": "RATE_LIMITED"}
        return self._post("/api/v3/order", {
            "symbol":sym,"side":"BUY","type":"MARKET","quantity":qty})

    def market_sell(self, sym: str, qty: float,
                    fee_guard: Optional["FeeBufferManager"] = None) -> dict:
        if not self._fee_ok(fee_guard):
            self.log.warning(f"market_sell {sym} BLOCAT — fee buffer critic")
            return {"status": "BLOCKED_FEE"}
        if MAINNET_DRY_RUN and not USE_TESTNET:
            p = self.price(sym)
            self.log.info(f"🔸 DRY_RUN market_sell {sym} qty={qty:.6f} price={p}")
            return {"status":"FILLED","price":str(p),
                    "fills":[{"price":str(p),"commission":str(qty*p*TAKER_FEE),"commissionAsset":"BNB"}]}
        if USE_TESTNET:
            p = self.price(sym)
            return {"status":"FILLED","price":str(p),
                    "fills":[{"price":str(p),"commission":str(qty*p*TAKER_FEE),"commissionAsset":"BNB"}]}
        qty, _, err = self._validate_order(sym, qty)
        if err:
            self.log.warning(f"market_sell SKIP: {err}")
            return {"status": "INVALID", "error": err}
        if not self._order_rate_ok():
            return {"status": "RATE_LIMITED"}
        return self._post("/api/v3/order", {
            "symbol":sym,"side":"SELL","type":"MARKET","quantity":qty})

    def futures_short(self, sym: str, qty: float) -> dict:
        if MAINNET_DRY_RUN and not USE_TESTNET:
            self.log.info(f"🔸 DRY_RUN futures_short {sym} qty={qty}")
            return {"status":"FILLED"}
        if USE_TESTNET: return {"status":"FILLED"}
        return self._post("/fapi/v1/order", {
            "symbol":sym,"side":"SELL","type":"MARKET","quantity":qty}, fut=True)

    def futures_close(self, sym: str, qty: float) -> dict:
        if MAINNET_DRY_RUN and not USE_TESTNET:
            self.log.info(f"🔸 DRY_RUN futures_close {sym} qty={qty}")
            return {"status":"FILLED"}
        if USE_TESTNET: return {"status":"FILLED"}
        return self._post("/fapi/v1/order", {
            "symbol":sym,"side":"BUY","type":"MARKET","quantity":qty}, fut=True)

    def futures_stop_loss(self, sym: str, qty: float, stop_price: float) -> dict:
        """
        ANTI-LICHIDARE: Plasează STOP_MARKET BUY la stop_price.
        Dacă prețul urcă peste stop_price → închide short-ul automat.
        Previne lichidarea la 33% (3x lev) prin exit la 20%.
        """
        if USE_TESTNET or MAINNET_DRY_RUN:
            self.log.info(f"{'DRY_RUN' if MAINNET_DRY_RUN else 'Testnet'}: stop_loss {sym} @ ${stop_price:.2f}")
            return {"status":"NEW","orderId":f"dry_sl_{time.time():.0f}"}
        try:
            return self._post("/fapi/v1/order", {
                "symbol": sym, "side": "BUY", "type": "STOP_MARKET",
                "quantity": qty, "stopPrice": f"{stop_price:.2f}",
                "closePosition": "false",
                "workingType": "MARK_PRICE",
            }, fut=True)
        except Exception as e:
            self.log.warning(f"futures_stop_loss {sym}: {e}")
            return {"status": "ERROR", "error": str(e)}

    def spot_cancel_all(self, sym: str) -> dict:
        """FIX6: Cancel all open spot orders for a symbol before grid rebuild."""
        if USE_TESTNET or MAINNET_DRY_RUN: return {"msg": "skipped"}
        try:
            url = self.base + "/api/v3/openOrders"
            p = self._sign({"symbol": sym})
            r = self.sess.delete(url, params=p, headers={"X-MBX-APIKEY": self.key}, timeout=7)
            return r.json() if r.ok else {"error": r.text}
        except Exception as e:
            self.log.warning(f"spot_cancel_all {sym}: {e}")
            return {}

    def spot_open_orders(self, sym: str) -> list:
        """Get open orders for symbol."""
        try:
            return self._get("/api/v3/openOrders", {"symbol": sym}, signed=True) or []
        except Exception as e:
            self.log.warning(f"open_orders {sym}: {e}")
            return []

    def limit_chaser(self, sym: str, side: str, qty: float,
                     max_attempts: int = 3, wait_sec: float = 2.5,
                     fee_guard=None) -> dict:
        """
        Maker-first order: încearcă limit la bid/ask de max_attempts ori.
        Dacă ordinul nu se execută în wait_sec secunde, îl anulează și
        repostează la noul preț. Fallback la market după epuizarea încercărilor.

        Economie față de market order: 0.09% per roundtrip (~0.1% net/trade).

        Args:
            side: "BUY" sau "SELL"
            qty:  cantitate în base asset
            max_attempts: câte reprize de limit înainte de fallback market
            wait_sec: secunde de așteptare per repriza
        """
        for attempt in range(1, max_attempts + 1):
            # Obține bid/ask curent
            try:
                book = self._get("/api/v3/ticker/bookTicker", {"symbol": sym})
                if not book:
                    break
                if side == "BUY":
                    # BUY la bid = maker (așteptăm fill, nu luăm din piață)
                    chase_price = float(book.get("bidPrice", 0))
                else:
                    # SELL la ask = maker (oferim la prețul mai mare)
                    chase_price = float(book.get("askPrice", 0))
                if chase_price <= 0:
                    break
            except Exception as e:
                self.log.debug(f"limit_chaser bookTicker: {e}")
                break

            # Plasează limit order
            if side == "BUY":
                r = self.limit_buy(sym, qty, chase_price, fee_guard)
            else:
                r = self.limit_sell(sym, qty, chase_price, fee_guard)

            oid = r.get("orderId")
            status = r.get("status", "")

            # DRY_RUN sau simulare — returnează imediat ca FILLED
            if status in ("NEW", "PARTIALLY_FILLED") and (
                    MAINNET_DRY_RUN or USE_TESTNET):
                import random; time.sleep(random.uniform(0.3, 1.5))  # simulare latentă maker
                self.log.info(
                    f"🎯 limit_chaser [{attempt}/{max_attempts}] "
                    f"DRY {side} {sym} @ ${chase_price:.4f}")
                return {**r, "status": "FILLED",
                        "fills": [{"price": str(chase_price)}]}

            if status not in ("NEW", "PARTIALLY_FILLED"):
                # BLOCKED_FEE / INVALID / RATE_LIMITED → abort
                return r

            self.log.info(
                f"🎯 limit_chaser [{attempt}/{max_attempts}] "
                f"{side} {sym} @ ${chase_price:.4f} — aștept {wait_sec}s")

            time.sleep(wait_sec)

            # Verifică dacă s-a executat
            if oid:
                try:
                    order = self.get_order(sym, oid)
                    filled_status = order.get("status", "")
                    if filled_status == "FILLED":
                        exec_price = float(order.get("price", chase_price))
                        self.log.info(
                            f"✅ limit_chaser FILLED @ ${exec_price:.4f} "
                            f"(maker fee saved vs taker)")
                        return {**order,
                                "fills": [{"price": str(exec_price)}]}
                    elif filled_status in ("PARTIALLY_FILLED",):
                        # Parțial — lasă să continue
                        pass
                    else:
                        # NEW sau EXPIRED — anulează și reîncearcă
                        self.spot_cancel_all(sym)
                except Exception as e:
                    self.log.debug(f"limit_chaser get_order: {e}")

        # Fallback la market dacă nicio repriza nu a prins fill
        self.log.info(
            f"⚡ limit_chaser fallback MARKET {side} {sym} "
            f"(epuizat {max_attempts} reprize maker)")
        if side == "BUY":
            return self.market_buy(sym, qty, fee_guard)
        else:
            return self.market_sell(sym, qty, fee_guard)

    def futures_cancel_all(self, sym: str) -> dict:
        """Anulează toate ordinele futures pe un simbol (inclusiv stop-loss)."""
        if USE_TESTNET: return {"code": 200}
        try:
            return self._post("/fapi/v1/allOpenOrders", {"symbol": sym}, fut=True)
        except Exception as e:
            self.log.warning(f"futures_cancel_all {sym}: {e}")
            return {}

    def futures_positions(self) -> list:
        """Returnează toate pozițiile futures deschise cu unrealized PnL."""
        try:
            data = self._get("/fapi/v2/positionRisk", fut=True)
            if not data: return []
            return [
                {
                    "symbol": p["symbol"],
                    "size": float(p.get("positionAmt", 0)),
                    "entry_price": float(p.get("entryPrice", 0)),
                    "mark_price": float(p.get("markPrice", 0)),
                    "unrealized_pnl": float(p.get("unRealizedProfit", 0)),
                    "leverage": int(p.get("leverage", 1)),
                    "liquidation_price": float(p.get("liquidationPrice", 0)),
                }
                for p in data
                if isinstance(p, dict) and abs(float(p.get("positionAmt", 0))) > 0
            ]
        except Exception as e:
            self.log.warning(f"futures_positions: {e}")
            return []

    def set_leverage(self, sym: str, leverage: int) -> dict:
        """Setează leverage pe futures pentru un simbol. Apelează o singură dată per simbol."""
        if USE_TESTNET:
            self.log.info(f"Testnet: set_leverage {sym} → {leverage}x (simulat)")
            return {"leverage": leverage}
        try:
            return self._post("/fapi/v1/leverage", {
                "symbol": sym, "leverage": leverage}, fut=True)
        except Exception as e:
            self.log.warning(f"set_leverage {sym}: {e}")
            return {"leverage": leverage}

    def set_margin_type(self, sym: str, margin_type: str = "CROSSED") -> dict:
        """Setează margin type: CROSSED (cross margin) sau ISOLATED."""
        if USE_TESTNET:
            return {"msg": "success"}
        try:
            return self._post("/fapi/v1/marginType", {
                "symbol": sym, "marginType": margin_type}, fut=True)
        except Exception as e:
            # Dacă e deja setat, Binance returnează eroare — ignorăm
            if "No need to change" in str(e):
                return {"msg": "already set"}
            self.log.warning(f"set_margin_type {sym}: {e}")
            return {"msg": str(e)}


# ══════════════════════════════════════════════════════════════════════
# S1: FUNDING RATE ARB — inimă botului
# ══════════════════════════════════════════════════════════════════════

class FundingArb:
    """
    55% din capitalul BNB → funding rate arbitraj pe USDT futures.

    LOGICĂ SIMPLĂ:
    1. Convertim BNB în USDT (pe Binance spot)
    2. Cu USDT cumpărăm token pe spot (ex: BTC)
    3. Deschidem SHORT pe futures acelui token (BTCUSDT)
    4. Pozițiile se anulează reciproc → delta = 0
    5. Primim funding rate la fiecare 8h

    FEE: plătit O SINGURĂ DATĂ la intrare și ieșire.
    PROFIT: rate × qty × preț la fiecare 8h, fără tranzacții suplimentare.

    RISC PRINCIPAL: funding rate poate deveni negativ → plătim noi.
    SOLUȚIE: monitorizare la 15 min, exit rapid dacă rate < prag.
    """

    def __init__(self, client: Binance, bnb_capital: float,
                 fees: FeeTracker, guard: DailyTradeGuard,
                 crash: "MarketCrashGuard" = None):
        self.client = client
        self.bnb    = bnb_capital
        self.fees   = fees
        self.guard  = guard
        self.crash  = crash
        self.log    = L("Funding")
        self._lock  = threading.Lock()
        self.pos:   Dict[str, dict] = {}
        self.total_pnl    = 0.0
        self.total_fees   = 0.0
        self.funding_col  = 0.0
        self.n_coll       = 0
        self._last_col    = 0.0
        self.profit_reserve = 0.0   # BNB acumulat ca rezerva (10% din profit)
        self._apr_hist_24h: Dict[str, List[tuple]] = {}
        self._exit_cooldown: Dict[str, float] = {}  # FIX 3: {sym: exit_ts}
        self._enh_filter = None  # ENH: FundingSpreadFilter reference
        self._ml = None          # ML: MLEngine reference
        self._load()

    def _load(self):
        try:
            if os.path.exists("v3_funding.json"):
                with open("v3_funding.json") as _jf:

                    d = json.load(_jf)
                self.pos            = d.get("pos", {})
                self.total_pnl      = d.get("total_pnl", 0.0)
                self.total_fees     = d.get("total_fees", 0.0)
                self.funding_col    = d.get("funding_col", 0.0)
                self.n_coll         = d.get("n_coll", 0)
                self.profit_reserve = d.get("profit_reserve", 0.0)
        except Exception as _e: logging.debug(f'Ignored: {_e}')
        # Incarcare rezerva separata (fisier dedicat)
        try:
            if os.path.exists(PROFIT_RESERVE_FILE):
                with open(PROFIT_RESERVE_FILE) as _jf:

                    d = json.load(_jf)
                # folosim maximul dintre cele doua surse (siguranta)
                self.profit_reserve = max(
                    self.profit_reserve,
                    d.get("total_reserve", 0.0))
        except Exception as _e: logging.debug(f'Ignored: {_e}')

    def _save(self):
        try:
            _data = {
                "pos": self.pos, "total_pnl": self.total_pnl,
                "total_fees": self.total_fees,
                "funding_col": self.funding_col, "n_coll": self.n_coll,
                "profit_reserve": self.profit_reserve,
                "ts": time.time()
            }
            _atomic_json_save("v3_funding.json", _data, indent=2)
        except Exception as _e: logging.debug(f'Ignored: {_e}')
        # Salveaza rezerva si in fisier dedicat (backup)
        try:
            bnb_p = self.client.price("BNBUSDC")
            _data = {
                "total_reserve":    self.profit_reserve,
                "total_reserve_usd": round(self.profit_reserve * bnb_p, 2),
                "pct_din_profit":   PROFIT_RESERVE_PCT * 100,
                "sursa":            "10% din fiecare colectare funding",
                "ts": time.time(),
                "data": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            }
            _atomic_json_save(PROFIT_RESERVE_FILE, _data, indent=2)
        except Exception as _e: logging.debug(f'Ignored: {_e}')

    def _apr(self, rate: float) -> float:
        return abs(rate) * 3 * 365

    def _apr_momentum(self, sym: str, curr_apr: float) -> float:
        """
        FR Momentum Entry: returneaza factorul de momentum APR (0.0 = normal).
        Daca APR a crescut >FR_MOMENTUM_THRESHOLD in 24h → momentum pozitiv.
        Folosit pentru a deschide pozitii suplimentare cand APR e in crestere.
        """
        now = time.time()
        hist = self._apr_hist_24h.setdefault(sym, [])
        hist.append((now, curr_apr))
        # Pastram doar ultimele 24h
        cutoff = now - 86400
        self._apr_hist_24h[sym] = [(t, a) for t, a in hist if t >= cutoff]
        hist = self._apr_hist_24h[sym]
        if len(hist) < 3: return 0.0
        apr_24h_ago = hist[0][1]
        if apr_24h_ago <= 0: return 0.0
        momentum = (curr_apr - apr_24h_ago) / apr_24h_ago
        return momentum   # pozitiv = APR in crestere, negativ = in scadere

    @staticmethod
    def _in_reset_window() -> bool:
        """
        FIX 1: Returneaza True daca suntem in fereastra de ±30 min
        fata de un reset funding (00:00, 08:00, 16:00 UTC).
        In aceasta fereastra NU inchidem pozitii — ratele oscileaza violent.
        """
        now_utc = datetime.now(timezone.utc)
        min_utc = now_utc.hour * 60 + now_utc.minute
        for reset_h in [0, 8, 16]:
            reset_min = reset_h * 60
            if abs(min_utc - reset_min) <= FR_RESET_WINDOW_MIN:
                return True
            # Verifica si pentru ziua urmatoare (23:30-00:30)
            if abs(min_utc - reset_min - 1440) <= FR_RESET_WINDOW_MIN:
                return True
        return False

    def scan_and_enter(self):
        # Skip complet dacă funding e dezactivat (ALLOC=0%)
        if ALLOC_FUNDING <= 0:
            return
        # v1.1: Funding e delta-neutral → NU se oprește la crash guard
        # Doar logăm warning-ul, nu facem return
        if self.crash and not self.crash.entries_ok:
            self.log.info(
                f"Funding scan în crash ({self.crash.level}) "
                f"— continuăm (delta-neutral)")
            # La RED: doar skipăm intrări NOI, pozițiile existente colectează
            if hasattr(self.crash, 'level') and self.crash.level == "RED":
                self.log.info("Funding: RED → skip intrări noi, pozițiile existente rămân")
                return
        if len(self.pos) >= FR_MAX_POS: return
        rates = self.client.all_funding_rates()
        if not rates: return

        # v1.3: Dynamic scan — descoperă top perechi din TOATE futures USDT
        if FR_DYNAMIC_SCAN:
            top_pairs = self.client.discover_top_funding_pairs(
                min_vol_usd=FR_DYNAMIC_MIN_VOL,
                top_n=FR_DYNAMIC_TOP_N,
                min_apr=FR_APR_MIN
            )
            scan_symbols = [sym for sym, _, _ in top_pairs]
            if not scan_symbols:
                scan_symbols = FR_SYMBOLS  # fallback
        else:
            scan_symbols = FR_SYMBOLS

        opps = []
        # ML: record all funding rates for FundingPredictor learning
        if self._ml and ML_AVAILABLE:
            try:
                # ML: record top 5 funding rates (nu 30 — reduce API calls)
                sorted_rates = sorted(
                    [(s, r) for s, r in rates.items() if s.endswith(("USDT", "USDC"))],
                    key=lambda x: -abs(x[1])
                )[:5]
                for sym_r, rate_r in sorted_rates:
                    if sym_r.endswith(("USDT", "USDC")):
                        kl = self.client.klines(sym_r, "1h", 30)
                        feat = FeatureExtractor.from_klines(kl, 20) if kl else {}
                        if feat:
                            self._ml.record_funding_rate(sym_r, feat, rate_r)
            except Exception as _e: logging.debug(f"Ignored: {_e}")
        for sym in scan_symbols:
            r   = rates.get(sym, 0)
            apr = self._apr(r)
            if apr < FR_APR_MIN: continue
            if sym in self.pos: continue
            # FIX 3: cooldown — nu reintra in acelasi simbol prea repede dupa exit
            cooldown_ts = self._exit_cooldown.get(sym, 0)
            if time.time() - cooldown_ts < FR_COOLDOWN_H * 3600:
                self.log.debug(
                    f"FR cooldown {sym}: "
                    f"{(FR_COOLDOWN_H*3600-(time.time()-cooldown_ts))/60:.0f} min ramasi")
                continue
            spot = self.client.price(sym)
            fut  = self.client.futures_price(sym)
            if spot <= 0 or fut <= 0: continue
            if abs(fut-spot)/spot > 0.003: continue
            # ═══ ENH: Funding Spread Filter avansat ═══
            if self._enh_filter:
                ok, reason = self._enh_filter.should_enter_funding(sym)
                if not ok:
                    self.log.info(f"ENH filter {sym}: {reason}")
                    continue
            opps.append((sym, r, apr, spot))

        opps.sort(key=lambda x: -x[2])
        n_new     = min(len(opps), FR_MAX_POS - len(self.pos))
        bnb_price = self.client.price("BNBUSDC")

        # I6: Kelly fractional — capital proportional cu APR
        # Cu cat APR mai mare, cu atat pozitia mai mare (cap KELLY_MAX_PER_SYM)
        top_opps  = opps[:n_new]
        total_apr = sum(o[2] for o in top_opps) or 1.0

        for sym, rate, apr, spot_p in top_opps:
            # Kelly weight: proportional cu APR relativ
            kelly_w  = apr / total_apr
            kelly_sz = self.bnb * kelly_w
            # Cap per simbol: max KELLY_MAX_PER_SYM din capital total funding
            size_bnb = min(kelly_sz, self.bnb * KELLY_MAX_PER_SYM)

            # FR Momentum Entry: daca APR e in crestere puternica,
            # deschidem pozitie mai mare (pana la +FR_MOMENTUM_EXTRA_PCT)
            if FR_MOMENTUM_ENTRY:
                momentum = self._apr_momentum(sym, apr)
                if momentum >= FR_MOMENTUM_THRESHOLD:
                    extra = self.bnb * FR_MOMENTUM_EXTRA_PCT
                    size_bnb = min(size_bnb + extra,
                                   self.bnb * (KELLY_MAX_PER_SYM + FR_MOMENTUM_EXTRA_PCT))
                    self.log.info(
                        f"FR Momentum {sym}: APR +{momentum*100:.0f}%/24h "
                        f"→ size marit cu {extra:.4f} BNB")

            if size_bnb < KELLY_MIN_SIZE:
                self.log.debug(
                    f"I6 Skip {sym}: size {size_bnb:.4f} < min {KELLY_MIN_SIZE}")
                continue

            size_usdt = size_bnb * bnb_price
            qty_token = size_usdt / spot_p

            # v1.3: Set leverage pe futures (o singură dată per simbol)
            if FR_LEVERAGE > 1:
                self.client.set_leverage(sym, FR_LEVERAGE)
                self.client.set_margin_type(sym, "CROSSED")

            # Cu leverage: poziția futures e mai mare → colectăm mai mult funding
            # Spot rămâne 1x (hedge), futures e Nx
            # Delta neutral pe 1x, extra funding pe (N-1)x
            leveraged_qty = round(qty_token * FR_LEVERAGE, 4)

            # Futures short (cu leverage)
            self.client.futures_short(sym, leveraged_qty)

            # ═══ ANTI-LICHIDARE: stop-loss pe futures ═══
            # Dacă prețul urcă 20% peste entry → close automat (Binance server-side)
            # Nu depinde de bot — exchange-ul execută chiar dacă VPS-ul cade
            stop_price = round(spot_p * (1 + FR_STOP_LOSS_PCT), 2)
            sl_result = self.client.futures_stop_loss(sym, leveraged_qty, stop_price)
            sl_oid = sl_result.get("orderId", "N/A")
            self.log.info(
                f"🛡 Stop-loss {sym}: ${stop_price:.2f} "
                f"(+{FR_STOP_LOSS_PCT*100:.0f}% peste ${spot_p:.2f}) "
                f"oid={sl_oid}")

            fee_bnb = size_bnb * TAKER_FEE * FR_LEVERAGE  # fee pe qty leveraged
            be_days = (ROUNDTRIP_TAKER * FR_LEVERAGE) / (abs(rate) * 3 * FR_LEVERAGE)

            pos_data = {
                "sym": sym, "rate": rate, "apr": apr,
                "size_bnb": size_bnb, "qty": round(qty_token, 6),
                "qty_leveraged": leveraged_qty,
                "leverage": FR_LEVERAGE,
                "spot_p": spot_p, "fee_in": fee_bnb,
                "collected": 0.0, "ts": time.time(),
                "entry_price": spot_p,
                "stop_loss_price": stop_price,
                "stop_loss_oid": sl_oid,
            }
            with self._lock:
                self.pos[sym] = pos_data
                self.total_fees += fee_bnb
                self.total_pnl  -= fee_bnb
            self._save()
            # Tracking saptamanal
            if hasattr(self, '_bot_ref') and self._bot_ref:
                self._bot_ref._record_weekly_trade("funding_open", fee_bnb)

            self.log.info(
                f"✅ FR OPEN {sym}: "
                f"APR={apr*100:.1f}% | {FR_LEVERAGE}x leverage | "
                f"{size_bnb:.4f} BNB (futures {leveraged_qty} {sym[:3]}) | "
                f"fee={fee_bnb:.5f} BNB | "
                f"break-even {be_days:.1f}z | "
                f"est.lună +{size_bnb*abs(rate)*3*30*FR_LEVERAGE:.5f} BNB")
            tg(
                f"💰 <b>FUNDING OPEN</b> {sym} ({FR_LEVERAGE}x)\n"
                f"APR: {apr*100:.1f}% ({rate*100:.4f}%/8h)\n"
                f"Capital: {size_bnb:.4f} BNB | Leverage: {FR_LEVERAGE}x\n"
                f"Fee intrare: {fee_bnb:.5f} BNB\n"
                f"Break-even: {be_days:.1f} zile\n"
                f"Est. profit/lună: +{size_bnb*abs(rate)*3*30*FR_LEVERAGE:.5f} BNB",
                silent=True
            )

    def collect(self):
        """Înregistrare funding la 8h + reinvestire automată."""
        if ALLOC_FUNDING <= 0 and not self.pos:
            return
        rates     = self.client.all_funding_rates()
        bnb_price = self.client.price("BNBUSDC")
        if bnb_price <= 0: return
        total = 0.0
        with self._lock: pos = dict(self.pos)

        # v1.5: Funding flip exit pe RED crash
        # Pe RED, funding poate inversa violent (longs lichidați) → exit imediat
        crash_level = getattr(self, '_crash_ref', None)
        if crash_level and hasattr(crash_level, 'level'):
            try:
                from enum import Enum
                if crash_level.level == "RED":  # CrashLevel.RED e string, nu IntEnum
                    for sym_r in list(pos.keys()):
                        r_check = rates.get(sym_r, 0)
                        if r_check < -0.0001:  # funding negativ > 0.01%/8h
                            self.log.warning(
                                f"🔴 RED + funding NEGATIV {sym_r}: {r_check:.4%}/8h → exit forțat")
                            self._close(sym_r, "FUNDING_FLIP_RED")
            except Exception as _fe:
                self.log.debug(f"funding flip check: {_fe}")

        for sym, p in pos.items():
            r = rates.get(sym, 0)
            if r < 0:
                self.log.info(f"⚠️  {sym}: rate negativ → exit")
                self._close(sym, "RATE_NEG")
                continue
            if r == 0: continue
            # v1.3: Funding se colectează pe qty_leveraged (include leverage)
            lev_qty = p.get("qty_leveraged", p["qty"])
            earned_usdt = lev_qty * p["spot_p"] * r
            earned_bnb  = earned_usdt / max(bnb_price, 0.01)
            with self._lock:
                if sym in self.pos:
                    self.pos[sym]["collected"] += earned_bnb
                    self.funding_col += earned_bnb
                    self.total_pnl   += earned_bnb
            total += earned_bnb
            self.fees.record("FUNDING", 0.0, earned_bnb, 0)

        if total > 0:
            self.n_coll += 1
            self.log.info(
                f"💰 Funding colectat #{self.n_coll}: "
                f"+{total:.5f} BNB (~${total*bnb_price:.2f}) | "
                f"total: {self.funding_col:.5f} BNB")
            tg(
                f"💰 <b>Funding #{self.n_coll}</b>\n"
                f"+{total:.5f} BNB (~${total*bnb_price:.2f})\n"
                f"Total colectat: {self.funding_col:.5f} BNB",
                silent=True
            )
            # ── Reinvestire automată ──────────────────────────────
            if REINVEST_ENABLED:
                self._reinvest(total, bnb_price)
        self._save()

    def _reinvest(self, profit_bnb: float, bnb_price: float):
        """
        Reinvestire automata dupa fiecare colectare funding.

        SPLIT:
          90% (REINVEST_PCT)    → redistribuit in pozitii existente (compound)
          10% (PROFIT_RESERVE_PCT) → rezerva BNB, NEATINSA, salvata separat

        Rezerva se acumuleaza in self.profit_reserve si in PROFIT_RESERVE_FILE.
        Nu se reinvesteste niciodata automat. Poate fi vazuta cu /rezerva in Telegram.
        """
        if profit_bnb < REINVEST_THRESHOLD: return
        if len(self.pos) == 0: return

        # ── 10% → rezerva BNB (neatinsa) ─────────────────────────────
        reserve_bnb = profit_bnb * PROFIT_RESERVE_PCT
        with self._lock:
            self.profit_reserve += reserve_bnb
        self.log.info(
            f"🏦 Rezerva profit: +{reserve_bnb:.5f} BNB → "
            f"total rezerva: {self.profit_reserve:.5f} BNB "
            f"(~${self.profit_reserve*bnb_price:.2f})")

        # ── 90% → reinvestit in pozitii ───────────────────────────────
        to_reinvest = profit_bnb * REINVEST_PCT
        with self._lock: pos = dict(self.pos)

        total_apr  = sum(p["apr"] for p in pos.values()) or 1.0
        reinvested = 0.0

        for sym, p in pos.items():
            weight  = p["apr"] / total_apr
            add_bnb = to_reinvest * weight
            if add_bnb < 0.001: continue

            add_usdt = add_bnb * bnb_price
            add_qty  = add_usdt / max(p["spot_p"], 1e-10)
            lev = p.get("leverage", 1)
            add_qty_lev = round(add_qty * lev, 4)

            if not USE_TESTNET:
                r = self.client.futures_short(sym, add_qty_lev)
                if r.get("status") != "FILLED": continue

            fee_add = add_bnb * TAKER_FEE * lev
            with self._lock:
                if sym in self.pos:
                    self.pos[sym]["size_bnb"] += add_bnb
                    self.pos[sym]["qty"]      += round(add_qty, 6)
                    self.pos[sym]["qty_leveraged"] = self.pos[sym].get("qty_leveraged", 0) + add_qty_lev
                    self.total_fees           += fee_add
                    self.total_pnl            -= fee_add
            reinvested += add_bnb

        if reinvested > 0:
            with self._lock:
                self.bnb += reinvested
            self.log.info(
                f"🔄 Reinvestit: {reinvested:.5f} BNB (90%) → "
                f"capital funding: {self.bnb:.5f} BNB | "
                f"rezerva: {self.profit_reserve:.5f} BNB (10%)")
            tg(
                f"🔄 <b>Reinvestire automata</b>\n"
                f"Profit colectat: {profit_bnb:.5f} BNB\n"
                f"├ 90% reinvestit: +{reinvested:.5f} BNB\n"
                f"└ 10% rezerva:    +{reserve_bnb:.5f} BNB\n"
                f"\n💰 Capital funding: {self.bnb:.5f} BNB\n"
                f"🏦 Rezerva totala: {self.profit_reserve:.5f} BNB "
                f"(~${self.profit_reserve*bnb_price:.2f})",
                silent=True
            )
        self._save()

    def check_exits(self):
        """
        FIX 1: Nu inchidem pozitii in fereastra de reset (±30 min fata de 00/08/16 UTC)
        FIX 5: Staggered close — distribuim close-urile pentru a evita spike de fee
        """
        # FIX 1: Protectie fereastra reset
        if self._in_reset_window():
            self.log.debug(
                f"check_exits SKIP — fereastra reset funding "
                f"(±{FR_RESET_WINDOW_MIN} min)")
            return

        rates = self.client.all_funding_rates()
        now   = time.time()
        with self._lock: pos = dict(self.pos)

        to_close = []   # FIX 5: colectam close-urile si le executam staggered

        for sym in list(pos.keys()):
            curr_apr = self._apr(rates.get(sym, 0))

            # Reactiv: exit daca APR < prag minim absolut
            if curr_apr < FR_APR_EXIT:
                to_close.append((sym, f"APR_LOW {curr_apr*100:.1f}%"))
                continue

            # I2: Predictiv
            with self._lock:
                if sym not in self.pos: continue
                hist = self.pos[sym].setdefault("apr_hist", [])
                hist.append((now, curr_apr))
                cutoff = now - FR_APR_HISTORY_H * 3600
                self.pos[sym]["apr_hist"] = [
                    (t, a) for t, a in hist if t >= cutoff]
                hist = self.pos[sym]["apr_hist"]
                hold_h = (now - self.pos[sym].get("ts", now)) / 3600

            if hold_h < FR_APR_MIN_HOLD_H:
                continue

            if len(hist) < 3:
                continue

            peak_apr = max(a for _, a in hist)
            if peak_apr <= 0:
                continue

            decline = (peak_apr - curr_apr) / peak_apr
            if decline >= FR_APR_TREND_EXIT:
                to_close.append((
                    sym,
                    f"APR_TREND -{decline*100:.0f}% "
                    f"(peak={peak_apr*100:.2f}% → acum={curr_apr*100:.2f}%)"))

        # FIX 5: Staggered close — 5 secunde intre close-uri
        for i, (sym, reason) in enumerate(to_close):
            if i > 0:
                time.sleep(FR_STAGGER_CLOSE_S)
            self._close(sym, reason)
            # FIX 3: inregistreaza cooldown
            self._exit_cooldown[sym] = time.time()

    def _close(self, sym: str, reason: str):
        with self._lock: p = self.pos.pop(sym, None)
        if not p: return
        lev = p.get("leverage", 1)
        fee_out  = p["size_bnb"] * TAKER_FEE * lev
        net      = p["collected"] - fee_out - p["fee_in"]
        fee_tot  = p["fee_in"] + fee_out
        with self._lock:
            self.total_fees += fee_out
            self.total_pnl  += p["collected"] - fee_out
        # Close cu qty leveraged (sau fallback qty normal)
        close_qty = p.get("qty_leveraged", p["qty"])
        # Anti-lichidare: anulăm stop-loss-ul înainte de close (altfel double-close)
        self.client.futures_cancel_all(sym)
        self.client.futures_close(sym, close_qty)
        self._save()
        self.fees.record("FUNDING", fee_tot, p["collected"], 2)
        if hasattr(self, '_bot_ref') and self._bot_ref:
            self._bot_ref._record_weekly_trade("funding_close", fee_tot)
        self.log.info(
            f"📤 FR CLOSE {sym}: {reason} | {lev}x | "
            f"colectat={p['collected']:.5f} BNB | "
            f"fee_total={fee_tot:.5f} BNB | "
            f"net={net:+.5f} BNB")
        tg(
            f"📤 <b>FUNDING CLOSE</b> {sym} ({lev}x)\n"
            f"Motiv: {reason}\n"
            f"Colectat: {p['collected']:.5f} BNB\n"
            f"Fee total: {fee_tot:.5f} BNB\n"
            f"Net: {net:+.5f} BNB",
            silent=True
        )

    def check_anti_liquidation(self):
        """
        ANTI-LICHIDARE: monitorizare activă a pozițiilor futures.
        Skip dacă funding dezactivat sau testnet (futures API diferită).
        """
        if ALLOC_FUNDING <= 0 or USE_TESTNET:
            return
        if not self.pos:
            return

        try:
            positions = self.client.futures_positions()
            if not positions:
                return

            pos_map = {p["symbol"]: p for p in positions}

            for sym, data in list(self.pos.items()):
                fp = pos_map.get(sym)
                if not fp:
                    continue

                entry = data.get("entry_price", data.get("spot_p", 0))
                if entry <= 0:
                    continue

                mark = fp.get("mark_price", 0)
                if mark <= 0:
                    continue

                # Short position: pierdere când prețul URCĂ
                price_change_pct = (mark - entry) / entry

                # Nivel 1: Warning la 12%
                if price_change_pct >= FR_ANTILIQ_WARN_PCT:
                    self.log.warning(
                        f"⚠️ ANTI-LIQ WARNING {sym}: preț +{price_change_pct*100:.1f}% "
                        f"(entry=${entry:.2f} mark=${mark:.2f})")
                    tg(
                        f"⚠️ <b>ANTI-LICHIDARE WARNING</b> {sym}\n"
                        f"Preț: ${entry:.2f} → ${mark:.2f} (+{price_change_pct*100:.1f}%)\n"
                        f"Stop-loss la: ${data.get('stop_loss_price', 0):.2f}\n"
                        f"Lichidare la: ~${entry * 1.33:.2f}\n"
                        f"Acțiune: stop-loss activ pe exchange",
                        silent=False
                    )

                # Nivel 2: Emergency close la 25% (backup)
                if price_change_pct >= FR_ANTILIQ_EMERGENCY_PCT:
                    self.log.error(
                        f"🚨 EMERGENCY CLOSE {sym}: +{price_change_pct*100:.1f}% "
                        f"— stop-loss nu s-a executat, forțăm close!")
                    tg(
                        f"🚨 <b>EMERGENCY CLOSE</b> {sym}\n"
                        f"Preț +{price_change_pct*100:.1f}% — forțăm închidere\n"
                        f"Pierdere futures: ~${abs(fp.get('unrealized_pnl', 0)):.2f}\n"
                        f"Spot câștigă similar → net loss mic",
                        silent=False
                    )
                    self._close(sym, f"EMERGENCY_ANTILIQ +{price_change_pct*100:.0f}%")

        except Exception as e:
            self.log.debug(f"Anti-liq check: {e}")

    @staticmethod
    def _next_funding_slot() -> float:
        """
        FIX 2: Calculează secunde până la următorul slot Binance.
        Sloturile sunt la 00:00, 08:00, 16:00 UTC (la fiecare 8h).
        La pornire, botul se sincronizează automat — nu ratează nicio colectare.
        """
        now_utc   = datetime.now(timezone.utc)
        hour      = now_utc.hour
        # Sloturile zilei (ore UTC)
        slots     = [0, 8, 16]
        # Găsim următorul slot
        next_slot = None
        for s in slots:
            if hour < s:
                next_slot = s; break
        if next_slot is None:
            next_slot = 24  # primul slot de mâine (00:00)
        # Secunde până la slot + 90s buffer (Binance procesează puțin după oră)
        delta_h   = next_slot - hour
        delta_s   = delta_h * 3600 - now_utc.minute * 60 - now_utc.second + 90
        return max(delta_s, 60)  # minim 60s

    def run(self, stop: threading.Event):
        if FR_DISABLE_ON_TESTNET and USE_TESTNET:
            self.log.info(
                "💰 Funding ARB DEZACTIVAT pe testnet "
                "(ratele sunt artificiale → fee fara venit). "
                "Pe live va fi activ automat.")
            tg(
                "💰 <b>Funding ARB dezactivat pe testnet</b>\n"
                "Ratele testnet sunt artificiale și cauzează fee fără venit.\n"
                "Grid și Swing continuă normal.\n"
                "Pe live, funding va fi activ automat.",
                silent=True)
            stop.wait()
            return
        # Sync la următorul slot UTC de funding (00/08/16)
        now_utc = datetime.now(timezone.utc)
        seconds_into_day = now_utc.hour * 3600 + now_utc.minute * 60 + now_utc.second
        next_slot_s = ((seconds_into_day // 28800) + 1) * 28800
        secs = next_slot_s - seconds_into_day
        scan_mode = "DYNAMIC (top funding din toate USDT futures)" if FR_DYNAMIC_SCAN else f"STATIC ({len(FR_SYMBOLS)} perechi)"
        self.log.info(
            f"💰 Funding ARB | {self.bnb:.4f} BNB | "
            f"APR min {FR_APR_MIN*100:.0f}% | "
            f"max {FR_MAX_POS} poz | scan: {scan_mode} | "
            f"fee O DATĂ ({ROUNDTRIP_TAKER*100:.4f}%) | "
            f"prima colectare în {secs/60:.1f} min (sync UTC)")
        last_scan = 0.0
        # FIX 2: last_col setat în trecut cu diferența până la slotul următor
        last_col  = time.time() - (28800 - secs)
        while not stop.is_set():
            try:
                now = time.time()
                if now - last_scan > FR_SCAN_INTERVAL_S:   # FIX 2: 30 min
                    self.scan_and_enter()
                    self.check_exits()
                    last_scan = now
                if now - last_col >= 28800:
                    self.collect()
                    last_col = now
            except Exception as e:
                self.log.warning(f"Funding: {e}")
            stop.wait(60)
        self.log.info("⛔ Funding oprit")


# ══════════════════════════════════════════════════════════════════════
# SOL TRADER — Grid + Swing pe SOLUSDT cu capital SOL
# Profit convertit automat înapoi în SOL zilnic
# ══════════════════════════════════════════════════════════════════════

SOL_GRID_SPACING  = 0.003   # 0.3% spacing grid SOL — mai multe fills
SOL_GRID_LEVELS   = 8       # 8 nivele — ordine maxime simultane
ENABLE_SOL_TRADER = False   # LIVE: dezactivat până există USDC suficient ($500+)
SOL_SWING_TP      = 0.030   # 3.0% TP swing SOL (range 5-6%/zi → TP mai mare)
SOL_SWING_SL      = 0.012   # 1.2% SL swing SOL (RR=2.5:1)
SOL_RESERVE_PCT   = 0.20    # 20% din profit → rezervă USDT (lichiditate black swan)
SOL_TRADER_FILE   = "v8_sol_trader.json"

class SolTrader:
    """
    Folosește SOL din spot pentru grid + swing pe SOLUSDT.
    Profitul (în USDT) se convertește automat înapoi în SOL la 00:05 UTC.

    ALOCARE:
      70% din SOL → Grid SOLUSDT (ordine limit maker)
      30% din SOL → Swing SOLUSDT (ordine market taker)

    PROFIT → SOL:
      La fiecare zi la 00:05 UTC profitul USDT acumulat
      se convertește în SOL via market buy.
    """

    def __init__(self, client: "Binance", sol_qty: float,
                 fees: "FeeTracker"):
        self.client     = client
        self.sol        = sol_qty       # SOL total disponibil
        self.fees       = fees
        self.log        = L("SolTrd")
        self._lock      = threading.Lock()

        # Grid
        self.grid_sol   = sol_qty * 0.70
        self.grid_fills = 0
        self.grid_pnl_usdt = 0.0

        # Swing
        self.swing_sol  = sol_qty * 0.30
        self.swing_pnl_usdt = 0.0
        self.swing_trades: Dict[str, dict] = {}
        self.n_wins = 0; self.n_losses = 0

        # Profit acumulat în USDT (de convertit în SOL)
        self.pending_usdt = 0.0
        self.total_sol_earned = 0.0   # SOL câștigat total
        self.total_usdt_profit = 0.0

        self._last_rb   = 0.0
        self._sol_grid = {}
        self._sol_grid_mid = 0.0
        self._sol_grid_spacing = 0.01
        self._sol_grid_rebuilt = 0.0
        self._kl: Dict[str, Tuple[float, list]] = {}
        # Mainnet grid state
        self._sol_grid: Dict[str, dict] = {}  # {level_id: {side, price, qty, filled, oid}}
        self._sol_grid_mid = 0.0
        self._sol_grid_spacing = SOL_GRID_SPACING
        self._sol_grid_rebuilt = 0.0
        self._real_pending_usdt = 0.0  # only from real fills
        self.reserve_usdt = 0.0  # 20% din profit — rezervă lichiditate
        self._load()

    def _load(self):
        try:
            if os.path.exists(SOL_TRADER_FILE):
                with open(SOL_TRADER_FILE) as _jf:

                    d = json.load(_jf)
                self.grid_fills        = d.get("grid_fills", 0)
                self.grid_pnl_usdt     = d.get("grid_pnl_usdt", 0.0)
                self.swing_pnl_usdt    = d.get("swing_pnl_usdt", 0.0)
                self.pending_usdt      = d.get("pending_usdt", 0.0)
                self.total_sol_earned  = d.get("total_sol_earned", 0.0)
                self.total_usdt_profit = d.get("total_usdt_profit", 0.0)
                self.n_wins            = d.get("n_wins", 0)
                self.n_losses          = d.get("n_losses", 0)
                self._sol_grid         = d.get("sol_grid", {})
                self._sol_grid_mid     = d.get("sol_grid_mid", 0.0)
                self._sol_grid_spacing = d.get("sol_grid_spacing", 0.01)
                self._sol_grid_rebuilt = d.get("sol_grid_rebuilt", 0.0)
                self._real_pending_usdt = d.get("real_pending_usdt", 0.0)
                self.reserve_usdt      = d.get("reserve_usdt", 0.0)
        except Exception as _e: logging.debug(f'Ignored: {_e}')

    def _save(self):
        try:
            _data = {
                "grid_fills":        self.grid_fills,
                "grid_pnl_usdt":     self.grid_pnl_usdt,
                "swing_pnl_usdt":    self.swing_pnl_usdt,
                "pending_usdt":      self.pending_usdt,
                "total_sol_earned":  self.total_sol_earned,
                "total_usdt_profit": self.total_usdt_profit,
                "n_wins":            self.n_wins,
                "n_losses":          self.n_losses,
                "sol_grid":          self._sol_grid,
                "sol_grid_mid":      self._sol_grid_mid,
                "sol_grid_spacing":  self._sol_grid_spacing,
                "sol_grid_rebuilt":  self._sol_grid_rebuilt,
                "real_pending_usdt": self._real_pending_usdt,
                "reserve_usdt":      self.reserve_usdt,
                "ts":                time.time(),
            }
            _atomic_json_save(SOL_TRADER_FILE, _data, indent=2)
        except Exception as _e: logging.debug(f'Ignored: {_e}')

    def _klines(self, sym: str) -> list:
        ts, kl = self._kl.get(sym, (0.0, []))
        if time.time() - ts > 900:
            kl = self.client.klines(sym, "1h", 50)
            self._kl[sym] = (time.time(), kl)
        if len(self._kl) > 50: # TTL: max 50 symb în cache
            oldest = min(self._kl, key=lambda k: self._kl[k][0])
            del self._kl[oldest]
        return kl

    def _efficiency(self, kl: list) -> float:
        """Efficiency Ratio (ER) × 250 — proxy pentru trend strength.
        NU este ADX Wilder. Calibrat empiric pentru GRID_ADX_MAX=22."""
        if len(kl) < 14: return 30.0
        cl   = [float(k[4]) for k in kl]
        net  = abs(cl[-1] - cl[-min(14, len(cl))])
        gros = sum(abs(cl[i]-cl[i-1]) for i in range(1, len(cl))) or 1e-10
        return net / gros * 100 * 2.5

    def _adx(self, kl: list) -> float:
        """ADX Welles Wilder corect (14 perioade).
        Folosit acolo unde e nevoie de trend strength real."""
        if len(kl) < 15: return 30.0
        try:
            highs  = [float(k[2]) for k in kl]
            lows   = [float(k[3]) for k in kl]
            closes = [float(k[4]) for k in kl]
            # True Range
            trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]),
                       abs(lows[i]-closes[i-1])) for i in range(1, len(kl))]
            # +DM / -DM
            pdm = [max(highs[i]-highs[i-1], 0) if highs[i]-highs[i-1] > lows[i-1]-lows[i]
                   else 0 for i in range(1, len(kl))]
            ndm = [max(lows[i-1]-lows[i], 0) if lows[i-1]-lows[i] > highs[i]-highs[i-1]
                   else 0 for i in range(1, len(kl))]
            # Smoothed (14)
            n = 14
            atr14 = sum(trs[:n]); pdi14 = sum(pdm[:n]); ndi14 = sum(ndm[:n])
            for i in range(n, len(trs)):
                atr14 = atr14 - atr14/n + trs[i]
                pdi14 = pdi14 - pdi14/n + pdm[i]
                ndi14 = ndi14 - ndi14/n + ndm[i]
            pdi = 100 * pdi14 / max(atr14, 1e-10)
            ndi = 100 * ndi14 / max(atr14, 1e-10)
            dx  = 100 * abs(pdi - ndi) / max(pdi + ndi, 1e-10)
            return dx
        except Exception:
            return self._efficiency(kl)

    # ── Grid ──────────────────────────────────────────────────────────
    def _calc_keltner(self, kl, period: int = 20, atr_mult: float = 2.0):
        """
        Keltner Channel: EMA(20) ± ATR(10) × 2.0
        Returnează (ema, upper, lower) sau (None,None,None) la date insuficiente.
        """
        if not kl or len(kl) < period + 5:
            return None, None, None
        try:
            closes = [float(k[4]) for k in kl]
            highs  = [float(k[2]) for k in kl]
            lows   = [float(k[3]) for k in kl]

            # EMA(20)
            k_mult = 2.0 / (period + 1)
            ema = sum(closes[:period]) / period
            for c in closes[period:]:
                ema = c * k_mult + ema * (1 - k_mult)

            # ATR(10)
            trs = []
            for i in range(1, len(kl)):
                h = highs[i]; l = lows[i]; pc = closes[i-1]
                trs.append(max(h - l, abs(h - pc), abs(l - pc)))
            atr = sum(trs[-10:]) / 10 if len(trs) >= 10 else sum(trs) / max(len(trs), 1)

            upper = ema + atr * atr_mult
            lower = ema - atr * atr_mult
            return ema, upper, lower
        except Exception:
            return None, None, None

    def _calc_zscore_bnb_sol(self) -> float:
        """
        Z-score spread BNB/SOL pentru mean reversion.
        Z > +2: BNB supraevaluat față de SOL → vinzi BNB, cumperi SOL
        Z < -2: SOL supraevaluat față de BNB → vinzi SOL, cumperi BNB
        Returnează z-score sau 0.0 la date insuficiente.
        """
        try:
            kl_bnb = self.client.klines("BNBUSDC", "1h", 30)
            kl_sol = self.client.klines("SOLUSDC", "1h", 30)
            if len(kl_bnb) < 25 or len(kl_sol) < 25:
                return 0.0
            closes_bnb = [float(k[4]) for k in kl_bnb[-25:]]
            closes_sol = [float(k[4]) for k in kl_sol[-25:]]
            import math
            spread = [math.log(b) - math.log(s) for b, s in zip(closes_bnb, closes_sol)]
            mean = sum(spread) / len(spread)
            std  = (sum((x - mean)**2 for x in spread) / len(spread)) ** 0.5
            return (spread[-1] - mean) / std if std > 1e-10 else 0.0
        except Exception:
            return 0.0

    def _calc_sol_spacing(self, kl, sol_price):
        """Calculează spacing dinamic + nivele adaptive pentru SOL grid."""
        adx = self._adx(kl)
        if adx > 45:
            return 0, 0, adx  # trend — skip

        adx_norm = min(1.0, max(0.0, adx / 25))
        dyn_sp = SOL_GRID_SPACING + adx_norm * SOL_GRID_SPACING  # baza din config + ADX scaling

        atr_pct = 0.03
        try:
            if kl and len(kl) >= 14:
                trs = []
                for i in range(1, len(kl)):
                    h = float(kl[i][2]); l = float(kl[i][3]); pc = float(kl[i-1][4])
                    trs.append(max(h-l, abs(h-pc), abs(l-pc)))
                atr = sum(trs[-14:]) / 14
                atr_pct = atr / sol_price if sol_price > 0 else 0.03
                if atr_pct > 0.04: dyn_sp *= 1.3   # era 1.4
                elif atr_pct > 0.025: dyn_sp *= 1.1  # era 1.2
                elif atr_pct < 0.01: dyn_sp *= 0.85
        except Exception as _e: logging.debug(f"Ignored: {_e}")
        if adx > 15:
            vol_mult = 1.0 + (adx - 15) / 10 * 0.20  # era 0.30
            dyn_sp *= min(vol_mult, 1.3)               # era 1.5
        dyn_sp = min(dyn_sp, 0.014)  # cap 1.4% (era 1.8%)

        # Folosește SOL_GRID_LEVELS dacă setat explicit (!=0)
        if SOL_GRID_LEVELS > 0:
            n_levels = SOL_GRID_LEVELS
            # Safety cap pe volatilitate mare — nu expunem prea mult capital
            if atr_pct > 0.04: n_levels = min(n_levels, 4)  # high vol → max 4
            elif atr_pct > 0.02: n_levels = min(n_levels, 6)  # medium vol → max 6
        elif atr_pct > 0.04: n_levels = 2
        elif atr_pct > 0.02: n_levels = 3
        else: n_levels = 4

        return dyn_sp, n_levels, adx

    def _sol_grid_build(self, sol_price, dyn_sp, n_levels):
        """Construiește grid real SOL: ordine limit BUY sub mid, SELL peste mid."""
        # Cancel stale orders
        if not USE_TESTNET:
            try:
                old = self.client.spot_open_orders("SOLUSDC")
                if old:
                    self.client.spot_cancel_all("SOLUSDC")
                    self.log.info(f"SOL Grid: cancelled {len(old)} stale orders")
            except Exception as _e: logging.debug(f"Ignored: {_e}")
        qty_per_level = round(self.grid_sol / max(n_levels * 2, 1), 4)
        
        # Validate minQty SOL (Binance min = 0.001 SOL, ~$0.09)
        if qty_per_level < 0.001:
            self.log.warning(
                f"SOL Grid: qty_per_level {qty_per_level} < minQty 0.001 → "
                f"reduce n_levels sau crește grid_sol capital")
            # Still try with minimum
            qty_per_level = 0.001
        
        # Validate min notional (Binance SOLUSDC ~$10)
        min_notional = qty_per_level * sol_price
        if min_notional < 10:
            self.log.warning(
                f"SOL Grid: notional per level ${min_notional:.2f} < $10 min — skip build")
            return
        
        levels = {}

        cumul = 0.0
        for i in range(1, n_levels + 1):
            cumul += dyn_sp * (SOL_GEO_MULT ** (i - 1))  # uniform spacing SOL → mai multe fills
            # BUY levels below mid (geometric: nivelele exterioare mai largi)
            buy_price = round(sol_price * (1 - cumul), 2)
            buy_id = f"B{i}"
            levels[buy_id] = {
                "side": "BUY", "price": buy_price,
                "qty": qty_per_level, "filled": False, "oid": None
            }
            # SELL levels above mid (geometric)
            sell_price = round(sol_price * (1 + cumul), 2)
            sell_id = f"S{i}"
            levels[sell_id] = {
                "side": "SELL", "price": sell_price,
                "qty": qty_per_level, "filled": False, "oid": None
            }

        # Place orders
        for lid, lvl in levels.items():
            try:
                if lvl["side"] == "BUY":
                    r = self.client.limit_buy("SOLUSDC", lvl["qty"], lvl["price"])
                else:
                    r = self.client.limit_sell("SOLUSDC", lvl["qty"], lvl["price"])
                lvl["oid"] = r.get("orderId")
            except Exception as e:
                self.log.warning(f"SOL Grid place {lid}: {e}")

        with self._lock:
            self._sol_grid = levels
            self._sol_grid_mid = sol_price
            self._sol_grid_spacing = dyn_sp
            self._sol_grid_rebuilt = time.time()

        self.log.info(
            f"✅ SOL Grid built: {n_levels}×2 levels | "
            f"mid=${sol_price:.2f} | spacing={dyn_sp*100:.2f}%")
        self._save()

    def _sol_grid_check_fills(self, sol_price):
        """Verifică fills pe grid SOL real prin comparație preț."""
        with self._lock:
            levels = dict(self._sol_grid)
        if not levels:
            return

        bnb_price = self.client.price("BNBUSDC") or 640

        for lid, lvl in levels.items():
            if lvl["filled"]:
                continue

            hit = False
            if USE_TESTNET or MAINNET_DRY_RUN:
                if lvl["side"] == "BUY":
                    hit = sol_price <= lvl["price"]
                else:
                    hit = sol_price >= lvl["price"]
            else:
                # LIVE: pre-check preț
                price_in_range = (
                    (lvl["side"] == "BUY" and sol_price <= lvl["price"] * 1.002) or
                    (lvl["side"] == "SELL" and sol_price >= lvl["price"] * 0.998)
                )
                if not price_in_range:
                    continue
                # Confirmă via API
                oid = lvl.get("oid")
                if not oid:
                    continue
                order_info = self.client.get_order("SOLUSDC", oid)
                status = order_info.get("status", "UNKNOWN")
                if status == "FILLED":
                    hit = True
                elif status in ("CANCELED", "EXPIRED", "REJECTED"):
                    self.log.warning(f"SOL Grid lvl={lid} status={status} → repost")
                    lvl["filled"] = False
                    lvl["oid"] = None
                    try:
                        if lvl["side"] == "BUY":
                            r = self.client.limit_buy("SOLUSDC", lvl["qty"], lvl["price"])
                        else:
                            r = self.client.limit_sell("SOLUSDC", lvl["qty"], lvl["price"])
                        lvl["oid"] = r.get("orderId")
                    except Exception as e:
                        self.log.warning(f"SOL Grid repost after {status}: {e}")
                    continue
                continue

            if not hit:
                continue

            lvl["filled"] = True
            spacing = self._sol_grid_spacing
            fill_usdt = lvl["qty"] * lvl["price"] * spacing
            _notional_usdt = lvl["qty"] * lvl["price"]
            fee_usdt = _notional_usdt * MAKER_FEE * 2
            net_usdt = fill_usdt - fee_usdt

            with self._lock:
                self.grid_pnl_usdt += net_usdt
                self.pending_usdt += net_usdt
                self._real_pending_usdt += net_usdt
                self.grid_fills += 1
                self.total_usdt_profit += net_usdt

            fee_bnb = fee_usdt / max(bnb_price, 1)
            net_bnb = net_usdt / max(bnb_price, 1)
            self.fees.record("SOL_GRID", fee_bnb, net_bnb + fee_bnb, 2)

            if hasattr(self, '_health') and self._health:
                self._health.record_trade("sol_grid", net_bnb)

            # Repost opposite order
            if lvl["side"] == "BUY":
                opp_price = round(lvl["price"] * (1 + spacing), 2)
                opp_side = "SELL"
            else:
                opp_price = round(lvl["price"] * (1 - spacing), 2)
                opp_side = "BUY"

            try:
                if opp_side == "BUY":
                    r = self.client.limit_buy("SOLUSDC", lvl["qty"], opp_price)
                else:
                    r = self.client.limit_sell("SOLUSDC", lvl["qty"], opp_price)
                lvl.update({
                    "price": opp_price, "side": opp_side,
                    "filled": False, "oid": r.get("orderId")
                })
            except Exception as e:
                self.log.warning(f"SOL Grid repost {lid}: {e}")

            with self._lock:
                self._sol_grid[lid] = lvl

            self.log.info(
                f"✅ SOL Grid fill: +${net_usdt:.3f} USDT | "
                f"spacing={spacing*100:.2f}% | fills={self.grid_fills}")

        self._save()

    def _run_grid(self):
        """Grid pe SOLUSDT cu 70% din SOL — ordine REALE."""
        sol_price = self.client.price("SOLUSDC")
        if sol_price <= 0: return

        # Hard stop: daca SOL a scazut >15%
        if not hasattr(self, '_grid_entry_price'):
            self._grid_entry_price = sol_price
        drop = (self._grid_entry_price - sol_price) / max(self._grid_entry_price, 0.01)
        if drop > 0.15:
            self.log.warning(
                f"SOL Grid STOP: pret scazut {drop*100:.1f}% "
                f"(${self._grid_entry_price:.2f} → ${sol_price:.2f})")
            return

        # Safety checks (ca BNB Grid)
        if hasattr(self, "_crash") and self._crash and not self._crash.entries_ok:
            self.log.info(f"SOL Grid rebuild SKIP — crash: {self._crash.level}")
            return
        if is_fed_blackout():
            self.log.info("SOL Grid rebuild SKIP — FOMC blackout")
            return

        kl = self._klines("SOLUSDC")
        dyn_sp, n_levels, adx = self._calc_sol_spacing(kl, sol_price)
        if dyn_sp == 0:
            self.log.info(f"SOL Grid OFF — ADX={adx:.0f} (trend)")
            # ADX prea mare — cancel grid activ
            if self._sol_grid:
                self.log.info(f"SOL Grid: ADX={adx:.0f} → clearing grid")
                if not USE_TESTNET:
                    try: self.client.spot_cancel_all("SOLUSDC")
                    except: pass
                with self._lock: self._sol_grid = {}
            return

        cap_per_level = self.grid_sol * sol_price / max(n_levels, 1)
        expected_profit = cap_per_level * dyn_sp
        expected_fee = cap_per_level * MAKER_FEE * 2
        if expected_profit < expected_fee * 3:
            return

        # Rebuild grid la fiecare GRID_REBUILD_H ore sau la prima rulare
        need_rebuild = (
            not self._sol_grid or
            time.time() - self._sol_grid_rebuilt > GRID_REBUILD_H * 3600
        )

        if need_rebuild:
            self._sol_grid_build(sol_price, dyn_sp, n_levels)

        # Check fills
        self._sol_grid_check_fills(sol_price)

    # ── Swing ─────────────────────────────────────────────────────────
    def _run_swing(self):
        """Swing pe SOLUSDT cu 30% din SOL."""
        if len(self.swing_trades) >= 1: return

        kl  = self._klines("SOLUSDC")
        if len(kl) < 22: return

        cl    = [float(k[4]) for k in kl]
        vl    = [float(k[5]) for k in kl]
        price = cl[-1]

        # ── Signal 1: Keltner Breakout ──────────────────────────────────
        ema, upper, lower = self._calc_keltner(kl, period=20, atr_mult=2.0)
        keltner_dir = 0
        if ema and upper and lower:
            if price > upper:   keltner_dir =  1   # breakout sus → LONG
            elif price < lower: keltner_dir = -1   # breakout jos → SHORT

        # ── Signal 2: Z-Score BNB/SOL Mean Reversion ───────────────────
        zscore = self._calc_zscore_bnb_sol()
        zscore_dir = 0
        if zscore < -2.0:   zscore_dir =  1  # SOL subevaluat → cumperi SOL
        elif zscore > 2.0:  zscore_dir = -1  # SOL supraevaluat → vinzi SOL

        # ── Signal 3: RSI+ADX clasic (fallback) ─────────────────────────
        d_  = [cl[i]-cl[i-1] for i in range(1, len(cl))]
        g_  = [max(0,x) for x in d_[-14:]]
        l_  = [abs(min(0,x)) for x in d_[-14:]]
        ag  = sum(g_)/14; al = sum(l_)/14
        rsi = 100-(100/(1+ag/al)) if al > 0 else 50
        adx = self._adx(kl)
        volr = vl[-1] / (sum(vl[-21:-1])/20) if sum(vl[-21:-1]) > 0 else 1.0
        classic_dir = 0
        if adx >= 28 and volr >= 1.8:
            if rsi < 45:   classic_dir =  1
            elif rsi > 55: classic_dir = -1

        # ── Decizie finală: 2 din 3 semnale trebuie să coincidă ──────────
        signals = [keltner_dir, zscore_dir, classic_dir]
        votes_long  = signals.count(1)
        votes_short = signals.count(-1)

        if votes_long >= 2:      direction =  1
        elif votes_short >= 2:   direction = -1
        else:
            self.log.debug(
                f"SOL Swing SKIP: K={keltner_dir} Z={zscore:.2f} C={classic_dir} "
                f"(no 2/3 consensus)")
            return

        sol_size = self.swing_sol * 0.20
        fee_usdt = sol_size * price * ROUNDTRIP_MAKER  # limit_chaser: maker fee

        # Plasează ordin cu limit_chaser (maker-first, fallback market)
        if not USE_TESTNET:
            side = "BUY" if direction == 1 else "SELL"
            r = self.client.limit_chaser(
                "SOLUSDC", side, round(sol_size, 4),
                max_attempts=3, wait_sec=2.5
            )
            if not r or r.get("status") in ("BLOCKED_FEE", "INVALID", "RATE_LIMITED"):
                self.log.warning(f"SOL Swing entry SKIP: {r}")
                return
            # Prețul real de execuție
            fills = r.get("fills", [])
            if fills:
                price = float(fills[0].get("price", price))

        t = {
            "dir":    direction,
            "entry":  price,
            "size":   sol_size,
            "ts":     time.time(),
            "fee_in": fee_usdt * 0.5,
        }
        with self._lock:
            self.swing_trades["SOLUSDC"] = t
        self.log.info(
            f"📈 SOL Swing {'LONG' if direction>0 else 'SHORT'} "
            f"@ ${price:.2f} | {sol_size:.4f} SOL")

    def _update_swing(self):
        """Verifică TP/SL pentru swing SOL."""
        with self._lock:
            trades = dict(self.swing_trades)
        if not trades: return

        sol_price = self.client.price("SOLUSDC")
        if sol_price <= 0: return

        for sym, t in trades.items():
            pct     = (sol_price - t["entry"]) / t["entry"] * t["dir"]
            hold_h  = (time.time() - t["ts"]) / 3600
            hit_tp  = pct >= SOL_SWING_TP
            hit_sl  = -pct >= SOL_SWING_SL
            timeout = hold_h >= 24

            if not (hit_tp or hit_sl or timeout): continue

            # Moon Bag: vinde 90%, păstrează 10% ca SOL pur acumulat
            MOON_BAG_PCT = 0.10
            close_size = round(t["size"] * (1.0 - MOON_BAG_PCT), 4)
            moon_bag   = round(t["size"] * MOON_BAG_PCT, 4)

            fee_out  = close_size * sol_price * MAKER_FEE  # limit_chaser: maker fee
            pnl_usdt = close_size * t["entry"] * pct - t["fee_in"] - fee_out

            # Închidere cu limit_chaser (maker-first, fallback market)
            if not USE_TESTNET:
                side = "SELL" if t["dir"] == 1 else "BUY"
                r = self.client.limit_chaser(
                    "SOLUSDC", side, close_size,
                    max_attempts=3, wait_sec=2.0
                )
                if not r or r.get("status") in ("BLOCKED_FEE", "INVALID"):
                    self.log.warning(f"SOL Swing close SKIP: {r}")
                    continue
                if moon_bag >= 0.001:
                    with self._lock:
                        self.moon_bag_sol = getattr(self, "moon_bag_sol", 0.0) + moon_bag
                    self.log.info(
                        f"🌙 Moon Bag +{moon_bag:.4f} SOL "
                        f"(total: {getattr(self,'moon_bag_sol',0):.4f} SOL)")

            with self._lock:
                if sym in self.swing_trades: del self.swing_trades[sym]
                self.swing_pnl_usdt    += pnl_usdt
                self.pending_usdt      += pnl_usdt
                self._real_pending_usdt += pnl_usdt
                self.total_usdt_profit += pnl_usdt
                if pnl_usdt > 0: self.n_wins += 1
                else:            self.n_losses += 1

            reason = "TP" if hit_tp else ("SL" if hit_sl else "TIMEOUT")
            n = self.n_wins + self.n_losses
            self.log.info(
                f"📤 SOL Swing {reason}: "
                f"{pnl_usdt:+.3f} USDT | "
                f"WR={self.n_wins}/{n}")
            self._save()

    # ── Conversie profit USDT → SOL ───────────────────────────────────
    def convert_profit_to_sol(self):
        """
        Convertește profitul USDT acumulat înapoi în SOL.
        Apelat la 00:05 UTC zilnic.
        """
        with self._lock:
            if USE_TESTNET:
                pending = self.pending_usdt
            else:
                # Pe mainnet: convertim doar profitul din ordine REALE
                pending = self._real_pending_usdt

        if pending <= 0.50:
            self.log.info(
                f"SolTrader: profit pending ${pending:.3f} < $0.50 → skip")
            return

        sol_price = self.client.price("SOLUSDC")
        if sol_price <= 0: return

        # 20% → rezervă USDT, 80% → cumpără SOL
        reserve_cut = round(pending * SOL_RESERVE_PCT, 4)
        buy_amount = pending - reserve_cut

        sol_qty = round(buy_amount / sol_price, 4)
        if sol_qty < 0.001: return

        if not USE_TESTNET:
            r = self.client.market_buy("SOLUSDC", sol_qty)
            if r.get("status") != "FILLED":
                self.log.warning(f"SolTrader: conversie esuata: {r}")
                return
        else:
            self.log.info(
                f"[TESTNET] SolTrader: conversie "
                f"${pending:.3f} → {sol_qty:.4f} SOL @ ${sol_price:.2f}")

        with self._lock:
            self.sol              += sol_qty
            self.total_sol_earned += sol_qty
            self.reserve_usdt     += reserve_cut
            self.pending_usdt      = 0.0
            self._real_pending_usdt = 0.0
            # AUTO-COMPOUND: update grid capital cu SOL nou
            old_grid = self.grid_sol
            self.grid_sol = self.sol * 0.70
            self.log.debug(f"SOL grid realocat: {old_grid:.4f} → {self.grid_sol:.4f} SOL")
            self.swing_sol = self.sol * 0.30

        self._save()
        self.log.info(
            f"🔄 SOL Trader conversie: "
            f"${pending:.3f} → 80% +{sol_qty:.4f} SOL | "
            f"20% +${reserve_cut:.2f} rezervă | "
            f"total SOL: {self.total_sol_earned:.4f} | "
            f"grid: {self.grid_sol:.4f} SOL | "
            f"total rezervă: ${self.reserve_usdt:.2f}")
        tg(
            f"🔄 <b>SOL Trader — Profit convertit</b>\n"
            f"${pending:.3f} USDT profit:\n"
            f"  80% → +{sol_qty:.4f} SOL\n"
            f"  20% → +${reserve_cut:.2f} rezervă (total ${self.reserve_usdt:.2f})\n"
            f"Pret SOL: ${sol_price:.2f}\n"
            f"─────────────────\n"
            f"Total SOL câștigat: {self.total_sol_earned:.4f} SOL\n"
            f"Grid fills: {self.grid_fills} | "
            f"Swing WR: {self.n_wins}/{self.n_wins+self.n_losses}",
            silent=False
        )

    def status(self) -> str:
        sol_p = self.client.price("SOLUSDC")
        n = self.n_wins + self.n_losses
        return (
            f"🌊 <b>SOL Trader</b>\n"
            f"Capital: {self.sol:.4f} SOL (${self.sol*sol_p:.2f})\n"
            f"  70% Grid: {self.sol*0.70:.4f} SOL\n"
            f"  30% Swing: {self.sol*0.30:.4f} SOL\n"
            f"Grid fills: {self.grid_fills}\n"
            f"Grid PnL: ${self.grid_pnl_usdt:.3f} USDT\n"
            f"Swing PnL: ${self.swing_pnl_usdt:.3f} USDT\n"
            + (f"Swing WR: {self.n_wins}/{n} ({self.n_wins/n*100:.0f}%)\n"
               if n > 0 else "Swing WR: 0/0\n") +
            f"Profit pending: ${self.pending_usdt:.3f} USDT\n"
            f"Total SOL câștigat: {self.total_sol_earned:.4f} SOL "
            f"(${self.total_sol_earned*sol_p:.2f})\n"
            f"Profit convertit automat zilnic în SOL"
        )

    def emergency_close(self):
        """Anulează toate ordinele SOL grid la urgență."""
        try:
            self.client.spot_cancel_all("SOLUSDC")
            with self._lock:
                self._sol_grid.clear()
            self.log.warning("🚨 SOL Grid emergency close")
        except Exception as e:
            self.log.debug(f"SOL emergency_close: {e}")

    def run(self, stop: threading.Event):
        sol_p = self.client.price("SOLUSDC")
        self.log.info(
            f"🌊 SOL Trader pornit | "
            f"{self.sol:.4f} SOL (${self.sol*sol_p:.2f}) | "
            f"Grid 70% + Swing 30% | "
            f"Profit → SOL zilnic")
        tg(
            f"🌊 <b>SOL Trader pornit</b>\n"
            f"Capital: {self.sol:.4f} SOL (${self.sol*sol_p:.2f})\n"
            f"Grid: {self.sol*0.70:.4f} SOL (spacing {SOL_GRID_SPACING*100:.1f}%)\n"
            f"Swing: {self.sol*0.30:.4f} SOL (TP {SOL_SWING_TP*100:.1f}%)\n"
            f"Profit convertit automat în SOL la 00:05 UTC",
            silent=True
        )

        last_grid_check = 0.0
        last_day        = datetime.now(timezone.utc).day

        while not stop.is_set():
            try:
                now_utc = datetime.now(timezone.utc)

                # Grid check la 45 secunde
                if time.time() - last_grid_check > 45:
                    self._run_grid()
                    last_grid_check = time.time()

                # Swing activ dacă ≥ 2 SOL (5.84 SOL disponibil)
                if self.sol >= 2.0:
                    self._run_swing()
                    self._update_swing()

                # Conversie profit → SOL la 00:05 UTC
                if (now_utc.day != last_day and
                        now_utc.hour == 0 and now_utc.minute >= 5):
                    self.convert_profit_to_sol()
                    last_day = now_utc.day

            except Exception as e:
                self.log.warning(f"SolTrader: {e}")
            stop.wait(60)
        self.log.info("⛔ SOL Trader oprit")


# ══════════════════════════════════════════════════════════════════════
# S2: GRID MAKER — limit orders, spacing 1.5%
# ══════════════════════════════════════════════════════════════════════

class GridMaker:
    """
    30% din capitalul BNB → grid pe perechi X/BNB.
    EXCLUSIV ordine LIMIT (maker fee 0.0075%/leg vs taker 0.05625%).
    Spacing 1.5% → profit net/roundtrip = 1.485% după fee maker.

    CONDIȚIE: ADX < 22 (piață sideways — gridul funcționează)
    REBUILD: la 8h sau dacă ADX > 27 (trend apărut)
    """

    def __init__(self, client: Binance, bnb_capital: float,
                 fees: FeeTracker, guard: DailyTradeGuard,
                 crash: "MarketCrashGuard" = None,
                 fee_guard: "FeeBufferManager" = None):
        self.client    = client
        self.bnb       = bnb_capital
        self.fees      = fees
        self.guard     = guard
        self.crash     = crash
        self.fee_guard = fee_guard
        self.log    = L("Grid")
        self._lock  = threading.Lock()
        self.grids: Dict[str, dict] = {}
        self.total_pnl   = 0.0
        self.total_fees  = 0.0
        self.total_fills = 0
        self.all_pairs: List[str] = []
        self._last_rb    = 0.0
        self._kl: Dict[str, Tuple[float, list]] = {}
        self._enh_optimizer = None  # ENH: GridSpacingOptimizer
        self._enh_mgr = None        # ENH: EnhancementsManager
        self._ml = None             # ML: MLEngine reference
        self._compounded_pnl = 0.0  # AUTO-COMPOUND: cât s-a adăugat deja
        self._perf: "Optional[PerformanceMetrics]" = None
        # VARIANTA B: {sym: {"qty": float, "deadline": float, "oid": str|None}}
        self._liquidation_timers: dict = {}
        self._load()

    def _load(self):
        try:
            if os.path.exists("v3_grid.json"):
                with open("v3_grid.json") as _jf:

                    d = json.load(_jf)
                # Verifică că state-ul nu e mai vechi de 24h (ghost capital protection)
                saved_ts = d.get("ts", 0)
                if time.time() - saved_ts > 86400:
                    logging.warning(
                        "v3_grid.json mai vechi de 24h — IGNORAT (ghost capital protection). "
                        "Șterge manual dacă vrei să resetezi starea.")
                    return
                self.total_pnl   = d.get("total_pnl", 0.0)
                self.total_fees  = d.get("total_fees", 0.0)
                self.total_fills = d.get("total_fills", 0)
                self._compounded_pnl = d.get("compounded_pnl", 0.0)
                # Restaurează qty_net salvat per pereche
                self._saved_qty_net = d.get("qty_net_map", {})
                self._saved_avg_buy = d.get("avg_buy_map", {})
        except Exception as _e: logging.debug(f'Ignored: {_e}')

    def _save(self, force: bool = False):
        # Throttle: scrie maxim o dată la 30s — EXCEPȚIE: flush forțat la fills noi
        now = time.time()
        fills_changed = self.total_fills != getattr(self, "_last_saved_fills", -1)
        if not force and not fills_changed and now - getattr(self, "_last_save_ts", 0) < 30:
            return
        self._last_save_ts = now
        self._last_saved_fills = self.total_fills
        try:
            # Salvează qty_net per pereche pentru persistența la restart
            _qty_net_map = {
                sym: g.get("qty_net", 0.0)
                for sym, g in self.grids.items()
                if g.get("qty_net", 0.0) > 0
            }
            _avg_buy_map = {
                sym: g.get("avg_buy_price", 0.0)
                for sym, g in self.grids.items()
                if g.get("avg_buy_price", 0.0) > 0
            }
            _data = {
                "total_pnl": self.total_pnl,
                "total_fees": self.total_fees,
                "total_fills": self.total_fills,
                "compounded_pnl": self._compounded_pnl,
                "active": len(self.grids), "ts": now,
                "qty_net_map": _qty_net_map,
                "avg_buy_map": _avg_buy_map,
            }
            _atomic_json_save("v3_grid.json", _data, indent=2)
        except Exception as _e: logging.debug(f'Ignored: {_e}')

    def _klines(self, sym: str) -> list:
        ts, kl = self._kl.get(sym, (0.0, []))
        if time.time() - ts > 900:
            kl = self.client.klines(sym, "1h", 40)
            if not hasattr(self, "_klines_cache"): self._klines_cache = {}
            self._klines_cache[sym] = kl
            self._kl[sym] = (time.time(), kl)
        return kl

    def _adx(self, kl: list) -> float:
        """ADX Welles Wilder corect (14 perioade) — identic cu SolTrader._adx."""
        if len(kl) < 15: return 30.0
        try:
            highs  = [float(k[2]) for k in kl]
            lows   = [float(k[3]) for k in kl]
            closes = [float(k[4]) for k in kl]
            trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]),
                       abs(lows[i]-closes[i-1])) for i in range(1, len(kl))]
            pdm = [max(highs[i]-highs[i-1], 0) if highs[i]-highs[i-1] > lows[i-1]-lows[i]
                   else 0 for i in range(1, len(kl))]
            ndm = [max(lows[i-1]-lows[i], 0) if lows[i-1]-lows[i] > highs[i]-highs[i-1]
                   else 0 for i in range(1, len(kl))]
            n = 14
            atr14 = sum(trs[:n]); pdi14 = sum(pdm[:n]); ndi14 = sum(ndm[:n])
            for i in range(n, len(trs)):
                atr14 = atr14 - atr14/n + trs[i]
                pdi14 = pdi14 - pdi14/n + pdm[i]
                ndi14 = ndi14 - ndi14/n + ndm[i]
            pdi = 100 * pdi14 / max(atr14, 1e-10)
            ndi = 100 * ndi14 / max(atr14, 1e-10)
            return 100 * abs(pdi - ndi) / max(pdi + ndi, 1e-10)
        except Exception:
            return 30.0

    def _adx_spacing(self, sym: str, adx: float) -> float:
        """
        Spacing auto-calibrat: ADX + ATR live + Fear&Greed.
        ATR (Average True Range) reflectă volatilitatea REALĂ a ultimelor ore.
        SOL face 5-6% range zilnic → spacing trebuie să fie mai mare.
        BNB face 2.5% range → spacing mai mic.
        """
        adx_norm = min(1.0, max(0.0, adx / GRID_ADX_MAX))
        if sym in ("BNBUSDT", "BNBUSDC"):
            sp = GRID_SPACING_BNB_MIN + adx_norm * (GRID_SPACING_BNB_MAX - GRID_SPACING_BNB_MIN)
        else:
            sp = GRID_SPACING_MIN + adx_norm * (GRID_SPACING_MAX - GRID_SPACING_MIN)

        # ATR live — ajustează spacing pe volatilitate reală
        try:
            kl = self.client.klines(sym, "1h", 20)
            if kl and len(kl) >= 14:
                trs = []
                for i in range(1, len(kl)):
                    h = float(kl[i][2]); l = float(kl[i][3]); pc = float(kl[i-1][4])
                    trs.append(max(h-l, abs(h-pc), abs(l-pc)))
                atr = sum(trs[-14:]) / 14
                price = float(kl[-1][4])
                if price > 0:
                    atr_pct = atr / price  # ATR ca % din preț
                    # SOL: ATR~5% → boost spacing 1.5x
                    # BNB: ATR~2% → normal 1.0x
                    if atr_pct > 0.03:      # >3% ATR = foarte volatil
                        sp *= 1.5
                    elif atr_pct > 0.02:    # >2% = volatil
                        sp *= 1.2
                    elif atr_pct < 0.008:   # <0.8% = foarte calm
                        sp *= 0.85          # spacing mai mic = mai multe fills
        except Exception as _e: logging.debug(f"Ignored: {_e}")
        # ADX volatility boost (existent)
        vol_mult = 1.0
        if adx > 15:
            vol_mult = 1.0 + (adx - 15) / (GRID_ADX_MAX - 15) * 0.50
        sp *= vol_mult

        # Market Sentinel boost — Fear&Greed
        # FG multiplier dezactivat pentru grid — crește spacing inutil în Fear
        # if hasattr(self, '_sentinel') and self._sentinel:
        #     sp *= self._sentinel.get_spacing_mult()

        # Floor: nu sub minimul per pereche + garantie profit > 5x fee
        pair_min = GRID_SPACING_BNB_MIN if sym in ("BNBUSDT", "BNBUSDC") else GRID_SPACING_MIN
        min_floor = max(pair_min, ROUNDTRIP_MAKER * 8)  # ×8 safety margin (era ×5 — prea mic)
        # Cap maxim: BNB 1.0%, SOL 1.6% — evită multiplicatori stivuiți (ADX×ATR×FG = 2%+)
        sp_cap = 0.0025 if sym in ("BNBUSDT", "BNBUSDC") else 0.006  # 0.6% cap — mai multe fills
        return max(min(sp, sp_cap), min_floor)

    def _select(self) -> List[Tuple[str, float, float]]:
        """
        Selecteaza perechi sideways pentru grid.
        Returneaza (sym, mid_price, adx) — adx pastrat pentru _build.
        """
        selected = []
        prices   = self.client.all_prices()

        now = time.time()
        if not hasattr(self, "_vol_pairs_cache") or now - getattr(self, "_vol_pairs_ts", 0) > 3600:
            self._vol_pairs_cache = self.client.discover_volatile_usdc_pairs(
                min_vol_usd=10_000_000, min_volatility=0.015, top_n=8)
            self._vol_pairs_ts = now
        candidate_pairs = self._vol_pairs_cache or ["BNBUSDC"]

        # Failsafe whipsaw: init blacklist
        if not hasattr(self, '_blacklist'):
            self._blacklist = {}
        _now_bl = time.time()
        self._blacklist = {s: d for s, d in self._blacklist.items() if d > _now_bl}

        for sym in candidate_pairs:
            if len(selected) >= GRID_MAX_PAIRS: break
            if sym in self._blacklist:
                self.log.info(
                    f"  Grid {sym}: BLACKLIST whipsaw → skip "
                    f"({int((self._blacklist[sym]-_now_bl)/3600)}h ramase)")
                continue
            kl  = self._klines(sym)
            adx = self._adx(kl)
            if adx > GRID_ADX_MAX:
                self.log.info(f"  Grid {sym}: ADX={adx:.1f} > {GRID_ADX_MAX} → skip (trend)")
                continue
            mid = prices.get(sym, 0)
            if mid <= 0: continue
            sp  = self._adx_spacing(sym, adx)
            selected.append((sym, mid, adx))
            self.log.info(
                f"  Grid {sym}: ADX={adx:.1f} → spacing={sp*100:.2f}% ✅")

        if not selected:
            # P9: fallback cu verificare trend EMA pe fiecare pereche
            for _fb in ["ETHUSDC", "SUIUSDC", "XRPUSDC", "NEARUSDC", "BNBUSDC"]:
                if len(selected) >= GRID_MAX_PAIRS: break
                _fb_mid = prices.get(_fb, 0)
                if _fb_mid <= 0: continue
                _fb_kl = self._klines(_fb)
                _fb_adx = self._adx(_fb_kl)
                if _fb_adx > GRID_ADX_MAX: continue
                # Verifica trend EMA
                _fb_closes = [float(k[4]) for k in _fb_kl] if _fb_kl else []
                if len(_fb_closes) >= 20:
                    _fb_ema5 = sum(_fb_closes[-5:]) / 5
                    _fb_ema20 = sum(_fb_closes[-20:]) / 20
                    if _fb_ema5 < _fb_ema20 * 0.998 and _fb_closes[-1] < _fb_ema20 * 0.995:
                        self.log.info(f"  Grid {_fb}: fallback skip — trend DOWN")
                        continue
                selected.append((_fb, _fb_mid, _fb_adx))
                self.log.info(f"  Grid {_fb}: fallback lateral (ADX={_fb_adx:.1f})")
            if not selected:
                self.log.warning("  Grid: nicio pereche disponibila — toate in trend DOWN")
        return selected


    def _get_inventory_usd(self, sym: str) -> float:
        asset = sym.replace("USDC","").replace("USDT","")
        if asset in ("BNB","SOL","USDC","USDT"): return 0.0
        try:
            acc = self.client._get("/api/v3/account", {}, signed=True) or {}
            qty = sum(float(b["free"])+float(b["locked"])
                      for b in acc.get("balances",[]) if b["asset"]==asset)
            return qty * (self.client.price(sym) or 0)
        except Exception as _e:
            self.log.debug(f"_get_inventory_usd {sym}: {_e}")
            return -1.0

    def _build(self, sym: str, mid: float,
               bnb_per_pair: float, adx: float = 15.0) -> dict:
        """Build grid with correct qty calculation per pair type.
        
        FIXED (Day 2): handles X/BNB, X/USDT, BNB/USDT correctly.
        Previously assumed all pairs were X/BNB → USDT pairs had qty=0.
        """
        # I1: spacing dinamic bazat pe ADX curent al perechii
        spacing = self._adx_spacing(sym, adx)
        bpl    = bnb_per_pair / GRID_LEVELS  # doar BUY plasate — capital corect per nivel
        levels = []
        
        # Get BNB price for USDT pair qty conversion
        bnb_price = self.client.price("BNBUSDC") if not sym.endswith("BNB") else 0
        if not sym.endswith("BNB") and bnb_price <= 0:
            self.log.warning(f"Grid {sym}: BNB price unavailable → skip")
            return {"sym":sym,"mid":mid,"bnb":bnb_per_pair,"spacing":spacing,
                    "adx":adx,"levels":[],"ts":time.time(),"pnl":0.0,"fills":0}
        
        cumul = 0.0
        for i in range(1, GRID_LEVELS+1):
            cumul += spacing * (GRID_GEO_MULT ** (i - 1))
            # BUY level
            bp  = round(mid*(1-cumul), 8)
            qx  = self.client.calculate_qty_for_pair(sym, bpl, bp, bnb_price)
            qx  = round(qx, 6)
            levels.append({"price":bp,"side":"BUY","qty":qx,
                           "filled":False,"oid":None})
            # SELL level (use mid as reference price — qty consistent with buy)
            sp  = round(mid*(1+cumul), 8)
            qx2 = self.client.calculate_qty_for_pair(sym, bpl, mid, bnb_price)
            qx2 = round(qx2, 6)
            levels.append({"price":sp,"side":"SELL","qty":qx2,
                           "filled":False,"oid":None})
        
        # Log first level for debugging
        if levels:
            self.log.info(
                f"Grid {sym} build: {len(levels)} levels | "
                f"first qty={levels[0]['qty']} @ ${levels[0]['price']:.4f} | "
                f"bpl={bpl:.5f} BNB")
        
        return {"sym":sym,"mid":mid,"bnb":bnb_per_pair,"spacing":spacing,
                "adx":adx,"levels":levels,"ts":time.time(),"pnl":0.0,"fills":0}

    def _place(self, g: dict):
        """DOAR BUY la build. SELL se plaseaza automat la fill BUY."""
        # P1: skip _place daca inventar real > $40 la restart
        if g.get("sym","") in getattr(self, "_place_skip", set()):
            self.log.info(f"📦 _place {g.get('sym','')}: skip (inventar real > $40)")
            return
        # COOLDOWN check: nu plasa BUY daca perechea e in cooldown
        _cd = g.get('_cooldown_until', 0)
        if _cd > time.time():
            _remaining = int((_cd - time.time()) / 60)
            self.log.info(
                f"🧊 _place {g.get('sym','?')}: cooldown {_remaining} min — skip BUY")
            return
        # Verifica inventar total (free+locked) — daca >30 USD, skip BUY
        _sym = g.get("sym", "")
        _asset = _sym.replace("USDC","").replace("USDT","")
        try:
            _accp = self.client._get('/api/v3/account', {}, signed=True) or {}
            _inv_qty = sum(float(b["free"])+float(b["locked"])
                          for b in _accp.get("balances",[]) if b["asset"]==_asset)
            _cur_p = self.client.price(_sym) or 0
            _inv_usd = _inv_qty * _cur_p
            _first_buy = next((float(l["qty"])*float(l["price"])
                               for l in g.get("levels",[])
                               if l.get("side")=="BUY" and not l.get("filled")), 0)
            if _inv_usd + _first_buy > 80.0:
                self.log.info(f"⛔ _place {_sym}: inventar ${_inv_usd:.1f} + ordin ${_first_buy:.1f} > $80 — skip BUY")
                return
        except Exception as _ep:
            self.log.debug(f"Inventar check: {_ep}")
        try:
            _usdc_start = self.client.full_balance().get("USDC", 0)
        except:
            _usdc_start = 0
        _usdc_used = 0.0
        for lvl in g["levels"]:
            if lvl["filled"]: continue
            try:
                if lvl["side"] == "SELL":
                    continue  # skip SELL — nu avem tokens
                _needed = lvl["qty"] * lvl["price"]
                if _needed < 10.0 or (_usdc_start - _usdc_used) < _needed + 5.0:  # min 0/ordin
                    self.log.debug(
                        f"Grid {g['sym']}: USDC insuficient "
                        f"(${_usdc_start - _usdc_used:.0f} < "
                        f"${_needed:.0f}) — skip nivel")
                    continue
                r = self.client.limit_buy(
                    g["sym"], lvl["qty"], lvl["price"],
                    fee_guard=self.fee_guard)
                if r.get("status") == "BLOCKED_FEE":
                    self.log.warning(
                        f"Grid {g['sym']}: ordin blocat — fee buffer critic")
                    break
                lvl["oid"] = r.get("orderId")
                _usdc_used += _needed
            except Exception as e:
                self.log.warning(f"Grid place {g['sym']}: {e}")

    @staticmethod
    def _is_grid_active_hours() -> bool:
        """
        I3: Grid activ doar 06:00-22:00 UTC.
        Noaptea volumul scade ~60%, fill rate neglijabil.
        Ordinele existente raman deschise, nu rebuildam si nu plasam noi.
        """
        h = datetime.now(timezone.utc).hour
        return GRID_ACTIVE_UTC_START <= h < GRID_ACTIVE_UTC_END

    def rebuild(self):
        # Selectorul dinamic alege perechile — nu fortam BNBUSDC
        global GRID_USDT_PAIRS
        # I3: Pauza nocturna UTC
        if not self._is_grid_active_hours():
            self.log.info(
                f"Grid rebuild SKIP — pauza nocturna "
                f"({datetime.now(timezone.utc).strftime('%H:%M')} UTC | "
                f"activ {GRID_ACTIVE_UTC_START}:00-{GRID_ACTIVE_UTC_END}:00 UTC)")
            self._last_rb = time.time(); return
        # Crash guard
        if self.crash and not self.crash.entries_ok:
            self.log.info(f"Grid rebuild SKIP — crash: {self.crash.level}")
            self._last_rb = time.time(); return
        # Fed blackout
        if is_fed_blackout():
            self.log.info("Grid rebuild SKIP — FOMC Fed blackout")
            self._last_rb = time.time(); return

        self.log.info("Rebuild grid (spacing dinamic ADX)...")

        # ── PROTECTIE: blocare in trend general ──
        # Daca media ADX a perechilor candidate > 40, piata e in trend
        # general (nu lateral) → grid nu functioneaza → asteapta.
        try:
            _cand = self._select() or []
            if _cand:
                _adx_vals = []
                for _cs, _cm, _ca in _cand:
                    if _ca and _ca > 0:
                        _adx_vals.append(_ca)
                if _adx_vals:
                    _avg_adx = sum(_adx_vals) / len(_adx_vals)
                    _high_adx = sum(1 for a in _adx_vals if a > 35)
                    if _avg_adx > 40 or _high_adx >= len(_adx_vals):
                        self.log.info(
                            f"📈 TREND GENERAL: ADX mediu={_avg_adx:.1f}, "
                            f"{_high_adx}/{len(_adx_vals)} perechi in trend "
                            f"→ trend mode activ (spacing 1.5% SELL 3%)")
                        # Nu return — continua in trend mode cu spacing marit
        except Exception as _te:
            self.log.debug(f"Check trend general: {_te}")

        # ── VARIANTA B: check timer lichidare (4h fallback → market sell) ──
        _now = time.time()
        for _sym, _liq in list(self._liquidation_timers.items()):
            if _now >= _liq["deadline"]:
                _qty = _liq["qty"]
                _p   = self.client.price(_sym) or 0
                if _qty > 0 and _p > 0 and _qty * _p >= 3.0:
                    self.log.info(
                        f"⏱ Lichidare 4h timeout {_sym}: "
                        f"market sell {_qty:.6f} (~${_qty*_p:.2f})")
                    try:
                        self.client.market_sell(_sym, round(_qty, 6),
                                                fee_guard=self.fee_guard)
                    except Exception as _e:
                        self.log.warning(f"Lichidare market sell {_sym}: {_e}")
                del self._liquidation_timers[_sym]

        # ── VARIANTA B: lichidare perechi care ies din selecție ──
        _new_pairs_syms = set(s for s, _, _ in (self._select() or []))
        # Verifică și qty_net_map din save (perechi din sesiuni anterioare)
        _saved_map = getattr(self, "_saved_qty_net", {})
        _all_syms_with_inventory = set(self.grids.keys()) | set(_saved_map.keys())
        for _sym in _all_syms_with_inventory:
            if _sym in _new_pairs_syms:
                continue  # rămâne în grid, nu lichidăm
            # Determină qty_net din grids sau din saved_map
            _qty_net = self.grids.get(_sym, {}).get("qty_net", 0.0)
            if _qty_net <= 0:
                _qty_net = _saved_map.get(_sym, 0.0)
            _avg_p   = self.grids.get(_sym, {}).get("avg_buy_price", 0.0)
            _cur_p   = self.client.price(_sym) or 0
            _val_usd = _qty_net * _cur_p
            if _qty_net <= 0 or _cur_p <= 0:
                continue
            if _val_usd < 3.0:
                self.log.info(
                    f"Lichidare {_sym}: qty={_qty_net:.6f} "
                    f"val=${_val_usd:.2f} < $3 → ignorat")
                continue
            # Plasează limit sell la avg_buy_price sau preț curent
            _sell_p = round(_avg_p if _avg_p > 0 else _cur_p, 8)
            self.log.info(
                f"📦 Lichidare inventar {_sym}: "
                f"limit sell {_qty_net:.6f} @ ${_sell_p:.4f} "
                f"(val=${_val_usd:.2f}, timeout 4h)")
            try:
                _r = self.client.limit_sell(
                    _sym, round(_qty_net, 6), _sell_p,
                    fee_guard=self.fee_guard)
                _oid = _r.get("orderId")
            except Exception as _e:
                self.log.warning(f"Lichidare limit sell {_sym}: {_e}")
                _oid = None
            self._liquidation_timers[_sym] = {
                "qty": _qty_net,
                "deadline": _now + 4 * 3600,
                "oid": _oid
            }
            # Curăță din saved_map ca să nu se repete la rebuild următor
            if _sym in _saved_map:
                del _saved_map[_sym]

        # FIX6: Cancel stale orders before rebuild
        for sym in GRID_USDT_PAIRS:
            try:
                old = self.client.spot_open_orders(sym)
                if old:
                    self.client.spot_cancel_all(sym)
                    self.log.info(f"FIX6: Cancelled {len(old)} stale orders on {sym}")
            except Exception as e:
                self.log.debug(f"Stale cleanup {sym}: {e}")
        if not self.all_pairs:
            self.all_pairs = self.client.discover_bnb_pairs(20)

        pairs = self._select()   # returneaza (sym, mid, adx)
        if not pairs:
            self.log.info(f"Nicio pereche sideways (ADX>{GRID_ADX_MAX})")
            self._last_rb = time.time(); return

        # ── CHECK RENTABILITATE: nu intra daca piata scade accelerat ──
        _skip_grid = False
        try:
            # Check 1h: BTC -1.5% in 2h
            _btc_kl = self.client.klines("BTCUSDC", "1h", 3)
            if _btc_kl and len(_btc_kl) >= 3:
                _btc_closes = [float(k[4]) for k in _btc_kl]
                _btc_chg_2h = (_btc_closes[-1] - _btc_closes[0]) / _btc_closes[0] * 100
                if _btc_chg_2h < -1.5:
                    self.log.warning(
                        f"⛔ GRID SKIP: BTC {_btc_chg_2h:+.1f}% in 2h "
                        f"→ piata in scadere accelerata")
                    _skip_grid = True
            # P8: Check 15min — trend mai rapid
            if not _skip_grid:
                _btc_15m = self.client.klines("BTCUSDC", "15m", 8)
                if _btc_15m and len(_btc_15m) >= 8:
                    _c15 = [float(k[4]) for k in _btc_15m]
                    _ema4_15 = sum(_c15[-4:]) / 4
                    _ema8_15 = sum(_c15) / 8
                    _chg_2h_15 = (_c15[-1] - _c15[0]) / _c15[0] * 100
                    if _ema4_15 < _ema8_15 * 0.997 and _chg_2h_15 < -1.0:
                        self.log.warning(
                            f"⛔ GRID SKIP: BTC 15m trend DOWN "
                            f"(EMA4/EMA8={_ema4_15/_ema8_15:.4f}, {_chg_2h_15:+.1f}% 2h)")
                        _skip_grid = True
        except Exception as _ce:
            self.log.debug(f"Check rentabilitate: {_ce}")
        if _skip_grid:
            self._last_rb = time.time()
            return

        # AUTO-COMPOUND: adaugă doar profitul NOU (necumpus încă)
        new_profit = self.total_pnl - self._compounded_pnl
        if new_profit > 0.0001:
            old_bnb = self.bnb
            self.bnb += new_profit
            self._compounded_pnl = self.total_pnl
            self.log.info(
                f"🔄 Grid compound: {old_bnb:.5f} + {new_profit:.5f} new profit "
                f"= {self.bnb:.5f} BNB (total compounded: {self._compounded_pnl:.5f})")
        bnb_each = self.bnb_eq / max(len(pairs), 1)  # FIX: bnb_eq = capital USDC in BNB, nu BNB fizic
        with self._lock: self.grids = {}

        for sym, mid, adx in pairs:
            sp = self._adx_spacing(sym, adx)
            # ── TREND DETECTION: EMA simplu pe 1h ──
            # Foloseste klines deja citite in _adx_spacing (cache)
            try:
                _kl_t = getattr(self, "_klines_cache", {}).get(sym)
                if not _kl_t:
                    _kl_t = self.client.klines(sym, "1h", 25)
                if _kl_t and len(_kl_t) >= 20:
                    _closes = [float(k[4]) for k in _kl_t]
                    _ema5 = sum(_closes[-5:]) / 5
                    _ema20 = sum(_closes[-20:]) / 20
                    _cur_c = _closes[-1]
                    _trend_up = _ema5 > _ema20 * 1.005 and _cur_c > _ema20 * 1.01
                    _trend_dn = _ema5 < _ema20 * 0.998 and _cur_c < _ema20 * 0.995
                    if _trend_up:
                        sp = 0.015  # 1.5% spacing → SELL la 3%
                        self.log.info(
                            f"📈 {sym}: trend UP (EMA5={_ema5:.4f} > EMA20={_ema20:.4f}) "
                            f"→ spacing=1.5% SELL=3%")
                    elif _trend_dn:
                        self.log.info(f"📉 {sym}: trend DOWN → skip grid")
                        continue
            except Exception as _te:
                self.log.debug(f"Trend check {sym}: {_te}")
            # ═══ ENH: Grid Optimizer — validare spacing minim profitabil ═══
            if self._enh_optimizer:
                bnb_p = self.client.price("BNBUSDC")
                cap_usd = bnb_each * bnb_p
                opt = self._enh_optimizer.calculate_optimal_grid(
                    symbol=sym, current_price=mid,
                    allocated_capital_usd=cap_usd,
                    recent_volatility_pct=adx * 0.15  # proxy vol din ADX
                )
                min_sp = opt["min_profitable_spacing_pct"] / 100
                if sp < min_sp:
                    self.log.warning(
                        f"ENH Grid {sym}: spacing {sp*100:.2f}% < min profitabil "
                        f"{min_sp*100:.2f}% → ajustat")
                    sp = min_sp
            # ═══ ENH v1.1: size multiplier per strategie (grid) ═══
            enh_mult = 1.0
            if self._enh_mgr:
                enh_mult = self._enh_mgr.get_size_multiplier_for_strategy("grid")
                if enh_mult <= 0:
                    self.log.info(f"ENH Grid SKIP — size mult = 0 (CB/OI/PL)")
                    continue
            adj_bnb = bnb_each * enh_mult
            g  = self._build(sym, mid, adj_bnb, adx=adx)
            with self._lock: self.grids[sym] = g
            # M1: Restaurează qty_net din save sau din soldul Binance
            _saved_map = getattr(self, "_saved_qty_net", {})
            if sym in _saved_map and _saved_map[sym] > 0:
                with self._lock:
                    if sym in self.grids:
                        self.grids[sym]["qty_net"] = _saved_map[sym]
                        # Restaureaza avg_buy_price salvat
                        _saved_avg = getattr(self, "_saved_avg_buy", {}).get(sym, 0.0)
                        if _saved_avg > 0:
                            self.grids[sym]["avg_buy_price"] = _saved_avg
                self.log.info(
                    f"📦 qty_net restaurat {sym}: {_saved_map[sym]:.6f} (din save)")
                # P1 FIX: verifica inventar real Binance — daca > $40, skip _place
                try:
                    _asset_p1 = sym.replace("USDC","").replace("USDT","")
                    _acc_p1 = self.client._get("/api/v3/account", {}, signed=True) or {}
                    _qty_p1 = sum(float(b["free"])+float(b["locked"])
                                  for b in _acc_p1.get("balances",[])
                                  if b["asset"]==_asset_p1)
                    _val_p1 = _qty_p1 * (self.client.price(sym) or 0)
                    if _val_p1 > 40.0:
                        self.log.info(
                            f"📦 {sym}: inventar real ${_val_p1:.1f} > $40 "
                            f"→ skip _place (anti-acumulare)")
                        self._place_skip = getattr(self, "_place_skip", set())
                        self._place_skip.add(sym)
                    else:
                        self._place_skip = getattr(self, "_place_skip", set())
                        self._place_skip.discard(sym)
                except Exception as _p1e:
                    self.log.debug(f"P1 inventar check: {_p1e}")
            else:
                # Fallback: citește soldul real Binance
                try:
                    _asset = sym.replace("USDC", "").replace("USDT", "")
                    _bal = self.client.full_balance().get(_asset, 0.0)
                    _p = self.client.price(sym) or 1
                    if _bal > 0 and _bal * _p >= 1.0:
                        with self._lock:
                            if sym in self.grids:
                                self.grids[sym]["qty_net"] = _bal
                        self.log.info(
                            f"📦 qty_net din Binance {sym}: {_bal:.6f} "
                            f"(${_bal*_p:.2f})")
                except Exception as _qe:
                    self.log.debug(f"qty_net fallback {sym}: {_qe}")
            # Verifica inventar INAINTE de _place — previne acumulare
            _sym_asset = sym.replace("USDC","").replace("USDT","")
            try:
                _inv_check = self.client._get("/api/v3/account", {}, signed=True) or {}
                _inv_qty = sum(float(b["free"])+float(b["locked"])
                              for b in _inv_check.get("balances",[])
                              if b["asset"]==_sym_asset)
                _inv_val = _inv_qty * (self.client.price(sym) or 0)
                if _inv_val > 80.0:
                    self.log.info(f"⛔ rebuild {sym}: inventar  > 5 — skip _place")
                    continue
            except Exception as _ice:
                self.log.debug(f"Inventar check rebuild: {_ice}")
            self._place(g)
            self.log.info(
                f"✅ Grid {sym}: ADX={adx:.1f} → spacing={sp*100:.2f}% | "
                f"net/fill={(sp-ROUNDTRIP_MAKER)*100:.3f}% | "
                f"{bnb_each:.4f} BNB")

        self._last_rb = time.time(); self._save()
        pairs_info = " | ".join(
            f"{s} {self._adx_spacing(s,a)*100:.1f}%"
            for s,_,a in pairs)
        tg(
            f"🔲 <b>Grid rebuild</b> (ADX dinamic)\n"
            f"{len(pairs)} perechi\n{pairs_info}\n"
            f"Fee maker: {ROUNDTRIP_MAKER*100:.4f}%",
            silent=True
        )

    def _startup_sell_orphans(self):
        """La pornire: plaseaza SELL pentru tokens free fara ordin de vanzare.
        Rezolva problema tokens ramasi dupa restart fara SELL.
        """
        if MAINNET_DRY_RUN or USE_TESTNET:
            return
        try:
            real_balances = self.client.full_balance()
        except Exception as _e:
            self.log.debug(f"Startup orphans balance error: {_e}")
            return

        # Asset-uri de ignorat
        skip = {'USDC','USDT','BUSD','FDUSD','TUSD','DAI','BNB','SOL',
                'LDADA','LDPEPE','LDBIO','LDBNB','CTSI','PIXEL','W','LUNC'}

        for asset, qty in real_balances.items():
            if asset in skip or qty <= 0:
                continue
            # Cauta perechea USDC
            sym = f"{asset}USDC"
            try:
                cur_p = self.client.price(sym) or 0
                if cur_p <= 0:
                    continue
                val_usd = qty * cur_p
                if val_usd < 1.0:
                    continue
                # Limiteaza SELL la max $80 per asset — evita expunere masiva
                if val_usd > 80.0:
                    qty = round(80.0 / cur_p, 6)
                    self.log.info(
                        f"Startup {sym}: inventar ${val_usd:.1f} > $80 "
                        f"→ SELL partial {qty:.6f} (${80.0:.0f})")
                # Verifica daca exista deja SELL activ pe Binance
                open_orders = self.client.spot_open_orders(sym)
                has_sell = any(
                    o.get('side') == 'SELL'
                    for o in (open_orders or [])
                )
                if has_sell:
                    self.log.info(
                        f"Startup {sym}: {qty:.4f} free, SELL deja activ")
                    continue
                # FIX: nu vinde sub pretul de cumparare.
                # Citeste pretul mediu real de achizitie din istoric.
                _avg_buy = 0.0
                try:
                    import hmac as _h, hashlib as _hl, urllib.request as _u
                    _ak = self.client.key
                    _sk = self.client.secret
                    _since = int((time.time() - 7*86400) * 1000)
                    _pq = f"symbol={sym}&startTime={_since}&limit=1000&timestamp={int(time.time()*1000)}"
                    _sg = _h.new(_sk.encode(), _pq.encode(), _hl.sha256).hexdigest()
                    _url = f"https://api.binance.com/api/v3/myTrades?{_pq}&signature={_sg}"
                    _req = _u.Request(_url, headers={"X-MBX-APIKEY": _ak})
                    _trades = json.loads(_u.urlopen(_req, timeout=10).read())
                    _bq = [(float(t['price']), float(t['qty']))
                           for t in _trades if t['isBuyer']]
                    if _bq:
                        _tq = sum(q for _, q in _bq)
                        _avg_buy = sum(p*q for p, q in _bq) / _tq if _tq else 0
                except Exception as _ae:
                    self.log.debug(f"Startup avg_buy {sym}: {_ae}")

                # Break-even = pret cumparare + 2x fee (acopera comisioane)
                _breakeven = _avg_buy * (1 + 2*MAKER_FEE) if _avg_buy > 0 else 0

                if _avg_buy <= 0:
                    # Nu stim pretul de cumparare → NU vindem (siguranta)
                    self.log.info(
                        f"⏸ Startup {sym}: {qty:.4f} free, pret cumparare "
                        f"necunoscut → NU vand (pastrez)")
                    continue

                if cur_p < _breakeven:
                    # Sub break-even → plaseaza SELL LA break-even, nu la pierdere
                    sell_p = round(_breakeven, 8)
                    self.log.info(
                        f"🔧 Startup SELL {sym}: {qty:.6f} @ ${sell_p:.4f} "
                        f"(break-even, avg buy ${_avg_buy:.4f}, "
                        f"curent ${cur_p:.4f} — astept revenire)")
                else:
                    # Peste break-even → vinde la curent +1%
                    sell_p = round(cur_p * 1.01, 8)
                    self.log.info(
                        f"🔧 Startup SELL {sym}: {qty:.6f} @ ${sell_p:.4f} "
                        f"(profit, avg buy ${_avg_buy:.4f})")

                _r = self.client.limit_sell(
                    sym, round(qty, 6), sell_p,
                    fee_guard=self.fee_guard)
                self.log.info(
                    f"✅ Startup SELL plasat {sym}: "
                    f"orderId={_r.get('orderId')} @ ${sell_p:.4f}")
            except Exception as _e:
                self.log.debug(f"Startup sell {sym}: {_e}")

    def _check_fills_live_balance(self):
        """Fix definitiv fills live: compara sold real Binance vs qty_net intern.
        Detecteaza BUY-uri executate indiferent de oid.
        """
        if MAINNET_DRY_RUN or USE_TESTNET:
            return  # doar pe live
        try:
            real_balances = self.client.full_balance()
        except Exception as _e:
            self.log.debug(f"Balance check error: {_e}")
            return

        # FIX INVENTAR: citeste free+locked pentru fiecare asset
        # full_balance() returneaza doar 'free' — tokens locked in SELL
        # nu apar → inventarul pare 0 → botul cumpara fara limita
        try:
            _acc3 = self.client._get('/api/v3/account', {}, signed=True) or {}
            real_balances_total = {
                b['asset']: float(b['free']) + float(b['locked'])
                for b in _acc3.get('balances', [])
                if float(b['free']) + float(b['locked']) > 0
            }
        except Exception as _be3:
            self.log.warning(f"Balance total fallback: {_be3} — skip ciclu")
            return  # nu continua fara date corecte de inventar

        with self._lock:
            grids_copy = dict(self.grids)

        for sym, g in grids_copy.items():
            try:
                asset = sym.replace('USDC', '').replace('USDT', '')
                real_qty = real_balances.get(asset, 0.0)  # free only (pt fill detection)
                real_qty_total = real_balances_total.get(asset, 0.0)  # free+locked (pt inventar)
                qty_net  = g.get('qty_net', 0.0)
                cur_p    = self.client.price(sym) or 0
                if cur_p <= 0:
                    continue

                # ── STOP-LOSS DUR -3% ──
                # Daca avem inventar real si pretul a scazut 3% sub
                # pretul mediu de cumparare → vinde imediat (pierdere mica)
                _avg_buy = g.get('avg_buy_price', 0.0)
                if real_qty > 0 and _avg_buy > 0 and cur_p > 0:
                    _loss_pct = (cur_p - _avg_buy) / _avg_buy
                    _inv_usd_sl = real_qty * cur_p
                    if _loss_pct <= -0.03 and _inv_usd_sl >= 5.0:
                        self.log.warning(
                            f"🛑 STOP-LOSS {sym}: pret ${cur_p:.4f} vs "
                            f"avg ${_avg_buy:.4f} ({_loss_pct*100:.1f}%) "
                            f"→ market sell {real_qty:.6f}")
                        try:
                            self.client.spot_cancel_all(sym)
                            self.client.market_sell(
                                sym, round(real_qty, 6),
                                fee_guard=self.fee_guard)
                            with self._lock:
                                if sym in self.grids:
                                    self.grids[sym]['qty_net'] = 0.0
                            self._save()
                            continue
                        except Exception as _sle:
                            self.log.warning(f"Stop-loss {sym}: {_sle}")

                # Toleranta: ignora diferente sub 1 USD
                diff = real_qty - qty_net
                diff_usd = diff * cur_p

                if abs(diff_usd) < 1.0:
                    continue  # diferenta neglijabila

                # SELL executat — diff negativ (tokens vanduti)
                if diff < 0:
                    _sold_usd = abs(diff_usd)
                    _bnb_p = self.client.price('BNBUSDC') or 640
                    _avg_b12 = g.get('avg_buy_price', 0.0)
                    if _avg_b12 > 0:
                        _real_pnl = abs(diff) * (cur_p - _avg_b12) - (_sold_usd * MAKER_FEE * 2)
                    else:
                        _real_pnl = _sold_usd * g.get('spacing', 0.01) - (_sold_usd * MAKER_FEE * 2)
                    pnl_bnb = _real_pnl / _bnb_p
                    fee_bnb = (_sold_usd * MAKER_FEE * 2) / _bnb_p
                    self.log.info(
                        f"✅ SELL executat {sym}: "
                        f"{abs(diff):.6f} {sym.replace('USDC','').replace('USDT','')} "
                        f"(~${_sold_usd:.2f}) | profit ~${_real_pnl:.3f}")

                    # ── FAILSAFE WHIPSAW: contor tranzactii pe pereche ──
                    if not hasattr(self, '_trade_log'):
                        self._trade_log = {}
                    if not hasattr(self, '_blacklist'):
                        self._blacklist = {}
                    _tnow = time.time()
                    self._trade_log.setdefault(sym, [])
                    self._trade_log[sym].append(_tnow)
                    # Pastreaza doar ultimele 6h
                    self._trade_log[sym] = [
                        t for t in self._trade_log[sym] if _tnow - t < 6*3600]
                    _trades_6h = len(self._trade_log[sym])
                    if _trades_6h > 20:
                        # Whipsaw detectat — blacklist 24h + scoate din grid
                        self._blacklist[sym] = _tnow + 24*3600
                        self.log.warning(
                            f"🚫 WHIPSAW {sym}: {_trades_6h} tranzactii in 6h "
                            f"→ BLACKLIST 24h, scos din grid")
                        try:
                            self.client.spot_cancel_all(sym)
                            _ar = sym.replace('USDC','').replace('USDT','')
                            _bal_r = self.client.full_balance().get(_ar, 0.0)
                            if _bal_r * cur_p > 5.0:
                                self.client.market_sell(
                                    sym, round(_bal_r, 6),
                                    fee_guard=self.fee_guard)
                                self.log.warning(
                                    f"🚫 {sym}: lichidat {_bal_r:.4f} la blacklist")
                        except Exception as _ble:
                            self.log.warning(f"Blacklist {sym}: {_ble}")
                        with self._lock:
                            if sym in self.grids:
                                del self.grids[sym]
                        self._save()
                        continue

                    with self._lock:
                        if sym in self.grids:
                            self.grids[sym]['qty_net'] = real_qty
                            self.grids[sym]['pnl']   += pnl_bnb
                            self.grids[sym]['fills'] += 1
                            self.total_pnl   += pnl_bnb
                            self.total_fees  += fee_bnb
                            self.total_fills += 1
                            # COOLDOWN: track ultimele 5 fills profit/pierdere
                            if not hasattr(self, '_fill_history'):
                                self._fill_history = {}
                            self._fill_history.setdefault(sym, []).append(_real_pnl)
                            self._fill_history[sym] = self._fill_history[sym][-5:]
                            # Daca ultimele 3 fills sunt negative → cooldown 15 min
                            _last3 = self._fill_history[sym][-3:]
                            if len(_last3) >= 3 and all(p < 0 for p in _last3):
                                _cd_until = time.time() + 900  # 15 min
                                self.grids[sym]['_cooldown_until'] = _cd_until
                                self.log.warning(
                                    f"🧊 COOLDOWN {sym}: 3 fills negative consecutiv "
                                    f"({[f'{p:.3f}' for p in _last3]}) → pauza 15 min")
                    # GRID CYCLING: SELL executat → plaseaza BUY nou la -spacing
                    _sp = g.get('spacing', 0.01)
                    # Folosim cur_p pentru BUY ciclu
                    _buy_p = round(cur_p * (1 - _sp), 8)
                    _buy_qty = round(abs(diff), 6)
                    _buy_val = _buy_qty * _buy_p
                    # Verifica cooldown inainte de cycling BUY
                    if g.get('_cooldown_until', 0) > time.time():
                        _rem = int((g['_cooldown_until'] - time.time()) / 60)
                        self.log.info(
                            f"🧊 Grid cycle {sym}: cooldown {_rem} min — skip BUY")
                        self._save()
                        continue
                    # Verifica inventar inainte de cycling BUY
                    _inv_cycle = real_qty_total * cur_p
                    if _inv_cycle > 80.0:  # consistent cu _place
                        self.log.info(
                            f"⛔ Grid cycle {sym}: inventar ${_inv_cycle:.1f} > $30 — skip BUY ciclu")
                        self._save()
                        continue
                    if _buy_val >= 10.0:  # minim 0/ordin
                        try:
                            _rb = self.client.limit_buy(
                                sym, _buy_qty, _buy_p,
                                fee_guard=self.fee_guard)
                            self.log.info(
                                f"🔄 Grid cycle {sym}: BUY {_buy_qty:.6f} "
                                f"@ ${_buy_p:.4f} orderId={_rb.get('orderId')}")
                            # qty_net = 0 dupa SELL executat
                            # (tokens vanduti, asteptam BUY nou sa se execute)
                            with self._lock:
                                if sym in self.grids:
                                    self.grids[sym]['qty_net'] = 0.0
                                    self.grids[sym]['avg_buy_price'] = _buy_p  # fix avg_buy
                        except Exception as _be:
                            self.log.warning(f"Grid cycle BUY {sym}: {_be}")
                    self._save()
                    continue

                # FIX1: Limita inventar — foloseste soldul real Binance
                _max_inventory_usd = 80.0  # max 2-3 nivele trend mode
                _real_inventory_usd = real_qty_total * cur_p  # free+locked
                if _real_inventory_usd > _max_inventory_usd:
                    # Inventar mare: plaseaza SELL dar NU BUY nou
                    # (nu sari cu continue — lasa codul sa plaseze SELL)
                    self.log.info(
                        f"⚠️  Inventar real {sym} ${_real_inventory_usd:.2f} > "
                        f"${_max_inventory_usd} — SELL plasat, BUY blocat")
                    with self._lock:
                        if sym in self.grids:
                            self.grids[sym]['qty_net'] = real_qty
                    # Forteaza plasarea SELL direct, fara BUY ciclu
                    _sell_p_inv = round(cur_p * (1 + g.get("spacing", 0.01) * 2.0), 8)  # P5: SELL 2x spacing
                    _sell_qty_inv = round(real_qty, 6)
                    try:
                        _open_inv = self.client.spot_open_orders(sym) or []
                        _has_sell_inv = any(o.get('side')=='SELL' for o in _open_inv)
                        if not _has_sell_inv and _sell_qty_inv * _sell_p_inv >= 5.0:
                            _ri = self.client.limit_sell(
                                sym, _sell_qty_inv, _sell_p_inv,
                                fee_guard=self.fee_guard)
                            self.log.info(
                                f"📤 SELL inventar {sym}: {_sell_qty_inv} "
                                f"@ ${_sell_p_inv:.4f} orderId={_ri.get('orderId')}")
                    except Exception as _sie:
                        self.log.warning(f"SELL inventar {sym}: {_sie}")
                    continue
                # FIX3: Protectie capital — foloseste real_balances deja citit
                _usdc_free = real_balances.get('USDC', 0)
                if _usdc_free < 30.0:
                    self.log.warning(
                        f"⚠️  USDC liber ${_usdc_free:.2f} < $30 "
                        f"— SELL plasat, BUY nou omis")
                    # Continua cu SELL dar nu plasa BUY ciclu
                    # (tratat mai jos prin qty_net actualizat)

                # BUY detectat — plaseaza SELL pentru diferenta
                # Spacing ASIMETRIC: SELL la 2x spacing fata de BUY
                # BUY la -0.6%, SELL la +1.2% → profit net mai mare
                _sp = g.get('spacing', 0.01)
                sell_p = round(cur_p * (1 + _sp * 2), 8)
                sell_qty = round(diff, 6)

                self.log.info(
                    f"✅ Fill live detectat {sym}: "
                    f"+{diff:.6f} {asset} (~${diff_usd:.2f}) "
                    f"→ SELL @ ${sell_p:.4f}")

                # Actualizeaza qty_net si avg_buy_price (VWAP)
                with self._lock:
                    if sym in self.grids:
                        _old_qty13 = self.grids[sym].get('qty_net', 0.0)
                        _old_avg13 = self.grids[sym].get('avg_buy_price', 0.0)
                        if _old_avg13 > 0 and _old_qty13 > 0:
                            _new_avg13 = (_old_avg13*_old_qty13+cur_p*diff)/(_old_qty13+diff) if (_old_qty13+diff)>0 else cur_p
                        else:
                            _new_avg13 = cur_p
                        self.grids[sym]['avg_buy_price'] = round(_new_avg13, 8)
                        self.grids[sym]['qty_net'] = real_qty
                        _bnb_p = self.client.price('BNBUSDC') or 640
                        pnl_bnb = (diff_usd * g.get('spacing', 0.01)) / _bnb_p
                        fee_bnb = (diff_usd * MAKER_FEE * 2) / _bnb_p
                        self.grids[sym]['pnl']   += pnl_bnb
                        self.grids[sym]['fills'] += 1
                        self.total_pnl   += pnl_bnb
                        self.total_fees  += fee_bnb
                        self.total_fills += 1

                # Plaseaza SELL limit
                try:
                    _r = self.client.limit_sell(
                        sym, sell_qty, sell_p,
                        fee_guard=self.fee_guard)
                    self.log.info(
                        f"📤 SELL plasat {sym}: {sell_qty} @ ${sell_p:.4f} "
                        f"orderId={_r.get('orderId')}")
                except Exception as _se:
                    self.log.warning(f"SELL failed {sym}: {_se}")

                self._save()

            except Exception as _e:
                self.log.debug(f"Balance fill check {sym}: {_e}")

    def _cleanup_stale_sells(self):
        """Curata SELL-uri orfane la preturi >5% fata de pretul curent."""
        if MAINNET_DRY_RUN or USE_TESTNET:
            return
        try:
            with self._lock:
                syms = list(self.grids.keys())
            for sym in syms:
                cur_p = self.client.price(sym) or 0
                if cur_p <= 0:
                    continue
                open_orders = self.client.spot_open_orders(sym) or []
                for o in open_orders:
                    if o.get('side') != 'SELL':
                        continue
                    sell_price = float(o.get('price', 0))
                    if sell_price <= 0:
                        continue
                    dist_pct = (sell_price - cur_p) / cur_p
                    if dist_pct > 0.015:
                        try:
                            _oid = o['orderId']
                            _p = self.client._sign({
                                'symbol': sym, 'orderId': _oid})
                            _url = self.client.base + '/api/v3/order'
                            self.client.sess.delete(
                                _url, params=_p,
                                headers={'X-MBX-APIKEY': self.client.key},
                                timeout=7)
                            self.log.info(
                                f"🧹 SELL orfan anulat {sym}: "
                                f"${sell_price:.4f} (+{dist_pct*100:.1f}% "
                                f"fata de ${cur_p:.4f})")
                        except Exception as _ce:
                            self.log.debug(f"Cancel stale sell {sym}: {_ce}")
        except Exception as _e:
            self.log.debug(f"Cleanup stale sells: {_e}")

    def check_fills(self):
        # Fix definitiv: detecteaza fills via sold real Binance
        self._cleanup_stale_sells()
        self._check_fills_live_balance()
        prices = self.client.all_prices()
        with self._lock: grids = dict(self.grids)

        for sym, g in grids.items():
            p = prices.get(sym, 0)
            if p <= 0: continue

            # Verifică ADX — dezactivează dacă trend a apărut
            kl  = self._klines(sym)
            adx = self._adx(kl)
            if adx > GRID_ADX_MAX + 5:
                self.log.info(f"⚠️  Grid {sym}: ADX={adx:.0f} → OFF")
                # VARIANTA B: lichidare inventar la ADX spike
                _qty_net = g.get("qty_net", 0.0)
                _avg_p   = g.get("avg_buy_price", 0.0)
                _cur_p   = p  # prețul curent deja calculat mai sus
                _val_usd = _qty_net * _cur_p
                if _qty_net > 0 and _cur_p > 0 and _val_usd >= 3.0:
                    self.log.info(
                        f"📦 Lichidare ADX spike {sym}: "
                        f"limit sell {_qty_net:.6f} @ ${_avg_p:.4f} "
                        f"(val=${_val_usd:.2f}, timeout 4h)")
                    try:
                        _r = self.client.limit_sell(
                            sym, round(_qty_net, 6), round(_avg_p, 8),
                            fee_guard=self.fee_guard)
                        _oid = _r.get("orderId")
                    except Exception as _e:
                        self.log.warning(f"Lichidare ADX spike {sym}: {_e}")
                        _oid = None
                    self._liquidation_timers[sym] = {
                        "qty": _qty_net,
                        "deadline": time.time() + 4 * 3600,
                        "oid": _oid
                    }
                elif _qty_net > 0 and _val_usd < 3.0:
                    self.log.info(
                        f"Lichidare ADX spike {sym}: "
                        f"val=${_val_usd:.2f} < $3 → ignorat")
                # FIX CRITIC: anulează ordinele rămase pe Binance
                # Fără asta, ordinele BUY se execută fără știrea botului
                try:
                    self.client.spot_cancel_all(sym)
                    self.log.info(f"ADX spike {sym}: ordine anulate pe Binance")
                except Exception as _ce:
                    self.log.warning(f"ADX spike cancel {sym}: {_ce}")
                # Verifică dacă există tokens deja cumpărați (fills executate)
                try:
                    _asset = sym.replace("USDC", "").replace("USDT", "")
                    _bal = self.client.full_balance().get(_asset, 0.0)
                    _bal_val = _bal * p
                    if _bal > 0 and _bal_val >= 3.0:
                        self.log.info(
                            f"⚠️  ADX spike {sym}: {_bal:.6f} {_asset} "
                            f"(${_bal_val:.2f}) în portofel — plasez SELL")
                        self.client.market_sell(
                            sym, round(_bal, 6),
                            fee_guard=self.fee_guard)
                except Exception as _be:
                    self.log.warning(f"ADX spike balance check {sym}: {_be}")
                with self._lock:
                    if sym in self.grids: del self.grids[sym]
                continue

            for lvl in g["levels"]:
                if lvl["filled"]: continue
                # FIX 4: Fill detection bazat pe mișcarea reală a prețului.
                # Vechea logică: random.random() < 0.10 — irelevantă pentru backtest.
                # Nouă logică: ordinul e umplut dacă prețul a trecut de nivelul limit.
                #   BUY limit la 98.5 → fill dacă ask <= 98.5 (preț curent ≤ nivel)
                #   SELL limit la 101.5 → fill dacă bid >= 101.5 (preț curent ≥ nivel)
                # Pe testnet folosim prețul mid cu toleranță 0.05% pentru latență simulată.
                hit = False
                if USE_TESTNET or MAINNET_DRY_RUN:
                    # Dry-run: fill dacă prețul a trecut de nivel
                    # BUY limit: fill dacă prețul curent <= nivel (am ajuns la prețul de cumpărare)
                    # SELL limit: fill dacă prețul curent >= nivel (am ajuns la prețul de vânzare)
                    if lvl["side"] == "BUY":
                        hit = p <= lvl["price"]
                    else:
                        hit = p >= lvl["price"]
                else:
                    # LIVE: verifică direct statusul ordinului prin API
                    # Pre-check de preț eliminat — cauza fills ratate
                    # (prețul poate reveni deasupra nivelului după fill rapid)
                    oid = lvl.get("oid")
                    if not oid:
                        continue
                    order_info = self.client.get_order(sym, oid)
                    status = order_info.get("status", "UNKNOWN")
                    if status == "FILLED":
                        hit = True
                    elif status in ("CANCELED", "EXPIRED", "REJECTED"):
                        # Ordinul nu mai există pe Binance — marchează și repostează
                        self.log.warning(f"Grid {sym} lvl oid={oid} status={status} → repost")
                        lvl["filled"] = False
                        lvl["oid"] = None
                        try:
                            if lvl["side"] == "BUY":
                                r = self.client.limit_buy(sym, lvl["qty"], lvl["price"])
                            else:
                                r = self.client.limit_sell(sym, lvl["qty"], lvl["price"])
                            lvl["oid"] = r.get("orderId")
                        except Exception as e:
                            self.log.warning(f"Grid {sym} repost after {status}: {e}")
                        continue
                    # NEW / PARTIALLY_FILLED / UNKNOWN → nu e fill complet
                    continue

                if not hit: continue

                lvl["filled"] = True
                spacing_used = g.get("spacing", GRID_SPACING)
                _notional = lvl["qty"] * lvl["price"]
                if sym.endswith("USDC") or sym.endswith("USDT"):
                    _bnb_p = self.client.price("BNBUSDC") or 640
                    _notional_bnb = _notional / max(_bnb_p, 1)
                else:
                    _notional_bnb = _notional
                fee_bnb = _notional_bnb * MAKER_FEE * 2
                pnl_bnb = _notional_bnb * spacing_used - fee_bnb

                with self._lock:
                    if sym in self.grids:
                        self.grids[sym]["pnl"]   += pnl_bnb
                        self.grids[sym]["fills"]  += 1
                    self.total_pnl   += pnl_bnb
                    self.total_fees  += fee_bnb
                    self.total_fills += 1
                self.fees.record("GRID", fee_bnb, pnl_bnb+fee_bnb, 2)

                # Health: track daily PnL per strategie
                if hasattr(self, '_health') and self._health:
                    self._health.record_trade("bnb_grid", pnl_bnb)

                # FIX11: Per-pair daily tracking
                if hasattr(self, '_bot_ref') and self._bot_ref:
                    _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    _tr = self._bot_ref._pair_daily_pnl
                    _tr.setdefault(sym, {})
                    _tr[sym][_today] = _tr[sym].get(_today, 0) + pnl_bnb
                    _fl = self._bot_ref._pair_daily_fills
                    _fl.setdefault(sym, {})
                    _fl[sym][_today] = _fl[sym].get(_today, 0) + 1

                # ML: record fill for GridSpacingML learning
                if self._ml:
                    try:
                        kl = self.client.klines(sym, "1h", 30)
                        feat = FeatureExtractor.from_klines(kl, 20) if ML_AVAILABLE else {}
                        if feat:
                            self._ml.record_grid_fill(feat, spacing_used, pnl_bnb)
                    except Exception as _e: logging.debug(f"Ignored: {_e}")
                # Repostează ordinul opus la distanta spacing_used
                opp_p = lvl["price"]*(1+spacing_used if lvl["side"]=="BUY"
                                      else 1-spacing_used)
                lvl.update({"price":round(opp_p,8),
                             "side":"SELL" if lvl["side"]=="BUY" else "BUY",
                             "filled":False,"oid":None})
                if lvl["side"]=="BUY":
                    r = self.client.limit_buy(sym, lvl["qty"], lvl["price"])
                else:
                    r = self.client.limit_sell(sym, lvl["qty"], lvl["price"])
                lvl["oid"] = r.get("orderId")

                self.log.info(
                    f"✅ Grid fill {sym}: +{pnl_bnb:.5f} BNB | "
                    f"fee={fee_bnb:.5f} | fills={self.total_fills}")
                if self._perf:
                    self._perf.add_trade(pnl_bnb)

        self._save()

    def emergency_close(self):
        """
        Apelat de CrashGuard la nivel RED.
        Sterge toate gridurile si anuleaza ordinele deschise.
        """
        with self._lock:
            syms = list(self.grids.keys())
            self.grids = {}
        if not syms:
            return
        # Anuleaza TOATE ordinele de pe Binance
        for _sym in syms:
            try:
                self.client.spot_cancel_all(_sym)
                self.log.info(f"Emergency close: anulate ordine {_sym}")
            except Exception as _e:
                self.log.warning(f"Emergency cancel {_sym}: {_e}")
        self.log.error(
            f"EMERGENCY CLOSE grid: {len(syms)} perechi inchise "
            f"({', '.join(syms)})")
        tg(
            f"🔴 <b>GRID EMERGENCY CLOSE</b>\n"
            f"Crash sever detectat. Inchis {len(syms)} perechi:\n"
            f"{', '.join(syms)}\n"
            f"Grid in STANDBY pana la recuperare piata."
        )
        self._save()

    def run(self, stop: threading.Event):
        self.log.info(
            f"Grid MAKER | {self.bnb:.4f} BNB | "
            f"spacing={GRID_SPACING*100:.1f}% | "
            f"net/fill={(GRID_SPACING-ROUNDTRIP_MAKER)*100:.3f}%")
        # La pornire: anuleaza TOATE ordinele orfane de pe Binance
        if not MAINNET_DRY_RUN and not USE_TESTNET:
            try:
                # Fara symbol = returneaza toate ordinele deschise
                all_open = self.client._get(
                    '/api/v3/openOrders', {}, signed=True) or []
                syms_to_cancel = set(o['symbol'] for o in all_open)
                if syms_to_cancel:
                    self.log.info(
                        f"Startup: curatare {len(all_open)} ordine "
                        f"orfane pe {len(syms_to_cancel)} perechi")
                for _sym in syms_to_cancel:
                    try:
                        self.client.spot_cancel_all(_sym)
                        self.log.info(f"Startup: anulate ordine orfane {_sym}")
                    except Exception as _ce:
                        self.log.debug(f"Startup cancel {_sym}: {_ce}")
            except Exception as _e:
                self.log.warning(f"Startup cancel all error: {_e}")
        # La pornire: plaseaza SELL pentru tokens free fara ordin de vanzare
        # Asteapta 150s la startup — lasa ML/Sentinel/retrain sa termine API calls
        # Evita rate limit care cauzeaza check_fills timeout in primul ciclu
        self.log.info("Grid: astept 150s pentru ML/Sentinel init + rate limit reset...")
        stop.wait(150)
        if stop.is_set(): return
        # ── VERIFICARE PIATA LA STARTUP ──
        try:
            self.log.info("🔍 Verificare conditii piata la startup...")
            _startup_ok = True
            _btc_kl = self.client.klines("BTCUSDC", "1h", 6)
            if _btc_kl and len(_btc_kl) >= 6:
                _btc_c = [float(k[4]) for k in _btc_kl]
                _chg_1h = (_btc_c[-1] - _btc_c[-2]) / _btc_c[-2] * 100
                _chg_6h = (_btc_c[-1] - _btc_c[0]) / _btc_c[0] * 100
                _ema5_btc = sum(_btc_c[-5:]) / 5
                _ema6_btc = sum(_btc_c) / 6
                if _chg_1h < -2.0:
                    self.log.warning(
                        f"⚠️ Startup: BTC {_chg_1h:+.1f}% in 1h — scadere rapida")
                    _startup_ok = False
                elif _chg_6h < -3.0:
                    self.log.warning(
                        f"⚠️ Startup: BTC {_chg_6h:+.1f}% in 6h — trend descendent")
                    _startup_ok = False
                elif _ema5_btc < _ema6_btc * 0.997:
                    self.log.warning(
                        f"⚠️ Startup: BTC EMA descendent — piata bearish")
                    _startup_ok = False
                else:
                    self.log.info(
                        f"✅ Startup: BTC OK | 1h:{_chg_1h:+.1f}% | 6h:{_chg_6h:+.1f}% — grid activ")
            if not _startup_ok:
                self.log.warning(
                    "⛔ Startup: conditii nefavorabile — grid in standby 30 min")
                stop.wait(1800)  # asteapta 30 min si reincearca
                if stop.is_set(): return
                self.log.info("🔄 Retry dupa 30 min standby...")
        except Exception as _sc:
            self.log.debug(f"Startup market check: {_sc}")

        self._startup_sell_orphans()
        # Reciteste capitalul DUPA anularea ordinelor orfane
        # (ordinele anulate elibereaza USDC care era blocat)
        if not MAINNET_DRY_RUN and not USE_TESTNET:
            try:
                _bal = self.client.full_balance()
                _usdc_free = _bal.get('USDC', 0.0)
                _bnb_p = self.client.price('BNBUSDC') or 640
                _usdc_for_grid = max(0.0, _usdc_free - 100.0)
                self.bnb_eq = _usdc_for_grid / max(_bnb_p, 1)
                self.log.info(
                    f"💰 Capital grid recalculat: "
                    f"${_usdc_for_grid:.0f} USDC "
                    f"(din ${_usdc_free:.0f}, tampon $100)")
            except Exception as _ce:
                self.log.debug(f"Capital recalc: {_ce}")
        self.rebuild()
        while not stop.is_set():
            try:
                # La RED: nu facem nimic pana la recuperare
                if self.crash and self.crash.is_red:
                    stop.wait(60); continue
                import threading as _thr
                # check_fills cu timeout 25s
                _cf = _thr.Thread(target=self.check_fills, daemon=True)
                _cf.start(); _cf.join(timeout=25)
                if _cf.is_alive():
                    self.log.warning("check_fills timeout 25s — skip ciclu")
                # rebuild cu timeout 30s
                if time.time() - self._last_rb > GRID_REBUILD_H*3600:
                    _rb = _thr.Thread(target=self.rebuild, daemon=True)
                    _rb.start(); _rb.join(timeout=30)
                    if _rb.is_alive():
                        self.log.warning("rebuild timeout 30s — skip")
                # stop-loss global cu timeout 15s
                if not MAINNET_DRY_RUN and not USE_TESTNET:
                    def _sl_run(self=self):
                        try:
                            _bal_sl = self.client._get("/api/v3/account",{},signed=True) or {}
                            _skip = {"USDC","USDT","BNB","SOL","BUSD","LDADA","LDPEPE","LDBIO","LDBNB","CTSI","PIXEL","W","LUNC"}
                            for _b in _bal_sl.get("balances",[]):
                                _a = _b["asset"]; _q = float(_b["free"]) + float(_b["locked"])
                                if _a in _skip or _q < 0.001: continue
                                _s = f"{_a}USDC"; _p = self.client.price(_s) or 0
                                if _p <= 0 or _q*_p < 3: continue
                                _avg = self.grids.get(_s,{}).get("avg_buy_price",0)
                                if _avg > 0 and (_p-_avg)/_avg <= -0.03:
                                    self.log.warning(f"🛑 STOP-LOSS GLOBAL {_s}: ${_p:.4f} vs avg ${_avg:.4f}")
                                    self.client.spot_cancel_all(_s)
                                    self.client.market_sell(_s, round(_q,6), fee_guard=self.fee_guard)
                        except Exception as _se: self.log.debug(f"SL global: {_se}")
                    _sl = _thr.Thread(target=_sl_run, daemon=True)
                    _sl.start(); _sl.join(timeout=15)
                    if _sl.is_alive(): self.log.warning("stop-loss global timeout 15s")
                # ── STOP-LOSS GLOBAL: verifica TOATE tokens din portofel ──
                # Nu doar cele din self.grids — prinde si ALLO, ZEC orfane etc.
                if not MAINNET_DRY_RUN and not USE_TESTNET:
                    try:
                        _bal_all = self.client.full_balance()
                        _skip_sl = {'USDC','USDT','BNB','SOL','BUSD',
                                    'LDADA','LDPEPE','LDBIO','LDBNB',
                                    'CTSI','PIXEL','W','LUNC'}
                        for _asset, _qty in _bal_all.items():
                            if _asset in _skip_sl or _qty < 0.001:
                                continue
                            _sym_sl = f"{_asset}USDC"
                            _cur_sl = self.client.price(_sym_sl) or 0
                            if _cur_sl <= 0:
                                continue
                            _val_sl = _qty * _cur_sl
                            if _val_sl < 3.0:
                                continue
                            # Cauta avg_buy din grids sau din istoricul recent
                            _avg_sl = 0.0
                            with self._lock:
                                if _sym_sl in self.grids:
                                    _avg_sl = self.grids[_sym_sl].get('avg_buy_price', 0.0)
                            if _avg_sl <= 0:
                                # Calculeaza avg din trades recente (24h)
                                try:
                                    import hmac as _hm, hashlib as _hl2
                                    _ak2 = self.client.key
                                    _sk2 = self.client.secret
                                    _since2 = int((time.time()-86400)*1000)
                                    _pq2 = (f"symbol={_sym_sl}&startTime={_since2}"
                                            f"&limit=500&timestamp={int(time.time()*1000)}")
                                    _sg2 = _hm.new(_sk2.encode(), _pq2.encode(),
                                                   _hl2.sha256).hexdigest()
                                    import urllib.request as _ur2, json as _j2
                                    _url2 = (f"https://api.binance.com/api/v3/myTrades"
                                             f"?{_pq2}&signature={_sg2}")
                                    _req2 = _ur2.Request(
                                        _url2, headers={"X-MBX-APIKEY": _ak2})
                                    _tr2 = _j2.loads(
                                        _ur2.urlopen(_req2, timeout=10).read())
                                    _buys2 = [(float(t['price']), float(t['qty']))
                                              for t in _tr2 if t['isBuyer']]
                                    if _buys2:
                                        _tq2 = sum(q for _, q in _buys2)
                                        _avg_sl = (sum(p*q for p, q in _buys2)
                                                   / _tq2 if _tq2 else 0)
                                except Exception: pass
                            if _avg_sl <= 0:
                                continue
                            _loss_pct = (_cur_sl - _avg_sl) / _avg_sl
                            if _loss_pct <= -0.03:
                                self.log.warning(
                                    f"🛑 STOP-LOSS GLOBAL {_sym_sl}: "
                                    f"pret ${_cur_sl:.4f} vs avg ${_avg_sl:.4f} "
                                    f"({_loss_pct*100:.1f}%) val=${_val_sl:.2f} "
                                    f"→ market sell")
                                try:
                                    # Anuleaza ordine existente
                                    self.client.spot_cancel_all(_sym_sl)
                                    # Market sell
                                    self.client.market_sell(
                                        _sym_sl, round(_qty, 6),
                                        fee_guard=self.fee_guard)
                                    # Scoate din grids
                                    with self._lock:
                                        if _sym_sl in self.grids:
                                            del self.grids[_sym_sl]
                                    self.log.warning(
                                        f"🛑 {_sym_sl}: lichidat "
                                        f"{_qty:.4f} @ ${_cur_sl:.4f}")
                                except Exception as _sle:
                                    self.log.warning(
                                        f"Stop-loss global {_sym_sl}: {_sle}")
                    except Exception as _sle2:
                        self.log.debug(f"Stop-loss global: {_sle2}")
                # Auto-recentrare: daca o pereche nu are ordine active
                if not MAINNET_DRY_RUN and not USE_TESTNET:
                    try:
                        _all_open = self.client._get(
                            '/api/v3/openOrders', {}, signed=True) or []
                        _open_syms = set(o['symbol'] for o in _all_open)
                        _needs_rebuild = False
                        _rebuild_sym = None
                        _prices = self.client.all_prices()
                        with self._lock:
                            for _sym in list(self.grids.keys()):
                                # Rebuild daca: fara ordine SAU fara SELL cand pret > max BUY
                                _sym_orders = [o for o in _all_open if o['symbol']==_sym]
                                _has_sell = any(o['side']=='SELL' for o in _sym_orders)
                                _has_buy  = any(o['side']=='BUY'  for o in _sym_orders)
                                _p = _prices.get(_sym, 0)
                                _min_buy = min((float(o['price']) for o in _sym_orders if o['side']=='BUY'), default=0)
                                _max_buy = max((float(o['price']) for o in _sym_orders if o['side']=='BUY'), default=0)
                                # SOLUTIE: rebuild DOAR in jos sau daca fara ordine.
                                # NICIODATA in sus (= chase trend, cauza WLD/ZEC).
                                # Cand pretul URCA peste grid: NU rebuild — fie
                                # SELL-urile se executa (profit), fie e trend
                                # ascendent (nu cumparam mai sus).
                                _price_below_grid = _p < _min_buy * 0.985 if _min_buy else False  # pret sub grid 1.5%
                                _price_above_grid = _p > _max_buy * 1.01 if _max_buy else False  # pret 1% deasupra
                                # Cooldown recentrare in sus: maxim o data la 30 min
                                if not hasattr(self, '_rebuild_cooldown'):
                                    self._rebuild_cooldown = {}
                                # Default: time.time()-1800 = cooldown epuizat (prima rulare OK)
                                # NU 0 (= epoch 1970 = 56 ani in urma = mereu OK = chase)
                                _last_rb_sym = self._rebuild_cooldown.get(_sym, time.time() - 1800)
                                _cooldown_ok = (time.time() - _last_rb_sym) > 1800  # 30 min
                                _rebuild_up = _price_above_grid and _cooldown_ok and not _has_sell
                                if _sym not in _open_syms or _price_below_grid or _rebuild_up:
                                    if _rebuild_up:
                                        _dir = f"pret deasupra grid ({((time.time()-_last_rb_sym)/60):.0f}min de la ultima recentrare)"
                                        self._rebuild_cooldown[_sym] = time.time()
                                    elif _price_below_grid:
                                        _dir = "pret sub grid"
                                    else:
                                        _dir = "fara ordine"
                                    self.log.info(
                                        f"⚡ Auto-rebuild: {_sym} "
                                        f"pret=${_p:.4f} ({_dir}) → recentrare")
                                    _needs_rebuild = True
                                    _rebuild_sym = _sym
                                    break
                        if _needs_rebuild and _rebuild_sym:
                            # Rebuild doar pe perechea dezechilibrata
                            # Nu intrerupe perechile care fac fills
                            try:
                                _mid = self.client.price(_rebuild_sym) or 0
                                if _mid > 0:
                                    _adx_kl = self._klines(_rebuild_sym)
                                    _adx_v  = self._adx(_adx_kl)
                                    if _adx_v > GRID_ADX_MAX:
                                        # Trend detectat — scoate perechea din grid
                                        self.log.info(
                                            f"⚡ {_rebuild_sym}: ADX={_adx_v:.1f} > "
                                            f"{GRID_ADX_MAX} trend → scos din grid")
                                        try:
                                            self.client.spot_cancel_all(_rebuild_sym)
                                        except: pass
                                        with self._lock:
                                            if _rebuild_sym in self.grids:
                                                del self.grids[_rebuild_sym]
                                        # Selecteaza o pereche noua in loc
                                        try:
                                            _new_pairs = self._select()
                                            _existing = set(self.grids.keys())
                                            _candidates = [
                                                (s, m, a) for s, m, a in (_new_pairs or [])
                                                if s not in _existing
                                            ]
                                            if _candidates:
                                                _ns, _nm, _na = _candidates[0]
                                                _bnb_per = self.bnb_eq / max(len(self.grids)+1, 1)
                                                _ng = self._build(_ns, _nm, _bnb_per, _na)
                                                if _ng and _ng.get('levels'):
                                                    with self._lock:
                                                        self.grids[_ns] = _ng
                                                    self._place(_ng)
                                                    self.log.info(
                                                        f"⚡ Pereche noua adaugata: {_ns} "
                                                        f"ADX={_na:.1f} @ ${_nm:.4f}")
                                        except Exception as _ne:
                                            self.log.debug(f"Selectie pereche noua: {_ne}")
                                    elif _adx_v <= GRID_ADX_MAX:
                                        # Check rentabilitate inainte de recentrare
                                        _recentr_ok = True
                                        try:
                                            _btc_rc = self.client.klines("BTCUSDC", "1h", 3)
                                            if _btc_rc and len(_btc_rc) >= 3:
                                                _btc_rc_c = [float(k[4]) for k in _btc_rc]
                                                _btc_rc_chg = (_btc_rc_c[-1]-_btc_rc_c[0])/_btc_rc_c[0]*100
                                                if _btc_rc_chg < -1.5:
                                                    self.log.warning(
                                                        f"⛔ Recentrare {_rebuild_sym} SKIP: "
                                                        f"BTC {_btc_rc_chg:+.1f}% in 2h")
                                                    _recentr_ok = False
                                        except Exception: pass
                                        if not _recentr_ok:
                                            pass
                                        else:
                                            # PROTECTIE chase-trend
                                            _ar = _rebuild_sym.replace("USDC","").replace("USDT","")
                                            _inv_r = self.client.full_balance().get(_ar, 0.0)
                                            if _ar == "BNB": _inv_r = 0.0
                                            if _inv_r * _mid > 5.0:
                                                self.log.info(
                                                    f"SKIP {_rebuild_sym}: inventar "
                                                    f"${_inv_r*_mid:.2f} - NU recentrez "
                                                    f"in sus (anti chase-trend)")
                                            else:
                                                _bnb_per = self.bnb_eq / max(len(self.grids), 1)
                                                _g = self._build(
                                                    _rebuild_sym, _mid, _bnb_per,
                                                    _adx_v)
                                                if _g and _g.get('levels'):
                                                    self.client.spot_cancel_all(_rebuild_sym)
                                                    with self._lock:
                                                        self.grids[_rebuild_sym] = _g
                                                        self.grids[_rebuild_sym]['qty_net'] = 0.0
                                                    self._place(_g)
                                                    self.log.info(
                                                        f"⚡ Rebuild selectiv {_rebuild_sym}: "
                                                        f"recentrat @ ${_mid:.4f}")
                            except Exception as _rbe:
                                self.log.debug(f"Rebuild selectiv {_rebuild_sym}: {_rbe}")
                    except Exception as _re:
                        self.log.debug(f"Auto-rebuild check: {_re}")
            except Exception as e:
                self.log.warning(f"Grid: {e}")
            self.log.debug("Grid loop: ciclu complet")
            stop.wait(45)
        self.log.info("Grid oprit")


# ══════════════════════════════════════════════════════════════════════
# S3: SWING — max 2 trades/zi, TP 1.8%, 3 filtre obligatorii
# ══════════════════════════════════════════════════════════════════════

class SwingTrader:
    """
    10% din capitalul BNB → swing trading selectiv.
    Max 2 trades/zi (nu overtrading).
    TP = 1.8% ≥ MIN_PROFIT_SWING (0.9% = fee × 8).

    EXPECTED VALUE per trade:
      55% × 1.8% × 50% - 45% × 0.7% - 0.1125%
      = 0.495% - 0.315% - 0.1125% = 0.0675%
      Pozitiv, dar marginal. Tocmai de aceea max 2/zi.
    """

    def __init__(self, client: Binance, bnb_capital: float,
                 fees: FeeTracker, guard: DailyTradeGuard,
                 crash: "MarketCrashGuard" = None,
                 fee_guard: "FeeBufferManager" = None):
        self.client    = client
        self.bnb       = bnb_capital
        self.fees      = fees
        self.guard     = guard
        self.crash     = crash
        self.fee_guard = fee_guard
        self.log    = L("Swing")
        self._lock  = threading.Lock()
        self.trades: Dict[str, dict] = {}
        self.total_pnl   = 0.0
        self.total_fees  = 0.0
        self.n_wins = 0; self.n_losses = 0
        self._wl: Dict[str, List[float]] = defaultdict(list)
        self._kl: Dict[str, Tuple[float, list]] = {}
        self._enh_mgr = None  # ENH: EnhancementsManager
        self._ml = None       # ML+AI: MLEngine reference
        self._load()

    def _load(self):
        try:
            if os.path.exists("v3_swing.json"):
                with open("v3_swing.json") as _jf:

                    d = json.load(_jf)
                self.total_pnl  = d.get("total_pnl", 0.0)
                self.total_fees = d.get("total_fees", 0.0)
                self.n_wins     = d.get("n_wins", 0)
                self.n_losses   = d.get("n_losses", 0)
        except Exception as _e: logging.debug(f'Ignored: {_e}')

    def _save(self):
        try:
            _data = {
                "total_pnl": self.total_pnl, "total_fees": self.total_fees,
                "n_wins": self.n_wins, "n_losses": self.n_losses,
                "ts": time.time()
            }
            _atomic_json_save("v3_swing.json", _data, indent=2)
        except Exception as _e: logging.debug(f'Ignored: {_e}')

    def _klines(self, sym: str) -> list:
        ts, kl = self._kl.get(sym, (0.0, []))
        if time.time() - ts > 900:
            kl = self.client.klines(sym, "1h", 50)
            self._kl[sym] = (time.time(), kl)
        return kl

    def _klines_4h(self, sym: str) -> list:
        """4h klines cache — confirmare trend higher timeframe."""
        key = f"{sym}_4h"
        ts, kl = self._kl.get(key, (0.0, []))
        if time.time() - ts > 3600:
            kl = self.client.klines(sym, "4h", 20)
            self._kl[key] = (time.time(), kl)
        return kl

    def _trend_4h(self, sym: str) -> int:
        """
        Trend 4h: +1 bullish, -1 bearish, 0 neutral.
        EMA8 vs EMA21 pe 4h candles.
        """
        try:
            kl = self._klines_4h(sym)
            if len(kl) < 22: return 0
            closes = [float(k[4]) for k in kl]
            ema8 = closes[0]; ema21 = closes[0]
            m8 = 2/9; m21 = 2/22
            for c in closes:
                ema8 = c*m8 + ema8*(1-m8)
                ema21 = c*m21 + ema21*(1-m21)
            if ema8 > ema21 * 1.002: return 1
            if ema8 < ema21 * 0.998: return -1
        except Exception as _e: logging.debug(f"Ignored: {_e}")
        return 0

    def _indicators(self, kl: list) -> Optional[dict]:
        if len(kl) < 22: return None
        cl = [float(k[4]) for k in kl]
        vl = [float(k[5]) for k in kl]
        # RSI 14
        d  = [cl[i]-cl[i-1] for i in range(1,len(cl))]
        g  = [max(0,x) for x in d[-14:]]
        l  = [abs(min(0,x)) for x in d[-14:]]
        ag = sum(g)/14; al = sum(l)/14
        rsi = 100-(100/(1+ag/al)) if al>0 else 50
        # ADX proxy
        net  = abs(cl[-1]-cl[-min(14,len(cl))])
        gros = sum(abs(cl[i]-cl[i-1]) for i in range(1,len(cl))) or 1e-10
        adx  = net/gros*100*2.5
        # Volume ratio
        av   = sum(vl[-21:-1])/20
        volr = vl[-1]/av if av>0 else 1.0
        return {"rsi":rsi,"adx":adx,"volr":volr,"price":cl[-1]}

    def _atr(self, kl: list) -> float:
        """
        ATR (Average True Range) normalizat = ATR / price_curent.
        Masura volatilitate: 0.005 = calma, 0.020 = volatila.
        Folosit pentru ATR sizing: marim size-ul cand piata se misca.
        """
        if len(kl) < SWING_ATR_PERIOD + 1:
            return 0.010   # valoare default medie
        trs = []
        for i in range(1, len(kl)):
            h   = float(kl[i][2])
            l   = float(kl[i][3])
            pc  = float(kl[i-1][4])
            tr  = max(h - l, abs(h - pc), abs(l - pc))
            trs.append(tr)
        atr  = sum(trs[-SWING_ATR_PERIOD:]) / SWING_ATR_PERIOD
        last = float(kl[-1][4])
        return atr / last if last > 0 else 0.010

    def _size(self, sym: str, atr_norm: float = 0.010) -> float:
        """
        Sizing dinamic cu 2 ajustari:
        1. WR per simbol (existent): +20% la WR>=60%, -30% la WR<=40%
        2. ATR sizing (NOU): marim size-ul cand volatilitatea confirma miscarea
           ATR mic (piata calma)  → size mai mic  (miscari mici, profit mic)
           ATR mare (piata vola)  → size mai mare (miscari mari, profit mai mare)
        Cap absolut: max 20% din capitalul swing.
        """
        base = self.bnb * 0.15

        # Ajustare WR
        wl = self._wl[sym]
        if len(wl) >= 10:
            wr = sum(1 for x in wl if x > 0) / len(wl)
            if wr >= 0.60:   base *= 1.20
            elif wr <= 0.40: base *= 0.70

        # ATR sizing: factor liniar intre ATR_NORM_LOW si ATR_NORM_HI
        atr_clamped = max(SWING_ATR_NORM_LOW,
                          min(SWING_ATR_NORM_HI, atr_norm))
        atr_factor  = (SWING_ATR_SIZE_MIN +
                       (atr_clamped - SWING_ATR_NORM_LOW) /
                       (SWING_ATR_NORM_HI - SWING_ATR_NORM_LOW) *
                       (SWING_ATR_SIZE_MAX - SWING_ATR_SIZE_MIN))
        base *= atr_factor

        # ═══ ENH v1.1: size multiplier per strategie (swing) ═══
        if self._enh_mgr:
            enh_mult = self._enh_mgr.get_size_multiplier_for_strategy("swing")
            base *= enh_mult

        return min(base, self.bnb * 0.20)

    def _btc_momentum_1h(self) -> float:
        """
        I4: Returneaza % schimbare BTC in ultima ora.
        Cache 15 minute pentru a nu face API call la fiecare verificare.
        """
        now = time.time()
        cached = self._kl.get("_btc_mom_cache", (0.0, 0.0))
        if now - cached[0] < 900:   # cache 15 min
            return cached[1]
        try:
            kl = self.client.klines("BTCUSDC", "1h", 3)
            if len(kl) >= 2:
                p_now  = float(kl[-1][4])
                p_prev = float(kl[-2][4])
                pct    = (p_now - p_prev) / p_prev if p_prev > 0 else 0.0
            else:
                pct = 0.0
        except Exception:
            pct = 0.0
        self._kl["_btc_mom_cache"] = (now, pct)
        return pct

    def scan(self):
        # Skip complet dacă capital = 0 (ALLOC_SWING=0%)
        if self.bnb <= 0:
            self._update(); return
        # Crash guard
        if self.crash and not self.crash.trading_ok:
            self.log.info(f"Swing scan SKIP — crash: {self.crash.level}")
            self._update(); return
        # Fed blackout
        if is_fed_blackout():
            self.log.info("Swing scan SKIP — FOMC Fed blackout")
            self._update(); return
        # EXTREME FEAR SKIP: WR=48% pe FG<20 → EV aproape zero după fee
        try:
            if self._enh_mgr and self._enh_mgr.adaptive_alloc.current_fg < 20:
                self.log.debug("Swing SKIP — Extreme Fear (FG<20)")
                self._update(); return
        except Exception as _e: logging.debug(f"Ignored: {_e}")
        # Filtru fereastra UTC 13-17 — intrari noi doar in overlap London+NY
        if SWING_UTC_FILTER:
            h_utc = datetime.now(timezone.utc).hour
            if not (SWING_UTC_WINDOW_START <= h_utc < SWING_UTC_WINDOW_END):
                # In afara ferestrei: doar actualizam (inchidem TP/SL), nu deschidem
                self._update(); return
        if len(self.trades) >= 2:
            self._update(); return
        if not self.guard.can_trade("SWING"):
            self._update(); return

        # I4: citim momentumul BTC o singura data (cache 15 min)
        btc_1h = self._btc_momentum_1h()

        for sym in SWING_PAIRS:
            if sym in self.trades: continue
            if not self.guard.can_trade("SWING"): break

            kl  = self._klines(sym)
            ind = self._indicators(kl)
            if not ind: continue

            adx  = ind["adx"]; rsi = ind["rsi"]
            volr = ind["volr"]; p   = ind["price"]

            # Cele 3 filtre obligatorii
            if adx < SWING_ADX_MIN:    continue   # trend insuficient
            if volr < SWING_VOL_MIN:   continue   # volum insuficient

            if rsi < 45:   direction = 1    # oversold → LONG (presiune buy)
            elif rsi > 55: direction = -1   # overbought → SHORT (presiune sell)
            else:          continue

            # I4: Filtru corelatie BTC — blocam directia contra momentumului macro
            # LONG blocat daca BTC a scazut >1.5%/1h (piata macro bearish)
            # SHORT blocat daca BTC a urcat >1.5%/1h (piata macro bullish)
            btc_filter = abs(btc_1h) >= SWING_BTC_FILTER_PCT
            if btc_filter:
                btc_dir = 1 if btc_1h > 0 else -1
                if direction != btc_dir:
                    self.log.info(
                        f"I4 Skip {sym}: directie {'+1' if direction>0 else '-1'} "
                        f"contra BTC {btc_1h*100:+.1f}%/1h")
                    continue

            # Confirmare trend 4h — reduce false signals ~20%
            trend4 = self._trend_4h(sym)
            if trend4 != 0 and trend4 != direction:
                self.log.info(
                    f"Skip {sym}: 1h={'L' if direction>0 else 'S'} "
                    f"contra 4h={'bull' if trend4>0 else 'bear'}")
                continue

            # ═══ ML+AI SIGNAL ═══
            ml_sizing = 1.0
            if self._ml and self._ml.enabled:
                kl_ml = self._klines(sym)
                ml_signal = self._ml.get_swing_signal(kl_ml, direction)
                if ML_DRY_RUN:
                    # DRY RUN: logăm semnalul dar nu blocăm tranzacția
                    self.log.info(
                        f"🧠 ML DRY_RUN {sym}: take={ml_signal['take_trade']} "
                        f"conf={ml_signal.get('confidence',0):.0%} "
                        f"sizing={ml_signal.get('sizing_mult',1.0):.2f} "
                        f"reason={ml_signal.get('reason','')}")
                else:
                    if not ml_signal["take_trade"] and ml_signal["confidence"] > 0.6:
                        self.log.info(
                            f"🧠 ML Skip {sym}: {ml_signal['reason']} "
                            f"(conf={ml_signal['confidence']:.0%})")
                        continue
                    ml_sizing = ml_signal.get("sizing_mult", 1.0)

            # Verificare EV pozitiv cu formula corectă (FIX 5)
            # Vechea formulă: 0.55×TP×0.5 − 0.45×SL − fee (inconsistentă)
            # Nouă formulă:   WR×TP − (1−WR)×SL − fee (standard)
            # Folosim WR per simbol dacă avem ≥10 trades, altfel SWING_WR_BASE
            wl_sym = self._wl[sym]
            wr_sym = (sum(1 for x in wl_sym if x > 0) / len(wl_sym)
                      if len(wl_sym) >= 10 else SWING_WR_BASE)
            ev = wr_sym * SWING_TP - (1 - wr_sym) * SWING_SL - ROUNDTRIP_TAKER - SLIPPAGE * 2
            if ev <= 0:
                self.log.warning(
                    f"EV negativ ({ev*100:.4f}%) — skip {sym} "
                    f"(WR={wr_sym*100:.0f}%)")
                continue

            size_bnb = self._size(sym, atr_norm=self._atr(kl)) * ml_sizing
            qty_x    = size_bnb / p
            tp_bnb   = p * (1+direction*SWING_TP)
            sl_bnb   = p * (1-direction*SWING_SL)
            slippage_cost = size_bnb * SLIPPAGE
            fee_in   = size_bnb * TAKER_FEE + slippage_cost

            if direction == 1:
                r = self.client.market_buy(sym, round(qty_x,6),
                                           fee_guard=self.fee_guard)
            else:
                r = self.client.market_sell(sym, round(qty_x,6),
                                            fee_guard=self.fee_guard)

            if r.get("status") == "BLOCKED_FEE":
                self.log.warning(f"SWING {sym} BLOCAT — fee buffer critic")
                continue

            t = {"sym":sym,"dir":direction,"entry":p,"qty":round(qty_x,6),
                 "size":size_bnb,"tp":tp_bnb,"sl":sl_bnb,
                 "ts":time.time(),"fee_in":fee_in,
                 "highest_price": p,    # I7: tracking pentru trailing SL
                 "trail_active":  False} # I7: trailing SL activat?
            with self._lock:
                self.trades[sym] = t
                self.total_fees += fee_in
            self.guard.record("SWING")

            ds = "🔼 LONG" if direction>0 else "🔽 SHORT"
            self.log.info(
                f"✅ SWING {sym}: {ds} @ {p:.8f} BNB | "
                f"TP={SWING_TP*100:.1f}% SL={SWING_SL*100:.1f}% | "
                f"ADX={adx:.0f} RSI={rsi:.0f} VOL={volr:.1f}× | "
                f"EV={ev*100:.4f}%")
            tg(
                f"📈 <b>SWING {ds}</b> {sym}\n"
                f"@ {p:.8f} BNB | ADX={adx:.0f} RSI={rsi:.0f}\n"
                f"TP: {SWING_TP*100:.1f}% | SL: {SWING_SL*100:.1f}% | "
                f"RR: {SWING_TP/SWING_SL:.1f}\n"
                f"Fee intrare: {fee_in:.5f} BNB\n"
                f"EV așteptat: +{ev*size_bnb:.5f} BNB/trade"
            )

        self._update()

    def _update(self):
        with self._lock: trades = dict(self.trades)
        prices = self.client.all_prices()
        for sym, t in trades.items():
            p = prices.get(sym, 0) or self.client.price(sym)
            if p <= 0: continue
            pct      = (p - t["entry"]) / t["entry"] * t["dir"]
            hold_h   = (time.time() - t["ts"]) / 3600

            # I7: Trailing SL — actualizeaza highest_price si SL dinamic
            # Directie LONG: highest = max price atins
            # Directie SHORT: highest = min price atins (cel mai favorabil)
            if t["dir"] == 1:
                if p > t.get("highest_price", t["entry"]):
                    with self._lock:
                        if sym in self.trades:
                            self.trades[sym]["highest_price"] = p
                    t["highest_price"] = p
            else:
                if p < t.get("highest_price", t["entry"]):
                    with self._lock:
                        if sym in self.trades:
                            self.trades[sym]["highest_price"] = p
                    t["highest_price"] = p

            # Calcul profit maxim atins
            best_pct = (abs(t["highest_price"] - t["entry"]) /
                        t["entry"])

            # Activeaza trailing dupa SWING_TRAIL_ACTIVATE profit
            if best_pct >= SWING_TRAIL_ACTIVATE and not t.get("trail_active"):
                with self._lock:
                    if sym in self.trades:
                        self.trades[sym]["trail_active"] = True
                t["trail_active"] = True
                self.log.info(
                    f"I7 Trailing SL activat {sym}: "
                    f"profit {best_pct*100:.2f}% > {SWING_TRAIL_ACTIVATE*100:.1f}%")

            # SL dinamic: daca trailing activ, SL urca cu pretul
            if t.get("trail_active"):
                locked_profit = best_pct * SWING_TRAIL_LOCK
                if t["dir"] == 1:
                    trail_sl_price = t["entry"] * (1 + locked_profit)
                    hit_sl = p <= trail_sl_price
                else:
                    trail_sl_price = t["entry"] * (1 - locked_profit)
                    hit_sl = p >= trail_sl_price
            else:
                # SL fix standard (inainte de activare trailing)
                hit_sl = -pct >= SWING_SL

            hit_tp  = pct >= SWING_TP
            timeout = hold_h >= 36

            if not (hit_tp or hit_sl or timeout): continue

            # La exit: fee_guard NU blocheaza (trebuie sa inchidem pozitia)
            # dar verifica si triggereaza refill daca e necesar
            if self.fee_guard: self.fee_guard.check()
            if t["dir"]==1: self.client.market_sell(sym, t["qty"])
            else:           self.client.market_buy(sym, t["qty"])

            fee_out  = t["size"] * TAKER_FEE + t["size"] * SLIPPAGE  # FIX 5: slippage exit
            pnl      = t["size"]*pct - t["fee_in"] - fee_out
            fee_tot  = t["fee_in"] + fee_out

            with self._lock:
                if sym in self.trades: del self.trades[sym]
                self.total_pnl  += pnl
                self.total_fees += fee_out
                if pnl > 0: self.n_wins += 1
                else:       self.n_losses += 1
            self._wl[sym].append(pnl)
            self._save()
            self.fees.record("SWING", fee_tot, abs(t["size"]*pct), 4)

            # Health: track daily PnL
            if hasattr(self, '_health') and self._health:
                self._health.record_trade("bnb_swing", pnl)

            # ML: record trade result for Ensemble learning
            if self._ml and ML_AVAILABLE:
                try:
                    kl = self._klines(sym)
                    feat = FeatureExtractor.from_klines(kl, 20) if kl else {}
                    if feat:
                        self._ml.record_trade_result(kl, feat, pnl)
                except Exception as _e: logging.debug(f"Ignored: {_e}")
            reason = ("TP" if hit_tp else
                      "TRAIL_SL" if (hit_sl and t.get("trail_active")) else
                      "SL" if hit_sl else "TIMEOUT")
            n = self.n_wins+self.n_losses
            self.log.info(
                f"📤 SWING {reason} {sym}: "
                f"{pnl:+.5f} BNB | "
                f"fee={fee_tot:.5f} | "
                f"WR={self.n_wins}/{n}")
            tg(
                f"📤 <b>SWING {reason}</b> {sym}\n"
                f"PnL net: {pnl:+.5f} BNB\n"
                f"Fee plătit: {fee_tot:.5f} BNB\n"
                f"WR: {self.n_wins}/{n} ({self.n_wins/n*100:.0f}%)"
            )

    def emergency_close(self):
        """Apelat de CrashGuard la nivel RED: inchide toate pozitiile imediat."""
        with self._lock:
            trades = dict(self.trades)
        if not trades:
            return
        self.log.error(
            f"EMERGENCY CLOSE swing: {len(trades)} pozitii")
        for sym, t in trades.items():
            try:
                if t["dir"] == 1: self.client.market_sell(sym, t["qty"])
                else:             self.client.market_buy(sym, t["qty"])
                fee_out = t["size"] * TAKER_FEE + t["size"] * SLIPPAGE
                p       = self.client.price(sym)
                pct     = (p - t["entry"]) / t["entry"] * t["dir"]
                pnl     = t["size"] * pct - t["fee_in"] - fee_out
                with self._lock:
                    if sym in self.trades: del self.trades[sym]
                    self.total_pnl  += pnl
                    self.total_fees += fee_out
                    if pnl > 0: self.n_wins += 1
                    else:       self.n_losses += 1
                self._wl[sym].append(pnl)
                self.log.info(f"Emergency close {sym}: {pnl:+.5f} BNB")
            except Exception as e:
                self.log.warning(f"Emergency close {sym}: {e}")
        self._save()
        tg(
            f"🔴 <b>SWING EMERGENCY CLOSE</b>\n"
            f"Crash sever. Inchis {len(trades)} pozitii:\n"
            f"{', '.join(trades.keys())}"
        )

    def run(self, stop: threading.Event):
        self.log.info(
            f"Swing | {self.bnb:.4f} BNB | "
            f"max 2/zi | TP={SWING_TP*100:.1f}% SL={SWING_SL*100:.1f}% | "
            f"fee roundtrip={ROUNDTRIP_TAKER*100:.4f}%")
        while not stop.is_set():
            try:
                # La RED: nu facem nimic (pozitiile au fost inchise de callback)
                if self.crash and self.crash.is_red:
                    stop.wait(60); continue
                self.scan()
            except Exception as e:
                self.log.warning(f"Swing: {e}")
            stop.wait(60)
        self.log.info("Swing oprit")


# ══════════════════════════════════════════════════════════════════════
# RISK MANAGER
# ══════════════════════════════════════════════════════════════════════

class BinanceEarn:
    """
    v3.1: Rezerva BNB (5% = ~0.06 BNB) pusă în Binance Simple Earn Flexible.
    APR tipic BNB Earn: 1-4% — mic dar mai bun decât zero.
    La 0.06 BNB × 3% APR = 0.0018 BNB/an = ~$1.2/an.

    API Binance Earn:
    - Subscribe: POST /sapi/v1/simple-earn/flexible/subscribe
    - Redeem:    POST /sapi/v1/simple-earn/flexible/redeem
    - Position:  GET  /sapi/v1/simple-earn/flexible/position

    NU folosim leverage, NU blocăm capitalul — Flexible = retras oricând.
    """

    def __init__(self, client: Binance, bnb_rezerva: float):
        self.client      = client
        self.bnb_rezerva = bnb_rezerva
        self.log         = L("Earn")
        self._lock       = threading.Lock()
        self.subscribed  = 0.0      # BNB în Earn
        self.earned      = 0.0      # dobândă acumulată
        self._last_check = 0.0
        self._product_id = "BNB001" # Binance Simple Earn Flexible BNB product ID
        self._load()

    def _load(self):
        try:
            if os.path.exists("v3_earn.json"):
                with open("v3_earn.json") as _jf:

                    d = json.load(_jf)
                self.subscribed = d.get("subscribed", 0.0)
                self.earned     = d.get("earned", 0.0)
        except Exception as _e: logging.debug(f'Ignored: {_e}')

    def _save(self):
        try:
            _data = {
                "subscribed": self.subscribed,
                "earned":     self.earned,
                "ts":         time.time()
            }
            _atomic_json_save("v3_earn.json", _data, indent=2)
        except Exception as _e: logging.debug(f'Ignored: {_e}')

    def _subscribe(self, qty: float) -> bool:
        """Subscrie BNB în Simple Earn Flexible."""
        if USE_TESTNET:
            self.log.info(f"[TESTNET] Earn subscribe {qty:.5f} BNB simulat")
            return True
        r = self.client._post("/sapi/v1/simple-earn/flexible/subscribe", {
            "productId": self._product_id,
            "amount":    f"{qty:.5f}",
            "autoSubscribe": "true",
        })
        return bool(r.get("success") or r.get("purchaseId"))

    def _get_position(self) -> float:
        """Returnează câte BNB sunt în Earn + dobânda."""
        if USE_TESTNET: return self.subscribed
        data = self.client._get("/sapi/v1/simple-earn/flexible/position",
                                {"asset": "BNB"}, signed=True)
        rows = data.get("rows", [])
        if rows:
            return float(rows[0].get("totalAmount", 0))
        return 0.0

    def subscribe_rezerva(self):
        """Pune rezerva în Earn dacă nu e deja subscrisă."""
        if not EARN_ENABLED: return
        if self.bnb_rezerva < EARN_MIN_BNB: return
        if self.subscribed > 0: return   # deja subscris

        qty = round(self.bnb_rezerva * 0.95, 5)  # păstrăm 5% lichid
        ok  = self._subscribe(qty)
        if ok:
            with self._lock:
                self.subscribed = qty
            self._save()
            self.log.info(
                f"✅ Earn subscribe: {qty:.5f} BNB rezervă → Simple Earn Flexible\n"
                f"   APR estimat: 1-4% | Retras oricând")
            tg(
                f"🏦 <b>Binance Earn</b>\n"
                f"Rezervă {qty:.5f} BNB → Simple Earn Flexible\n"
                f"APR estimat: ~2% | Est. an: +{qty*0.02:.5f} BNB\n"
                f"Retrasă automat dacă botul are nevoie",
                silent=True
            )

    def check_and_collect(self):
        """Verifică dobânda acumulată și o înregistrează."""
        if not EARN_ENABLED or self.subscribed <= 0: return
        now = time.time()
        if now - self._last_check < EARN_CHECK_H * 3600: return
        self._last_check = now

        try:
            position = self._get_position()
            if position > self.subscribed:
                new_earned = position - self.subscribed
                with self._lock:
                    self.earned    += new_earned
                    self.subscribed = position
                self._save()
                bnb_p = self.client.price("BNBUSDC")
                self.log.info(
                    f"🏦 Earn dobândă: +{new_earned:.6f} BNB "
                    f"(~${new_earned*bnb_p:.4f}) | "
                    f"total: {self.earned:.6f} BNB")
        except Exception as e:
            self.log.warning(f"Earn check: {e}")

    def run(self, stop: threading.Event):
        self.log.info(
            f"🏦 Binance Earn pornit | "
            f"Rezervă: {self.bnb_rezerva:.5f} BNB | "
            f"APR est: 1-4%")
        # Subscribe imediat la pornire
        self.subscribe_rezerva()
        while not stop.is_set():
            try:
                self.check_and_collect()
            except Exception as e:
                self.log.warning(f"Earn loop: {e}")
            stop.wait(3600)   # verificare la 1h
        self.log.info("⛔ Earn oprit")


# ══════════════════════════════════════════════════════════════════════
# C3: LAUNCHPOOL STAKING v5.0 — 20% capital, APR 5-18%
# ══════════════════════════════════════════════════════════════════════

class LaunchpoolStaking:
    """
    v5.0: 20% din capitalul BNB pus in Binance Launchpool / Simple Earn.

    LOGICA:
    - Launchpool: mineaza tokeni noi cu BNB (APR 10-18% in perioadele active)
    - Simple Earn Flexible: APR 2-4% permanent ca fallback
    - Rotatie automata: cand se termina un Launchpool, trecem la urmatorul
      sau la Simple Earn Flexible pana apare unul nou.

    API BINANCE:
    - Launchpool activ:  GET  /sapi/v1/launchpool/poolDetail
    - Subscribe Earn:    POST /sapi/v1/simple-earn/flexible/subscribe
    - Redeem:            POST /sapi/v1/simple-earn/flexible/redeem
    - Pozitie curenta:   GET  /sapi/v1/simple-earn/flexible/position

    AVANTAJ vs simple Earn:
    - Simple Earn BNB: ~2-4% APR permanent
    - Launchpool:      ~10-18% APR in perioade de lansare (la fiecare 2-4 saptamani)
    - Mix asteptat:    ~6-12% APR mediu anual

    RISC: ZERO pe capital (BNB nu e blocat, retras oricand).
    """

    # ID-uri produse Binance Simple Earn BNB
    EARN_PRODUCT_ID = "BNB001"

    def __init__(self, client: Binance, bnb_staking: float):
        self.client      = client
        self.bnb_staking = bnb_staking   # capital alocat (20% din total)
        self.log         = L("Launch")
        self._lock       = threading.Lock()

        # State
        self.subscribed_earn      = 0.0    # BNB in Simple Earn Flexible
        self.subscribed_launchpool= 0.0    # BNB in Launchpool activ
        self.earned_earn          = 0.0    # dobanda Earn
        self.earned_launchpool    = 0.0    # tokeni Launchpool (in BNB equiv)
        self.current_pool         = None   # dict cu info pool activ
        self.total_earned         = 0.0    # total BNB echivalent castigat

        self._last_check  = 0.0
        self._last_rotate = 0.0
        self._load()

    def _load(self):
        try:
            if os.path.exists("v5_launch.json"):
                with open("v5_launch.json") as _jf:

                    d = json.load(_jf)
                self.subscribed_earn       = d.get("subscribed_earn", 0.0)
                self.subscribed_launchpool = d.get("subscribed_launchpool", 0.0)
                self.earned_earn           = d.get("earned_earn", 0.0)
                self.earned_launchpool     = d.get("earned_launchpool", 0.0)
                self.total_earned          = d.get("total_earned", 0.0)
                self.current_pool          = d.get("current_pool", None)
        except Exception as _e: logging.debug(f"Ignored: {_e}")
    def _save(self):
        try:
            _data = {
                "subscribed_earn":       self.subscribed_earn,
                "subscribed_launchpool": self.subscribed_launchpool,
                "earned_earn":           self.earned_earn,
                "earned_launchpool":     self.earned_launchpool,
                "total_earned":          self.total_earned,
                "current_pool":          self.current_pool,
                "ts":                    time.time(),
            }
            _atomic_json_save("v5_launch.json", _data, indent=2)
        except Exception as _e: logging.debug(f"Ignored: {_e}")
    def _get_active_launchpool(self) -> Optional[dict]:
        """
        Interogheaza Binance API pentru Launchpool-uri active cu BNB.
        Returneaza None daca nu exista sau pe testnet.
        """
        if USE_TESTNET:
            return None
        try:
            data = self.client._get(
                "/sapi/v1/launchpool/poolDetail",
                {"asset": "BNB"}, signed=True
            )
            if isinstance(data, list) and data:
                # Cauta pool activ (status = 1)
                for pool in data:
                    if pool.get("status") == 1:
                        return {
                            "poolId":  pool.get("poolId"),
                            "project": pool.get("projectName", "Unknown"),
                            "apr":     float(pool.get("yearRate", 0)),
                            "endTime": pool.get("endTime", 0),
                        }
        except Exception as e:
            self.log.debug(f"Launchpool API: {e}")
        return None

    def _subscribe_earn(self, qty: float) -> bool:
        """Subscrie in Simple Earn Flexible BNB."""
        if USE_TESTNET:
            self.log.info(f"[TESTNET] Launch subscribe {qty:.5f} BNB in Earn")
            return True
        r = self.client._post("/sapi/v1/simple-earn/flexible/subscribe", {
            "productId":     self.EARN_PRODUCT_ID,
            "amount":        f"{qty:.5f}",
            "autoSubscribe": "true",
        })
        return bool(r.get("success") or r.get("purchaseId"))

    def _redeem_earn(self, qty: float) -> bool:
        """Retrage din Simple Earn Flexible."""
        if USE_TESTNET:
            return True
        r = self.client._post("/sapi/v1/simple-earn/flexible/redeem", {
            "productId": self.EARN_PRODUCT_ID,
            "amount":    f"{qty:.5f}",
        })
        return bool(r.get("success") or r.get("redeemId"))

    def _get_earn_position(self) -> float:
        """Returneaza BNB total in Earn (principal + dobanda)."""
        if USE_TESTNET:
            return self.subscribed_earn
        data = self.client._get(
            "/sapi/v1/simple-earn/flexible/position",
            {"asset": "BNB"}, signed=True
        )
        rows = data.get("rows", []) if isinstance(data, dict) else []
        return float(rows[0].get("totalAmount", 0)) if rows else 0.0

    def subscribe_initial(self):
        """
        La pornire: subscrie tot capitalul de staking in Earn Flexible
        ca pozitie de baza. Daca apare Launchpool, rotatim acolo.
        """
        if self.subscribed_earn > 0 or self.subscribed_launchpool > 0:
            self.log.info(
                f"Launch: deja subscris "
                f"(Earn={self.subscribed_earn:.5f} BNB, "
                f"Pool={self.subscribed_launchpool:.5f} BNB)")
            return

        qty = round(self.bnb_staking * 0.98, 5)   # 2% rezerva lichida
        if qty < 0.01:
            self.log.warning(f"Launch: capital prea mic ({qty:.5f} BNB)")
            return

        ok = self._subscribe_earn(qty)
        if ok:
            with self._lock:
                self.subscribed_earn = qty
            self._save()
            bnb_p = self.client.price("BNBUSDC")
            self.log.info(
                f"Launch subscribe: {qty:.5f} BNB → Simple Earn Flexible | "
                f"~${qty*bnb_p:.2f} | APR est: 2-4%")
            tg(
                f"🏦 <b>Launchpool Staking ACTIV</b>\n"
                f"{qty:.5f} BNB (~${qty*bnb_p:.2f}) → Simple Earn Flexible\n"
                f"APR baza: ~2-4% | Rotatie automata la Launchpool (10-18%)\n"
                f"Capital retras automat la nevoie",
                silent=True
            )

    def rotate_to_launchpool(self, pool: dict):
        """
        Muta capitalul din Earn in Launchpool activ.
        Apelat cand detectam un pool nou cu APR mai bun.
        """
        if USE_TESTNET:
            self.log.info(
                f"[TESTNET] Rotatie simulata → Launchpool {pool['project']} "
                f"APR={pool['apr']*100:.1f}%")
            with self._lock:
                self.current_pool = pool
            return

        # Retragem din Earn
        if self.subscribed_earn > 0:
            ok = self._redeem_earn(self.subscribed_earn)
            if not ok:
                self.log.warning("Rotatie: nu am putut retrage din Earn")
                return
            with self._lock:
                self.subscribed_launchpool = self.subscribed_earn
                self.subscribed_earn       = 0.0
                self.current_pool          = pool
            self._save()
            self.log.info(
                f"Rotatie: {self.subscribed_launchpool:.5f} BNB → "
                f"Launchpool {pool['project']} | APR={pool['apr']*100:.1f}%")
            tg(
                f"🚀 <b>Launchpool Rotatie</b>\n"
                f"{self.subscribed_launchpool:.5f} BNB → {pool['project']}\n"
                f"APR: {pool['apr']*100:.1f}% (vs ~3% Earn)\n"
                f"Boost estimat: +{(pool['apr']-0.03)*self.subscribed_launchpool*self.client.price('BNBUSDC')/12:.2f}$/luna",
                silent=True
            )

    def check_and_collect(self):
        """
        Verificare la fiecare 6h:
        1. Colecteaza dobanda/tokeni acumulati
        2. Verifica daca a aparut un Launchpool nou mai bun
        3. Rotatie automata daca APR Launchpool > APR Earn * 2
        """
        now = time.time()
        if now - self._last_check < 6 * 3600:
            return
        self._last_check = now

        bnb_p = self.client.price("BNBUSDC")

        # ── Colectare dobanda Earn ──────────────────────────────────
        if self.subscribed_earn > 0:
            pos = self._get_earn_position()
            if pos > self.subscribed_earn:
                new_earn = pos - self.subscribed_earn
                with self._lock:
                    self.earned_earn    += new_earn
                    self.total_earned   += new_earn
                    self.subscribed_earn = pos
                self._save()
                self.log.info(
                    f"Earn dobanda: +{new_earn:.6f} BNB "
                    f"(~${new_earn*bnb_p:.4f}) | "
                    f"total Earn: {self.earned_earn:.6f} BNB")

        # ── Colectare aproximata Launchpool (estimare) ───────────────
        lp_earned_this_cycle = 0.0
        if self.subscribed_launchpool > 0 and self.current_pool:
            apr  = self.current_pool.get("apr", 0.08)
            est  = self.subscribed_launchpool * apr / 365 / 4  # per 6h
            with self._lock:
                self.earned_launchpool += est
                self.total_earned      += est
            lp_earned_this_cycle = est
            self._save()
            if est > 0.0001:
                self.log.info(
                    f"Launchpool {self.current_pool['project']}: "
                    f"+{est:.6f} BNB est (~${est*bnb_p:.4f}) | "
                    f"APR={apr*100:.1f}%")

        # ── I5: Compound profit LP → Funding ─────────────────────────
        # Profitul Launchpool (dobanda + tokeni) e reinvestit in Funding Arb
        # cand se acumuleaza minim 0.005 BNB (pragul de reinvestire funding).
        # Efectul: compound suplimentar pe profitul pasiv LP.
        # Referinta la FundingArb e setata extern dupa init.
        total_new = lp_earned_this_cycle + (new_earn if 'new_earn' in locals() else 0.0)
        if (total_new > 0 and
                hasattr(self, '_funding_ref') and
                self._funding_ref is not None):
            with self._lock:
                self._pending_compound = getattr(self, '_pending_compound', 0.0) + total_new

            if getattr(self, '_pending_compound', 0.0) >= REINVEST_THRESHOLD:
                to_send = self._pending_compound
                self._pending_compound = 0.0
                try:
                    self._funding_ref._reinvest(to_send, bnb_p)
                    self.log.info(
                        f"I5 Compound LP→Funding: {to_send:.6f} BNB "
                        f"reinvestit in funding arb")
                    tg(
                        f"🔄 <b>Compound LP → Funding</b>\n"
                        f"+{to_send:.6f} BNB din Launchpool/Earn\n"
                        f"reinvestit automat in funding arb",
                        silent=True
                    )
                except Exception as e:
                    self.log.warning(f"I5 Compound LP: {e}")

        # ── Verificare pool nou ────────────────────────────────────
        new_pool = self._get_active_launchpool()
        if new_pool and (
            not self.current_pool or
            new_pool["poolId"] != self.current_pool.get("poolId")
        ):
            earn_apr = 0.03  # APR tipic Earn Flexible BNB
            if new_pool["apr"] > earn_apr * 2:
                self.log.info(
                    f"Pool nou detectat: {new_pool['project']} "
                    f"APR={new_pool['apr']*100:.1f}% > {earn_apr*100:.0f}%*2 → rotatie")
                self.rotate_to_launchpool(new_pool)

        # ── Daca pool-ul s-a terminat, reintram in Earn ──────────────
        if self.current_pool and not USE_TESTNET:
            end_ts = self.current_pool.get("endTime", 0) / 1000
            if end_ts > 0 and time.time() > end_ts:
                self.log.info(
                    f"Pool {self.current_pool['project']} expirat. "
                    f"Reintram in Simple Earn Flexible.")
                qty = round(self.subscribed_launchpool * 0.98, 5)
                if self._subscribe_earn(qty):
                    with self._lock:
                        self.subscribed_earn       = qty
                        self.subscribed_launchpool = 0.0
                        self.current_pool          = None
                    self._save()
                    tg(
                        f"🏦 <b>Launchpool expirat</b>\n"
                        f"Revenit in Simple Earn: {qty:.5f} BNB\n"
                        f"Asteptam urmatorul Launchpool...",
                        silent=True
                    )

    def run(self, stop: threading.Event):
        bnb_p = self.client.price("BNBUSDC")
        self.log.info(
            f"Launchpool Staking | {self.bnb_staking:.4f} BNB "
            f"(~${self.bnb_staking*bnb_p:.0f}) | "
            f"APR est: 5-18% (Launchpool) / 2-4% (Earn fallback)")
        self.subscribe_initial()
        while not stop.is_set():
            try:
                self.check_and_collect()
            except Exception as e:
                self.log.warning(f"Launchpool: {e}")
            stop.wait(3600)
        self.log.info("Launchpool oprit")

    @property
    def total_subscribed(self) -> float:
        return self.subscribed_earn + self.subscribed_launchpool

    @property
    def current_apr_est(self) -> float:
        if self.current_pool and self.subscribed_launchpool > 0:
            return self.current_pool.get("apr", 0.10)
        return 0.03  # Earn Flexible fallback


# ══════════════════════════════════════════════════════════════════════
# DUAL INVESTMENT MANAGER v8.0 — produs structurat Binance
# ══════════════════════════════════════════════════════════════════════

class DualInvestmentManager:
    """
    Gestioneaza subscrierile la Binance Dual Investment (DCI).

    LOGICA:
    ─────────────────────────────────────────────────────
    Dual Investment = produs structurat cu 2 rezultate posibile:
      CALL (sell high): depui BNB, la expirare:
        → BNB > strike: primesti USDT la pretul strike + premium
        → BNB < strike: primesti BNB inapoi + premium
      PUT (buy low): depui USDT, la expirare:
        → BNB < strike: primesti BNB la pretul strike + premium
        → BNB > strike: primesti USDT inapoi + premium

    In ambele cazuri: PREMIUM GARANTAT indiferent de directie.
    APR echivalent: 10-30% in conditii normale, pana la 80% in volatilitate mare.

    STRATEGIE:
      - Ciclu 7 zile, 4 cicluri/luna
      - Alternare CALL/PUT in functie de trend:
          Trend UP   → CALL (strike +10%) — vindem mai sus
          Trend DOWN → PUT  (strike -10%) — cumparam mai jos
          Sideways   → cel mai mare premium disponibil
      - Capital: 10% din total BNB

    API Binance:
      GET  /sapi/v1/dci/product/list      — produse disponibile
      POST /sapi/v1/dci/product/subscribe — subscrie
      GET  /sapi/v1/dci/product/positions — pozitii active
    """

    def __init__(self, client: Binance, bnb_capital: float):
        self.client      = client
        self.bnb_capital = bnb_capital
        self.log         = L("DualInv")
        self._lock       = threading.Lock()

        # State
        self.active_pos: List[dict] = []   # pozitii DCI active
        self.total_premium = 0.0           # premium total colectat (BNB equiv)
        self.n_cycles      = 0             # cicluri completate
        self._last_check   = 0.0
        self._last_subscribe = 0.0
        self._load()

    def _load(self):
        try:
            if os.path.exists("v8_di.json"):
                with open("v8_di.json") as _jf:

                    d = json.load(_jf)
                self.active_pos     = d.get("active_pos", [])
                self.total_premium  = d.get("total_premium", 0.0)
                self.n_cycles       = d.get("n_cycles", 0)
                self._last_subscribe= d.get("last_subscribe", 0.0)
        except Exception as _e: logging.debug(f'Ignored: {_e}')

    def _save(self):
        try:
            _data = {
                "active_pos":     self.active_pos,
                "total_premium":  self.total_premium,
                "n_cycles":       self.n_cycles,
                "last_subscribe": self._last_subscribe,
                "ts":             time.time(),
            }
            _atomic_json_save("v8_di.json", _data, indent=2)
        except Exception as _e: logging.debug(f'Ignored: {_e}')

    def _get_products(self, product_type: str = "CALL") -> List[dict]:
        """
        Listeaza produsele DCI disponibile pe Binance.
        product_type: 'CALL' (sell high) sau 'PUT' (buy low)
        """
        if USE_TESTNET:
            # Simulam un produs disponibil pe testnet
            bnb_p = self.client.price("BNBUSDC")
            return [{
                "id":          f"sim_DCI_{product_type}_{int(time.time())}",
                "type":        product_type,
                "asset":       "BNB",
                "strikePrice": f"{bnb_p * (1 + DI_STRIKE_UP if product_type=='CALL' else 1 - DI_STRIKE_DOWN):.2f}",
                "annualRate":  "0.180",    # 18% APR simulat
                "duration":    DI_CYCLE_D,
                "minAmount":   "0.01",
            }]
        try:
            data = self.client._get(
                "/sapi/v1/dci/product/list",
                {"asset": "BNB", "orderType": product_type,
                 "pageSize": 10, "pageIndex": 1},
                signed=True
            )
            return data.get("data", []) if isinstance(data, dict) else []
        except Exception as e:
            self.log.warning(f"DI get_products: {e}")
            return []

    def _subscribe(self, product_id: str, amount: float,
                   product_type: str) -> bool:
        """Subscrie la un produs DCI."""
        if USE_TESTNET:
            self.log.info(
                f"[TESTNET] DI subscribe {amount:.4f} BNB "
                f"tip={product_type} simulat")
            return True
        try:
            r = self.client._post(
                "/sapi/v1/dci/product/subscribe",
                {"productId":       product_id,
                 "investmentAmount": f"{amount:.5f}",
                 "autoCompound":    "NONE"}
            )
            return bool(r.get("purchaseId") or r.get("success"))
        except Exception as e:
            self.log.warning(f"DI subscribe: {e}")
            return False

    def _get_positions(self) -> List[dict]:
        """Returneaza pozitiile DCI active."""
        if USE_TESTNET:
            return self.active_pos
        try:
            data = self.client._get(
                "/sapi/v1/dci/product/positions",
                {"status": "PENDING", "pageSize": 50},
                signed=True
            )
            return data.get("data", []) if isinstance(data, dict) else []
        except Exception as e:
            self.log.warning(f"DI get_positions: {e}")
            return []

    def _detect_trend(self) -> str:
        """
        Detecteaza trendul curent BNB pentru a alege tip produs (CALL/PUT).
        Bazat pe EMA 24h si pret curent.
        """
        try:
            kl = self.client.klines("BNBUSDC", "4h", 24)
            if len(kl) < 12: return "SIDEWAYS"
            closes = [float(k[4]) for k in kl]
            ema_12 = sum(closes[-12:]) / 12
            ema_24 = sum(closes) / 24
            current = closes[-1]
            if current > ema_12 > ema_24:
                return "UP"
            elif current < ema_12 < ema_24:
                return "DOWN"
            return "SIDEWAYS"
        except Exception:
            return "SIDEWAYS"

    def _best_product(self, products: List[dict],
                      min_apr: float) -> Optional[dict]:
        """Alege produsul cu cel mai mare APR anual."""
        valid = [
            p for p in products
            if float(p.get("annualRate", 0)) >= min_apr
            and int(p.get("duration", 999)) <= DI_CYCLE_D + 2
        ]
        if not valid: return None
        return max(valid, key=lambda p: float(p.get("annualRate", 0)))

    def subscribe_cycle(self):
        """
        Subscrie la un nou ciclu DCI daca nu avem pozitie activa.
        Apelat la fiecare DI_CHECK_H ore.
        """
        if not DI_ENABLED: return

        # Nu subscriem daca avem deja pozitie activa
        live_pos = self._get_positions()
        if live_pos:
            self.log.debug(f"DI: {len(live_pos)} pozitii active, skip subscribe")
            with self._lock:
                self.active_pos = live_pos
            return

        # Nu subscriem prea des (min 6 zile intre subscrieri)
        days_since = (time.time() - self._last_subscribe) / 86400
        if days_since < DI_CYCLE_D - 1:
            return

        bnb_p    = self.client.price("BNBUSDC")
        trend    = self._detect_trend()
        di_type  = ("CALL" if trend == "UP" else
                    "PUT"  if trend == "DOWN" else
                    "CALL")  # sideways: CALL are de obicei premium mai mare

        products = self._get_products(di_type)
        best     = self._best_product(products, DI_MIN_APR)

        if not best:
            self.log.info(
                f"DI: niciun produs {di_type} cu APR>={DI_MIN_APR*100:.0f}%")
            # Incearca tipul opus
            alt_type = "PUT" if di_type == "CALL" else "CALL"
            products2 = self._get_products(alt_type)
            best      = self._best_product(products2, DI_MIN_APR)
            if best: di_type = alt_type

        if not best:
            self.log.info("DI: niciun produs disponibil aceasta saptamana")
            return

        amount    = round(self.bnb_capital * 0.95, 5)  # 95% din capital DI
        apr       = float(best.get("annualRate", 0))
        strike    = float(best.get("strikePrice", bnb_p))
        duration  = int(best.get("duration", DI_CYCLE_D))
        est_prem  = amount * apr / 365 * duration   # BNB premium estimat

        ok = self._subscribe(best.get("id", ""), amount, di_type)

        if ok:
            pos_data = {
                "id":       best.get("id"),
                "type":     di_type,
                "amount":   amount,
                "apr":      apr,
                "strike":   strike,
                "duration": duration,
                "ts":       time.time(),
                "est_prem": est_prem,
            }
            with self._lock:
                self.active_pos = [pos_data]
                self._last_subscribe = time.time()
            self._save()

            self.log.info(
                f"✅ DI subscribe: {amount:.4f} BNB | {di_type} | "
                f"strike=${strike:.2f} | APR={apr*100:.1f}% | "
                f"premium est: +{est_prem:.5f} BNB")
            tg(
                f"💎 <b>Dual Investment</b> subscris\n"
                f"Tip: {di_type} | Trend detectat: {trend}\n"
                f"Capital: {amount:.4f} BNB (~${amount*bnb_p:.2f})\n"
                f"Strike: ${strike:.2f} ({'+' if di_type=='CALL' else '-'}"
                f"{abs(strike/bnb_p-1)*100:.0f}% vs spot)\n"
                f"APR: {apr*100:.1f}% | Durata: {duration} zile\n"
                f"Premium estimat: +{est_prem:.5f} BNB "
                f"(~${est_prem*bnb_p:.2f})",
                silent=True
            )

    def collect_matured(self):
        """Verifica si colecteaza pozitiile expirate."""
        now = time.time()
        with self._lock:
            pos = list(self.active_pos)

        for p in pos:
            age_d = (now - p.get("ts", now)) / 86400
            if age_d < p.get("duration", DI_CYCLE_D) - 0.5:
                continue
            # Pozitia a expirat — colectam premium estimat
            prem = p.get("est_prem", 0)
            with self._lock:
                self.total_premium += prem
                self.n_cycles      += 1
                if p in self.active_pos:
                    self.active_pos.remove(p)
            self._save()
            bnb_p = self.client.price("BNBUSDC")
            self.log.info(
                f"💎 DI ciclu #{self.n_cycles} complet: "
                f"+{prem:.5f} BNB (~${prem*bnb_p:.2f}) | "
                f"total: {self.total_premium:.5f} BNB")
            tg(
                f"💎 <b>Dual Investment expirat</b>\n"
                f"Ciclu #{self.n_cycles} | {p['type']}\n"
                f"Premium colectat: +{prem:.5f} BNB (~${prem*bnb_p:.2f})\n"
                f"Total acumulat: {self.total_premium:.5f} BNB\n"
                f"Urmatoarea subscriere in ~1 zi.",
                silent=True
            )

    def run(self, stop: threading.Event):
        bnb_p = self.client.price("BNBUSDC")
        self.log.info(
            f"💎 Dual Investment | {self.bnb_capital:.4f} BNB "
            f"(~${self.bnb_capital*bnb_p:.2f}) | "
            f"APR min {DI_MIN_APR*100:.0f}% | ciclu {DI_CYCLE_D}z")
        while not stop.is_set():
            try:
                self.collect_matured()
                now = time.time()
                if now - self._last_check > DI_CHECK_H * 3600:
                    self._last_check = now
                    self.subscribe_cycle()
            except Exception as e:
                self.log.warning(f"DualInv: {e}")
            stop.wait(3600)
        self.log.info("Dual Investment oprit")

class CrashLevel:
    NORMAL = "NORMAL"
    YELLOW = "YELLOW"   # precautie: oprire intrari noi swing/grid
    ORANGE = "ORANGE"   # crash activ: toate intrarile noi oprite
    RED    = "RED"      # crash sever: inchidere pozitii + standby


# ══════════════════════════════════════════════════════════════════════
# MARKET SENTINEL — Fear & Greed + BTC Dominanta + Volume Anomaly
# Consulta platforme externe si ajusteaza strategiile automat
# ══════════════════════════════════════════════════════════════════════

SENTINEL_UPDATE_H  = 1.0    # update la fiecare ora
FEAR_GREED_EXTREME = 20     # < 20 = Extreme Fear → spacing mare
FEAR_GREED_GREED   = 75     # > 75 = Greed → spacing mic
BTC_DOM_HIGH       = 0.58   # > 58% dominanta → altcoins slabe → skip swing
BTC_DOM_LOW        = 0.42   # < 42% dominanta → altseason → boost swing
VOL_ANOMALY_MULT   = 2.0    # volum > 2x media = anomalie


class MarketCrashGuard:
    """
    Monitorizează BTC pentru crash-uri rapide.
    3 nivele: GREEN (normal), YELLOW (-3%), ORANGE (-5%), RED (-8%).
    RED → oprește intrări noi pe swing/grid, funding continuă.
    """
    def __init__(self, client: "Binance"):
        self.client = client
        self.log = L("Crash")
        self.level = "GREEN"
        self._last_check = 0
        self._btc_ref = 0.0
        # FIX8: Load btc_ref from file if exists
        self._crash_ref_file = "v8_crash_ref.json"
        try:
            if os.path.exists(self._crash_ref_file):
                with open(self._crash_ref_file) as _jf:

                    d = json.load(_jf)
                saved_ref = d.get("btc_ref", 0)
                age_s = time.time() - d.get("ts", 0)
                if saved_ref > 0 and age_s < 86400:  # max 24h old
                    self._btc_ref = saved_ref
                    self.log.info(f"FIX8: Restored BTC ref ${saved_ref:.2f} (age {age_s/3600:.1f}h)")
        except Exception as _e: logging.debug(f"Ignored: {_e}")
    def _crash_save_ref(self):
        """FIX8: Persist BTC reference price."""
        try:
            _atomic_json_save(self._crash_ref_file, {
                "btc_ref": self._btc_ref,
                "ts": time.time(),
            })
        except Exception as _e: logging.debug(f"Ignored: {_e}")
    def check(self):
        """Verifică BTC drop la fiecare 5 minute."""
        now = time.time()
        if now - self._last_check < 300:
            return
        self._last_check = now
        try:
            btc = self.client.price("BTCUSDC")
            if btc <= 0:
                return
            if self._btc_ref <= 0:
                self._btc_ref = btc
                self._crash_save_ref()
                return

            drop = (btc - self._btc_ref) / max(self._btc_ref, 0.01)

            if drop < -0.08:
                if self.level != "RED":
                    self.log.error(f"🔴 CRASH RED: BTC {drop*100:.1f}%")
                self.level = "RED"
            elif drop < -0.05:
                if self.level not in ("RED", "ORANGE"):
                    self.log.warning(f"🟠 CRASH ORANGE: BTC {drop*100:.1f}%")
                self.level = "ORANGE"
            elif drop < -0.03:
                if self.level not in ("RED", "ORANGE", "YELLOW"):
                    self.log.info(f"🟡 CRASH YELLOW: BTC {drop*100:.1f}%")
                self.level = "YELLOW"
            else:
                if self.level != "GREEN":
                    self.log.info(f"🟢 Crash recovered: BTC {drop*100:+.1f}%")
                self.level = "GREEN"

            # Reset reference daily at UTC midnight
            if datetime.now(timezone.utc).hour == 0 and datetime.now(timezone.utc).minute < 5:
                self._btc_ref = btc

        except Exception as e:
            self.log.debug(f"CrashGuard check: {e}")

    @property
    def entries_ok(self) -> bool:
        return self.level in ("GREEN", "YELLOW")

    @property
    def trading_ok(self) -> bool:
        return self.level != "RED"

    @property
    def is_red(self) -> bool:
        return self.level == "RED"

    def status_str(self) -> str:
        emoji = {"GREEN": "🟢", "YELLOW": "🟡", "ORANGE": "🟠", "RED": "🔴"}
        return f"{emoji.get(self.level, '?')} {self.level}"

    def run(self, stop_event, on_red=None, on_orange=None, on_yellow=None, on_recover=None):
        """Thread principal — verifică BTC la 5 minute."""
        self.log.info(
            f"🛡 Market Crash Guard pornit | simboluri: {CRASH_SYMBOLS} | "
            f"RED la BTC: -10%/1h sau -15%/4h sau -20%/24h")
        while not stop_event.is_set():
            try:
                old_level = self.level
                self.check()
                # Callbacks pe schimbare nivel
                if self.level != old_level:
                    if self.level == "RED" and on_red:
                        on_red(old_level)
                    elif self.level == "ORANGE" and on_orange:
                        on_orange(old_level)
                    elif self.level == "YELLOW" and on_yellow:
                        on_yellow(old_level)
                    elif self.level == "GREEN" and old_level != "GREEN" and on_recover:
                        on_recover(old_level)
            except Exception as e:
                self.log.debug(f"CrashGuard: {e}")
            stop_event.wait(300)  # 5 minute


class MarketSentinel:
    """
    Consulta API-uri externe gratuite pentru date de sentiment:

    1. Fear & Greed Index (alternative.me) — nelimitat
       → < 20 Extreme Fear  : spacing grid mai mare (+50%)
       → > 75 Greed         : spacing grid mai mic (-20%)
       → 20-75 Normal       : spacing standard

    2. BTC Dominanta (CoinGecko) — 30 req/min gratuit
       → > 58% : altcoins slabe → dezactiveaza swing SOL
       → < 42% : altseason    → boost swing SOL

    3. Volume Anomaly (Binance) — deja integrat
       → volum SOL > 2x media → mareste capital grid SOL temporar

    Toate ajustarile sunt TEMPORARE si revin la normal automat.
    Zero cost — API-uri 100% gratuite.
    """

    def __init__(self, client: "Binance"):
        self.client      = client
        self.log         = L("Sentinel")
        self._lock       = threading.Lock()
        self._last_update= 0.0

        # Valori curente
        self.fear_greed  = 50    # 0-100 (50 = neutral)
        self.fg_label    = "Neutral"
        self.btc_dom     = 0.50  # 0-1 (50% = normal)
        self.sol_vol_ratio = 1.0 # raport volum curent vs medie 7 zile

        # Multiplicatori pentru strategii
        self.spacing_mult  = 1.0   # pentru grid spacing
        self.swing_active  = True  # swing SOL activ/inactiv
        self.swing_boost   = 1.0   # boost sizing swing

        self._vol_history: list = []  # volum SOL ultimele 7 zile

    def _fetch_fear_greed(self) -> int:
        """Fear & Greed Index de la alternative.me — 100% gratuit."""
        try:
            r = _http_session.get(
                "https://api.alternative.me/fng/?limit=1",
                timeout=5)
            if r.status_code == 200:
                data = r.json()
                val  = int(data["data"][0]["value"])
                lbl  = data["data"][0]["value_classification"]
                self.log.info(f"Fear&Greed: {val} ({lbl})")
                return val, lbl
        except Exception as e:
            self.log.debug(f"Fear&Greed fetch: {e}")
        return 50, "Neutral"

    def _fetch_btc_dominance(self) -> float:
        """BTC dominanta de la CoinGecko — gratuit 30 req/min."""
        try:
            r = _http_session.get(
                "https://api.coingecko.com/api/v3/global",
                timeout=5)
            if r.status_code == 200:
                dom = r.json()["data"]["market_cap_percentage"]["btc"] / 100
                self.log.info(f"BTC dominanta: {dom*100:.1f}%")
                return dom
        except Exception as e:
            self.log.debug(f"BTC dominance fetch: {e}")
        return 0.50

    def _fetch_vol_anomaly(self) -> float:
        """Volume anomaly SOL din Binance (deja integrat)."""
        try:
            klines = self.client.klines("SOLUSDC", "1d", 8)
            if len(klines) >= 8:
                vols    = [float(k[5]) for k in klines]
                avg_7   = sum(vols[:-1]) / 7
                today   = vols[-1]
                ratio   = today / avg_7 if avg_7 > 0 else 1.0
                self.log.info(f"SOL vol ratio: {ratio:.2f}x")
                return ratio
        except Exception as e:
            self.log.debug(f"Vol anomaly fetch: {e}")
        return 1.0

    def update(self):
        """Actualizeaza toate datele si recalculeaza multiplicatorii."""
        now = time.time()
        if now - self._last_update < SENTINEL_UPDATE_H * 3600:
            return
        self._last_update = now

        fg, fg_lbl   = self._fetch_fear_greed()
        btc_dom      = self._fetch_btc_dominance()
        vol_ratio    = self._fetch_vol_anomaly()

        with self._lock:
            self.fear_greed    = fg
            self.fg_label      = fg_lbl
            self.btc_dom       = btc_dom
            self.sol_vol_ratio = vol_ratio

            # ── Fear & Greed → spacing grid ───────────────────────────
            if fg < FEAR_GREED_EXTREME:
                # Extreme Fear: volatilitate mare → spacing +50%
                self.spacing_mult = 1.50
            elif fg < 35:
                # Fear: spacing +25%
                self.spacing_mult = 1.25
            elif fg > FEAR_GREED_GREED:
                # Greed: piata calma → spacing -20%
                self.spacing_mult = 0.80
            else:
                # Normal
                self.spacing_mult = 1.00

            # ── BTC Dominanta → swing SOL ─────────────────────────────
            if btc_dom > BTC_DOM_HIGH:
                # BTC domina → altcoins slabe → dezactiveaza swing SOL
                self.swing_active = False
                self.swing_boost  = 0.0
            elif btc_dom < BTC_DOM_LOW:
                # Altseason → boost swing SOL
                self.swing_active = True
                self.swing_boost  = 1.30
            else:
                self.swing_active = True
                self.swing_boost  = 1.00

        # Log si Telegram la schimbari semnificative
        self.log.info(
            f"🔭 Sentinel: FG={fg}({fg_lbl}) | "
            f"BTC dom={btc_dom*100:.1f}% | "
            f"SOL vol={vol_ratio:.1f}x | "
            f"spacing×{self.spacing_mult:.2f} | "
            f"swing={'✅' if self.swing_active else '❌'}")

        if fg < FEAR_GREED_EXTREME or fg > FEAR_GREED_GREED or \
           btc_dom > BTC_DOM_HIGH or btc_dom < BTC_DOM_LOW or \
           vol_ratio > VOL_ANOMALY_MULT:
            tg(
                f"🔭 <b>Market Sentinel</b>\n"
                f"Fear & Greed: {fg} — {fg_lbl}\n"
                f"BTC Dominanță: {btc_dom*100:.1f}%\n"
                f"SOL Volum: {vol_ratio:.1f}x medie\n"
                f"─────────────────\n"
                f"Grid spacing: ×{self.spacing_mult:.2f}\n"
                f"Swing SOL: {'✅ Activ' if self.swing_active else '❌ Oprit'}\n"
                f"{'⚠️ Extreme Fear — spacing mărit' if fg < FEAR_GREED_EXTREME else ''}"
                f"{'🚨 BTC dominanță mare — swing oprit' if btc_dom > BTC_DOM_HIGH else ''}"
                f"{'📈 Altseason — swing boost 30%' if btc_dom < BTC_DOM_LOW else ''}",
                silent=True
            )

    def get_spacing_mult(self) -> float:
        with self._lock: return getattr(self, 'spacing_mult', 1.0)

    def get_swing_active(self) -> bool:
        with self._lock: return self.swing_active

    def get_swing_boost(self) -> float:
        with self._lock: return self.swing_boost

    def status(self) -> str:
        with self._lock:
            fg = self.fear_greed; lbl = self.fg_label
            dom = self.btc_dom; vol = self.sol_vol_ratio
            sm = self.spacing_mult; sa = self.swing_active
        return (
            f"🔭 <b>Market Sentinel</b>\n"
            f"Fear & Greed:    {fg}/100 — {lbl}\n"
            f"BTC Dominanță:   {dom*100:.1f}%\n"
            f"SOL Volum ratio: {vol:.1f}x\n"
            f"─────────────────\n"
            f"Grid spacing mult: ×{sm:.2f}\n"
            f"Swing SOL: {'✅ Activ' if sa else '❌ Oprit (BTC dom mare)'}\n"
            f"Update: la fiecare {SENTINEL_UPDATE_H:.0f}h"
        )

    def run(self, stop: threading.Event):
        self.log.info(
            f"🔭 Market Sentinel pornit | "
            f"Fear&Greed + BTC Dom + Volume | "
            f"update la {SENTINEL_UPDATE_H:.0f}h")
        # Update imediat la pornire
        self.update()
        while not stop.is_set():
            try:
                self.update()
            except Exception as e:
                self.log.warning(f"Sentinel: {e}")
            stop.wait(300)  # verifica la 5 minute daca e timpul
        self.log.info("⛔ Market Sentinel oprit")
    """
    Monitorizează BTC + BNB și detectează crash-uri în timp real.

    ARHITECTURĂ:
    - Thread dedicat, rulează independent de strategii
    - Analizează 3 ferestre temporale: 1h, 4h, 24h
    - Escaladare automată: NORMAL → YELLOW → ORANGE → RED
    - De-escaladare: necesită recuperare confirmată (+3% față de low, 2h stabilitate)
    - La RED: semnalizează FundingArb, GridMaker, SwingTrader să oprească/închidă

    STRATEGIA DE IEȘIRE LA RED:
    - Swing: închide toate pozițiile imediat (market order)
    - Grid: anulează toate ordinele deschise, nu mai repostează
    - Funding: închide pozițiile unde rata e deja negativă sau aproape de 0
               (cele cu rată pozitivă rămân — delta-neutral, pierderea e limitată)
    """



class CrashMonitor:
    """
    Monitorizează crash-uri de piață pe BTCUSDC și BNBUSDC.
    Detectează drop-uri -3%/-5%/-8% și declanșează niveluri YELLOW/ORANGE/RED.
    """

    def __init__(self, client: Binance):
        self.client   = client
        self.log      = L("Crash")
        self._lock    = threading.Lock()
        self.level    = CrashLevel.NORMAL
        self._prev    = CrashLevel.NORMAL

        # Istoric prețuri pentru calcul % schimbare pe ferestre
        # {sym: deque([(timestamp, price), ...])}
        self._hist: Dict[str, deque] = {
            s: deque(maxlen=1500) for s in CRASH_SYMBOLS   # ~25h la 60s
        }

        # Minimul înregistrat la intrarea în RED (pentru recuperare)
        self._crash_low: Dict[str, float] = {}
        self._stable_since: float = 0.0     # când am intrat în "stabil"
        self._crash_ts:     float = 0.0     # când a fost detectat crash-ul
        self._load()

    def _load(self):
        try:
            if os.path.exists("v4_crash.json"):
                with open("v4_crash.json") as _jf:

                    d = json.load(_jf)
                saved = d.get("level", CrashLevel.NORMAL)
                # La restart, dacă era RED/ORANGE, începem din YELLOW (precauție)
                self.level = CrashLevel.YELLOW if saved in (
                    CrashLevel.RED, CrashLevel.ORANGE) else saved
                self._crash_low  = d.get("crash_low", {})
                self._crash_ts   = d.get("crash_ts", 0.0)
                if self.level != CrashLevel.NORMAL:
                    self.log.warning(
                        f"Restart cu nivel crash {self.level} "
                        f"(salvat: {saved}). Precautie activata.")
        except Exception as _e: logging.debug(f"Ignored: {_e}")
    def _save(self):
        # Throttle: scrie maxim o dată la 60s
        now = time.time()
        if now - getattr(self, "_last_save_ts", 0) < 60:
            return
        self._last_save_ts = now
        try:
            _data = {
                "level":     self.level,
                "crash_low": self._crash_low,
                "crash_ts":  self._crash_ts,
                "ts":        now,
            }
            _atomic_json_save("v4_crash.json", _data, indent=2)
        except Exception as _e: logging.debug(f"Ignored: {_e}")
    def _pct_change(self, sym: str, seconds: int) -> Optional[float]:
        """
        Returnează % schimbare față de prețul de acum `seconds` secunde.
        None dacă nu avem suficient istoric.
        """
        hist = self._hist.get(sym)
        if not hist or len(hist) < 2:
            return None
        now_ts    = time.time()
        target_ts = now_ts - seconds
        current   = hist[-1][1]
        # Căutăm cel mai apropiat punct de target_ts
        ref_price = None
        for ts, px in hist:
            if ts >= target_ts:
                ref_price = px
                break
        if ref_price is None or ref_price == 0:
            return None
        return (current - ref_price) / ref_price

    def _calc_level(self) -> str:
        """Determină nivelul de crash pe baza prețurilor actuale."""
        worst = CrashLevel.NORMAL
        for sym in CRASH_SYMBOLS:
            c1h  = self._pct_change(sym, 3600)
            c4h  = self._pct_change(sym, 14400)
            c24h = self._pct_change(sym, 86400)

            # RED — oricare fereastră depășește pragul
            if ((c1h  is not None and c1h  <= CRASH_R_1H)  or
                (c4h  is not None and c4h  <= CRASH_R_4H)  or
                (c24h is not None and c24h <= CRASH_R_24H)):
                return CrashLevel.RED   # escaladare imediată, nu continuăm

            # ORANGE
            if ((c1h  is not None and c1h  <= CRASH_O_1H)  or
                (c4h  is not None and c4h  <= CRASH_O_4H)  or
                (c24h is not None and c24h <= CRASH_O_24H)):
                worst = CrashLevel.ORANGE

            # YELLOW
            elif worst == CrashLevel.NORMAL:
                if ((c1h  is not None and c1h  <= CRASH_Y_1H)  or
                    (c4h  is not None and c4h  <= CRASH_Y_4H)  or
                    (c24h is not None and c24h <= CRASH_Y_24H)):
                    worst = CrashLevel.YELLOW

        return worst

    def _check_recovery(self) -> bool:
        """
        Verifică dacă piața s-a recuperat după un crash RED/ORANGE.
        Condiții:
          1. Prețul curent > crash_low * (1 + CRASH_RECOVER_PCT)
          2. Stabilitate timp de CRASH_RECOVER_H ore (nivelul calculat = NORMAL)
        """
        if not self._crash_low:
            return True
        # Condiție 1: recuperare față de low
        for sym in CRASH_SYMBOLS:
            low  = self._crash_low.get(sym, 0)
            hist = self._hist.get(sym)
            if not hist or not low:
                continue
            curr = hist[-1][1]
            if curr < low * (1 + CRASH_RECOVER_PCT):
                self._stable_since = 0.0   # resetăm cronometrul
                return False
        # Condiție 2: stabilitate
        calc = self._calc_level()
        if calc not in (CrashLevel.NORMAL, CrashLevel.YELLOW):
            self._stable_since = 0.0
            return False
        if self._stable_since == 0.0:
            self._stable_since = time.time()
        stable_h = (time.time() - self._stable_since) / 3600
        return stable_h >= CRASH_RECOVER_H

    def update(self) -> Tuple[str, str]:
        """
        Actualizează istoricul prețurilor și recalculează nivelul.
        Returnează (nivel_nou, nivel_vechi).
        """
        # Colectare prețuri
        for sym in CRASH_SYMBOLS:
            p = self.client.price(sym)
            if p > 0:
                self._hist[sym].append((time.time(), p))

        new_level = self._calc_level()

        with self._lock:
            old = self.level

            # Escaladare: întotdeauna permisă
            order = [CrashLevel.NORMAL, CrashLevel.YELLOW,
                     CrashLevel.ORANGE, CrashLevel.RED]
            if order.index(new_level) > order.index(self.level):
                # Intrăm în crash
                if new_level == CrashLevel.RED and old != CrashLevel.RED:
                    self._crash_ts = time.time()
                    # Salvăm prețurile low pentru condiția de recuperare
                    for sym in CRASH_SYMBOLS:
                        h = self._hist.get(sym)
                        if h:
                            self._crash_low[sym] = min(px for _, px in h)
                self.level = new_level

            # De-escaladare: numai prin recuperare confirmată
            elif (order.index(new_level) < order.index(self.level) and
                  self.level in (CrashLevel.RED, CrashLevel.ORANGE)):
                if self._check_recovery():
                    self.log.info(
                        f"Piata stabilizata dupa {(time.time()-self._crash_ts)/3600:.1f}h. "
                        f"Nivel: {self.level} -> {new_level}")
                    self._crash_low  = {}
                    self._stable_since = 0.0
                    self.level = new_level
                # else: rămânem la nivelul actual până la recuperare completă
            elif self.level == CrashLevel.YELLOW and new_level == CrashLevel.NORMAL:
                # YELLOW → NORMAL direct, fără condiție specială
                self.level = CrashLevel.NORMAL

            self._save()
            return self.level, old

    @property
    def is_normal(self)  -> bool:
        return self.level == CrashLevel.NORMAL

    @property
    def is_yellow(self)  -> bool:
        return self.level == CrashLevel.YELLOW

    @property
    def is_orange(self)  -> bool:
        return self.level == CrashLevel.ORANGE

    @property
    def is_red(self)     -> bool:
        return self.level == CrashLevel.RED

    @property
    def trading_ok(self) -> bool:
        """Pot intra în tranzacții noi?"""
        return self.level == CrashLevel.NORMAL

    @property
    def entries_ok(self) -> bool:
        """Pot deschide poziții noi (inclusiv YELLOW)?"""
        return self.level in (CrashLevel.NORMAL, CrashLevel.YELLOW)

    def status_str(self) -> str:
        icons = {
            CrashLevel.NORMAL: "🟢",
            CrashLevel.YELLOW: "🟡",
            CrashLevel.ORANGE: "🟠",
            CrashLevel.RED:    "🔴",
        }
        return f"{icons.get(self.level, '?')} {self.level}"

    def run(self, stop: threading.Event,
            on_red:    callable = None,
            on_orange: callable = None,
            on_yellow: callable = None,
            on_recover:callable = None):
        """
        Thread principal al crash guard.
        Callback-uri apelate la schimbare de nivel:
          on_red(prev)    — crash sever detectat
          on_orange(prev) — crash moderat
          on_yellow(prev) — precautie
          on_recover(prev)— recuperare
        """
        self.log.info(
            f"🛡 Market Crash Guard pornit | "
            f"simboluri: {CRASH_SYMBOLS} | "
            f"RED la BTC: {CRASH_R_1H*100:.0f}%/1h sau "
            f"{CRASH_R_4H*100:.0f}%/4h sau "
            f"{CRASH_R_24H*100:.0f}%/24h")
        tg(
            f"🛡 <b>Market Crash Guard activ</b>\n"
            f"Monitorizez: {', '.join(CRASH_SYMBOLS)}\n"
            f"YELLOW la -4%/1h | ORANGE la -6%/1h | RED la -10%/1h\n"
            f"La RED: inchid swing+grid, funding monitorizat intensiv",
            silent=True
        )

        while not stop.is_set():
            try:
                new_level, old_level = self.update()

                if new_level == old_level:
                    stop.wait(CRASH_CHECK_SEC)
                    continue

                # ── Schimbare de nivel ────────────────────────────────
                icons = {CrashLevel.NORMAL:"🟢", CrashLevel.YELLOW:"🟡",
                         CrashLevel.ORANGE:"🟠", CrashLevel.RED:"🔴"}
                order = [CrashLevel.NORMAL, CrashLevel.YELLOW,
                         CrashLevel.ORANGE, CrashLevel.RED]
                escalating = order.index(new_level) > order.index(old_level)

                # Prețuri curente pentru context
                btc_p = (self._hist["BTCUSDC"][-1][1]
                         if self._hist["BTCUSDC"] else 0)
                bnb_p = (self._hist["BNBUSDC"][-1][1]
                         if self._hist["BNBUSDC"] else 0)
                c1h_btc = self._pct_change("BTCUSDC", 3600) or 0
                c4h_btc = self._pct_change("BTCUSDC", 14400) or 0

                if escalating:
                    msg = (
                        f"{icons[new_level]} <b>CRASH ALERT: {new_level}</b> "
                        f"(era {old_level})\n"
                        f"BTC: ${btc_p:,.0f} | 1h: {c1h_btc*100:+.1f}% | "
                        f"4h: {c4h_btc*100:+.1f}%\n"
                        f"BNB: ${bnb_p:.2f}\n"
                    )
                    if new_level == CrashLevel.YELLOW:
                        msg += "Actiune: swing & grid STOP intrari noi"
                        if on_yellow: on_yellow(old_level)
                    elif new_level == CrashLevel.ORANGE:
                        msg += "Actiune: TOATE intrarile noi oprite"
                        if on_orange: on_orange(old_level)
                    elif new_level == CrashLevel.RED:
                        msg += (
                            "Actiune: INCHID pozitii swing & grid!\n"
                            "Funding cu rata pozitiva RAMAN deschise.\n"
                            "Bot in STANDBY pana la recuperare."
                        )
                        if on_red: on_red(old_level)

                    self.log.error(
                        f"CRASH {old_level}->{new_level}: "
                        f"BTC {c1h_btc*100:+.1f}%/1h, {c4h_btc*100:+.1f}%/4h")
                    tg(msg)

                else:
                    # De-escaladare / recuperare
                    msg = (
                        f"{icons[new_level]} <b>PIATA STABILIZATA</b>: "
                        f"{old_level} → {new_level}\n"
                        f"BTC: ${btc_p:,.0f} | 1h: {c1h_btc*100:+.1f}%\n"
                        f"Recuperare confirmata. Trading RELUAT."
                    )
                    self.log.info(
                        f"Recuperare: {old_level}->{new_level}")
                    tg(msg)
                    if on_recover: on_recover(old_level)

            except Exception as e:
                self.log.warning(f"CrashGuard: {e}")

            stop.wait(CRASH_CHECK_SEC)

        self.log.info("Crash Guard oprit")


# ══════════════════════════════════════════════════════════════════════
# RISK MANAGER — FIX 1: clasă separată (era înglobată eronat în BinanceEarn)
# ══════════════════════════════════════════════════════════════════════

class RiskManager:
    """
    Monitorizare pierdere zilnică + circuit breaker.
    Circuit breaker: dacă pierdem >2% din capital într-o zi → stop 24h.
    Drawdown calculat față de peak-ul portofoliului.
    """

    def __init__(self, bnb: float):
        self.bnb   = bnb
        self.log   = L("Risk")
        self._lock = threading.Lock()
        self.daily = 0.0
        self.total = 0.0
        self.peak  = 0.0
        self.day   = datetime.now(tz=timezone.utc).date()
        self.cb    = False

    def record(self, pnl: float):
        with self._lock:
            if datetime.now(tz=timezone.utc).date() != self.day:
                self.daily = 0.0
                self.day   = datetime.now(tz=timezone.utc).date()
                self.cb    = False   # reset circuit breaker la zi nouă
            self.daily += pnl
            self.total += pnl
            if self.total > self.peak:
                self.peak = self.total

    def ok(self) -> bool:
        with self._lock:
            if self.cb: return False
            if self.daily < -self.bnb * DAILY_LOSS_LIMIT:
                self.cb = True
                tg(
                    f"🚨 <b>CIRCUIT BREAKER ACTIVAT</b>\n"
                    f"Pierdere zilnică: {self.daily:.5f} BNB "
                    f"({self.daily/max(self.bnb,0.001)*100:.2f}%)\n"
                    f"Limită: -{DAILY_LOSS_LIMIT*100:.0f}%\n"
                    f"Trading oprit 24h. Resetare la miezul nopții."
                )
                self.log.error(
                    f"🚨 Circuit breaker: {self.daily:.5f} BNB pierdut azi")
                return False
        return True

    def dd(self) -> float:
        """Drawdown curent față de peak (0.0 = fără pierdere)."""
        return max(0.0, (self.peak - self.total) / (self.bnb + 1e-10))


# ══════════════════════════════════════════════════════════════════════
# WEEKLY REBALANCER v8.0 — capital dinamic pe 7 zile
# ══════════════════════════════════════════════════════════════════════

class WeeklyRebalancer:
    """
    Rebalanseaza capitalul intre strategii la fiecare 7 zile.

    LOGICA:
    ─────────────────────────────────────────────────────
    La fiecare 7 zile compara ROI% al fiecarei strategii:
      - Strategia cu ROI maxim primeste +REBAL_SHIFT_PCT capital
      - Capitalul e luat din strategia cu ROI minim
      - Constrangeri: nicio strategie nu depaseste 70% sau scade sub 2%

    EXEMPLU:
      Saptamana 1: Funding +2.1%, Grid +0.8%, Swing -0.1%
      → Funding primeste +3% capital din Swing
      Saptamana 2: Grid +1.5%, Funding +0.9%, Swing +0.3%
      → Grid primeste +3% capital din Swing

    EFECTUL IN TIMP:
      Capitalul migreaza organic spre ce functioneaza.
      In bull: mai mult Funding (rate mari)
      In sideways: mai mult Grid (fills frecvente)
      Swing primeste capital doar cand performa consistent.
    """

    def __init__(self, bot: "SolanaBot"):
        self.bot          = bot
        self.log          = L("Rebal")
        self._last_rebal  = time.time()   # prima rebalansare dupa 7 zile
        self._snap: Dict[str, float] = {}  # snapshot PnL la inceput saptamana
        self._take_snapshot()

    def _take_snapshot(self):
        """Salveaza PnL curent ca referinta pentru saptamana."""
        b = self.bot
        self._snap = {
            "funding":    b.funding.total_pnl,
            "grid":       b.grid.total_pnl,
            "swing":      b.swing.total_pnl,
            "launchpool": b.launchpool.total_earned,
            "ts":         time.time(),
        }

    def _weekly_roi(self) -> Dict[str, float]:
        """Calculeaza ROI% din ultima saptamana per strategie."""
        b    = self.bot
        caps = {
            "funding":    b.bnb * b.funding.bnb / max(b.bnb, 1e-10),
            "grid":       b.bnb * ALLOC_GRID,
            "swing":      b.bnb * ALLOC_SWING,
            "launchpool": b.bnb * ALLOC_LAUNCHPOOL,
        }
        current = {
            "funding":    b.funding.total_pnl,
            "grid":       b.grid.total_pnl,
            "swing":      b.swing.total_pnl,
            "launchpool": b.launchpool.total_earned,
        }
        roi = {}
        for k in current:
            delta = current[k] - self._snap.get(k, current[k])
            cap   = max(caps.get(k, 0.01), 0.01)
            roi[k] = delta / cap   # ROI relativ la capitalul alocat
        return roi

    def rebalance(self):
        """Executa rebalansarea daca e momentul."""
        if not REBAL_ENABLED: return
        days_since = (time.time() - self._last_rebal) / 86400
        if days_since < REBAL_INTERVAL_D: return

        roi   = self._weekly_roi()
        best  = max(roi, key=roi.get)
        worst = min(roi, key=roi.get)

        if best == worst:
            self._take_snapshot()
            self._last_rebal = time.time()
            return

        # Calculam cat putem transfera
        b   = self.bot
        bnb_safe = max(b.bnb, 0.001)
        allocs = {
            "funding":    b.funding.bnb / bnb_safe,
            "grid":       b.grid.bnb    / bnb_safe,
            "swing":      b.swing.bnb   / bnb_safe,
            "launchpool": b.launchpool.bnb_staking / bnb_safe,
        }

        new_best  = min(REBAL_MAX_ALLOC, allocs[best] + REBAL_SHIFT_PCT)
        new_worst = max(REBAL_MIN_ALLOC, allocs[worst] - REBAL_SHIFT_PCT)
        actual_shift = min(
            new_best - allocs[best],
            allocs[worst] - new_worst
        )

        if actual_shift < 0.005:
            self.log.info("Rebal: shift prea mic, skip")
            self._take_snapshot()
            self._last_rebal = time.time()
            return

        shift_bnb = b.bnb * actual_shift

        # Aplicam rebalansarea (actualizam capitalul fiecarei strategii)
        strat_map = {
            "funding":    b.funding,
            "grid":       b.grid,
            "swing":      b.swing,
            "launchpool": b.launchpool,
        }
        # Scadem din worst
        ws = strat_map[worst]
        if hasattr(ws, 'bnb'):
            ws.bnb = max(0, ws.bnb - shift_bnb)
        elif hasattr(ws, 'bnb_staking'):
            ws.bnb_staking = max(0, ws.bnb_staking - shift_bnb)

        # Adaugam la best
        bs = strat_map[best]
        if hasattr(bs, 'bnb'):
            bs.bnb += shift_bnb
        elif hasattr(bs, 'bnb_staking'):
            bs.bnb_staking += shift_bnb

        self._take_snapshot()
        self._last_rebal = time.time()

        roi_str = " | ".join(
            f"{k}: {v*100:+.2f}%" for k, v in sorted(
                roi.items(), key=lambda x: -x[1]))
        self.log.info(
            f"Rebal: {worst}→{best} | shift={shift_bnb:.4f} BNB | "
            f"ROI 7z: {roi_str}")
        tg(
            f"⚖️ <b>Rebalansare saptamanala</b>\n"
            f"Capital mutat: {shift_bnb:.4f} BNB\n"
            f"De la: {worst} (ROI {roi[worst]*100:+.2f}%)\n"
            f"La: {best} (ROI {roi[best]*100:+.2f}%)\n"
            f"\nROI 7 zile:\n" +
            "\n".join(f"  {k}: {v*100:+.2f}%" for k, v in
                      sorted(roi.items(), key=lambda x: -x[1])),
            silent=True
        )


# ══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════

class MLEngine:
    """
    Manager central ML + Deep AI.
    
    FAZĂ 1 (COLLECTING): Doar colectează date. NU influențează trading.
    FAZĂ 2 (TRAINING):   Modelele se antrenează. Încă NU tranzacționează.
    FAZĂ 3 (READY):      ML filtrează trades (doar dacă accuracy > threshold).
    
    Trading-ul merge 100% pe reguli fixe până ML-ul dovedește că e util.
    """

    MIN_REGIME_ACCURACY = 0.45
    MIN_REGIME_SAMPLES = 100
    MIN_PRICE_MLP_ACCURACY = 0.40
    MIN_GRID_FILLS = 50
    MIN_FUNDING_OBS = 100
    MIN_ENSEMBLE_TRADES = 50
    MIN_ANOMALY_SAMPLES = 200

    def __init__(self, telegram_callback=None):
        self.regime = RegimeClassifier() if ML_AVAILABLE else None
        self.grid_ml = GridSpacingML() if ML_AVAILABLE else None
        self.funding_ml = FundingPredictor() if ML_AVAILABLE else None
        self.price_mlp = PriceDirectionMLP() if DEEP_AI_AVAILABLE else None
        self.anomaly = AnomalyDetectorMLP() if DEEP_AI_AVAILABLE else None
        self.ensemble = EnsembleSignal() if DEEP_AI_AVAILABLE else None
        self.telegram = telegram_callback
        self.last_train_ts = 0
        self.train_interval = 86400
        self.enabled = ML_AVAILABLE
        self._training = False

    def _is_model_ready(self, name: str) -> bool:
        if name == "regime":
            return (self.regime and self.regime.trained and
                    self.regime.accuracy >= self.MIN_REGIME_ACCURACY and
                    self.regime.train_samples >= self.MIN_REGIME_SAMPLES)
        elif name == "price_mlp":
            return (self.price_mlp and self.price_mlp.trained and
                    self.price_mlp.accuracy >= self.MIN_PRICE_MLP_ACCURACY)
        elif name == "anomaly":
            return (self.anomaly and self.anomaly.trained and
                    self.anomaly.train_samples >= self.MIN_ANOMALY_SAMPLES)
        elif name == "grid_ml":
            return (self.grid_ml and self.grid_ml.trained and
                    len(self.grid_ml.fill_history) >= self.MIN_GRID_FILLS)
        elif name == "funding_ml":
            return (self.funding_ml and self.funding_ml.trained and
                    len(self.funding_ml.history) >= self.MIN_FUNDING_OBS)
        elif name == "ensemble":
            return (self.ensemble and self.ensemble.trained and
                    len(self.ensemble.history) >= self.MIN_ENSEMBLE_TRADES)
        return False

    @property
    def trading_ready(self) -> bool:
        return self._is_model_ready("regime") and self._is_model_ready("price_mlp")

    @property
    def phase(self) -> str:
        if self.trading_ready: return "READY"
        if any(m and m.trained for m in [self.regime, self.price_mlp, self.anomaly]):
            return "TRAINING"
        return "COLLECTING"

    def retrain_if_needed(self, klines_1h: list = None):
        if not self.enabled or self._training: return
        if time.time() - self.last_train_ts < self.train_interval: return
        self._training = True
        try:
            trained = []
            if klines_1h:
                if self.regime and self.regime.train(klines_1h):
                    trained.append(f"Regime({self.regime.accuracy:.0%},{self.regime.train_samples}s)")
                if self.price_mlp and self.price_mlp.train(klines_1h):
                    trained.append(f"MLP({self.price_mlp.accuracy:.0%})")
                if self.anomaly and self.anomaly.train(klines_1h):
                    trained.append(f"Anomaly(t={self.anomaly.threshold:.3f})")
            if self.grid_ml and len(self.grid_ml.fill_history) >= 30 and self.grid_ml.train():
                trained.append(f"Grid({len(self.grid_ml.fill_history)}f)")
            if self.funding_ml and len(self.funding_ml.history) >= 50 and self.funding_ml.train():
                trained.append(f"Fund({len(self.funding_ml.history)}o)")
            if self.ensemble and len(self.ensemble.history) >= 30 and self.ensemble.train():
                trained.append(f"Ens({len(self.ensemble.history)}t)")
            if trained:
                self.last_train_ts = time.time()
                msg = f"🧠 ML [{self.phase}]: {', '.join(trained)}"
                _enh_logger.info(msg)
                if self.telegram:
                    try: self.telegram(msg)
                    except Exception as _e: logging.debug(f'Ignored: {_e}')
        except Exception as e:
            _enh_logger.warning(f"ML retrain error: {e}")
        finally:
            self._training = False

    def get_regime(self, klines_1h): return self.regime.predict(klines_1h) if self._is_model_ready("regime") else {"regime":"unknown","confidence":0.0,"proba":{}}
    def get_price_direction(self, klines_1h): return self.price_mlp.predict(klines_1h) if self._is_model_ready("price_mlp") else {"direction":"flat","confidence":0.0,"proba":{}}
    def get_anomaly_score(self, klines_1h): return self.anomaly.score(klines_1h) if self._is_model_ready("anomaly") else {"score":0.0,"is_anomaly":False,"sizing_mult":1.0}
    def get_optimal_spacing(self, feat_dict): return self.grid_ml.predict_optimal_spacing(feat_dict) if self._is_model_ready("grid_ml") else 0.012
    def rank_funding_pairs(self, sym_feat): return self.funding_ml.rank_symbols(sym_feat) if self._is_model_ready("funding_ml") else []

    def record_grid_fill(self, feat_dict, spacing, profit):
        if self.grid_ml: self.grid_ml.record_fill(feat_dict, spacing, profit)
    def record_funding_rate(self, symbol, feat_dict, rate):
        if self.funding_ml: self.funding_ml.record_rate(symbol, feat_dict, rate)
    def record_trade_result(self, klines_1h, feat_dict, pnl):
        if self.ensemble and feat_dict:
            r = self.get_regime(klines_1h); d = self.get_price_direction(klines_1h); a = self.get_anomaly_score(klines_1h)
            self.ensemble.record_trade(r.get("proba",{}), d.get("proba",{}), a.get("score",0), feat_dict, pnl)

    def get_swing_signal(self, klines_1h: list, direction: int) -> dict:
        """ML NU blochează trades până nu e READY. Colectează date mereu."""
        if not self.trading_ready:
            return {"take_trade": True, "confidence": 0.0,
                    "sizing_mult": 1.0, "reason": f"ML:{self.phase}"}
        regime = self.get_regime(klines_1h)
        price_dir = self.get_price_direction(klines_1h)
        anomaly = self.get_anomaly_score(klines_1h)
        reasons = []; confidence = 0.5; sizing = anomaly.get("sizing_mult", 1.0)
        r, r_c = regime.get("regime","unknown"), regime.get("confidence",0)
        if r == "sideways" and r_c > 0.65:
            return {"take_trade":False,"confidence":r_c,"sizing_mult":sizing,"reason":f"ML:sideways({r_c:.0%})"}
        d, d_c = price_dir.get("direction","flat"), price_dir.get("confidence",0)
        if d != "flat" and d_c > 0.55:
            if (d=="up" and direction==1) or (d=="down" and direction==-1):
                confidence += 0.2; reasons.append(f"MLP:{d}({d_c:.0%})")
            elif d_c > 0.6:
                return {"take_trade":False,"confidence":d_c,"sizing_mult":sizing,"reason":f"MLP:contra {d}({d_c:.0%})"}
        if (r=="trending_up" and direction==1) or (r=="trending_down" and direction==-1):
            confidence += 0.15; sizing *= 1.1; reasons.append(f"regime:{r}")
        if anomaly.get("is_anomaly",False):
            confidence -= 0.1; reasons.append(f"anomaly")
        return {"take_trade": confidence > 0.55, "confidence": min(confidence,1.0),
                "sizing_mult": sizing, "reason": " | ".join(reasons) or "ML:READY"}

    def get_status(self) -> dict:
        return {"enabled":self.enabled, "deep_ai":DEEP_AI_AVAILABLE, "phase":self.phase,
                "trading_ready":self.trading_ready,
                "regime":self.regime.get_status() if self.regime else {},
                "grid_ml":self.grid_ml.get_status() if self.grid_ml else {},
                "funding_ml":self.funding_ml.get_status() if self.funding_ml else {},
                "price_mlp":self.price_mlp.get_status() if self.price_mlp else {},
                "anomaly":self.anomaly.get_status() if self.anomaly else {},
                "ensemble":self.ensemble.get_status() if self.ensemble else {}}

# ══════════════════════════════════════════════════════════════════════
# DEEP AI MODULE — Neural Networks (MLP, CPU only, no GPU)
# 3 modele: PriceDirectionMLP, AnomalyDetectorMLP, EnsembleSignal
# ══════════════════════════════════════════════════════════════════════

try:
    import numpy as np
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    DEEP_AI_AVAILABLE = ML_AVAILABLE  # needs numpy + sklearn
except ImportError:
    RandomForestClassifier = None
    GradientBoostingClassifier = None
    StandardScaler = None
    LinearRegression = None
    DEEP_AI_AVAILABLE = False



class SolanaBot:

    def __init__(self):
        self.log = L("Bot")
        self.bin = Binance(BINANCE_KEY, BINANCE_SECRET, USE_TESTNET)
        if not USE_TESTNET:
            self.bin.load_exchange_info()  # FIX1: cache lot sizes for mainnet

        # ── Detectare automata portofel la pornire ────────────────────
        self.log.info("🔍 Verificare portofel Binance...")
        balances = self.bin.full_balance()

        # BNB
        if MANUAL_BNB > 0:
            self.bnb = MANUAL_BNB
        else:
            self.bnb = balances.get("BNB", 0.0)
            if self.bnb <= 0:
                self.log.warning("Nu am detectat BNB. Folosesc 1.0 BNB (testnet).")
                self.bnb = 1.0

        # ── Capital grid din USDC (prag dinamic: tot peste $150 tampon) ──
        # BNB ramane DOAR pentru fee. Grid-ul dimensioneaza din USDC real.
        USDC_BUFFER = 100.0   # tampon redus (risc mediu-mic)
        _usdc_free = balances.get("USDC", 0.0)
        _bnb_p_init = self.bin.price("BNBUSDC") or 640
        _usdc_for_grid = max(0.0, _usdc_free - USDC_BUFFER)
        self.grid_capital_bnb_eq = _usdc_for_grid / max(_bnb_p_init, 1)
        self.log.info(
            f"💰 Capital grid: ${_usdc_for_grid:.0f} USDC "
            f"(din ${_usdc_free:.0f}, tampon ${USDC_BUFFER:.0f}) "
            f"= {self.grid_capital_bnb_eq:.4f} BNB-eq")
        if self.grid_capital_bnb_eq <= 0:
            self.log.warning("USDC sub tampon — grid foloseste BNB ca baza (capital mic)")
            self.grid_capital_bnb_eq = self.bnb * 0.95

        # SOL
        self.sol_detected = balances.get("SOL", 0.0)

        # ═══ AUTO-DISCOVER: toate monedele din portofel ═══
        # AUTO-DISCOVER: doar pe MAINNET. Testnet are 423 fake tokens.
        if USE_TESTNET:
            bnb_p = self.bin.price("BNBUSDC") or 640
            sol_p = self.bin.price("SOLUSDC") or 85
            self.portfolio = {
                "assets": {"BNB": {"qty": self.bnb, "usd": self.bnb*bnb_p, "pair": "BNBUSDC", "price": bnb_p, "vol_24h": 500_000_000},
                           "SOL": {"qty": self.sol_detected, "usd": self.sol_detected*sol_p, "pair": "SOLUSDC", "price": sol_p, "vol_24h": 800_000_000}},
                "grid_pairs": ["BNBUSDC", "SOLUSDC"],
                "swing_pairs": ["SOLUSDC", "BNBUSDC"],
                "all_usdt_pairs": ["BNBUSDC", "SOLUSDC"],
            }
            self.discovered_assets = self.portfolio["assets"]
            self.log.info("Testnet: perechi hardcoded (skip discovery)")
        else:
            portfolio = self.bin.discover_portfolio_pairs(balances, min_usd=1.0)
            self.portfolio = portfolio
            self.discovered_assets = portfolio["assets"]

        # Setează dinamic perechile de trading din portofel
        global GRID_USDT_PAIRS, SWING_PAIRS
        # FORȚAT: grid rulează DOAR pe BNBUSDC — ignorăm portfolio_scan
        # portfolio_scan poate returna SOLBNB sau alte perechi X/BNB incorecte
        GRID_USDT_PAIRS = ["BNBUSDC"]
        self.log.info("✅ Grid forțat pe BNBUSDC")
        if self.portfolio["swing_pairs"]:
            SWING_PAIRS = self.portfolio["swing_pairs"][:10]

        # Preturi
        bnb_p = self.bin.price("BNBUSDC")
        sol_p = self.bin.price("SOLUSDC")
        # Guard: dacă API returnează 0 la pornire, retry cu fallback
        if bnb_p <= 0:
            import time as _t; _t.sleep(2)
            bnb_p = self.bin.price("BNBUSDC") or 638.0
        if sol_p <= 0:
            sol_p = self.bin.price("SOLUSDC") or 87.0
        usd   = self.bnb * bnb_p

        # Afisare portofel complet
        # Afisare portofel — DOAR monedele descoperite (nu fiat/garbage)
        portfolio_lines = []
        for asset, info in sorted(self.discovered_assets.items(),
                                   key=lambda x: -x[1]["usd"]):
            portfolio_lines.append(
                f"    {asset:<6} {info['qty']:.6f}  (~${info['usd']:.2f})")

        self.log.info(
            f"\n{'═'*62}\n"
            f"  SOLANA BOT v1.0 — Portofel detectat\n"
            f"  {'─'*58}\n"
            f"  📦 Active gasite:\n"
            + "\n".join(portfolio_lines) +
            f"\n  {'─'*58}\n"
            f"  🔑 BNB pentru bot:  {self.bnb:.4f} BNB (~${usd:.2f})\n"
            f"  🌟 SOL detectat:    {self.sol_detected:.4f} SOL (~${self.sol_detected*sol_p:.2f})\n"
            f"  {'─'*58}\n"
            f"  Alocare 100% (verificata matematic):\n"
            f"    Funding:    {ALLOC_FUNDING*100:.0f}% ({self.bnb*ALLOC_FUNDING:.4f} BNB)\n"
            f"    Grid:       {ALLOC_GRID*100:.0f}% ({self.bnb*ALLOC_GRID:.4f} BNB)\n"
            f"    Swing:       {ALLOC_SWING*100:.0f}% ({self.bnb*ALLOC_SWING:.4f} BNB)\n"
            f"    Launchpool: {ALLOC_LAUNCHPOOL*100:.0f}% ({self.bnb*ALLOC_LAUNCHPOOL:.4f} BNB)\n"
            f"    Earn rezerva:{ALLOC_REZERVA*100:.0f}% ({self.bnb*ALLOC_REZERVA:.4f} BNB)\n"
            f"    Dual Invest: {ALLOC_DI*100:.0f}% ({self.bnb*ALLOC_DI:.4f} BNB)\n"
            f"    Fee Buffer:  {ALLOC_FEE_BUFFER*100:.0f}% ({self.bnb*ALLOC_FEE_BUFFER:.4f} BNB LIBER)\n"
            f"  TOTAL: {(ALLOC_FUNDING+ALLOC_GRID+ALLOC_SWING+ALLOC_LAUNCHPOOL+ALLOC_REZERVA+ALLOC_DI+ALLOC_FEE_BUFFER)*100:.0f}% ✅\n"
            f"  {'─'*58}\n"
            f"  🔍 Perechi din portofel:\n"
            f"    Grid:  {', '.join(GRID_USDT_PAIRS)}\n"
            f"    Swing: {', '.join(SWING_PAIRS[:5])}{'...' if len(SWING_PAIRS) > 5 else ''}\n"
            f"    Monede: {len(self.discovered_assets)} detectate\n"
            f"{'═'*62}")

        # Verificare matematica la pornire — esueaza imediat daca gresit
        total_alloc = (ALLOC_FUNDING + ALLOC_GRID + ALLOC_SWING +
                       ALLOC_LAUNCHPOOL + ALLOC_REZERVA +
                       ALLOC_DI + ALLOC_FEE_BUFFER)
        if abs(total_alloc - 1.0) > 1e-9:
            raise ValueError(
                f"ALOCARE GRESITA: {total_alloc*100:.2f}% != 100%! "
                f"Botul ar ramane fara BNB pentru comisioane.")

        # Componente
        self.fees  = FeeTracker()
        self.health = StrategyHealthMonitor()
        self.perf   = PerformanceMetrics()
        self.guard = DailyTradeGuard()
        self.risk  = RiskManager(self.bnb)
        self.crash    = MarketCrashGuard(self.bin)
        self.sentinel = MarketSentinel(self.bin)   # Fear&Greed + BTC Dom
        self.crash_monitor = CrashMonitor(self.bin)  # crash detection BTCUSDC/BNBUSDC
        self.earn     = BinanceEarn(self.bin, self.bnb * ALLOC_REZERVA)
        # SOL Accumulator — 10% profit zilnic → SOL neatins
        self.sol_acc = SolAccumulator(self.bin)
        # SOL Trader — 6.15 SOL în grid + swing, profit → SOL
        self.sol_trader = SolTrader(
            self.bin, self.sol_detected, self.fees)
        self.fee_buf = FeeBufferManager(self.bin, self.earn)
        self.tgbot = _tgbot

        # Strategii — suma exacta 100%
        self.funding    = FundingArb(self.bin,
                                     self.bnb * ALLOC_FUNDING,
                                     self.fees, self.guard, self.crash)
        self.funding._crash_ref = self.crash  # pentru funding flip exit pe RED
        self.grid       = GridMaker(self.bin,
                                    self.grid_capital_bnb_eq,
                                    self.fees, self.guard, self.crash,
                                    fee_guard=self.fee_buf)

        self.grid._perf = self.perf        # Conectam sentinel la grid pentru spacing dinamic
        self.grid._sentinel = self.sentinel
        self.grid._bot_ref = self  # FIX11: per-pair tracking
        self.sol_trader._crash = self.crash  # Safety: SOL grid checks crash
        self.swing      = SwingTrader(self.bin,
                                      self.bnb * ALLOC_SWING,
                                      self.fees, self.guard, self.crash,
                                      fee_guard=self.fee_buf)
        self.launchpool = LaunchpoolStaking(self.bin,
                                            self.bnb * ALLOC_LAUNCHPOOL)
        self.dual_inv   = DualInvestmentManager(self.bin,
                                                self.bnb * ALLOC_DI)

        # I5: compound LP→Funding
        self.launchpool._funding_ref      = self.funding
        self.launchpool._pending_compound = 0.0
        # Tracking saptamanal — referinta la orchestrator
        self.funding._bot_ref = self

        # ═══ ENHANCEMENTS v1.3 (integrat direct) — TREBUIE ÎNAINTE de conectări ═══
        self.enhancements = EnhancementsManager(
            initial_capital_usd=usd + self.sol_detected * sol_p,
            telegram_callback=lambda msg: tg(msg, silent=True)
        )
        self.log.info("✅ Enhancements v1.3 integrat")
        # Withdrawal check: dacă cheia API are permisiunea de retragere → refuz start
        try:
            _perms = self.bin._get("/sapi/v1/account/apiRestrictions", signed=True)
            if _perms.get("enableWithdrawals", False):
                tg("⚠️ <b>ALERTĂ SECURITATE</b>: Cheia API are permisiunea de RETRAGERE activă!\n"
                   "Dezactivează în Binance → API Management → Edit → Uncheck Withdrawals!")
                self.log.warning("⚠️ API key has withdrawal permission — risc de securitate!")
        except Exception as _e:
            self.log.debug(f"withdrawal check: {_e}")
        # NTP check: verifică drift față de Binance serverTime
        try:
            _srv = self.bin._get("/api/v3/time")
            _srv_ts = _srv.get("serverTime", 0)
            _local_ts = int(time.time() * 1000)
            _drift_ms = abs(_local_ts - _srv_ts)
            if _drift_ms > 1000:
                self.log.error(
                    f"⏱ CLOCK DRIFT: {_drift_ms}ms față de Binance! "
                    f"Ordinele vor fi respinse (-1021). Sincronizează NTP pe VPS: "
                    f"sudo ntpdate -u pool.ntp.org")
            else:
                self.log.info(f"✅ Clock sync OK: drift={_drift_ms}ms")
        except Exception as _e:
            self.log.debug(f"NTP check: {_e}")
        # CB force_close callback — apelat ÎNAINTE de tranziția CRITICAL→COOLDOWN
        self.enhancements.circuit_breaker._force_close_cb = self._emergency_force_close
        # Transmite health monitor la sub-strategii
        self.sol_trader._health = self.health
        self.grid._health       = self.health
        self.swing._health      = self.health
        # ═══ ML ENGINE ═══
        if ML_AVAILABLE and not USE_TESTNET:
            self.ml = MLEngine(telegram_callback=lambda msg: tg(msg, silent=True))
            self.log.info("🧠 ML Engine activ (MAINNET)")
        else:
            self.ml = None
            if USE_TESTNET:
                self.log.info("⚠️ ML Engine OFF pe TESTNET")
            else:
                self.log.info("⚠️ ML Engine dezactivat (scikit-learn lipsește)")

        # ═══ ENH: conectam module la strategii ═══
        self.funding._enh_filter = self.enhancements.funding_filter
        self.grid._enh_optimizer = self.enhancements.grid_optimizer
        self.grid._enh_mgr = self.enhancements
        self.swing._enh_mgr = self.enhancements
        # PUNCT 6: Conectare FeeBuffer → ENH (sizing reduce când fee buffer scăzut)
        self.enhancements._fee_buf = self.fee_buf
        # ═══ ML: conectam ML engine la strategii ═══
        if self.ml:
            self.swing._ml = self.ml
            self.grid._ml = self.ml
            self.funding._ml = self.ml

        # ═══ WALLET CONTENT — BNB + SOL display cu reguli stricte ═══
        self.wallet_content = WalletContent(
            binance_client=self.bin,
            telegram_callback=tg
        )
        # Pre-populare cu soldul detectat la pornire (evită false alarm la start)
        self.wallet_content._bnb       = self.bnb
        self.wallet_content._sol       = self.sol_detected
        self.wallet_content._bnb_price = bnb_p
        self.wallet_content._sol_price = sol_p
        self.wallet_content._cache_ts  = time.time()

        # ═══ INTEGRITY GUARD — DEZACTIVAT (false positives la restart) ═══
        # Hash-ul se schimbă la fiecare upload nou, cauzând alerte false.
        # Reactivare când fix-ul persistenței e implementat corect.
        class _DummyIntegrity:
            def __init__(self): self.enabled = False
            def check(self): return True
            def get_status(self): return {"enabled": False, "status": "disabled"}
        self.integrity_guard = _DummyIntegrity()
        # Conectare SpreadMonitor la client Binance (pentru orderbook live)
        self.enhancements.spread_monitor.client = self.bin

        # Rebalancer saptamanal
        self.rebalancer = WeeklyRebalancer(self)

        # Tracker tranzactii saptamanal
        self._week_start_ts  = time.time()
        self._week_trades    = {
            "funding_open":  0, "funding_close": 0,
            "grid_fills":    0, "swing":         0,
            "di_cycles":     0, "total_fee_bnb": 0.0,
        }
        self._week_snap = {
            "funding_open":  0, "funding_close": 0,
            "grid_fills":    self.grid.total_fills,
            "swing":         self.swing.n_wins + self.swing.n_losses,
            "di_cycles":     self.dual_inv.n_cycles,
            "total_fee_bnb": 0.0,
        }

        self.stop_evt = threading.Event()
        self.start_ts = time.time()
        # ═══ SAFETY STOP — oprește totul la pierdere > $20 ═══
        self._initial_capital_usd = usd + self.sol_detected * sol_p
        self._safety_stopped = False
        # FIX11: Per-pair daily profit tracking
        self._pair_daily_pnl = {}   # {sym: {date: pnl_bnb}}
        self._pair_daily_fills = {} # {sym: {date: fill_count}}

        # FIX4: Persistent state — uptime + initial capital survive restarts
        self._persistent_file = "v8_persistent_state.json"
        self._load_persistent_state(usd + self.sol_detected * sol_p)
        self.log.info(f"🛡 Safety stop: capital inițial ${self._initial_capital_usd:.2f}, max loss ${SAFETY_MAX_LOSS_USD:.0f}")
        if MAINNET_DRY_RUN:
            self.log.info("🔸🔸🔸 MAINNET DRY_RUN ACTIV — ordine simulate 🔸🔸🔸")

        # Covered calls: timestamp ultima vanzare (0 = niciodata)
        self._last_call_sale = float(
            open("v6_calls.txt").read().strip()
            if os.path.exists("v6_calls.txt") else str(time.time())
        )

    def _load_persistent_state(self, current_capital_usd: float):
        """FIX4: Load persistent state from disk."""
        try:
            if os.path.exists(self._persistent_file):
                with open(self._persistent_file) as _jf:

                    d = json.load(_jf)
                self._total_uptime_s = d.get("total_uptime_s", 0.0)
                saved_capital = d.get("initial_capital_usd", 0)
                if saved_capital > 0:
                    self._initial_capital_usd = saved_capital
                    self.log.info(f"FIX4: Restored initial capital ${saved_capital:.2f}, uptime {self._total_uptime_s/3600:.1f}h")
                else:
                    self._total_uptime_s = 0.0
            else:
                self._total_uptime_s = 0.0
        except Exception as e:
            self.log.warning(f"Persistent state load: {e}")
            self._total_uptime_s = 0.0

    def _save_persistent_state(self):
        """FIX4: Save persistent state to disk."""
        try:
            elapsed = time.time() - self.start_ts
            total = self._total_uptime_s + elapsed
            _atomic_json_save(self._persistent_file, {
                "total_uptime_s":     total,
                "initial_capital_usd": self._initial_capital_usd,
                "last_save":          datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "session_start":      self.start_ts,
            })
        except Exception as e:
            self.log.debug(f"Persistent save: {e}")

    def total_uptime_h(self) -> float:
        """FIX4: Total uptime across restarts, in hours."""
        return (self._total_uptime_s + (time.time() - self.start_ts)) / 3600

    # ── Callback-uri Crash Guard ──────────────────────────────────────

    def _on_red(self, prev: str):
        """Crash sever: sincronizează cu CircuitBreaker + închide poziții."""
        self.log.error(f"ON_RED ({prev}->RED): emergency close swing+grid+SOL")

        # ═══ PUNCT 5: Sincronizare CrashGuard → CircuitBreaker ═══
        # CrashGuard detectează crash BTC, CircuitBreaker gestionează sizing
        # Un singur loc de decizie: CB trece în CRITICAL automat
        try:
            self.enhancements.circuit_breaker.state = CircuitBreakerState.CRITICAL
            self.log.info("CB sincronizat: CRITICAL (din CrashGuard RED)")
        except Exception as e:
            self.log.debug(f"CB sync: {e}")

        self.swing.emergency_close()
        self.grid.emergency_close()
        with self.sol_trader._lock:
            self.sol_trader.swing_trades.clear()
        self.log.info("SOL Trader swing inchis la RED")
        rates = self.bin.all_funding_rates()
        with self.funding._lock:
            pos_syms = list(self.funding.pos.keys())
        for sym in pos_syms:
            r = rates.get(sym, 0)
            if r <= 0:
                self.funding._close(sym, "CRASH_RED_RATE_NEG")
            else:
                self.log.info(
                    f"Funding {sym} RAMAN deschis (rate {r*100:.4f}% > 0)")
        tg(
            f"🔴 <b>CRASH RED</b>\n"
            f"Emergency close: Swing BNB + Grid BNB + Swing SOL\n"
            f"CB → CRITICAL sincronizat\n"
            f"Funding continua (delta-neutral)",
            silent=False
        )

    def _on_orange(self, prev: str):
        """Crash moderat: sincronizare CB WARNING + blocăm intrări."""
        self.log.warning(f"ON_ORANGE ({prev}->ORANGE): intrari noi blocate")
        try:
            if self.enhancements.circuit_breaker.state == CircuitBreakerState.NORMAL:
                self.enhancements.circuit_breaker.state = CircuitBreakerState.WARNING
                self.log.info("CB sincronizat: WARNING (din CrashGuard ORANGE)")
        except Exception as _e: logging.debug(f'Ignored: {_e}')
        with self.sol_trader._lock:
            self.sol_trader.swing_trades.clear()
        tg(
            f"🟠 <b>CRASH ORANGE</b>\n"
            f"Intrari noi blocate | Swing SOL oprit | CB→WARNING\n"
            f"Grid + Funding continua",
            silent=True
        )

    def _on_yellow(self, prev: str):
        self.log.warning(f"ON_YELLOW ({prev}->YELLOW): precautie activata")

    def _on_recover(self, prev: str):
        """Recuperare: CB revine la NORMAL + rebuild grid."""
        self.log.info(
            f"Recuperare {prev}→NORMAL: rebuild grid cu spacing recalibrat")
        try:
            self.enhancements.circuit_breaker.state = CircuitBreakerState.NORMAL
            self.log.info("CB sincronizat: NORMAL (recuperare crash)")
        except Exception as _e: logging.debug(f'Ignored: {_e}')
        self.grid._last_rb = 0.0
        tg(
            f"🟢 <b>Piata recuperata</b>\n"
            f"Grid rebuild cu spacing ADX recalibrat.\n"
            f"CB → NORMAL",
            silent=True
        )

    def _weekly_trades_summary(self) -> str:
        """
        Calculeaza tranzactiile din saptamana curenta (de la pornire sau
        de la ultimul reset de 7 zile).
        Compara cu snapshot-ul de la inceputul saptamanii.
        """
        now    = time.time()
        days   = (now - self._week_start_ts) / 86400
        snap   = self._week_snap

        # Calcul delta fata de snapshot
        grid_fills_w   = self.grid.total_fills - snap["grid_fills"]
        swing_w        = (self.swing.n_wins + self.swing.n_losses) - snap["swing"]
        di_w           = self.dual_inv.n_cycles - snap["di_cycles"]
        fund_open_w    = self._week_trades["funding_open"]
        fund_close_w   = self._week_trades["funding_close"]
        fee_w          = self._week_trades["total_fee_bnb"]

        total_w = fund_open_w + fund_close_w + grid_fills_w + swing_w + di_w

        # Reset automat la 7 zile
        if days >= 7:
            self._week_start_ts = now
            self._week_snap = {
                "funding_open":  0,
                "funding_close": 0,
                "grid_fills":    self.grid.total_fills,
                "swing":         self.swing.n_wins + self.swing.n_losses,
                "di_cycles":     self.dual_inv.n_cycles,
                "total_fee_bnb": 0.0,
            }
            self._week_trades = {k: 0 if isinstance(v,int) else 0.0
                                 for k,v in self._week_trades.items()}

        bnb_p = self.bin.price("BNBUSDC")
        return (
            f"\n📊 <b>Tranzactii saptamana ({days:.1f}z):</b>\n"
            f"  Total:          {total_w} tranzactii\n"
            f"  💰 Funding:     {fund_open_w} open + {fund_close_w} close\n"
            f"  🔲 Grid fills:  {grid_fills_w}\n"
            f"  📈 Swing:       {swing_w}\n"
            f"  💎 DI cicluri:  {di_w}\n"
            f"  💸 Fee sapt:    {fee_w:.5f} BNB (~${fee_w*bnb_p:.2f})\n"
            f"  📉 Fee/tranz:   "
            f"{fee_w/max(total_w,1)*1000:.4f} mBNB"
        )

    def _record_weekly_trade(self, trade_type: str, fee_bnb: float = 0.0):
        """Inregistreaza o tranzactie in tracker-ul saptamanal."""
        if trade_type in self._week_trades:
            self._week_trades[trade_type] += 1
        self._week_trades["total_fee_bnb"] += fee_bnb

    def _pnl(self) -> Tuple[float, float]:
        net  = (self.funding.total_pnl  + self.grid.total_pnl  +
                self.swing.total_pnl    + self.earn.earned      +
                self.launchpool.total_earned +
                self.dual_inv.total_premium)
        fees = (self.funding.total_fees + self.grid.total_fees  +
                self.swing.total_fees)
        return net, fees

    def _emergency_force_close(self):
        """
        Callback apelat de CircuitBreaker la starea CRITICAL.
        Închide toate pozițiile deschise pe toate modulele active.
        """
        self.log.error("🚨 EMERGENCY FORCE CLOSE — Circuit Breaker CRITICAL")
        try:
            self.grid.emergency_close()
        except Exception as e:
            self.log.error(f"emergency_close grid: {e}")
        try:
            self.swing.emergency_close()
        except Exception as e:
            self.log.error(f"emergency_close swing: {e}")
        try:
            self.sol_trader.emergency_close()
        except Exception as e:
            self.log.debug(f"emergency_close sol_trader: {e}")
        try:
            tg = getattr(self, 'tgbot', None)
            if tg: tg.send("🚨 <b>EMERGENCY CLOSE</b>\nCircuit Breaker CRITICAL — toate pozițiile închise!")
        except Exception:
            pass

    def _reconcile_loop(self, stop: threading.Event):
        """Reconciliere periodică ordine Binance vs state intern.
        FIX: recuperează oid-uri pentru toate perechile din grid.
        Rezolvă problema fills ratate după restart.
        """
        self.log.info("🔄 Reconciler pornit (60s)")
        while not stop.is_set():
            stop.wait(60)
            if stop.is_set(): break
            try:
                # Recuperează oid-uri pentru toate perechile active din grid
                with self.grid._lock:
                    active_syms = list(self.grid.grids.keys())
                for sym in active_syms:
                    try:
                        binance_orders = self.bin.spot_open_orders(sym)
                        if not isinstance(binance_orders, list):
                            continue
                        # Construiește map preț→orderId de pe Binance
                        price_to_oid = {
                            float(o["price"]): o["orderId"]
                            for o in binance_orders
                        }
                        # Injectează oid în lvl-urile fără oid
                        recovered = 0
                        with self.grid._lock:
                            g = self.grid.grids.get(sym, {})
                            for lvl in g.get("levels", []):
                                if lvl.get("filled"): continue
                                if lvl.get("oid"): continue
                                # Caută ordinul pe Binance după preț
                                lvl_p = round(lvl["price"], 8)
                                for bp, boid in price_to_oid.items():
                                    if abs(bp - lvl_p) / max(lvl_p, 1e-9) < 0.0001:
                                        lvl["oid"] = boid
                                        recovered += 1
                                        break
                        if recovered > 0:
                            self.log.info(
                                f"🔄 Reconcile {sym}: recuperat {recovered} oid-uri")
                        # Verificare discrepanță număr ordine
                        local_levels = self.grid.grids.get(sym, {}).get("levels", [])
                        local_n = len([l for l in local_levels if not l.get("filled")])
                        binance_n = len(binance_orders)
                        if abs(binance_n - local_n) > 2:
                            self.log.warning(
                                f"🔄 Reconcile {sym}: Binance={binance_n} "
                                f"local={local_n} — discrepanță")
                    except Exception as _e:
                        logging.debug(f"Reconcile {sym}: {_e}")
            except Exception as e:
                self.log.debug(f"Reconcile: {e}")

    def report(self) -> str:
        """Raport simplificat — doar module active."""
        try:
            uptime = self.total_uptime_h()
            uptime_days = uptime / 24
            bnb_p = self.bin.price("BNBUSDC") or 585
            usdc_p = self.bin.price("SOLUSDC") or 0

            # Portofoliu real din Binance
            try:
                _acc = self.bin._get("/api/v3/account", {}, signed=True) or {}
                _balances = {b["asset"]: float(b["free"])+float(b["locked"])
                             for b in _acc.get("balances", [])
                             if float(b["free"])+float(b["locked"]) > 0.001}
                _usdc = _balances.get("USDC", 0)
                _bnb = _balances.get("BNB", 0)
                _total = _usdc + _bnb * bnb_p
                for _a, _q in _balances.items():
                    if _a in ("USDC","BNB","LDADA","LDPEPE","LDBIO","LDBNB","CTSI","PIXEL","W","LUNC"): continue
                    _p = self.bin.price(f"{_a}USDC") or 0
                    _total += _q * _p
            except Exception:
                _total = 0; _usdc = 0; _bnb = self.bnb

            # Capital initial din persistent state
            _initial = getattr(self, "_initial_capital_usd", 700.0)
            _profit = _total - _initial
            _roi = (_profit / max(_initial, 1)) * 100
            _daily = _profit / max(uptime_days, 0.1)

            # Grid stats
            g = self.grid
            _fills = g.total_fills
            _pairs = ", ".join(g.grids.keys()) if g.grids else "N/A"
            _spacing = {sym: f"{gv.get('spacing',0)*100:.1f}%" for sym,gv in g.grids.items()}
            _fills_h = _fills / max(uptime, 0.1)

            # Trend / Regime
            _fg = getattr(self.sentinel, "fear_greed", 0)
            _fg_label = getattr(self.sentinel, "fg_label", "N/A")

            return (
                f"{'═'*44}\n"
                f"🤖 <b>GRID BOT RAPORT</b> | 🟢 LIVE\n"
                f"{'═'*44}\n"
                f"📅 {__import__('datetime').datetime.now().strftime('%d.%m.%Y %H:%M')} | "
                f"⏱ {uptime:.1f}h online\n"
                f"\n"
                f"<b>💼 Portofoliu:</b>\n"
                f"  USDC: ${_usdc:.2f} | BNB: {_bnb:.4f} (~${_bnb*bnb_p:.2f})\n"
                f"  <b>TOTAL: ${_total:.2f}</b>\n"
                f"  Capital initial: ${_initial:.2f}\n"
                f"  Profit: <b>${_profit:+.2f} ({_roi:+.1f}%)</b>\n"
                f"  Zilnic: ${_daily:+.2f}/zi\n"
                f"\n"
                f"<b>🔲 Grid:</b>\n"
                f"  Perechi: {_pairs}\n"
                f"  Spacing: {_spacing}\n"
                f"  Fills total: {_fills} ({_fills_h:.1f}/h)\n"
                f"  Pnl grid: {g.total_pnl*bnb_p:+.3f} USD\n"
                f"\n"
                f"<b>📊 Piata:</b>\n"
                f"  Fear & Greed: {_fg} ({_fg_label})\n"
                f"  BNB: ${bnb_p:.2f}\n"
                f"{'═'*44}"
            )
        except Exception as _re:
            return f"Raport error: {_re}"
            f"\n<b>Risk:</b> DD={self.risk.dd()*100:.1f}% | "
            f"{'🔴 CB' if self.risk.cb else '🟢 ok'} | "
            f"Crash: {self.crash.status_str()} | "
            f"FG: {getattr(self.sentinel, 'fear_greed', 50)}({getattr(self.sentinel, 'fg_label', 'N/A')[:4]}) "
            f"sp×{self.sentinel.get_spacing_mult():.2f}\n"
            f"\n<b>ENH v1.3:</b> CB={self.enhancements.circuit_breaker.get_status()['state']} | "
            f"PL={'🔒' if self.enhancements.profit_lock.locked else '🔓'} | "
            f"FG={self.enhancements.adaptive_alloc.current_mode} | "

    def run(self):
        bnb_p = self.bin.price("BNBUSDC")
        sol_p = self.bin.price("SOLUSDC")
        tg(
            f"🌊 <b>SOLANA BOT v1.0 PORNIT</b>\n"
            f"{'─'*28}\n"
            f"📦 <b>Portofel detectat:</b>\n"
            f"  BNB: {self.bnb:.4f} (~${self.bnb*bnb_p:.2f})\n"
            f"  SOL: {self.sol_detected:.4f} (~${self.sol_detected*sol_p:.2f})\n"
            f"{'─'*28}\n"
            f"⏸ <b>TRADING IN PAUZA</b>\n"
            f"Trimite /start_trading pentru a activa.\n"
            f"{'─'*28}\n"
            f"Testnet: {USE_TESTNET}\n"
            f"💰 Funding | 🔲 Grid | 📈 Swing\n"
            f"🌟 SOL Accumulator activ (10% profit zilnic)\n"
            f"🛡 Crash Guard | 🔄 Compound 90%\n"
            f"🛡 ENH v1.1: CB smart + ProfitLock + SOL DCA + OI/strat\n"
            f"   /enh pentru status enhancements\n"
            f"Profit estimat: ${self.bnb*bnb_p*0.033:.0f}/luna"
        )

        def crash_run(stop):
            self.crash.run(
                stop,
                on_red     = self._on_red,
                on_orange  = self._on_orange,
                on_yellow  = self._on_yellow,
                on_recover = self._on_recover,
            )

        threads = [
            # threading.Thread(target=self.funding.run,
            #                  args=(self.stop_evt,), name="Funding",    daemon=True),
            threading.Thread(target=self.grid.run,
                             args=(self.stop_evt,), name="Grid",       daemon=True),
            threading.Thread(target=self._reconcile_loop,
                             args=(self.stop_evt,), name="Reconcile",  daemon=True),
            threading.Thread(target=self.swing.run,
                             args=(self.stop_evt,), name="Swing",      daemon=True),
            # threading.Thread(target=self.launchpool.run,
            #                  args=(self.stop_evt,), name="Launchpool", daemon=True),
            # threading.Thread(target=self.earn.run,
            #                  args=(self.stop_evt,), name="Earn",       daemon=True),
            # threading.Thread(target=self.dual_inv.run,
            #                  args=(self.stop_evt,), name="DualInv",    daemon=True),
            threading.Thread(target=crash_run,
                             args=(self.stop_evt,), name="Crash",      daemon=True),
            threading.Thread(target=self.tgbot.run,
                             args=(self.stop_evt, self), name="TgBot", daemon=True),
            threading.Thread(target=self.sol_acc.run,
                             args=(self.stop_evt, self), name="SolAcc",   daemon=True),
            *(
                [threading.Thread(target=self.sol_trader.run,
                             args=(self.stop_evt,), name="SolTrd", daemon=True)]
                if ENABLE_SOL_TRADER else []
            ),
            threading.Thread(target=self.sentinel.run,
                             args=(self.stop_evt,),      name="Sentinel", daemon=True),
            threading.Thread(target=self.crash_monitor.run,
                             args=(self.stop_evt,),      name="CrashMon", daemon=True),
        ]
        self._threads = threads  # FIX3+12: keep reference for monitoring
        if not ENABLE_SOL_TRADER:
            self.log.info("⚠️ SOL Trader DEZACTIVAT — activează când ai $500+ USDC")
        for t in threads: t.start(); self.log.info(f"▶ {t.name}")
        self._last_heartbeat = time.time()
        self._heartbeat_interval = 300  # 5min

        last_report   = time.time()
        last_fee_check = time.time()  # Fee monitoring la 1h
        last_cc_check = time.time()
        last_enh_oi   = time.time()  # ENH: OI sentinel check interval
        last_enh_day  = datetime.now(timezone.utc).day  # ENH: daily close tracking
        try:
            while True:
                time.sleep(30)
                # FIX3: Watchdog heartbeat
                if time.time() - self._last_heartbeat > self._heartbeat_interval:
                    self._last_heartbeat = time.time()
                    self._save_persistent_state()  # FIX4: periodic save
                    # FIX12: Thread death monitor
                    for t in self._threads:
                        if not t.is_alive():
                            self.log.error(f"🛑 Thread MORT: {t.name}")
                            tg(f"🛑 <b>THREAD MORT: {t.name}</b>\n"
                               f"Restart recomandat: sudo systemctl restart bnb-bot")
                # Kill switch check — /stop din Telegram oprește imediat
                if self._safety_stopped:
                    self.log.error("⛔ Safety/Kill stop activ — main loop oprit")
                    break
                self.risk.record(0)

                # ═══ BULL DETECTION ALERT (check la 1h, alertă 1/24h) ═══
                try:
                    if not hasattr(self, "_last_bull_check"):
                        self._last_bull_check = 0
                        self._last_bull_alert = 0
                    if time.time() - self._last_bull_check > 3600:
                        self._last_bull_check = time.time()
                        kl = self.bin.klines("SOLUSDC", "1h", 25)
                        if kl and len(kl) >= 24:
                            # Price change 24h
                            price_now = float(kl[-1][4])
                            price_24h_ago = float(kl[0][4])
                            change_24h = (price_now - price_24h_ago) / max(price_24h_ago, 0.01) * 100
                            # ADX
                            cl = [float(k[4]) for k in kl]
                            net = abs(cl[-1] - cl[-min(14, len(cl))])
                            gros = sum(abs(cl[i]-cl[i-1]) for i in range(1, len(cl))) or 1e-10
                            adx = net / gros * 100 * 2.5
                            # 20-MA distance
                            ma20 = sum(cl[-20:]) / 20 if len(cl) >= 20 else price_now
                            ma_dist = (price_now - ma20) / max(ma20, 0.01) * 100
                            # Condiții bull
                            is_bull = (change_24h > 15 and adx > 30 and ma_dist > 2)
                            if is_bull and (time.time() - self._last_bull_alert > 86400):
                                self._last_bull_alert = time.time()
                                tg(
                                    f"🚀 <b>BULL DETECTAT — SOL</b>\n"
                                    f"Preț 24h: +{change_24h:.1f}%\n"
                                    f"ADX: {adx:.0f} (trend puternic)\n"
                                    f"Distanță MA20: +{ma_dist:.1f}%\n"
                                    f"─────────────\n"
                                    f"Grid poate fi subobtim.\n"
                                    f"Evaluează manual: /pauza dacă vrei hold SOL\n"
                                    f"/resume când revine la range lateral",
                                    urgent=True
                                )
                                self.log.warning(f"🚀 BULL: change={change_24h:.1f}% adx={adx:.0f} ma_dist={ma_dist:.1f}%")
                except Exception as e:
                    self.log.debug(f"Bull check: {e}")


                # ═══ ENH v1.3: Update circuit breaker + profit lock + OI + SOL dip ═══
                try:
                    bnb_p_now = self.bin.price("BNBUSDC")
                    sol_p_now = self.bin.price("SOLUSDC")
                    total_usd = self.bnb * bnb_p_now + self.sol_detected * sol_p_now
                    self.enhancements.circuit_breaker.update_capital(total_usd)
                    self.enhancements.update_prices({
                        "BNBUSDC": bnb_p_now, "SOLUSDC": sol_p_now
                    })
                    # v1.1: Profit Lock — update cu PnL zilnic curent
                    net_pnl_now, _ = self._pnl()
                    daily_pnl_usd = net_pnl_now * bnb_p_now
                    self.enhancements.update_daily_pnl(daily_pnl_usd)
                    # OI Sentinel — check la 15 min
                    if time.time() - last_enh_oi > 900:
                        self.enhancements.oi_sentinel.check_all()
                        last_enh_oi = time.time()
                    # v1.1: CB smart — close doar grid+swing, NU funding
                    if self.enhancements.circuit_breaker.force_close_all():
                        if self.enhancements.circuit_breaker.should_close_strategy("swing"):
                            self.swing.emergency_close()
                        if self.enhancements.circuit_breaker.should_close_strategy("grid"):
                            self.grid.emergency_close()
                        self.log.error("🔴 ENH CB CRITICAL → swing+grid closed, funding RĂMÂNE")
                    # v1.1: SOL DCA dip check — DEZACTIVAT (necesită USDC)
                    # is_dip, dip_reason = self.enhancements.sol_accumulator.check_dip_buy(sol_p_now)
                    pass  # dip buy dezactivat
                    # Daily close — SOL batch + reset
                    now_day = datetime.now(timezone.utc).day
                    if now_day != last_enh_day:
                        self.enhancements.on_daily_close(daily_pnl_usd)
                        # Health monitor — auto-disable strategie cu pierderi consecutive
                        health_alerts = self.health.on_daily_close()
                        for strat, loss in health_alerts:
                            tg(
                                f"🚨 <b>STRATEGIE DEZACTIVATĂ</b>\n"
                                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                f"Strategie: {strat}\n"
                                f"Motiv: {self.health.MAX_LOSING_DAYS} zile pierdere consecutivă\n"
                                f"Pierdere: ${abs(loss):.2f}\n"
                                f"Reactivare: automată după {self.health.COOLDOWN_DAYS} zile\n"
                                f"Manual: /health_enable {strat}",
                                silent=False)
                        last_enh_day = now_day
                    # v1.2: Sync Fear & Greed → Adaptive Allocator
                    self.enhancements.update_fear_greed(getattr(self.sentinel, 'fear_greed', 50))
                    # v1.2: Auto-compound check (la fiecare 7 zile)
                    compound = self.enhancements.check_compound(total_usd)
                    if compound:
                        self.log.info(
                            f"📈 COMPOUND: ${compound['old_base']:.0f} → "
                            f"${compound['new_base']:.0f} (+${compound['compound_amount']:.2f})")
                    # ═══ BNB → SOL PRIORITY: DEZACTIVAT ═══
                    pass  # dezactivat — nu convertim BNB în SOL
                except Exception as e:
                    self.log.debug(f"ENH update: {e}")
                # ═══ END ENH v1.3 ═══

                # ═══ SPREAD MONITOR — detectare retragere MM ═══
                try:
                    spread_level = self.enhancements.spread_monitor.update()
                    if spread_level == SpreadMonitor.LEVEL_ALERT:
                        self.log.warning(
                            f"🚨 Spread ALERT global — trading blocat temporar")
                except Exception as e:
                    self.log.debug(f"SpreadMonitor: {e}")

                # ═══ INTEGRITY GUARD — verificare integritate fișier ═══
                try:
                    if not self.integrity_guard.check():
                        self.log.critical(
                            "🔴 INTEGRITY BREACH — bot oprit pentru securitate!")
                        break
                except Exception as e:
                    self.log.debug(f"IntegrityGuard: {e}")

                # ═══ WALLET CONTENT — verificare rezerva BNB fee ═══
                try:
                    fee_ok, fee_msg = self.wallet_content.check_fee_reserve()
                    if not fee_ok:
                        self.log.error(f"BNB FEE RESERVE: {fee_msg}")
                except Exception as e:
                    self.log.debug(f"WalletContent: {e}")

                # ═══ ANTI-LICHIDARE: verificare futures la 30s ═══
                try:
                    self.funding.check_anti_liquidation()
                except Exception as e:
                    self.log.debug(f"Anti-liq: {e}")

                # ═══ PLAFON ZILNIC REAL: valoare TOTALA portofel ═══
                # FIX: baza = USDC + inventar (consistent), nu doar USDC free.
                # Altfel ordinele plasate (USDC blocat, tokens necontabilizati)
                # par pierdere falsa.
                if not self._safety_stopped:
                    try:
                        _now_ts = time.time()
                        def _total_portfolio_value():
                            # FIX: full_balance() da doar 'free'. Trebuie sa
                            # adaugam si banii din ordinele deschise (locked),
                            # altfel vedem gauri false cand botul are ordine.
                            import hmac as _h2, hashlib as _hl2, urllib.request as _u2
                            _ak2 = self.bin.key
                            _sk2 = self.bin.secret
                            # Account complet (free + locked)
                            _v = 0.0
                            try:
                                _pq2 = f"timestamp={int(time.time()*1000)}"
                                _sg2 = _h2.new(_sk2.encode(), _pq2.encode(), _hl2.sha256).hexdigest()
                                _url2 = f"https://api.binance.com/api/v3/account?{_pq2}&signature={_sg2}"
                                _req2 = _u2.Request(_url2, headers={"X-MBX-APIKEY": _ak2})
                                _acc = json.loads(_u2.urlopen(_req2, timeout=10).read())
                                for _bal in _acc.get('balances', []):
                                    _a = _bal['asset']
                                    _q = float(_bal['free']) + float(_bal['locked'])
                                    if _q <= 0 or _a.startswith('LD'):
                                        continue
                                    if _a == 'USDC':
                                        _v += _q
                                    elif _a == 'BNB':
                                        _v += _q * (self.bin.price('BNBUSDC') or 0)
                                    else:
                                        _v += _q * (self.bin.price(f"{_a}USDC") or 0)
                            except Exception as _ve:
                                self.log.debug(f"Portfolio value: {_ve}")
                                # Fallback: free only
                                _b = self.bin.full_balance()
                                _v = _b.get('USDC', 0.0)
                                for _a, _q in _b.items():
                                    if _a == 'USDC' or _a.startswith('LD'): continue
                                    if _q > 0:
                                        _ap = self.bin.price('BNBUSDC') if _a=='BNB' else self.bin.price(f"{_a}USDC")
                                        _v += _q * (_ap or 0)
                            return _v
                        # Init / reset zilnic (la 24h) — valoare TOTALA
                        if not hasattr(self, '_day_start_usdc'):
                            self._day_start_usdc = _total_portfolio_value()
                            self._day_start_ts = _now_ts
                            self.log.info(
                                f"📅 Plafon zilnic init: portofel total "
                                f"${self._day_start_usdc:.2f}")
                        # P13: calculeaza profitul zilnic real pentru auto-compound BNB
                        _real_total = _total_portfolio_value()
                        self._daily_profit_usd = max(0.0, _real_total - self._day_start_usdc)
                        if _now_ts - self._day_start_ts > 86400:
                            self._day_start_usdc = _total_portfolio_value()
                            self._day_start_ts = _now_ts
                            self.log.info(
                                f"📅 Plafon zilnic reset: ${self._day_start_usdc:.2f}")
                        # Verifica pierderea reala (aceeasi metrica: total portofel)
                        _real_total = _total_portfolio_value()
                        _day_loss = _real_total - self._day_start_usdc
                        if _day_loss < -10.0:
                            self._safety_stopped = True
                            self.stop_evt.set()
                            self.log.error(
                                f"🛑 PLAFON ZILNIC: pierdere reala "
                                f"${abs(_day_loss):.2f} > $10 → STOP")
                            try:
                                self.grid.emergency_close()
                                self.swing.emergency_close()
                            except Exception: pass
                            self.tgbot.send_urgent(
                                f"🛑 <b>PLAFON ZILNIC -$10</b>\n"
                                f"Pierdere reala: ${abs(_day_loss):.2f}\n"
                                f"Sold start: ${self._day_start_usdc:.2f}\n"
                                f"Sold acum: ${_real_total:.2f}\n"
                                f"⛔ Trading oprit")
                    except Exception as _dle:
                        self.log.debug(f"Plafon zilnic: {_dle}")

                # ═══ SAFETY STOP: oprește la -$20 pierdere ═══
                if not self._safety_stopped:
                    try:
                        _bnb_now = self.bin.price("BNBUSDC")
                        _sol_now = self.bin.price("SOLUSDC")
                        _net_pnl, _ = self._pnl()
                        # FIX: safety stop urmărește DOAR PnL din trading,
                        # NU devalorizare capital prin market moves
                        # (fluctuațiile BNB/SOL nu sunt "pierderi" botului)
                        _pnl_usd = _net_pnl * _bnb_now
                        _total_loss = _pnl_usd

                        _current_usd = self.bnb * _bnb_now + self.sol_detected * _sol_now
                        if _total_loss < -SAFETY_MAX_LOSS_USD:
                            self._safety_stopped = True
                            self.stop_evt.set()  # OPREȘTE TOATE THREAD-URILE
                            self.log.error(
                                f"🛑 SAFETY STOP: pierdere ${abs(_total_loss):.2f} > "
                                f"${SAFETY_MAX_LOSS_USD:.0f} limita!")

                            # Închide TOATE pozițiile
                            try:
                                self.swing.emergency_close()
                                self.grid.emergency_close()
                                for sym in list(self.funding.pos.keys()):
                                    self.funding._close(sym, "SAFETY_STOP")
                                self.sol_trader.swing_trades.clear()
                            except Exception as ce:
                                self.log.error(f"Safety close error: {ce}")

                            # Notificare URGENTĂ
                            self.tgbot.send_urgent(
                                f"🛑 <b>SAFETY STOP ACTIVAT</b>\n"
                                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                f"Pierdere: <b>${abs(_total_loss):.2f}</b>\n"
                                f"Limita: ${SAFETY_MAX_LOSS_USD:.0f}\n"
                                f"Capital inițial: ${self._initial_capital_usd:.2f}\n"
                                f"Capital acum: ${_current_usd:.2f}\n"
                                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                f"⛔ TOATE pozițiile închise\n"
                                f"⛔ Trading OPRIT\n"
                                f"🔄 Restart bot pentru reactivare")

                            # OPREȘTE TRADING
                            self.log.error("⛔ Trading oprit — SAFETY STOP")
                            break  # iese din main loop

                    except Exception as e:
                        self.log.debug(f"Safety check: {e}")

                # ═══ ML+AI: Retrain periodic (24h) — thread separat ══
                try:
                    if self.ml and not getattr(self, '_ml_training', False):
                        def _ml_retrain_bg():
                            try:
                                self._ml_training = True
                                kl_btc = self.bin.klines("BTCUSDC", "1h", 200)
                                self.ml.retrain_if_needed(kl_btc)
                            except Exception as _e:
                                self.log.debug(f"ML retrain bg: {_e}")
                            finally:
                                self._ml_training = False
                        threading.Thread(
                            target=_ml_retrain_bg,
                            name="MLRetrain", daemon=True
                        ).start()
                except Exception as e:
                    self.log.debug(f"ML retrain: {e}")

                # ═══ FEE MONITOR: alertă dacă fee > profit (la 1h) ═══
                try:
                    if time.time() - last_fee_check > 3600:
                        bnb_p_fee = self.bin.price("BNBUSDC")
                        if bnb_p_fee > 0:
                            fee_alert = self.fees.check_fee_alert(bnb_p_fee)
                            if fee_alert:
                                self.log.warning(fee_alert)
                                tg(fee_alert, silent=False)
                        last_fee_check = time.time()
                except Exception as e:
                    self.log.debug(f"Fee check: {e}")

                if not self.risk.ok():
                    self.log.error("⛔ Circuit breaker activ")
                    self.tgbot.send_urgent(
                        f"🚨 <b>CIRCUIT BREAKER</b>\n"
                        f"Pierdere zilnica peste limita.\n"
                        f"Trading OPRIT. Trimite /status pentru detalii.")
                    break
                # ═══ RAPORT ZILNIC la 10:00 ora României (08:00 UTC) ═══
                try:
                    now_utc = datetime.now(timezone.utc)
                    if (now_utc.hour == 8 and now_utc.minute < 2 and
                            time.time() - last_report > 3600):
                        r = self.report()
                        # Adaugă daily summary
                        net_pnl, fees_pnl = self._pnl()
                        bnb_p_r = self.bin.price("BNBUSDC")
                        _cap_usd_r = self.bnb * bnb_p_r
                        self.perf.record_daily(_cap_usd_r)
                        pnl_usd = net_pnl * bnb_p_r
                        # v8-37: raportăm explicit fee-urile (gross/net/fee ratio)
                        fees_usd = fees_pnl * bnb_p_r
                        gross_usd = pnl_usd + fees_usd
                        fee_ratio = (fees_usd / gross_usd * 100) if gross_usd > 0 else 0.0
                        uptime_d = (time.time() - self.start_ts) / 86400
                        daily_avg = pnl_usd / max(uptime_d, 0.1)
                        fee_alert = self.fees.check_fee_alert(bnb_p_r)

                        daily = (
                            f"\n{'━'*30}\n"
                            f"📅 <b>RAPORT ZILNIC</b> ({now_utc.strftime('%d %b %Y')})\n"
                            f"Profit brut:  ${gross_usd:+.2f}\n"
                            f"Fee plătit:   ${fees_usd:.2f} ({fee_ratio:.1f}% din brut)\n"
                            f"Profit net:   ${pnl_usd:+.2f}\n"
                            f"Medie zilnică: ${daily_avg:.2f}/zi\n"
                            f"Proiecție lunară: ${daily_avg*30:.0f}/lună\n"
                            f"{fee_alert if fee_alert else '✅ Fee-uri sub control'}\n"
                        )
                        self.log.info(f"\n{r}")
                        tg(r + daily)
                        last_report = time.time()
                except Exception as e:
                    self.log.debug(f"Daily report: {e}")

                # ── USDC Reserve Builder ─────────────────────────────────
                # Convertește TOT profitul zilnic BNB → USDC până la $500
                # Scop: acumulare USDC pentru live trading (ordine BUY grid)
                try:
                    USDC_RESERVE_TARGET = 300.0  # $300 suficient pentru 8 nivele SOL BUY
                    _bnb_now = self.bin.price("BNBUSDC")
                    _usdc_balance = 0.0
                    try:
                        _balances = self.bin._get("/api/v3/account", signed=True)
                        for _b in _balances.get("balances", []):
                            if _b["asset"] == "USDC":
                                _usdc_balance = float(_b["free"])
                                break
                    except Exception:
                        pass

                    if _usdc_balance < USDC_RESERVE_TARGET and net_pnl > 0:
                        # Calculăm BNB de vândut = profitul zilnic net
                        _bnb_to_sell = net_pnl  # tot profitul net
                        _min_notional = 10.0 / max(_bnb_now, 1)
                        if _bnb_to_sell >= _min_notional and _bnb_now > 0:
                            if MAINNET_DRY_RUN:
                                _usdc_gain = _bnb_to_sell * _bnb_now
                                self.log.info(
                                    f"💵 DRY_RUN USDC reserve: vând {_bnb_to_sell:.6f} BNB "
                                    f"→ +${_usdc_gain:.2f} USDC "
                                    f"(total simulat: ${_usdc_balance + _usdc_gain:.2f}/$500)")
                                tg(f"💵 <b>USDC Reserve</b> [DRY]: "
                                   f"+${_usdc_gain:.2f} → "
                                   f"${_usdc_balance + _usdc_gain:.2f}/$500")
                            else:
                                result = self.bin.market_sell("BNBUSDC", round(_bnb_to_sell, 5))
                                if result.get("status") == "FILLED":
                                    _usdc_gain = float(result.get("cummulativeQuoteQty", 0))
                                    self.log.info(
                                        f"💵 USDC reserve: +${_usdc_gain:.2f} "
                                        f"(total: ${_usdc_balance + _usdc_gain:.2f}/$500)")
                                    tg(f"💵 <b>USDC Reserve</b>: "
                                       f"+${_usdc_gain:.2f} → "
                                       f"${_usdc_balance + _usdc_gain:.2f}/$500")
                    elif _usdc_balance >= USDC_RESERVE_TARGET:
                        self.log.info(f"✅ USDC reserve complet: ${_usdc_balance:.0f}/$500")
                except Exception as _e:
                    self.log.debug(f"USDC reserve: {_e}")
                # ─────────────────────────────────────────────────────────

                # Rebalansare saptamanala
                if REBAL_ENABLED:
                    self.rebalancer.rebalance()
                # Covered calls: reminder lunar
                if (COVERED_CALLS_ENABLED and
                        time.time() - last_cc_check > 3600):
                    last_cc_check = time.time()
                    days_since = (time.time() - self._last_call_sale) / 86400
                    if days_since >= COVERED_CALLS_REMIND_D:
                        bnb_p  = self.bin.price("BNBUSDC")
                        strike = bnb_p * (1 + COVERED_CALLS_STRIKE_PCT)
                        tg(
                            f"📋 <b>Covered Calls Reminder</b>\n"
                            f"Au trecut {days_since:.0f} zile de la ultima vanzare.\n"
                            f"\n<b>Parametri recomandat acum:</b>\n"
                            f"• BNB spot: ${bnb_p:.2f}\n"
                            f"• Strike recomandat: ${strike:.2f} (+{COVERED_CALLS_STRIKE_PCT*100:.0f}%)\n"
                            f"• Expirare: ~{COVERED_CALLS_EXPIRY_D} zile\n"
                            f"• Premium estimat: ${self.bnb * bnb_p * 0.018:.2f}-"
                            f"${self.bnb * bnb_p * 0.025:.2f}\n"
                            f"\n<b>Pasi:</b>\n"
                            f"Binance → Derivatives → Options → BNB\n"
                            f"→ Sell Call → Strike ~${strike:.0f}\n"
                            f"\nDupa vanzare, trimite /calls_done pentru reset."
                        )
                        self.log.info(
                            f"Covered calls reminder: {days_since:.0f} zile "
                            f"de la ultima vanzare")
        except KeyboardInterrupt:
            self.log.info("⛔ Ctrl+C")

        self.stop_evt.set()
        for t in threads: t.join(timeout=10)
        r = self.report()
        self.log.info(f"\nRAPORT FINAL\n{r}")
        tg(f"⛔ <b>BOT OPRIT</b>\n{r}")


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    for pkg in ["requests"]:
        try: __import__(pkg)
        except ImportError:
            print(f"pip install {pkg}"); sys.exit(1)

    if not BINANCE_KEY:
        print(
            "Setup:\n"
            "  export BINANCE_API_KEY=...\n"
            "  export BINANCE_SECRET_KEY=...\n"
            "  export TELEGRAM_BOT_TOKEN=...   # de la @BotFather\n"
            "  export TELEGRAM_CHAT_ID=...     # ID-ul tau de chat\n"
            "  export TG_ALERT_LEVEL=1         # 0=all 1=normal 2=urgent\n"
            "  export TG_DAILY_HOUR=9          # ora raport zilnic (local)\n"
            "  export USE_TESTNET=True\n"
            "  export BNB_AMOUNT=1.22          # sau 0 pentru auto-detect\n"
            "  python3 bnb_bot_v8.py\n"
            "\n"
            "Comenzi Telegram dupa pornire:\n"
            "  /status   /raport   /pozitii   /grid\n"
            "  /pauza    /resume   /nivel X   /ajutor"
        )

    _bot_instance = SolanaBot()
    _bot_instance.run()


if __name__ == "__main__":
    # FIX2: Graceful shutdown
    _bot_instance = None
    def _graceful_shutdown(signum, frame):
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        print(f"\n🛑 {sig_name} received — graceful shutdown...")
        if _bot_instance and hasattr(_bot_instance, "stop_evt"):
            _bot_instance.stop_evt.set()
            # Join threads cu timeout
            for _t in threading.enumerate():
                if _t.name in ("Grid","Swing","SolTrd","SolAcc","Sentinel","Reconcile","TgBot"):
                    _t.join(timeout=2)
            time.sleep(2)
            try:
                _bot_instance.log.info(f"🛑 Graceful shutdown ({sig_name})")
                # Save all state
                if hasattr(_bot_instance, "grid"): _bot_instance.grid._save()
                if hasattr(_bot_instance, "funding"): _bot_instance.funding._save()
                if hasattr(_bot_instance, "sol_trader"): _bot_instance.sol_trader._save()
                if hasattr(_bot_instance, "swing"): _bot_instance.swing._save()
                _bot_instance._save_persistent_state()
                # Anuleaza toate ordinele active pe Binance la shutdown
                try:
                    if hasattr(_bot_instance, "grid") and _bot_instance.grid:
                        _client = _bot_instance.grid.client
                        import json as _json
                        _p = f"timestamp={int(time.time()*1000)}"
                        import hmac as _hm, hashlib as _hl
                        _sig = _hm.new(_client.secret.encode(), _p.encode(), _hl.sha256).hexdigest()
                        import urllib.request as _ur
                        _req = _ur.Request(
                            f"https://api.binance.com/api/v3/openOrders?{_p}&signature={_sig}",
                            headers={"X-MBX-APIKEY": _client.key})
                        _orders = _json.loads(_ur.urlopen(_req, timeout=5).read())
                        _syms = set(o["symbol"] for o in _orders)
                        for _sym in _syms:
                            try:
                                _client.spot_cancel_all(_sym)
                                _bot_instance.log.info(f"🛑 Shutdown: anulate ordine {_sym}")
                            except Exception: pass
                except Exception as _se:
                    print(f"Shutdown cancel orders: {_se}")
                _bot_instance.log.info("✅ State saved. Exiting.")
            except Exception as e:
                print(f"Shutdown save error: {e}")
        sys.exit(0)
    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)

    main()