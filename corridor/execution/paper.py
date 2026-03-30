from __future__ import annotations

import csv
import json
import math
import socket
import time
from configparser import ConfigParser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from corridor.config import CorridorConfig
from corridor.models import ActionRecord, ActionType, CorridorState, Regime
from corridor.options.butterfly_selector import ButterflyCandidate, select_butterflies
from corridor.options.chain_loader import IBOptionChainLoader
from corridor.options.combo_builder import ComboLegSpec, build_butterfly_combo
from corridor.strategy.center_estimator import CenterEstimator
from corridor.strategy.corridor_state_machine import CorridorStateMachine
from corridor.strategy.regime import RangeRegimeDetector


try:
    from ib_insync import IB, LimitOrder, Option, Stock, util
except ImportError:  # pragma: no cover - optional dependency
    IB = None
    LimitOrder = None
    Option = None
    Stock = None
    util = None


@dataclass(slots=True)
class PaperRunnerConfig:
    symbol: str = "SPY"
    mode: str = "delayed"
    host: str = "127.0.0.1"
    port: int = 4001
    client_id: int = 71
    chain_client_id_offset: int = 100
    quantity: int = 1
    poll_seconds: int = 30
    history_days: int = 5
    start_flat: bool = True
    paper_execution: bool = False
    once: bool = False
    check_only: bool = False
    output_dir: Path = Path("corridor_outputs") / "paper_runner"
    order_tif: str = "DAY"
    log_prefix: str = "paper"


@dataclass(slots=True)
class ManagedPosition:
    layer_id: int
    candidate: ButterflyCandidate
    quantity: int
    opened_at: pd.Timestamp
    open_limit: float
    open_status: str
    source_action: str
    order_id: Optional[int] = None
    close_order_id: Optional[int] = None
    closed_at: Optional[pd.Timestamp] = None
    close_limit: Optional[float] = None
    close_status: str = ""


def _require_ib() -> None:
    if IB is None or LimitOrder is None or Option is None or Stock is None or util is None:
        raise RuntimeError("ib_insync is required for run_paper_corridor.py")


def _ensure_utc_timestamp(value: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _timeframe_to_delta(label: str) -> pd.Timedelta:
    value, unit = label.strip().split(maxsplit=1)
    amount = int(value)
    unit = unit.lower()
    if unit.startswith("min"):
        return pd.Timedelta(minutes=amount)
    if unit.startswith("hour"):
        return pd.Timedelta(hours=amount)
    if unit.startswith("day"):
        return pd.Timedelta(days=amount)
    raise ValueError(f"Unsupported timeframe: {label}")


class CsvEventLogger:
    """Append runtime transitions, actions, and orders to CSV files."""

    def __init__(self, output_dir: Path, prefix: str) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths = {
            "transitions": self.output_dir / f"{self.prefix}_transitions.csv",
            "actions": self.output_dir / f"{self.prefix}_actions.csv",
            "orders": self.output_dir / f"{self.prefix}_orders.csv",
            "state": self.output_dir / f"{self.prefix}_state.json",
        }

    def write_transition(self, record: dict[str, Any]) -> None:
        self._append(self.paths["transitions"], record)

    def write_action(self, record: dict[str, Any]) -> None:
        self._append(self.paths["actions"], record)

    def write_order(self, record: dict[str, Any]) -> None:
        self._append(self.paths["orders"], record)

    def write_state(self, payload: dict[str, Any]) -> None:
        self.paths["state"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _append(path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if needs_header:
                writer.writeheader()
            writer.writerow(row)


class PaperCorridorRunner:
    """Run the corridor logic on live IBKR bars and optionally place paper combo orders."""

    def __init__(self, corridor_cfg: CorridorConfig, runner_cfg: PaperRunnerConfig) -> None:
        _require_ib()
        self.cfg = corridor_cfg
        self.runner_cfg = runner_cfg
        self.ib = IB()
        self.detector = RangeRegimeDetector(corridor_cfg)
        self.estimator = CenterEstimator(corridor_cfg)
        self.machine = CorridorStateMachine(corridor_cfg)
        self.logger = CsvEventLogger(runner_cfg.output_dir, runner_cfg.log_prefix)
        self.history = pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
        self.underlying_contract = None
        self.last_processed_ts: Optional[pd.Timestamp] = None
        self.positions: dict[int, ManagedPosition] = {}
        self.bar_delta = _timeframe_to_delta(self.cfg.timeframe)
        self.market_data_type = 3 if self.runner_cfg.mode == "delayed" else 1

    def run(self) -> int:
        self.connect()
        try:
            frame = self.fetch_recent_history()
            self.seed_from_history(frame)
            if self.runner_cfg.check_only:
                self.print_snapshot(label="startup-check", persist=True)
                return 0

            if self.runner_cfg.once:
                self.poll_once()
                self.print_snapshot(label="single-pass", persist=True)
                return 0

            print(
                f"Paper corridor runner started | symbol={self.cfg.symbol} | mode={self.runner_cfg.mode} | "
                f"execution={'paper' if self.runner_cfg.paper_execution else 'dry-run'} | timeframe={self.cfg.timeframe}"
            )
            while True:
                self.poll_once()
                time.sleep(max(5, self.runner_cfg.poll_seconds))
        except KeyboardInterrupt:
            print("Paper corridor runner stopped.")
            return 0
        finally:
            self.disconnect()

    def connect(self) -> None:
        port = self._resolve_ib_port(self.runner_cfg.host, self.runner_cfg.port)
        if port != self.runner_cfg.port:
            print(f"Requested IB port {self.runner_cfg.port} was not reachable; using {port} instead.")
            self.runner_cfg.port = port
        self.ib.connect(self.runner_cfg.host, self.runner_cfg.port, clientId=self.runner_cfg.client_id, timeout=10)
        self.ib.reqMarketDataType(self.market_data_type)
        self.underlying_contract = Stock(self.cfg.symbol, self.cfg.ib_exchange, self.cfg.ib_currency)
        self.ib.qualifyContracts(self.underlying_contract)

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()

    def fetch_recent_history(self) -> pd.DataFrame:
        duration = f"{max(1, self.runner_cfg.history_days)} D"
        bars = self.ib.reqHistoricalData(
            self.underlying_contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=self.cfg.timeframe,
            whatToShow=self.cfg.ib_what_to_show,
            useRTH=self.cfg.ib_use_rth,
            formatDate=1,
        )
        frame = util.df(bars)
        if frame.empty:
            raise RuntimeError("IB returned no bars for the paper runner.")
        frame["timestamp"] = pd.to_datetime(frame["date"], utc=True)
        frame["symbol"] = self.cfg.symbol
        frame = frame[["timestamp", "symbol", "open", "high", "low", "close", "volume"]].copy()
        frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp", "symbol"])
        return frame.reset_index(drop=True)

    def seed_from_history(self, frame: pd.DataFrame) -> None:
        eligible = self._completed_bars(frame)
        if eligible.empty:
            raise RuntimeError("No completed bars available to seed the paper runner.")
        self.history = eligible.copy().reset_index(drop=True)
        self.last_processed_ts = pd.Timestamp(self.history["timestamp"].iloc[-1])

        if self.runner_cfg.start_flat:
            self.machine = CorridorStateMachine(self.cfg)
            self.positions.clear()
            print(f"Seeded {len(self.history)} completed bars and reset to flat start.")
            return

        for _, row in self.history.iterrows():
            self._process_bar(row, allow_orders=False)
        print(f"Seeded {len(self.history)} completed bars and synced live state from history.")

    def poll_once(self) -> None:
        fresh = self.fetch_recent_history()
        eligible = self._completed_bars(fresh)
        if self.last_processed_ts is not None:
            eligible = eligible[eligible["timestamp"] > self.last_processed_ts]
        if eligible.empty:
            print("No new completed bars.")
            self._write_state_snapshot()
            return

        for _, row in eligible.iterrows():
            self.history = (
                pd.concat([self.history, pd.DataFrame([row])], ignore_index=True)
                .drop_duplicates(subset=["timestamp", "symbol"], keep="last")
                .sort_values("timestamp")
                .tail(max(300, self.cfg.regime_lookback * 4))
                .reset_index(drop=True)
            )
            self._process_bar(row, allow_orders=True)
            self.last_processed_ts = pd.Timestamp(row["timestamp"])

        self._write_state_snapshot()

    def print_snapshot(self, label: str, persist: bool = False) -> None:
        payload = self._build_state_snapshot()
        if persist:
            self.logger.write_state(payload)
        latest_ts = payload["timestamp"]
        candidates = payload.get("candidates", [])
        print(
            f"{label} | ts={latest_ts} | price={payload['price']:.2f} | "
            f"regime={payload['regime']} | center={payload['center']} | "
            f"state={payload['state']} | open_positions={len(self.positions)} | "
            f"candidates={len(candidates)}"
        )
        if payload.get("candidate_status"):
            print(f"Candidates | {payload['candidate_status']}")
        if payload.get("candidate_error"):
            print(f"Candidate Error | {payload['candidate_error']}")
        for idx, candidate in enumerate(candidates[:3], start=1):
            print(
                f"Candidate {idx} | exp={candidate['expiry']} | "
                f"{candidate['lower_strike']}/{candidate['body_strike']}/{candidate['upper_strike']} {candidate['right']} | "
                f"debit={candidate['net_debit']:.2f} | spread={candidate['total_spread']:.2f} | "
                f"max_reward={candidate['max_reward']:.2f}"
            )

    def _process_bar(self, row: pd.Series, allow_orders: bool) -> None:
        timestamp = pd.Timestamp(row["timestamp"])
        price = float(row["close"])
        regime = self.detector.evaluate(self.history)
        center = self.estimator.estimate(self.history)
        step = self.machine.process_bar(self.cfg.symbol, timestamp, price, regime, center)

        for transition in step.transitions:
            self.logger.write_transition(
                {
                    "timestamp": transition.timestamp.isoformat(),
                    "symbol": transition.symbol,
                    "from_state": transition.from_state.value,
                    "to_state": transition.to_state.value,
                    "reason": transition.reason,
                    "regime": transition.regime.value,
                    "price": round(transition.price, 4),
                    "center_price": transition.center_price,
                    "drift_count": transition.drift_count,
                    "layer_count": transition.layer_count,
                }
            )
            print(
                f"Transition | {transition.from_state.value} -> {transition.to_state.value} | "
                f"price={transition.price:.2f} | reason={transition.reason}"
            )

        for action in step.actions:
            action_row = {
                "timestamp": action.timestamp.isoformat(),
                "symbol": action.symbol,
                "action": action.action.value,
                "state": action.state.value,
                "price": round(action.price, 4),
                "center_price": action.center_price,
                "layer_id": action.layer_id,
                "detail": action.detail,
                **action.metadata,
            }
            self.logger.write_action(action_row)
            print(
                f"Action | {action.action.value} | layer={action.layer_id} | "
                f"price={action.price:.2f} | detail={action.detail}"
            )
            if allow_orders:
                self._handle_action(action, center, regime)

    def _handle_action(self, action: ActionRecord, center, regime) -> None:
        if action.action in {ActionType.DRIFT_STARTED, ActionType.DRIFT_RESOLVED, ActionType.REBUILD_REQUESTED}:
            return

        if action.action in {ActionType.SESSION_FLUSH, ActionType.ABORTED}:
            if action.layer_id is not None:
                self._close_position(action)
            return

        if action.action == ActionType.REBUILT:
            if action.detail.startswith("Established"):
                self._open_position(action, center, regime)
            else:
                if action.layer_id is not None:
                    self._close_position(action)
            return

        if action.action in {ActionType.ENTER_PRIMARY, ActionType.ADD_SUPPLEMENTAL}:
            self._open_position(action, center, regime)

    def _open_position(self, action: ActionRecord, center, regime) -> None:
        if action.layer_id is None or action.layer_id in self.positions:
            return
        if center is None or regime is None or regime.regime != Regime.RANGE:
            return

        target_body = float(action.metadata.get("body_strike") or action.metadata.get("center_price") or action.center_price or 0.0)
        candidate = self._select_candidate(target_body)
        if candidate is None:
            self._log_order(
                {
                    "timestamp": action.timestamp.isoformat(),
                    "layer_id": action.layer_id,
                    "symbol": self.cfg.symbol,
                    "side": "OPEN",
                    "mode": "paper" if self.runner_cfg.paper_execution else "dry-run",
                    "status": "skipped",
                    "reason": "No candidate butterfly matched the corridor center.",
                }
            )
            return

        limit_price = self._combo_limit_price(candidate, side="BUY")
        status = "dry_run"
        order_id = None
        if self.runner_cfg.paper_execution:
            trade = self._place_combo_order(candidate, "BUY", limit_price)
            status = trade.orderStatus.status or "submitted"
            order_id = getattr(trade.order, "orderId", None)

        self.positions[action.layer_id] = ManagedPosition(
            layer_id=action.layer_id,
            candidate=candidate,
            quantity=self.runner_cfg.quantity,
            opened_at=action.timestamp,
            open_limit=limit_price,
            open_status=status,
            source_action=action.action.value,
            order_id=order_id,
        )
        self._log_order(
            {
                "timestamp": action.timestamp.isoformat(),
                "layer_id": action.layer_id,
                "symbol": self.cfg.symbol,
                "side": "OPEN",
                "mode": "paper" if self.runner_cfg.paper_execution else "dry-run",
                "status": status,
                "order_id": order_id,
                "quantity": self.runner_cfg.quantity,
                "expiry": candidate.expiry,
                "right": candidate.right,
                "lower_strike": candidate.lower_strike,
                "body_strike": candidate.body_strike,
                "upper_strike": candidate.upper_strike,
                "limit_price": round(limit_price, 2),
                "net_debit": round(candidate.net_debit, 4),
                "total_spread": round(candidate.total_spread, 4),
                "reason": action.detail,
            }
        )

    def _close_position(self, action: ActionRecord) -> None:
        position = self.positions.get(action.layer_id or -1)
        if position is None:
            self._log_order(
                {
                    "timestamp": action.timestamp.isoformat(),
                    "layer_id": action.layer_id,
                    "symbol": self.cfg.symbol,
                    "side": "CLOSE",
                    "mode": "paper" if self.runner_cfg.paper_execution else "dry-run",
                    "status": "skipped",
                    "reason": "No tracked position for the closing action.",
                }
            )
            return

        limit_price = self._combo_limit_price(position.candidate, side="SELL")
        status = "dry_run"
        order_id = None
        if self.runner_cfg.paper_execution:
            trade = self._place_combo_order(position.candidate, "SELL", limit_price)
            status = trade.orderStatus.status or "submitted"
            order_id = getattr(trade.order, "orderId", None)

        position.closed_at = action.timestamp
        position.close_limit = limit_price
        position.close_status = status
        position.close_order_id = order_id
        self._log_order(
            {
                "timestamp": action.timestamp.isoformat(),
                "layer_id": action.layer_id,
                "symbol": self.cfg.symbol,
                "side": "CLOSE",
                "mode": "paper" if self.runner_cfg.paper_execution else "dry-run",
                "status": status,
                "order_id": order_id,
                "quantity": position.quantity,
                "expiry": position.candidate.expiry,
                "right": position.candidate.right,
                "lower_strike": position.candidate.lower_strike,
                "body_strike": position.candidate.body_strike,
                "upper_strike": position.candidate.upper_strike,
                "limit_price": round(limit_price, 2),
                "net_debit": round(position.candidate.net_debit, 4),
                "total_spread": round(position.candidate.total_spread, 4),
                "reason": action.detail,
            }
        )
        del self.positions[position.layer_id]

    def _select_candidate(self, target_body: float) -> Optional[ButterflyCandidate]:
        candidates = self._load_candidates(target_body)
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda candidate: (
                abs(candidate.body_strike - target_body),
                candidate.total_spread,
                candidate.net_debit,
            ),
        )

    def _load_candidates(self, target_body: float) -> list[ButterflyCandidate]:
        loader = IBOptionChainLoader(
            self.runner_cfg.host,
            self.runner_cfg.port,
            self.runner_cfg.client_id + self.runner_cfg.chain_client_id_offset,
            self.cfg.ib_exchange,
            self.cfg.ib_currency,
        )
        quotes = loader.load_candidates(
            self.cfg.symbol,
            target_body,
            self.cfg.butterfly_width,
            self.cfg.dte_min,
            self.cfg.dte_max,
            market_data_type=self.market_data_type,
        )
        return select_butterflies(quotes, target_body, self.cfg.butterfly_width, self.cfg)

    def _combo_limit_price(self, candidate: ButterflyCandidate, side: str) -> float:
        spread_buffer = min(max(0.01, candidate.total_spread * 0.25), max(0.02, self.cfg.max_acceptable_option_spread))
        if side == "BUY":
            return max(0.01, round(candidate.net_debit + spread_buffer, 2))
        return max(0.01, round(max(0.01, candidate.net_debit - spread_buffer), 2))

    def _place_combo_order(self, candidate: ButterflyCandidate, side: str, limit_price: float):
        right = "C" if candidate.right == "CALL" else "P"
        lower = Option(
            symbol=self.cfg.symbol,
            lastTradeDateOrContractMonth=candidate.expiry,
            strike=candidate.lower_strike,
            right=right,
            exchange=self.cfg.ib_exchange,
            currency=self.cfg.ib_currency,
        )
        body = Option(
            symbol=self.cfg.symbol,
            lastTradeDateOrContractMonth=candidate.expiry,
            strike=candidate.body_strike,
            right=right,
            exchange=self.cfg.ib_exchange,
            currency=self.cfg.ib_currency,
        )
        upper = Option(
            symbol=self.cfg.symbol,
            lastTradeDateOrContractMonth=candidate.expiry,
            strike=candidate.upper_strike,
            right=right,
            exchange=self.cfg.ib_exchange,
            currency=self.cfg.ib_currency,
        )
        qualified = self.ib.qualifyContracts(lower, body, upper)
        if len(qualified) != 3:
            raise RuntimeError("Unable to qualify option legs for combo order.")

        if side == "BUY":
            leg_specs = [
                ComboLegSpec(con_id=qualified[0].conId, ratio=1, action="BUY", exchange=self.cfg.ib_exchange),
                ComboLegSpec(con_id=qualified[1].conId, ratio=2, action="SELL", exchange=self.cfg.ib_exchange),
                ComboLegSpec(con_id=qualified[2].conId, ratio=1, action="BUY", exchange=self.cfg.ib_exchange),
            ]
        else:
            leg_specs = [
                ComboLegSpec(con_id=qualified[0].conId, ratio=1, action="SELL", exchange=self.cfg.ib_exchange),
                ComboLegSpec(con_id=qualified[1].conId, ratio=2, action="BUY", exchange=self.cfg.ib_exchange),
                ComboLegSpec(con_id=qualified[2].conId, ratio=1, action="SELL", exchange=self.cfg.ib_exchange),
            ]

        combo = build_butterfly_combo(self.cfg.symbol, self.cfg.ib_currency, self.cfg.ib_exchange, leg_specs)
        order = LimitOrder(side, self.runner_cfg.quantity, limit_price, tif=self.runner_cfg.order_tif)
        order.orderRef = f"corridor:{self.cfg.symbol}:{side}:{candidate.expiry}:{candidate.body_strike}"
        trade = self.ib.placeOrder(combo, order)
        self.ib.sleep(1.0)
        return trade

    def _completed_bars(self, frame: pd.DataFrame) -> pd.DataFrame:
        now_utc = _ensure_utc_timestamp(pd.Timestamp.utcnow())
        out = frame.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        out = out[out["timestamp"] + self.bar_delta <= now_utc]
        return out.sort_values("timestamp").reset_index(drop=True)

    def _write_state_snapshot(self) -> None:
        payload = self._build_state_snapshot()
        self.logger.write_state(payload)

    def _build_state_snapshot(self) -> dict[str, Any]:
        regime = self.detector.evaluate(self.history)
        center = self.estimator.estimate(self.history)
        latest = self.history.iloc[-1] if not self.history.empty else None
        candidates: list[dict[str, Any]] = []
        candidate_status = "Candidate selection skipped."
        candidate_error: Optional[str] = None
        if regime is not None and regime.regime == Regime.RANGE and center is not None:
            try:
                loaded = self._load_candidates(center.center_price)
                candidates = [
                    {
                        "expiry": candidate.expiry,
                        "right": candidate.right,
                        "lower_strike": candidate.lower_strike,
                        "body_strike": candidate.body_strike,
                        "upper_strike": candidate.upper_strike,
                        "net_debit": round(candidate.net_debit, 4),
                        "total_spread": round(candidate.total_spread, 4),
                        "max_risk": round(candidate.max_risk, 2),
                        "max_reward": round(candidate.max_reward, 2),
                    }
                    for candidate in loaded[:5]
                ]
                if candidates:
                    candidate_status = f"Loaded {len(candidates)} candidate butterflies for the current center."
                else:
                    candidate_status = "No qualifying butterflies matched the current center and spread filters."
            except Exception as exc:
                candidate_error = str(exc).strip() or exc.__class__.__name__
                candidate_status = "Option-chain lookup failed; see candidate error."
        elif regime is not None and regime.regime != Regime.RANGE:
            candidate_status = f"Skipped candidate selection because regime={regime.regime.value}."
        elif center is None:
            candidate_status = "Skipped candidate selection because no valid center is available."

        payload = {
            "timestamp": latest["timestamp"].isoformat() if latest is not None else None,
            "symbol": self.cfg.symbol,
            "price": round(float(latest["close"]), 4) if latest is not None else None,
            "regime": regime.regime.value if regime is not None else None,
            "center": center.center_price if center is not None else None,
            "state": self.machine.context.state.value,
            "open_positions": [
                {
                    "layer_id": position.layer_id,
                    "expiry": position.candidate.expiry,
                    "right": position.candidate.right,
                    "lower_strike": position.candidate.lower_strike,
                    "body_strike": position.candidate.body_strike,
                    "upper_strike": position.candidate.upper_strike,
                    "quantity": position.quantity,
                    "open_limit": position.open_limit,
                    "open_status": position.open_status,
                    "source_action": position.source_action,
                }
                for position in sorted(self.positions.values(), key=lambda item: item.layer_id)
            ],
            "execution_mode": "paper" if self.runner_cfg.paper_execution else "dry-run",
            "candidates": candidates,
            "candidate_status": candidate_status,
            "candidate_error": candidate_error,
        }
        return payload

    def _log_order(self, record: dict[str, Any]) -> None:
        self.logger.write_order(record)

    @staticmethod
    def _port_is_listening(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            return False

    def _resolve_ib_port(self, host: str, requested_port: int) -> int:
        local_hosts = {"127.0.0.1", "localhost", "::1"}
        if host not in local_hosts:
            return requested_port
        if self._port_is_listening(host, requested_port):
            return requested_port

        configured_port = self._detect_local_gateway_port()
        candidates: list[int] = []
        if configured_port is not None:
            candidates.append(configured_port)
        candidates.extend([4002, 4001, 4000, 7497, 7496])

        for candidate in candidates:
            if candidate == requested_port:
                continue
            if self._port_is_listening(host, candidate):
                return candidate
        return requested_port

    @staticmethod
    def _detect_local_gateway_port() -> Optional[int]:
        config_candidates = [
            Path("C:/Jts/ibgateway/1045/jts.ini"),
            Path.home() / "Jts" / "jts.ini",
        ]
        config_candidates.extend(Path("C:/Jts/ibgateway").glob("*/jts.ini"))
        config_candidates.extend((Path.home() / "Jts" / "ibgateway").glob("*/jts.ini"))

        seen: set[Path] = set()
        for path in config_candidates:
            resolved = path.resolve()
            if resolved in seen or not resolved.exists():
                continue
            seen.add(resolved)

            parser = ConfigParser()
            try:
                parser.read(resolved, encoding="utf-8")
                if parser.has_option("IBGateway", "LocalServerPort"):
                    value = parser.get("IBGateway", "LocalServerPort").strip()
                    return int(value)
            except (OSError, ValueError):
                continue
        return None
