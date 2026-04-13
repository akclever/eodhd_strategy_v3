#!/usr/bin/env python3
from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.wrapper import EWrapper


class IBKRWhatIfClient(EWrapper, EClient):
    def __init__(self):
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)

        self._thread: threading.Thread | None = None

        self.connected_event = threading.Event()
        self.done_event = threading.Event()

        self.connection_error: str | None = None
        self.errors: List[dict] = []
        self.next_order_id: int | None = None

        self.expected_order_ids: set[int] = set()
        self.received_order_ids: set[int] = set()

        self.results: List[dict] = []

    def connect_and_start(self, host: str, port: int, client_id: int, timeout: int = 15) -> None:
        self.connect(host, int(port), int(client_id))
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

        ok = self.connected_event.wait(timeout=timeout)
        if not ok:
            raise RuntimeError(
                f"Timed out connecting to IB Gateway at {host}:{port}. "
                f"Make sure Gateway is running, logged in, and API access is enabled."
            )

    def disconnect_and_stop(self) -> None:
        try:
            if self.isConnected():
                self.disconnect()
        finally:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2)

    def nextValidId(self, orderId: int):
        self.next_order_id = int(orderId)
        self.connected_event.set()

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = ""):
        row = {
            "reqId": reqId,
            "errorCode": errorCode,
            "errorString": errorString,
        }
        self.errors.append(row)

        if int(errorCode) == 502 and not self.connected_event.is_set():
            self.connection_error = f"{errorCode}: {errorString}"
            self.connected_event.set()

    def openOrder(self, orderId, contract, order, orderState):
        symbol = str(getattr(contract, "symbol", "") or "").upper().strip()

        result = {
            "order_id": int(orderId),
            "symbol": symbol,
            "action": str(getattr(order, "action", "") or ""),
            "order_type": str(getattr(order, "orderType", "") or ""),
            "quantity": float(getattr(order, "totalQuantity", 0) or 0),
            "limit_price": getattr(order, "lmtPrice", None),
            "status": getattr(orderState, "status", None),
            "init_margin_before": getattr(orderState, "initMarginBefore", None),
            "init_margin_change": getattr(orderState, "initMarginChange", None),
            "init_margin_after": getattr(orderState, "initMarginAfter", None),
            "maint_margin_before": getattr(orderState, "maintMarginBefore", None),
            "maint_margin_change": getattr(orderState, "maintMarginChange", None),
            "maint_margin_after": getattr(orderState, "maintMarginAfter", None),
            "equity_with_loan_before": getattr(orderState, "equityWithLoanBefore", None),
            "equity_with_loan_change": getattr(orderState, "equityWithLoanChange", None),
            "equity_with_loan_after": getattr(orderState, "equityWithLoanAfter", None),
            "commission": getattr(orderState, "commission", None),
            "min_commission": getattr(orderState, "minCommission", None),
            "max_commission": getattr(orderState, "maxCommission", None),
            "commission_currency": getattr(orderState, "commissionCurrency", None),
            "warning_text": getattr(orderState, "warningText", None),
        }

        self.results.append(result)
        self.received_order_ids.add(int(orderId))

        if self.expected_order_ids and self.received_order_ids >= self.expected_order_ids:
            self.done_event.set()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IBKR What-If preview for blotter trades")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002, help="IB Gateway paper is commonly 4002; live is commonly 4001")
    parser.add_argument("--client-id", type=int, default=102)

    parser.add_argument("--blotter-input", required=True)
    parser.add_argument("--only-flagged", action="store_true", help="Only preview rows where trade_flag == True")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output", default="ibkr_whatif.csv")
    return parser.parse_args(argv)


def _safe_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def build_stock_contract(symbol: str) -> Contract:
    root = str(symbol).upper().split(".")[0]

    c = Contract()
    c.symbol = root
    c.secType = "STK"
    c.exchange = "SMART"
    c.currency = "USD"
    return c


def build_order(row: pd.Series) -> Order:
    qty = int(abs(float(row["delta_shares"])))
    if qty <= 0:
        raise ValueError(f"Non-positive quantity for {row.get('symbol')}")

    side = str(row["side"]).upper().strip()
    order_type = str(row.get("suggested_order_type") or "LMT").upper().strip()
    tif = str(row.get("suggested_tif") or "DAY").upper().strip()

    o = Order()
    o.action = side
    o.totalQuantity = qty
    o.orderType = order_type
    o.tif = tif
    o.whatIf = True

    if order_type == "LMT":
        lmt = _safe_float(row.get("suggested_limit_price"))
        if lmt is None or lmt <= 0:
            raise ValueError(f"Missing/invalid suggested_limit_price for {row.get('symbol')}")
        o.lmtPrice = float(lmt)

    # Important IBKR compatibility fix:
    # community reports indicate these can trigger 10268 unless explicitly disabled
    if hasattr(o, "eTradeOnly"):
        o.eTradeOnly = False
    if hasattr(o, "firmQuoteOnly"):
        o.firmQuoteOnly = False

    return o


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    blotter = pd.read_csv(args.blotter_input)
    if blotter.empty:
        raise SystemExit("Blotter is empty.")

    if args.only_flagged and "trade_flag" in blotter.columns:
        blotter = blotter[blotter["trade_flag"] == True].copy()

    blotter = blotter[blotter["side"].isin(["BUY", "SELL"])].copy()
    blotter = blotter[pd.to_numeric(blotter["delta_shares"], errors="coerce").fillna(0).abs() > 0].copy()

    if blotter.empty:
        raise SystemExit("No BUY/SELL trades to preview.")

    client = IBKRWhatIfClient()

    try:
        client.connect_and_start(args.host, args.port, args.client_id, timeout=args.timeout)

        if client.connection_error:
            raise RuntimeError(
                f"IB Gateway connection failed: {client.connection_error}. "
                f"Check Gateway login and port."
            )

        if client.next_order_id is None:
            raise RuntimeError("Did not receive nextValidId from IB Gateway.")

        next_id = int(client.next_order_id)

        for _, row in blotter.iterrows():
            order_id = next_id
            next_id += 1

            contract = build_stock_contract(str(row["symbol"]))
            order = build_order(row)

            client.expected_order_ids.add(order_id)
            client.placeOrder(order_id, contract, order)

        ok = client.done_event.wait(timeout=args.timeout)
        if not ok:
            print("Warning: timed out waiting for all What-If responses. Partial results may have been returned.")

    finally:
        client.disconnect_and_stop()

    results_df = pd.DataFrame(client.results)

    if results_df.empty:
        print("No What-If responses returned.")
        if client.errors:
            err_df = pd.DataFrame(client.errors)
            print(err_df.to_string(index=False))
            if (err_df["errorCode"] == 10268).any():
                print("\nIBKR rejected the preview because unsupported order attributes were sent.")
                print("The usual fix is to explicitly set order.eTradeOnly = False and order.firmQuoteOnly = False.")
        raise SystemExit(1)

    out = blotter.merge(
        results_df,
        left_on=blotter["symbol"].astype(str).str.upper().str.split(".").str[0],
        right_on="symbol",
        how="left",
        suffixes=("", "_whatif"),
    )

    out.to_csv(args.output, index=False)

    print(f"Saved What-If preview to {args.output}")

    preview_cols = [
        "symbol",
        "side",
        "delta_shares",
        "suggested_order_type",
        "suggested_limit_price",
        "commission",
        "commission_currency",
        "init_margin_change",
        "maint_margin_change",
        "equity_with_loan_change",
        "warning_text",
        "status",
    ]
    preview_cols = [c for c in preview_cols if c in out.columns]
    print("\nWhat-If preview:")
    print(out[preview_cols].to_string(index=False))

    if client.errors:
        print("\nIBKR API notices/errors seen during What-If:")
        print(pd.DataFrame(client.errors).head(20).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())