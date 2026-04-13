from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper


def _normalize_symbol_from_contract(contract: Contract) -> str:
    symbol = str(getattr(contract, "symbol", "") or "").upper().strip()
    primary_exchange = str(getattr(contract, "primaryExchange", "") or "").upper().strip()
    exchange = str(getattr(contract, "exchange", "") or "").upper().strip()
    currency = str(getattr(contract, "currency", "") or "").upper().strip()

    if not symbol:
        return ""

    exchange_hint = primary_exchange or exchange

    us_exchanges = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS", "IEX", "SMART"}
    if currency == "USD" or exchange_hint in us_exchanges:
        return f"{symbol}.US"

    if exchange_hint in {"SEHK", "HKFE", "HKEX", "XHKG"} or currency == "HKD":
        return f"{symbol}.HK"

    if exchange_hint in {"ASX", "XASX"} or currency == "AUD":
        return f"{symbol}.AU"

    if exchange_hint in {"XETRA", "XETR"}:
        return f"{symbol}.XETRA"

    if exchange_hint in {"XPAR", "PA"}:
        return f"{symbol}.PA"

    if exchange_hint in {"XAMS", "AS"}:
        return f"{symbol}.AS"

    if exchange_hint in {"XSWX", "SWX", "SW"}:
        return f"{symbol}.SW"

    if exchange_hint in {"XSES", "SES", "SGX"} or currency == "SGD":
        return f"{symbol}.SG"

    return symbol


class IBGatewayClient(EWrapper, EClient):
    def __init__(self):
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)

        self._thread: Optional[threading.Thread] = None

        self.connected_event = threading.Event()
        self.managed_accounts_event = threading.Event()
        self.positions_event = threading.Event()
        self.account_summary_event = threading.Event()

        self.connection_error: Optional[str] = None
        self.errors: List[Dict[str, Any]] = []

        self.next_valid_id_value: Optional[int] = None
        self.managed_accounts_list: List[str] = []

        self.positions_rows: List[Dict[str, Any]] = []
        self.account_summary_rows: List[Dict[str, Any]] = []

        self._active_account_summary_req_id: Optional[int] = None

    def connect_and_start(self, host: str, port: int, client_id: int, timeout: int = 15) -> None:
        self.connect(host, int(port), int(client_id))

        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

        ok = self.connected_event.wait(timeout=timeout)
        if not ok:
            raise RuntimeError(
                f"Timed out connecting to IB Gateway at {host}:{port}. "
                f"Make sure IB Gateway is running, logged in, and the socket port matches."
            )

        self.managed_accounts_event.wait(timeout=5)

    def disconnect_and_stop(self) -> None:
        try:
            if self.isConnected():
                self.disconnect()
        finally:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2)

    def nextValidId(self, orderId: int):
        self.next_valid_id_value = int(orderId)
        self.connected_event.set()

    def managedAccounts(self, accountsList: str):
        accounts = [x.strip() for x in str(accountsList).split(",") if x.strip()]
        self.managed_accounts_list = accounts
        self.managed_accounts_event.set()

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

    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        self.positions_rows.append(
            {
                "account": account,
                "conid": getattr(contract, "conId", None),
                "symbol_ibkr": str(getattr(contract, "symbol", "") or "").upper().strip(),
                "symbol": _normalize_symbol_from_contract(contract),
                "secType": getattr(contract, "secType", None),
                "currency": getattr(contract, "currency", None),
                "exchange": getattr(contract, "exchange", None),
                "primaryExchange": getattr(contract, "primaryExchange", None),
                "localSymbol": getattr(contract, "localSymbol", None),
                "tradingClass": getattr(contract, "tradingClass", None),
                "position": float(position),
                "average_cost": float(avgCost) if avgCost is not None else None,
            }
        )

    def positionEnd(self):
        self.positions_event.set()

    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        self.account_summary_rows.append(
            {
                "reqId": reqId,
                "account": account,
                "tag": tag,
                "value": value,
                "currency": currency,
            }
        )

    def accountSummaryEnd(self, reqId: int):
        if self._active_account_summary_req_id is None:
            return

        if int(reqId) == int(self._active_account_summary_req_id):
            self.account_summary_event.set()

    def resolve_account_id(self, account_id: str | None = None) -> str:
        if account_id:
            return str(account_id)

        if self.managed_accounts_list:
            return self.managed_accounts_list[0]

        raise RuntimeError(
            "No IBKR account ID was received from managedAccounts. "
            "Check that IB Gateway is logged in and API access is enabled."
        )

    def fetch_positions(self, timeout: int = 20) -> List[Dict[str, Any]]:
        self.positions_rows = []
        self.positions_event.clear()

        self.reqPositions()

        ok = self.positions_event.wait(timeout=timeout)
        self.cancelPositions()

        if not ok:
            raise RuntimeError("Timed out waiting for positions from IB Gateway.")
        return list(self.positions_rows)

    def fetch_account_summary_single(
        self,
        req_id: int,
        group: str,
        tags: str,
        timeout: int = 20,
    ) -> List[Dict[str, Any]]:
        self.account_summary_rows = []
        self.account_summary_event.clear()
        self._active_account_summary_req_id = int(req_id)

        self.reqAccountSummary(int(req_id), group, tags)
        ok = self.account_summary_event.wait(timeout=timeout)
        self.cancelAccountSummary(int(req_id))
        self._active_account_summary_req_id = None

        if not ok:
            raise RuntimeError(f"Timed out waiting for account summary for req_id={req_id}, tags={tags}")
        return list(self.account_summary_rows)

    def fetch_account_summary_bundle(self, timeout: int = 20) -> List[Dict[str, Any]]:
        """
        Sequentially request:
          1) core summary tags
          2) base-currency ledger ($LEDGER)
          3) USD ledger ($LEDGER:USD)
          4) all-currency ledger ($LEDGER:ALL)

        We do these one after another because IBKR allows only two active
        reqAccountSummary subscriptions at a time.
        """
        bundle: List[Dict[str, Any]] = []

        requests = [
            (
                9001,
                "All",
                "AccountType,NetLiquidation,TotalCashValue,SettledCash,BuyingPower,"
                "EquityWithLoanValue,GrossPositionValue,InitMarginReq,MaintMarginReq,"
                "AvailableFunds,ExcessLiquidity,Cushion,Leverage",
                "core",
            ),
            (9002, "All", "$LEDGER", "ledger_base"),
            (9003, "All", "$LEDGER:USD", "ledger_usd"),
            (9004, "All", "$LEDGER:ALL", "ledger_all"),
        ]

        for req_id, group, tags, bucket in requests:
            try:
                rows = self.fetch_account_summary_single(
                    req_id=req_id,
                    group=group,
                    tags=tags,
                    timeout=timeout,
                )
                for row in rows:
                    x = dict(row)
                    x["summary_bucket"] = bucket
                    x["summary_tags_request"] = tags
                    bundle.append(x)
            except Exception as exc:
                self.errors.append(
                    {
                        "reqId": req_id,
                        "errorCode": -999,
                        "errorString": f"Account summary request failed for {bucket} ({tags}): {exc}",
                    }
                )

        return bundle