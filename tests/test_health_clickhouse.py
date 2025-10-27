"""
test_health_clickhouse.py
Description: Unit tests for ClickHouse connection and queries.
Author: Elham Esmaeilnia (elham.e.shirvani@gmail.com)
Date: 2025 Sep 30
Version: 1.1.1
"""
from unittest.mock import patch
import pytest
from clickhouse_driver import Client

def test_clickhouse_connection_success():
    with patch.object(Client, "execute", return_value=[("ok",)]) as mock_execute:
        ch = Client(host="mock_host", port=9000, user="mock_user", password="mock_pass", database="mock_db")
        result = ch.execute("SELECT 1")
        assert result == [("ok",)]
        mock_execute.assert_called_once_with("SELECT 1")


def test_clickhouse_query_symbols():
    expected = [("BINANCE:BTCUSDT",), ("BINANCE:ETHUSDT",)]
    with patch.object(Client, "execute", return_value=expected) as mock_execute:
        ch = Client(host="mock_host", port=9000, user="mock_user", password="mock_pass", database="mock_db")
        result = ch.execute("SELECT DISTINCT symbol FROM novoxpert.tradingview_ohlcv")
        assert result == expected
        mock_execute.assert_called_once()


def test_clickhouse_query_failure():
    with patch.object(Client, "execute", side_effect=Exception("Connection failed")) as mock_execute:
        ch = Client(host="mock_host", port=9000, user="mock_user", password="mock_pass", database="mock_db")
        with pytest.raises(Exception, match="Connection failed"):
            ch.execute("SELECT 1")
        mock_execute.assert_called_once()
