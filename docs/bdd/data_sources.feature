Feature: Market-news data sources
  The system should provide consistent market and news data streams
  For downstream feature generation and modeling

  Scenario: OHLCV market data
    Given ClickHouse contains OHLCV candles at 1-minute resolution
    And a symbol "AAPL" exists in the configuration
    When OHLCV data is requested for the time range
      | start_time | end_time |
      | 2025-12-22T14:30:00Z | 2025-12-22T18:30:00Z |
    Then the result should contain 1-minute candles
    And each candle should include open, high, low, close, volume, and timestamp

  Scenario: News data
    Given MongoDB contains news items at 1-minute resolution
    When news data is requested for the same time range
    Then the result should include {
        "releasedAt"
        "assets"
        "content"
        "news_count"
    }
