Feature: Live Window Metric Snapshot Service
  The system should compute minute-level live portfolio performance metrics
  For the active 4-hour trading window and persist them for real-time dashboards

  Background:
    Given MongoDB is available with windows and live_metrics collections
    And ClickHouse OHLCV price data is available
    And AlphaFusionNet final portfolio weights exist
    And risk controls (stop-loss, take-profit, confidence) are configured
    And the scheduler triggers the service every minute

  Scenario: Initialize or reuse a 4-hour trading window
    Given the current UTC time
    When the service determines today’s trading windows
    Then an existing window_id for today should be reused if present
    Or a new window_id should be created with t0 equal to the current minute
    And the window duration should be exactly 4 hours
    And the window document should include entry prices, weights, quantities, and initial NAV

  Scenario: Skip execution before window start
    Given a window exists with start time t0
    And the current as_of time is before t0
    When the live metric snapshot service runs
    Then no prices should be fetched
    And no snapshot should be written to MongoDB

  Scenario: End the window after expiration
    Given a window exists with end time t1
    And the current as_of time is after t1
    When the live metric snapshot service runs
    Then the window status should be marked as ENDED
    And no further snapshots should be computed or persisted

  Scenario: Fetch live prices with fallback and stale detection
    Given an active window and portfolio symbols
    When the service fetches minute close prices from ClickHouse
    Then prices should be retrieved for each symbol at the current minute
    And missing prices should fallback to the last known snapshot price
    Or fallback to the entry price if no snapshot exists
    And any fallback price should be marked as stale
    And stale symbols should be tracked per snapshot

  Scenario: Fetch benchmark price and detect staleness
    Given a configured benchmark symbol
    When the benchmark close price is requested
    Then the benchmark price should be fetched from ClickHouse
    And fallback logic should apply if the price is missing
    And benchmark staleness should be recorded if applicable

  Scenario: Compute live portfolio performance metrics
    Given current prices, entry prices, and portfolio weights
    When the metric computation is executed
    Then a live snapshot should be produced containing:
      | Field                    | Description                                      |
      | as_of                   | current minute timestamp (UTC)                  |
      | prices                  | per-symbol close prices                         |
      | symbol_values           | per-symbol market value                         |
      | portfolio_value         | total portfolio NAV                             |
      | cumulative_return       | Rp(t) since window start                        |
      | pnl                     | profit and loss                                 |
      | benchmark_price         | benchmark close price                           |
      | benchmark_return        | benchmark cumulative return                    |
      | alpha                   | portfolio return minus benchmark return        |
      | weights                 | fixed portfolio weights                        |
      | quantities              | per-symbol quantities                          |

  Scenario: Persist live snapshot to MongoDB
    Given a computed live metrics snapshot
    When the snapshot is persisted
    Then it should be appended to windows.live_history
    And a flattened document should be inserted into live_metrics
    And the snapshot should be indexed by window_id and as_of timestamp

  Scenario: Record stale symbol information
    Given one or more symbols have stale prices
    When the snapshot is persisted
    Then the stale symbols list should be stored with the snapshot
    And a warning log should be emitted identifying stale symbols

  Scenario: First snapshot fallback to entry prices
    Given no previous live snapshot exists for the window
    When the service fetches prices
    Then entry prices should be used as fallback values
    And the snapshot should still be computed successfully

  Scenario: Include risk controls in window state
    Given risk controls from AlphaFusionNet predictions
    When the window state is initialized
    Then stop-loss, take-profit, and confidence should be stored per symbol
    And these risk controls should be available to downstream consumers

  Scenario: Stateless minute-by-minute execution
    Given the service is triggered by a scheduler
    When each invocation runs
    Then exactly one minute-level snapshot should be computed
    And the service should exit immediately after persistence
    And no in-memory state should be required between runs

  Scenario: Support real-time dashboard visualization
    Given live_metrics contains minute-level snapshots
    When the dashboard queries the collection
    Then it should be able to render:
      | live metric charts          |
      | live return curve           |
      ...
