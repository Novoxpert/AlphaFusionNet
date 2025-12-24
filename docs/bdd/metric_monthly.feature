Feature: Metric Monthly Service (Month-To-Date Performance from Live Snapshots)
  The system should compute month-to-date portfolio performance metrics
  Using stored minute-level live snapshots from trading windows
  To support AlphaFusionNet monthly dashboard analytics

  Background:
    Given MongoDB is available with windows and monthly collections
    And trading windows contain minute-level live_history snapshots
    And each window stores portfolio weights for its trading day
    And the service is scheduled to run once per trading day
    And configuration is loaded from AFN_config.yaml

  Scenario: Determine the current month for MTD computation
    Given the current UTC date
    When the monthly metric service starts
    Then the target year should be the current UTC year
    And the target month should be the current UTC month
    And the service should compute metrics month-to-date only

  Scenario: Load all trading windows for the current month
    Given a target year and month
    When the service queries the windows collection
    Then all windows whose t0 falls within the calendar month should be loaded
    And each window should represent one trading day
    And windows outside the month should be ignored

  Scenario: Handle months with no trading windows
    Given no windows exist for the target month
    When the service runs
    Then default zero-valued monthly metrics should be produced
    And portfolio_daily_returns should be empty
    And a monthly document should still be upserted
    And a metrics_history snapshot should be appended

  Scenario: Build daily portfolio weights from window documents
    Given monthly windows are loaded
    When the service extracts portfolio weights
    Then weights should be mapped by trading date and symbol
    And weights should be treated as fixed for that trading day
    And only configured trading symbols should be included

  Scenario: Extract minute-level prices from windows.live_history
    Given monthly windows contain live_history snapshots
    When the service extracts price series
    Then prices should be taken from the actual stored live snapshots
    And fallback logic used during live computation should be preserved
    And no direct ClickHouse queries should be performed

  Scenario: Handle missing or empty price history
    Given no usable price data can be extracted from live_history
    When the service runs
    Then default zero-valued monthly metrics should be produced
    And portfolio_daily_returns should be empty
    And a metrics_history snapshot should be appended

  Scenario: Aggregate minute-level prices into daily open and close
    Given minute-level prices from live snapshots
    When prices are aggregated by trading day
    Then a daily open price should be computed per symbol
    And a daily close price should be computed per symbol
    And aggregation should respect calendar day boundaries

  Scenario: Compute per-symbol daily returns
    Given daily open and close prices
    When per-symbol daily returns are computed
    Then returns should be calculated as:
      """
      (daily_close - daily_open) / daily_open
      """
    And only dates within the target month should be included

  Scenario: Compute daily portfolio returns
    Given per-symbol daily returns and daily portfolio weights
    When portfolio daily returns are computed
    Then symbol returns should be combined using that day’s weights
    And one portfolio return should be produced per trading day
    And results should be indexed by trading date

  Scenario: Handle empty portfolio daily returns
    Given no portfolio daily returns can be computed
    When the service runs
    Then default zero-valued monthly metrics should be produced
    And portfolio_daily_returns should be empty
    And a metrics_history snapshot should be appended

  Scenario: Compute month-to-date performance metrics
    Given portfolio daily returns for the month
    When monthly metrics are computed
    Then the following metrics should be produced:
      | Metric                              | Description                                      |
      | winning_percentage_pct              | % of trading days with positive returns          |
      | consistency_score_periods           | longest consecutive positive-return streak       |
      | rolling_return_consistency_pct      | rolling return consistency over N days           |
      | n_trading_days_in_month             | number of trading days observed                  |
      | first_trading_day                   | first trading date in ISO format                 |
      | last_trading_day                    | last trading date in ISO format                  |

  Scenario: Persist monthly metrics with historical tracking
    Given month-to-date metrics are computed
    When metrics are persisted to MongoDB
    Then a single document per month_id should exist
    And last_metrics should be updated with the newest values
    And last_snapshot_date should reflect the execution date
    And portfolio_daily_returns should be refreshed
    And a new entry should be appended to metrics_history

  Scenario: Preserve monthly metrics history across executions
    Given the service runs multiple times during the same month
    When each execution completes
    Then previous metrics_history entries should remain unchanged
    And each run should append a new snapshot
    And the dashboard should be able to visualize metric evolution over time

  Scenario: Ensure idempotent daily execution
    Given the service runs more than once on the same day
    When metrics are recomputed
    Then no historical data should be deleted
    And the latest metrics should reflect the most recent computation
    And the service should complete safely without corruption

  Scenario: Support monthly dashboard visualizations
    Given monthly metrics are stored in MongoDB
    When the dashboard queries the monthly collection
    Then it should be able to render:
      | month-to-date performance              |
      | daily return timeline                  |
      | winning-day percentage indicator       |
      | consistency and streak analytics       |
      | rolling return stability visualization |
      ...
