Feature: Portfolio monitoring dashboard
  The system should present portfolio outputs and metrics
  Through a real-time React-based interface

  Scenario: Display latest portfolio decisions
    Given final portfolio output exists for the latest window
    When the dashboard loads
    Then portfolio weights should be displayed per symbol
    And live metric parameters should be visible
    And reasoning text should be shown

  Scenario: Display metrics time series
    Given metrics windows collection exists
    When metrics charts are rendered
    Then entry, return, PnL, confidence, Sl, Tp, ... should be visualized over time

