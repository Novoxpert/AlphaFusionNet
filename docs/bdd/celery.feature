Feature: Scheduler pipeline 
  The system should have scheduler for all pipeline stages
  To guarantee deterministic and aligned execution

  Scenario: Enforce a shared window definition for prediction at 14:30 UTC
    When compute_trading_days_service runs
    Then trading_days_cache.json should be produced in data folder in alphafusionnet
    Given a pipeline window is defined by start and end UTC timestamps
    And trading_days_chache.json file
    When tasks worker
    And scheduler beat are triggered
    When it is a trading day
    Then ChronoBridge should use the exact window
    And NeuralFusionCore should consume the same window features
    And NetWeaver should consume the same window features
    And AlphaFusionNet should produce output for the same window identifier
    And Metrice system should calculate live and monthly metrics for the same window