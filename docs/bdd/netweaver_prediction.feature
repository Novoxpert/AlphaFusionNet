Feature: NetWeaver graph-based return prediction
  The system should model cross-asset relationships
  To produce return forecasts per symbol and top-k

  Scenario: Predict returns for all symbols
    Given Chronobridge_api_service runs
    When NetWeaver runs netweaver_prediction_service
    Then requests to chronobridge api for the target data window
    Given NetWeaver Graph model weights
    Then a return prediction for each symbol
    And top-k should be produced
    And the results should be saved in csv in results prediction folder