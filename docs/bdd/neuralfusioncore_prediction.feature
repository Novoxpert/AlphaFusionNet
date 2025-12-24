Feature: NeuralFusionCore weight prediction
  The system should produce risk-aware portfolio weights
  Using Cross‑Gated Attention Fusion model

  Scenario: Generate symbol weights for a window
    Given online_test.parquet exists for the target window 
    And NeuralFusionCore model weights
    When NeuralFusionCore runs prediction_service
    Then a weight should be produced for each configured symbol
    Then all weights should be saved in mongodb NeuralFusionCore_predictions collection by its predicted timestamp
    
  
