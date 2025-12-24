Feature: AlphaFusionNet policy decision layer
  The system should combine multiple model outputs
  To produce final portfolio decisions with reasoning

  Scenario: Combine NeuralFusionCore and NetWeaver outputs and Produce explainable reasoning
    Given NeuralFusionCore_predictions collection exist
    And NetWeaver return predictions and top-k csv exist
    When alphafusionnet_service runs
    Then NetWeaver prediction should be saved to netweaver_predictions collection in mongodb 
    Given OPENAI GPT for producing decision policy
    When llm applies the policy method produced
    Then final portfolio weights should be produced
     {
      "alpha": float,
      "method": str,
      "gross_net": 1.0,
      "topk_net": int | null,
      "topk_final": int | null,
      "overrides": dict,
      "sector_multipliers": dict,
      "reasoning": str
      }
    And portfolio constraints should be satisfied 
    Given stop-loss and take-profit parameters are configured
    And confidence random bound
    Then stop-loss and take-profit levels should be included per symbol
    And Generate risk controls
    Given final weights, risk controls, OPENAI GPT, notes="Moderate-high volatility, tech sector resilience" and timestamp
    Then a trading agent reasoning runs
    And the reasoning should reference contributing market signals for each produced weights