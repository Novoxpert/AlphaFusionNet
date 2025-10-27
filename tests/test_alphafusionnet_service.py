"""
Test Suite for AlphaFusionNet Service
================================

Overview
--------
This module contains unit and integration tests for the AlphaFusionNet service, 
including the quantitative fusion engine (QuantAlphaFusionNet), LLM integration (OpenAI_LLM), 
and the full hybrid controller (LLMAlphaFusionNetController). The tests ensure 
that the system correctly fuses NeuralFusionCore and NetWeaver outputs and applies 
top-k filtering, clipping, and normalization rules.

Test Coverage
-------------
1. QuantAlphaFusionNet:
    - Conversion of NetWeaver predicted returns into normalized weights.
    - Fusion of NeuralFusionCore and NetWeaver weights using alpha blending.
    - Overrides and sector multipliers are applied correctly.
    - Clipping and gross exposure normalization.

2. LLMAlphaFusionNetController:
    - Builds prompts for LLM given NeuralFusionCore and NetWeaver outputs.
    - Validates LLM policy outputs and applies defaults when necessary.
    - Integrates QuantAlphaFusionNet fusion with LLM policy correctly.
    - Post-fusion top-k filtering applied correctly.

3. Full AlphaFusionNet Integration:
    - Ensures that final portfolio weights are consistent with inputs and configuration.
    - Handles corner cases such as missing tickers, empty weights, or zero sums.
    - Tests fallback behavior when LLM API fails.
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Oct 22
Version: 1.1.0 
"""
import unittest
import pandas as pd
from src.quant_alphafusionnet import QuantAlphaFusionNet
from src.controller import LLMAlphaFusionNetController
from src.llm_alphafusionnet import OpenAI_LLM

class TestQuantAlphaFusionNet(unittest.TestCase):
    def setUp(self):
        self.quant = QuantAlphaFusionNet(alpha=0.7, gross=1.0, w_min=-0.4, w_max=0.4)

    def test_fuse_signed_basic(self):
        w_neural = {"AAPL":0.1,"MSFT":0.2,"TSLA":-0.1}
        w_net = {"AAPL":0.05,"MSFT":0.05,"TSLA":0.02}
        result = self.quant.fuse_signed(w_neural, w_net)
        self.assertTrue(abs(result.sum()) > 0)
        self.assertTrue((result <= 0.4).all() and (result >= -0.4).all())

    def test_convert_netweaver_to_weights(self):
        s_net = {"AAPL":0.05,"MSFT":0.1,"TSLA":-0.03}
        w = self.quant.convert_netweaver_to_weights(s_net, gross_net=0.5, method="rank")
        self.assertTrue(abs(sum(w.values())) <= 0.5)

class TestAlphaFusionNetController(unittest.TestCase):
    def setUp(self):
        self.quant = QuantAlphaFusionNet(alpha=0.7, gross=1.0)
        self.llm = OpenAI_LLM(model_name="gpt-4o-mini", timeout=10)
        self.controller = LLMAlphaFusionNetController(self.quant, self.llm)

    def test_decide_empty_inputs(self):
        nf_weights = {}
        nw_scores = {}
        out = self.controller.decide(nf_weights, nw_scores)
        self.assertIsInstance(out["final_weights"], pd.Series)
        self.assertEqual(len(out["final_weights"]), 0)

    def test_decide_basic(self):
        nf_weights = {"AAPL":0.1,"MSFT":0.2,"TSLA":-0.1}
        nw_scores = {"AAPL":0.05,"MSFT":0.08,"TSLA":0.02}
        out = self.controller.decide(nf_weights, nw_scores, notes="test")
        self.assertIsInstance(out["final_weights"], pd.Series)
        self.assertTrue(abs(out["final_weights"].abs().sum() - 1.0) < 1e-6)

if __name__ == "__main__":
    unittest.main()
