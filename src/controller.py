from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
from .quant_alphafusionnet import QuantAlphaFusionNet
from .llm_alphafusionnet import OpenAI_LLM
import json

class LLMAlphaFusionNetController:
    def __init__(self, quant_core: QuantAlphaFusionNet, llm_client: OpenAI_LLM):
        self.quant = quant_core
        self.llm = llm_client
        self.default_alpha = 0.7
        self.default_method = "rank"
        self.default_gross_net = 0.5

    def build_prompt(self, w_neural: Dict[str,float], s_net: Dict[str,float], notes: Optional[str]=None) -> str:
        return json.dumps({"neuralfusioncore_weights": w_neural,"netweaver_scores":s_net,"notes": notes or ""})

    def validate_policy(self, raw_policy: Dict[str, Any]) -> Dict[str, Any]:
        policy = {}
        policy["alpha"] = float(np.clip(raw_policy.get("alpha", self.default_alpha), 0,1))
        method = raw_policy.get("method", self.default_method)
        policy["method"] = method if method in {"rank","softmax","proportional"} else self.default_method
        policy["gross_net"] = float(np.clip(raw_policy.get("gross_net", self.default_gross_net),0,2))
        policy["topk_net"] = int(raw_policy["topk_net"]) if str(raw_policy.get("topk_net")).isdigit() else None
        policy["topk_final"] = int(raw_policy["topk_final"]) if str(raw_policy.get("topk_final")).isdigit() else None
        policy["overrides"] = raw_policy.get("overrides",{})
        policy["sector_multipliers"] = raw_policy.get("sector_multipliers",{})
        policy["reasoning"] = str(raw_policy.get("reasoning",""))[:2000]
        return policy

    def decide(
        self, w_neural: Dict[str, float], s_net: Dict[str, float], notes: Optional[str] = None
    ) -> Dict[str, Any]:

        prompt_json = self.build_prompt(w_neural, s_net, notes)
        try:
            raw_policy = self.llm.request_policy(prompt_json)
        except Exception as e:
            raw_policy = {"reasoning": f"LLM call failed: {e}"}

        policy = self.validate_policy(raw_policy)

        # Convert NetWeaver scores to weights
        w_net_conv = self.quant.convert_netweaver_to_weights(
            s_net,
            gross_net=policy.get("gross_net", 0.5),
            method=policy.get("method", "rank"),
            topk=policy.get("topk_net"),
        )

        # Apply fusion coefficient alpha and fuse
        self.quant.alpha = policy.get("alpha", 0.7)
        final_weights = self.quant.fuse_signed(
            w_neural=w_neural,
            w_net=w_net_conv,
            overrides=policy.get("overrides"),
            sector_multipliers=policy.get("sector_multipliers"),
            renormalize_after_clip=False,  # normalize at the very end
        )

        # Apply post-filter top-k final if specified
        if policy.get("topk_final") is not None:
            abs_sorted = final_weights.abs().sort_values(ascending=False)
            topk_idxs = abs_sorted.head(policy["topk_final"]).index
            final_weights = final_weights.reindex(topk_idxs).copy()

        # Normalize final weights to gross exposure
        total_abs = final_weights.abs().sum()
        if total_abs > 0:
            final_weights = (self.quant.gross / total_abs) * final_weights
        else:
            final_weights[:] = 0.0

        return {"policy": policy, "final_weights": final_weights, "w_net_converted": pd.Series(w_net_conv)}
