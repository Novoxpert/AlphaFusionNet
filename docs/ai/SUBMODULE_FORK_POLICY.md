# Submodule + Fork Policy

## Policy
- External code should be consumed via Submodule or Fork.
- Best practice: rewrite each submodule URL to your fork (HTTPS).
- Make changes on your fork/branch.
- In your main repo, only update submodule pointers.
- Keep architecture outputs only in this repo: docs/ai/** and memory-bank/** and commit them.
- If a submodule is private: HTTPS requires credentials (Git Credential Manager / PAT). This is an access constraint, not an agent bug.

## Overrides
Edit: memory-bank/submodule_overrides.json

## Discovered .gitmodules

### .gitmodules
- submodule.apps/NeuralFusionCore.url = https://github.com/Novoxpert/NeuralFusionCore.git  (recommended: https://github.com/Novoxpert/NeuralFusionCore.git)
- submodule.apps/ChronoBridge.url = https://github.com/Novoxpert/ChronoBridge.git  (recommended: https://github.com/Novoxpert/ChronoBridge.git)
- submodule.apps/NetWeaver.url = https://github.com/Novoxpert/NetWeaver.git  (recommended: https://github.com/Novoxpert/NetWeaver.git)

### apps/ChronoBridge/.gitmodules
- submodule.apps/NeuralFusionCore.url = https://github.com/Novoxpert/NeuralFusionCore.git  (recommended: https://github.com/Novoxpert/NeuralFusionCore.git)

### apps/ChronoBridge/apps/NeuralFusionCore/.gitmodules
- submodule.apps/TimesNet.url = https://github.com/Novoxpert/Time-Series-Library.git  (recommended: https://github.com/Novoxpert/Time-Series-Library.git)

### apps/NetWeaver/.gitmodules
- submodule.apps/Financial-GraphAttention.url = https://github.com/NovoXpertCo/Financial-GraphAttention.git  (recommended: https://github.com/NovoXpertCo/Financial-GraphAttention.git)

### apps/NeuralFusionCore/.gitmodules
- submodule.apps/TimesNet.url = https://github.com/Novoxpert/Time-Series-Library.git  (recommended: https://github.com/Novoxpert/Time-Series-Library.git)

