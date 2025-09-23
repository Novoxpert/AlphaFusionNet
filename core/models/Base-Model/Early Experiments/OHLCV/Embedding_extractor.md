# TimesNet Embedding Extractor

This notebook extracts backbone embeddings from a trained TimesNet model for time series analysis, specifically designed for Bitcoin OHLCV (Open, High, Low, Close, Volume) data classification tasks which is trained in the `thuml_timesnet.ipynb`.

## Purpose

Now that the TimesNet model is nearly trained, we can extract its embeddings and prepare them for fusion with the news embeddings we'll generate next.

## Dataset

The notebook works with Bitcoin OHLCV data containing:
- **Date**: Timestamp information
- **OHLCV**: Open, High, Low, Close, Volume price data
- **Returns**: Pre-computed return values (`pre_return`, `return`)
- **Total Features**: 7 input features + 1 target variable


## Model Configuration

### TimesNet Parameters
```python
config = {
    'model': 'TimesNet',
    'task_name': 'long_term_forecast',
    'seq_len': 48,           # Input sequence length
    'label_len': 24,         # Label sequence length  
    'pred_len': 1,           # Prediction horizon
    'd_model': 64,           # Model dimension
    'e_layers': 1,           # Encoder layers
    'num_kernels': 6,        # Inception kernels
    'enc_in': 12,            # Input features
    'batch_size': 32,        # Batch size
    'dropout': 0.1           # Dropout rate
    ...
}
```

## Key Functions

### `debug_model_structure(checkpoint_path, data_config)`
Analyzes the TimesNet model architecture and prints all available layers for hook registration.

**Parameters:**
- `checkpoint_path`: Path to model checkpoint or 'auto' for random initialization
- `data_config`: Dictionary containing model configuration

**Returns:**
- Initialized experiment object

### `extract_embeddings_with_multiple_hooks(exp, target_layers=None)`
Extracts embeddings from multiple model layers simultaneously.

**Parameters:**
- `exp`: Experiment object from `debug_model_structure()`
- `target_layers`: List of layer names to target (optional)

**Returns:**
- Dictionary mapping layer names to extracted embeddings

### `extract_final_embeddings(exp, layer_name)`
Extracts embeddings from a specific layer across all test data.

**Parameters:**
- `exp`: Experiment object
- `layer_name`: Name of the target layer

**Returns:**
- NumPy array of shape `(num_samples, embedding_dim)`

## Model Architecture

The TimesNet model consists of:

1. **Data Embedding Layer**: Converts input sequences to embeddings
   - Token embedding via 1D convolution
   - Positional embedding
   - Temporal embedding
   
2. **TimesBlock**: Core processing module
   - Inception blocks with multiple kernel sizes
   - GELU activation
   - Multi-scale feature extraction

3. **Output Layers**:
   - Layer normalization
   - Linear projection for final predictions


## Fixed Issues & Solutions

### Find the Best Match Embeddings
By hooking into all the layers of the TimesNet we found the layer that give us the most
meaningful embeddings and then extract them from there

## What to Do

### Train Better
Earned a more accurate and better trained model before strating embedding extractions