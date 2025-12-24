Feature: ChronoBridge synchronized data pipeline
  The system should build time-aligned datasets from market and news data
  For training, inference, and backtesting workflows

  Scenario: Resample OHLCV data to 3-minute resolution
    Given OHLCV and News data exists at 1-minute resolution 
    When ChronoBridge performs data_ingest_service
    Then the OHLCV data should be fetched and resampled to 3-minute bars
    And the chuncked News data should be fetched
    And time order should be preserved
    And gaps should be explicitly represented 
    And push resample OHLCV and News to Redis
    
  Scenario: Build joint features from OHLCV and news
    Given resampled OHLCV data exists
    And news data exists in the same time window
    When the features_service runs in its mode
    Then OHLCV-based features should be generated
    And news-based features should be time-aligned
    And the final feature set should be keyed by timestamp
    Given news items contain text content
    When the embedding stage runs
    Then BigBird embeddings should be computed
    And embedding vectors should be attached to feature rows
    When the embedding is NaN for some timestamp 
    Then it filled it by "no news vec" embedding vector

  Scenario: Produce training artifacts
    Given ChronoBridge features_service runs in train mode
    When dataset construction completes
    Then train.parquet and val.parquet for NeuralFusionCore model should be written
    And normalizer.pkl and meta.json should describe schema, symbols, and normalization
    Given train.parquet, val.parque, normalizer.pkl and meta.json exist
    Then NeuralFusionCore train_service used this files for its training loop.
    Given ChronoBridge features_service runs in bridge mode
    When dataset construction completes
    Then bridge_not_norm.parquet
    And bridge.parquet for NetWeaver model should be written
    Given bridge parquet files
    When Chronobridge_service runs 
    Then this bridge parquet files should be save to mongodb chrono_bridge collection
    Given chrono_bridge collection
    When chronobridge_api_service runs
    When NetWeaver_train_service runs
    Then this NetWeaver_train_service requests to chronobridge_api_service for its target training window 

    

  Scenario: Produce online inference artifacts
    Given ChronoBridge_service runs in synchronized mode
    When dataset construction completes
    And the dataset should include only the latest inference window
    Then online_test.parquet for NeuralFusionCore predictin_service should be produced
    And online_bridge_not_norm.parquet
    And online_bridge.parquet should be produced
    And saved in chrono_bridge collection
    When chronobridge_api_service runs
    Then NetWeaver_prediction_service should request request for traget window
    