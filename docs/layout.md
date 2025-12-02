## Repository Layout
This repository is organized into a modular, multi-application architecture designed for scalability and clear separation of concerns. The apps/ directory contains the three main components of the system—ChronoBridge, NeuralFusionCore, and NetWeaver—each with its own data pipelines, models, feature engines, services, and utilities. Shared resources such as database helpers, trading calendar utilities, and metric tools reside in the top-level lib/ folder, while global configuration files, orchestration scripts, logs, and scheduler logic are grouped into their respective directories. The tests/ suite provides comprehensive coverage across services, models, and integrations (Mongo, Redis, ClickHouse). This structured layout ensures maintainability, modular development, and efficient navigation across all subsystems of the AlphaFusionNet platform.

```
AlphaFusionNet
│
│
├── apps/
│    ├── ChronoBridge/
│    │       ├── data/      
│    │       │   └── processed/
│    │       │       └── show_files.py  
│    │       ├── models/
│    │       ├── src/
│    │       │     └── inference.py  
│    │       ├── scripts/
│    │       │     ├── chronobridge_api_service.py 
│    │       │     ├──  chronobridge_service.py
│    │       │     ├──  data_ingest_service.py
│    │       │     └── features_service.py
│    │       ├── lib/
│    │       │   ├── features.py         
│    │       │   ├── market.py
│    │       │   ├── news.py
│    │       │   ├── redis_utils.py
│    │       │   └── utils.py
│    │       └──apps/
│    │              └──NeuralFusionCore
│    │
│    ├── NeuralFusionCore/
│    │     ├── data/
│    │     │   ├── outputs/
│    │     │   │   └── model_weights.pt        
│    │     │   └── processed/
│    │     │       └── show_files.py                   
│    │     │   
│    │     ├── lib/
│    │     │   ├── backtest.py
│    │     │   ├── backtest_weights.py        
│    │     │   ├── dataset.py
│    │     │   ├── features.py
│    │     │   ├── loss_weights.py            
│    │     │   ├── market.py
│    │     │   ├── model.py
│    │     │   ├── news.py
│    │     │   ├── redis_utils.py
│    │     │   ├── train.py
│    │     │   └── utils.py
│    │     ├──_init__.py
│    │     ├── README.md
│    │     ├── requirements.txt
│    │     ├── config.py
│    │     └── scripts/
│    │           ├── train_service.py
│    │           ├── finetune_service.py
│    │           ├── prediction_service.py 
│    │           ├── backtesting_service.py
│    │           └── api_service.py
│    │
│    └── NetWeaver/
│          ├── src/
│          │   ├── data_pipeline.py        
│          │   ├── train.py                 
│          │   ├── run_analysis.py          
│          │   ├── parse_arg.py            
│          │   └── utils/
│          │       ├── data_utils.py        
│          │       ├── metrics_utils.py     
│          │       ├── analysis_utils.py    
│          │       └── training_logger.py   
│          ├── data/
│          │   ├── raw/
│          │   │   ├── SP500_dataset/
│          │   │   └── *.npy (graph files)
│          │   └── processed/
│          │       └── model_data.pickle
│          └── results/
│              ├── saved_models/            
│              ├── train/                   
│              └── analysis/                
│
├── data/
│    └──trading_days_cache.json
├── lib/
│    ├──db_utils.py
│    ├──trading_calendar_utils.py
│    └──metric_utils.py 
│ 
│
├── scripts/
│    ├── _init__.py
│    ├── alphafusionnet_api_service.py
│    ├── alphafusionnet_service.py
│    ├── compute_trading_days_service.py
│    ├── metric_backtesting.py
│    ├── show_backtest_metrics.py
│    ├── metric_live_service.py
│    ├── metric_monthly_service.py 
│    ├── future_testing_service.py 
│    └── furure_testing_api_service.py 
│  
├── src/
│     ├── contoller.py
│     ├── TradingAgent.py
│     ├── llm_alphafusionnet.py
│     └── quant_alphafusionnet.py
│
├── config/
│     └──AFN_config.yml
│
├── logs/
│     └──alphafusionnet_service.log
│
├── tests/
│    ├──_init__.py
│    ├── test_alphafusionnet_api_service.py
│    ├── test_alphafusionnet_service.py
│    ├── test_api_service.py
│    ├── test_chronobridge_api_service.py
│    ├── test_chronobridge_service.py
│    ├── test_future_testing_api_service.py 
│    ├── test_future_testing_service.py 
│    ├── test_netweaver_finetune_service.py
│    ├── test_netweaver_train_service.py
│    ├── test_netweaver_prediction_service.py
│    ├── test_neuralfusioncore_api_service.py
│    ├── test_neuralfusioncore_backtesting_service.py
│    ├── test_neuralfusioncore_data_ingest_service.py
│    ├── test_neuralfusioncore_features_service.py
│    ├── test_neuralfusioncore_finetune_service.py
│    ├── test_health_clickhouse.py
│    ├── test_health_mongo.py
│    ├── test_health_redis.py
│    ├── test_neuralfusioncore_model.py
│    ├── test_neuralfusioncore_prediction_service.py
│    ├── test_redis_clickhouse_integration.py
│    ├── test_redis_mongo_integration.py
│    └── test_neuralfusioncore_train_service.py
│
├── README.md
├── scheduler/
│     ├── schaduler.py
│     ├── tasks.py
│     ├── test_celery_func.py
│     └──trigger.py
├── requirements.txt
└── pytest.ini
```
> Any folders missing on your machine will be created by the scripts if needed.
