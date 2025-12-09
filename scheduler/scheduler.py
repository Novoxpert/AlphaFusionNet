"""
scheduler.py
Description: Defines all Celery tasks for data ingestion, feature processing,
             model training, fine-tuning, and prediction 
             — executed sequentially or on schedule using Celery workers.
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Oct 4
Version: 1.1.0 
"""
from .tasks import app
from celery.schedules import crontab

# Set timezone to UTC
app.conf.timezone = 'UTC'
app.conf.enable_utc = True

# Beat schedule
app.conf.beat_schedule = {
    # Daily workflow at 00:00 UTC time
    'daily-update': {
        'task': 'tasks.daily_update',
        'schedule': crontab(hour=0, minute=0),
    },
    # Prediction task at 14:30 UTC time
    'prediction-14-30pm': {
        'task': 'tasks.prediction_14_30PM',
        'schedule': crontab(hour=13, minute=50),
    },
    # Refresh trading-days cache once per day (02:00 UTC)
    'refresh-trading-days-cache': {
        'task': 'tasks.refresh_trading_days_cache',
        'schedule': crontab(hour=2, minute=0),
    },
    
}