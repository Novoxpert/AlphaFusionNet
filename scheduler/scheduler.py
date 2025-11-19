"""
scheduler.py
Description: Defines all Celery tasks for data ingestion, feature processing,
             model training, fine-tuning, and prediction 
             — executed sequentially or on schedule using Celery workers.
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Oct 4
Version: 1.1.0 
"""
from tasks import app
from celery.schedules import crontab

# Set timezone to UTC
app.conf.timezone = 'UTC'
app.conf.enable_utc = True

# Beat schedule
app.conf.beat_schedule = {
    # Daily workflow at 00:00 London time
    'daily-update': {
        'task': 'tasks.daily_update',
        'schedule': crontab(hour=0, minute=0),
    },
    # Prediction task at 14:00 London time
    'prediction-14pm': {
        'task': 'tasks.prediction_14PM',
        'schedule': crontab(hour=14, minute=0),
    },
    
    # Live test task at 18:05 UTC time
    'live-test-18-15': {
        'task': 'tasks.live_test_18PM_pluse_10min',
        'schedule': crontab(hour=18, minute=5),
    },
}