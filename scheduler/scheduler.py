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

# Beat schedule
app.conf.beat_schedule = {
    # Daily workflow at 00:00
    'daily-update': {
        'task': 'tasks.daily_update',
        'schedule': crontab(hour=0, minute=0),
    },
    # 4-hourly predictions
    'prediction-4h': {
        'task': 'tasks.prediction_4h',
        'schedule': crontab(minute=0, hour='*/4'),
    },
}

app.conf.timezone = 'Asia/Tehran'  # change to custom timezone
