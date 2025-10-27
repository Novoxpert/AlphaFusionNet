"""
trigger.py
Description: Manually triggers the one-time initial workflow
             (historical or first-time pipeline run) by sending the initial_run task to Celery.
Author: Elham Esmaeilnia(elham.e.shirvani@gmail.com)
Date: 2025 Oct 4
Version: 1.0.0 
"""
from tasks import initial_run
print("Triggering initial_run task...")
initial_run.delay()

