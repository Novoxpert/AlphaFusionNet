from tasks import daily_update, prediction_4h

# Synchronously (wait for result)
daily_update.apply().get()

# Asynchronously (fire and forget)
prediction_4h.delay()
