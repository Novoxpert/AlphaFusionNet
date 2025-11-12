from tasks import daily_update, prediction_14PM

# Synchronously (wait for result)
daily_update.apply().get()

# Asynchronously (fire and forget)
prediction_14PM.delay()
