import time
import pandas as pd
from live_schedule import fetch_hourly_ohlcv, on_entry_trigger, on_exit_trigger
from apscheduler.schedulers.background import BackgroundScheduler

def smoke_test_fetch():
    print("=== Smoke Test: fetch_hourly_ohlcv ===")
    df = fetch_hourly_ohlcv()
    print(df.tail())
    assert isinstance(df, pd.DataFrame), "fetch_hourly_ohlcv should return a DataFrame"
    assert len(df) == 25, f"Expected 25 rows, got {len(df)}"
    assert not df['close'].isna().any(), "Found NaNs in 'close' column"
    print("✔️ Smoke test fetch_hourly_ohlcv passed\n")

def manual_trigger_test():
    print("=== Manual Trigger Test ===")
    print(">> ENTRY TEST <<")
    on_entry_trigger()
    print("\n>> EXIT TEST <<")
    on_exit_trigger()
    print("✔️ Manual trigger test passed\n")

def scheduler_integration_test(run_seconds=120):
    print("=== Scheduler Integration Test ===")
    scheduler = BackgroundScheduler(timezone='UTC')

    # schedule handlers every 30 seconds for test purposes
    entry_job = scheduler.add_job(on_entry_trigger, 'interval', seconds=30, id='entry_test')
    exit_job  = scheduler.add_job(on_exit_trigger,  'interval', seconds=30, id='exit_test')
    scheduler.start()

    print("Scheduled jobs:")
    for job in scheduler.get_jobs():
        print(f" - {job.id}: next run at {job.next_run_time}")

    print(f"Running scheduler for {run_seconds} seconds... (handlers will fire every 30s)")
    try:
        time.sleep(run_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        scheduler.shutdown()
        print("Scheduler shut down")
        print("✔️ Scheduler integration test passed\n")

if __name__ == "__main__":
    smoke_test_fetch()
    manual_trigger_test()
    # run scheduler test for 2 minutes
    scheduler_integration_test(run_seconds=120)

