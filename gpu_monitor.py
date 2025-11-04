import pynvml
import time
import csv
import datetime

LOG_INTERVAL = 1.0   # seconds between logs
LOG_SECONDS = 100     # total duration
LOG_FILENAME = f'gpu_basic_telemetry_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

def get_gpu_stats(handle):
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # milliwatts to watts
    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "temperature_C": temp,
        "power_W": power,
        "mem_total_MB": mem.total / 1024**2,
        "mem_used_MB": mem.used / 1024**2,
        "mem_free_MB": mem.free / 1024**2
    }

def main():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    fields = ["timestamp", "temperature_C", "power_W", "mem_total_MB", "mem_used_MB", "mem_free_MB"]
    with open(LOG_FILENAME, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        print(f"Started simplified GPU telemetry logging: {LOG_FILENAME}")
        start_time = time.time()
        while (time.time() - start_time) < LOG_SECONDS:
            stats = get_gpu_stats(handle)
            writer.writerow(stats)
            print(stats)
            time.sleep(LOG_INTERVAL)
    pynvml.nvmlShutdown()
    print("Logging complete.")

if __name__ == "__main__":
    main()