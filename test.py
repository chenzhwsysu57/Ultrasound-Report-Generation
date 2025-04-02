import time
from concurrent.futures import ThreadPoolExecutor
import threading

def long_running_task(name, duration):
    print(f"Task {name} started.")
    time.sleep(duration)
    print(f"Task {name} completed.")

def run_executor():
    with ThreadPoolExecutor() as executor:
        executor.submit(long_running_task, "A", 5)
        executor.submit(long_running_task, "B", 3)

def main():
    # Run the executor in a separate thread
    executor_thread = threading.Thread(target=run_executor)
    executor_thread.start()

    # Main thread continues immediately
    print("Main thread is doing other work...")
    print("Main thread is still working...")

if __name__ == "__main__":
    main()
    print("done")
