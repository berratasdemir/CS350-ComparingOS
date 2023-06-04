import subprocess
import os
import time

import psutil as psutil

class BootTimeAnalyzer:
    def measure_boot_time(self):
        # Reboot the system
        subprocess.call(["shutdown", "-r", "-t", "0"])

        # Measure the time taken for the system to become responsive after reboot
        start_time = time.time()
        while True:
            try:
                # Try to execute a command that should be available after boot-up
                subprocess.check_output("dir", shell=True)
                end_time = time.time()
                boot_time = end_time - start_time
                break
            except subprocess.CalledProcessError:
                # Command failed, system is not yet ready
                pass

        return boot_time


class FileAnalyzer:
    def analyze_file(self, file_path):
        start_time = time.time()

        # Perform file analysis tasks
        file_size = os.path.getsize(file_path)
        file_extension = os.path.splitext(file_path)[1]
        file_modified_time = os.path.getmtime(file_path)

        end_time = time.time()
        execution_time = end_time - start_time

        return file_size, file_extension, file_modified_time, execution_time


class SystemAnalyzer:
    def measure_memory_usage(self):
        memory_usage = psutil.virtual_memory().used / (1024 ** 2)  # Memory usage in megabytes
        return memory_usage

    def measure_cpu_utilization(self):
        cpu_utilization = psutil.cpu_percent()
        return cpu_utilization


if __name__ == "__main__":
    boot_time_analyzer = BootTimeAnalyzer()
    file_analyzer = FileAnalyzer()
    system_analyzer = SystemAnalyzer()

    #boot time
    boot_time = boot_time_analyzer.measure_boot_time()
    # Save the boot-up time to a text file
    with open("boot_time.txt", "w") as file:
        file.write(f"Boot-up Time: {boot_time} seconds")

    # file analyzer
    file_path = "test.txt"


    file_size, file_extension, file_modified_time, execution_time = file_analyzer.analyze_file(file_path)

    print("File Size:", file_size, "bytes")
    print("File Extension:", file_extension)
    print("File Modified Time:", file_modified_time)
    print("Execution Time:", execution_time, "seconds")

    # Measure memory usage
    memory_usage = system_analyzer.measure_memory_usage()
    print("Memory Usage:", memory_usage, "MB")

    # Measure CPU utilization
    cpu_utilization = system_analyzer.measure_cpu_utilization()
    print("CPU Utilization:", cpu_utilization, "%")
