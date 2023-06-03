import os
import csv
import datetime
import time

class FileSystemStats:
    def __init__(self):
        self.total_files = 0
        self.total_dirs = 0
        self.total_size = 0  # in bytes

    def analyze(self, start_path):
        start_time = time.time()
        root_dirs = os.listdir(start_path)
        for count, root_dir in enumerate(root_dirs, 1):
            for root, dirs, files in os.walk(os.path.join(start_path, root_dir)):
                self.total_dirs += len(dirs)
                self.total_files += len(files)
                for name in files:
                    try:
                        self.total_size += os.path.getsize(os.path.join(root, name))
                    except OSError as e:
                        print(f"OS error occurred: {e}")
            elapsed_time = time.time() - start_time
            avg_time_per_dir = elapsed_time / count
            remaining_time = avg_time_per_dir * (len(root_dirs) - count)
            print(f"Processed {count} out of {len(root_dirs)} root directories. Estimated time remaining: {remaining_time} seconds.")

    def to_dict(self):
        return {
            'total_dirs': self.total_dirs,
            'total_files': self.total_files,
            'total_size': self.total_size,
        }

def write_stats_to_csv(stats, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=stats.keys())
        writer.writeheader()
        writer.writerow(stats)

def main():
    start_path = os.path.abspath(os.sep)  # get root directory
    stats = FileSystemStats()
    stats.analyze(start_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    write_stats_to_csv(stats.to_dict(), f'FileSystemStats_{timestamp}.csv')

if __name__ == "__main__":
    main()
