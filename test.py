import os
import sys

def count_files(directory):
    return sum(1 for entry in os.scandir(directory) if entry.is_file())

if __name__ == '__main__':
        
    directory = "merged_dataset"
    
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)
    
    file_count = count_files(directory)
    print(f"The number of files in directory {directory} is: {file_count}")