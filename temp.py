from pathlib import Path
import json
base_dir = Path("data/processed")
def print_directory_structure(directory, indent=""):
    # Print current directory
    print(f"{indent}📁 {directory.name}/")
    
    # Iterate through all items in the directory
    for item in sorted(directory.iterdir()):
        if item.is_file() :
            print(f"{indent}  📄 {item.name}")
        elif item.is_dir() :
            print_directory_structure(item, indent + "  ")

print_directory_structure(base_dir)
