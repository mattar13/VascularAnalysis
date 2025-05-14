import sys
print("\nBefore cleanup:")
for p in sys.path:
    print(p)

# Remove the old project path
sys.path = [p for p in sys.path if 'VesselTracing' not in p]

print("\nAfter cleanup:")
for p in sys.path:
    print(p)

print("\nTrying to import:")
try:
    from DataManager import DataManager
    print("DataManager import successful!")
except ImportError as e:
    print(f"Import error: {e}")

print("\nTrying to list src directory:")
import os
print(os.listdir("src"))