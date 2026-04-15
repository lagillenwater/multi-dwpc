#!/usr/bin/env python3
"""Read a value from a JSON file by key, or print a list space-separated.

Usage:
    python3 scripts/read_json_value.py <json_path> <key>
    python3 scripts/read_json_value.py <json_path>
"""
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)

if len(sys.argv) > 2:
    print(data[sys.argv[2]])
else:
    print(" ".join(str(x) for x in data))
