#!/usr/bin/env python3
"""Generate src/olmoe/tokenizer_data.inc from tokenizer.json."""
import sys

def main():
    with open(sys.argv[1], "rb") as f:
        data = f.read()
    with open(sys.argv[2], "w") as f:
        f.write("static const uint8_t kTokenizerJsonData[] = {\n")
        for i, b in enumerate(data):
            if i % 12 == 0:
                f.write("  ")
            f.write(f"0x{b:02x}, ")
            if i % 12 == 11:
                f.write("\n")
        if len(data) % 12 != 0:
            f.write("\n")
        f.write("};\n")
        f.write(f"static const size_t kTokenizerJsonDataLen = {len(data)};\n")

if __name__ == "__main__":
    main()
