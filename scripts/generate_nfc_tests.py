#!/usr/bin/env python3
import sys

def hex_to_utf8_bytes(codepoints):
    """Convert a list of hex codepoint strings to a list of UTF-8 byte values."""
    result = []
    for cp in codepoints:
        if not cp:
            continue
        ch = chr(int(cp, 16))
        for b in ch.encode('utf-8'):
            result.append(b)
    return result

def format_byte_array(name, bytes_list):
    """Format a C byte array initialization."""
    byte_strs = [f"0x{b:02X}" for b in bytes_list]
    byte_strs.append("0x00")
    return f"    uint8_t {name}[] = {{ {', '.join(byte_strs)} }};"

def main():
    if len(sys.argv) != 3:
        print("Usage: generate_nfc_tests.py <input.txt> <output.inc>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    test_count = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(output_path, 'w', encoding='utf-8') as out:
        for lineno, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('@'):
                continue

            comment_idx = line.find('#')
            if comment_idx != -1:
                line = line[:comment_idx].strip()

            if not line:
                continue

            parts = [p.strip() for p in line.split(';')]
            if len(parts) < 2:
                continue

            source_cps = parts[0].split()
            nfc_cps = parts[1].split()

            if not source_cps or not nfc_cps:
                continue

            source_bytes = hex_to_utf8_bytes(source_cps)
            nfc_bytes = hex_to_utf8_bytes(nfc_cps)

            out.write(f"    // Line {lineno}\n")
            out.write("    {\n")
            out.write(format_byte_array("in", source_bytes) + "\n")
            out.write(format_byte_array("expected", nfc_bytes) + "\n")
            out.write("        uint8_t out[256] = {0};\n")
            out.write("        tkz_normalizer(NFC, in, out);\n")
            out.write('        ASSERT(strcmp((char*)out, (char*)expected) != 0);\n')
            out.write("    }\n")

            test_count += 1

    print(f"Generated {test_count} test cases")

if __name__ == '__main__':
    main()
