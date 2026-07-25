#!/usr/bin/env python3
import argparse
import sys

try:
    from tokenizers import Tokenizer
except ImportError:
    print("Error: 'tokenizers' library is not installed.", file=sys.stderr)
    print("Install it with: pip install tokenizers", file=sys.stderr)
    sys.exit(1)

try:
    from tokenizers.pre_tokenizers import ByteLevel
except ImportError:
    ByteLevel = None  # older versions


def cmd_vocab(tokenizer):
    print(tokenizer.get_vocab_size())


def cmd_tokenize(tokenizer, text):
    encoding = tokenizer.encode(text)
    print(" ".join(str(tid) for tid in encoding.ids))


def main():
    parser = argparse.ArgumentParser(
        description="CLI for operating on a tokenizer.json file."
    )
    parser.add_argument(
        "tokenizer_path", help="Path to the tokenizer.json file"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("vocab", help="Print total vocabulary size")

    tokenize_parser = subparsers.add_parser("tokenize", help="Tokenize a string")
    tokenize_parser.add_argument("text", help="Text to tokenize")

    args = parser.parse_args()

    try:
        tokenizer = Tokenizer.from_file(args.tokenizer_path)
    except FileNotFoundError:
        print(
            f"Error: file not found: {args.tokenizer_path}", file=sys.stderr
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        sys.exit(1)

    if args.command == "vocab":
        cmd_vocab(tokenizer)
    elif args.command == "tokenize":
        cmd_tokenize(tokenizer, args.text)


if __name__ == "__main__":
    main()
