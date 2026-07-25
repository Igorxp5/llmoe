CC = gcc
CFLAGS = -Wall -I. -Isrc -Ivendor

SRCS = $(wildcard src/*.c vendor/*.c)
BUILD_DIR = ./build

.PHONY: all clean prepare test

all: prepare $(BUILD_DIR)/main

prepare:
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/main: $(SRCS)
	$(CC) $(CFLAGS) -static -o $@ $(SRCS)

TEST_SRCS = tests/main.c tests/test_normalizer.c src/tokenizer/normalizer.c \
            vendor/utf8proc/utf8proc.c
TEST_INC  = tests/generated/nfc_tests.inc

$(TEST_INC): tests/resources/NormalizationTest.txt scripts/generate_nfc_tests.py
	@mkdir -p tests/generated
	python3 scripts/generate_nfc_tests.py tests/resources/NormalizationTest.txt $@

test: $(TEST_INC)
	$(CC) $(CFLAGS) -o $(BUILD_DIR)/test_runner $(TEST_SRCS)
	$(BUILD_DIR)/test_runner

clean:
	rm -rf $(BUILD_DIR)
