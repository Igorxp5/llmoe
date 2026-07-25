CC       = gcc
CFLAGS   = -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
           -Wno-type-limits -Wno-unused-function \
           -DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl \
           -fPIC -ffunction-sections -fdata-sections \
           -g -O2
LDFLAGS  = -static -Wl,--gc-sections -lpthread -lm

IREE_DIR  = vendor/iree/runtime/src

INCS      = -I$(IREE_DIR) -I. -Isrc

# ── IREE Tokenizer sources ──────────────────────────────────────────────────
IREE_TKZ  = $(IREE_DIR)/iree/tokenizer
IREE_TKZ_SRCS := $(shell find $(IREE_TKZ) -name '*.c' \
                 ! -path '*_test.cc'    ! -path '*_fuzz.cc' \
                 ! -path '*_benchmark*' ! -path '*/testing/*' \
                 ! -path '*/testdata/*' ! -path '*/tools/*')

# ── IREE Base sources (only what the tokenizer needs) ───────────────────────
IREE_BASE = $(IREE_DIR)/iree/base
IREE_BASE_SRCS = \
	$(IREE_BASE)/allocator.c          $(IREE_BASE)/allocator_libc.c \
	$(IREE_BASE)/bitfield.c           $(IREE_BASE)/bitmap.c \
	$(IREE_BASE)/printf.c             $(IREE_BASE)/status.c \
	$(IREE_BASE)/status_stack_trace.c $(IREE_BASE)/string_builder.c \
	$(IREE_BASE)/string_view.c        $(IREE_BASE)/time.c \
	$(IREE_BASE)/wait_source.c \
	$(IREE_BASE)/internal/unicode.c   $(IREE_BASE)/internal/unicode_tables.c \
	$(IREE_BASE)/internal/arena.c     $(IREE_BASE)/internal/atomic_slist.c \
	$(IREE_BASE)/internal/json.c      $(IREE_BASE)/internal/memory.c \
	$(IREE_BASE)/internal/time.c      $(IREE_BASE)/internal/base64.c \
	$(IREE_BASE)/internal/path.c      $(IREE_BASE)/internal/cpu.c \
	$(IREE_BASE)/internal/fpu_state.c $(IREE_BASE)/internal/csprng.c \
	$(IREE_BASE)/internal/spsc_queue.c $(IREE_BASE)/internal/mpsc_queue.c \
	$(IREE_BASE)/threading/thread.c   $(IREE_BASE)/threading/thread_pthreads.c \
	$(IREE_BASE)/threading/mutex.c    $(IREE_BASE)/threading/notification.c \
	$(IREE_BASE)/threading/affinity.c $(IREE_BASE)/threading/numa_fallback.c \
	$(IREE_BASE)/threading/wait_address.c \
	$(IREE_BASE)/tracing/console.c

# ── Project sources ─────────────────────────────────────────────────────────
PROJ_SRCS = $(wildcard src/*.c)

# ── All sources ─────────────────────────────────────────────────────────────
ALL_SRCS  = $(PROJ_SRCS) $(IREE_TKZ_SRCS) $(IREE_BASE_SRCS)

# ── Object files ────────────────────────────────────────────────────────────
BUILD_DIR = ./build
OBJS      = $(patsubst $(IREE_DIR)/%.c,$(BUILD_DIR)/%.o,$(IREE_TKZ_SRCS) $(IREE_BASE_SRCS))
OBJS     += $(patsubst src/%.c,$(BUILD_DIR)/src/%.o,$(PROJ_SRCS))

IREE_OBJS = $(patsubst $(IREE_DIR)/%.c,$(BUILD_DIR)/%.o,$(IREE_TKZ_SRCS) $(IREE_BASE_SRCS))

TEST_SRCS = $(wildcard tests/*.c)
TEST_OBJS = $(patsubst %.c,$(BUILD_DIR)/%.o,$(TEST_SRCS))

.PHONY: all clean prepare test

all: prepare $(BUILD_DIR)/main

prepare:
	@mkdir -p $(BUILD_DIR) $(sort $(dir $(OBJS) $(TEST_OBJS) $(IREE_OBJS)))

# ── Compile IREE sources ────────────────────────────────────────────────────
$(BUILD_DIR)/iree/%.o: $(IREE_DIR)/iree/%.c
	$(CC) $(CFLAGS) $(INCS) -c -o $@ $<

$(BUILD_DIR)/iree/tokenizer/%.o: $(IREE_DIR)/iree/tokenizer/%.c
	$(CC) $(CFLAGS) $(INCS) -c -o $@ $<

# ── Compile project sources ─────────────────────────────────────────────────
$(BUILD_DIR)/src/%.o: src/%.c
	$(CC) $(CFLAGS) $(INCS) -c -o $@ $<

$(BUILD_DIR)/tests/%.o: tests/%.c
	$(CC) $(CFLAGS) $(INCS) -c -o $@ $<

# ── Link ────────────────────────────────────────────────────────────────────
$(BUILD_DIR)/main: $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJS)

# ── Tests ───────────────────────────────────────────────────────────────────
$(BUILD_DIR)/test_runner: $(TEST_OBJS) $(IREE_OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_OBJS) $(IREE_OBJS)

test: $(BUILD_DIR)/test_runner
	$(BUILD_DIR)/test_runner

clean:
	rm -rf $(BUILD_DIR)
