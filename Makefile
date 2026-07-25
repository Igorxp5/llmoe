CC       = gcc
CFLAGS   = -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers \
           -Wno-type-limits -Wno-unused-function \
           -fPIC -ffunction-sections -fdata-sections \
           -g -O2 \
           -DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl

# ── IREE installation (built via CMake) ──────────────────────────────────────
IREE_SRC_DIR  = vendor/iree/runtime/src
IREE_FLATCC   = vendor/iree/third_party/flatcc/include
IREE_PREFIX   = build/iree-install
IREE_LIB_DIR  = $(IREE_PREFIX)/lib
IREE_STAMP    = $(IREE_PREFIX)/.stamp

INCS      = -I$(IREE_SRC_DIR) -I$(IREE_FLATCC) -I. -Isrc

# ── IREE static libraries (tokenizer, base, flatcc) ─────────────────────────
# Only tokenizer+base are linked; HAL/VM/ukernel are excluded to avoid
# mutually-exclusive arch-specific implementations causing duplicate symbols.
IREE_LIBS  = -Wl,--start-group \
             $(shell find $(IREE_LIB_DIR) -name 'libiree_tokenizer*.a' 2>/dev/null | sort) \
             $(shell find $(IREE_LIB_DIR) -name 'libiree_base*.a' 2>/dev/null | sort) \
             $(shell find $(IREE_LIB_DIR) -name 'libiree_testing_benchmark.a' 2>/dev/null) \
             $(shell find $(BUILD_DIR)/iree-build/build_tools/third_party/flatcc -name 'libflatcc*.a' 2>/dev/null | sort) \
             -Wl,--end-group

LDFLAGS  = -static -Wl,--gc-sections
LDLIBS   = $(IREE_LIBS) -lpthread -lm -ldl

# ── Project sources ─────────────────────────────────────────────────────────
PROJ_SRCS = $(shell find src -name '*.c' 2>/dev/null | sort)

# ── Object files ────────────────────────────────────────────────────────────
BUILD_DIR = ./build
OBJS      = $(patsubst src/%.c,$(BUILD_DIR)/src/%.o,$(PROJ_SRCS))

TEST_SRCS = $(wildcard tests/*.c)
TEST_OBJS = $(patsubst %.c,$(BUILD_DIR)/%.o,$(TEST_SRCS))

.PHONY: all clean distclean prepare test iree

all: prepare $(IREE_STAMP) $(BUILD_DIR)/main

prepare:
	@mkdir -p $(BUILD_DIR) $(sort $(dir $(OBJS) $(TEST_OBJS)))

# ── IREE auto-build (runs CMake configure + build + install once) ────────────
$(IREE_STAMP):
	@echo "==> Building IREE from vendor/iree (this may take a while)..."
	cmake -G Ninja -B $(BUILD_DIR)/iree-build -S vendor/iree \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DCMAKE_C_COMPILER=clang \
	    -DCMAKE_CXX_COMPILER=clang++ \
	    -DIREE_BUILD_COMPILER=OFF \
	    -DIREE_BUILD_TESTS=OFF \
	    -DIREE_BUILD_SAMPLES=OFF \
	    -DIREE_BUILD_BENCHMARKS=OFF \
	    -DIREE_BUILD_PYTHON_BINDINGS=OFF \
	    -DIREE_BUILD_BINDINGS_TFLITE=OFF \
	    -DIREE_BUILD_BINDINGS_TFLITE_JAVA=OFF \
	    -DIREE_HAL_DRIVER_DEFAULTS=OFF \
	    -DIREE_ERROR_ON_MISSING_SUBMODULES=OFF
	cmake --build $(BUILD_DIR)/iree-build -j$$(nproc)
	cmake --install $(BUILD_DIR)/iree-build --prefix $(IREE_PREFIX) \
	    --component IREEDevLibraries-Runtime
	@touch $@
	@echo "==> IREE build complete"

iree: $(IREE_STAMP)

# ── Compile project sources ─────────────────────────────────────────────────
$(BUILD_DIR)/src/%.o: src/%.c
	$(CC) $(CFLAGS) $(INCS) -c -o $@ $<

$(BUILD_DIR)/tests/%.o: tests/%.c
	$(CC) $(CFLAGS) $(INCS) -c -o $@ $<

# ── Link ────────────────────────────────────────────────────────────────────
$(BUILD_DIR)/main: $(OBJS) | $(IREE_STAMP)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

$(BUILD_DIR)/test_runner: $(TEST_OBJS) $(filter-out $(BUILD_DIR)/src/main.o,$(OBJS)) | $(IREE_STAMP)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(TEST_OBJS) $(filter-out $(BUILD_DIR)/src/main.o,$(OBJS)) $(LDLIBS)

# ── Tests ───────────────────────────────────────────────────────────────────
test: prepare $(BUILD_DIR)/test_runner
	$(BUILD_DIR)/test_runner

# ── Clean ───────────────────────────────────────────────────────────────────
clean:
	rm -rf $(BUILD_DIR)/*.o $(BUILD_DIR)/src $(BUILD_DIR)/tests
	rm -f $(BUILD_DIR)/main $(BUILD_DIR)/test_runner

distclean:
	rm -rf $(BUILD_DIR)
