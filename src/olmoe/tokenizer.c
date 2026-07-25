#include "olmoe/tokenizer.h"

#include "iree/base/api.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/tokenizer.h"

#include "tokenizer_data.inc"

static iree_tokenizer_t* g_tokenizer = NULL;

static void ensure_tokenizer(void) {
  if (g_tokenizer) return;

  iree_string_view_t json = iree_make_string_view(
      (const char*)kTokenizerJsonData, kTokenizerJsonDataLen);
  iree_status_t status = iree_tokenizer_from_huggingface_json(
      json, iree_allocator_system(), &g_tokenizer);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    g_tokenizer = NULL;
  }
}

void olmoe_tokenize(uint8_t* in, uint16_t* out) {
  ensure_tokenizer();

  if (!g_tokenizer) {
    out[0] = 0xFFFF;
    return;
  }

  iree_tokenizer_token_id_t token_ids[LLMOE_OLMOE_MAX_OUTPUT_TOKENS];
  iree_tokenizer_token_output_t output =
      iree_tokenizer_make_token_output(token_ids, NULL, NULL, LLMOE_OLMOE_MAX_OUTPUT_TOKENS);

  iree_host_size_t token_count = 0;
  iree_status_t status = iree_tokenizer_encode(
      g_tokenizer,
      iree_make_cstring_view((const char*)in),
      IREE_TOKENIZER_ENCODE_FLAG_NONE,
      output, iree_allocator_system(), &token_count);

  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    out[0] = 0xFFFF;
    return;
  }

  iree_host_size_t limit =
      token_count < LLMOE_OLMOE_MAX_OUTPUT_TOKENS ? token_count : LLMOE_OLMOE_MAX_OUTPUT_TOKENS;
  for (iree_host_size_t i = 0; i < limit; i++) {
    out[i] = (uint16_t)token_ids[i];
  }
  out[limit] = 0xFFFF;
}
