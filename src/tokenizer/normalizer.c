#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "vendor/utf8proc/utf8proc.h"

#include "tokenizer/limits.h"
#include "normalizer.h"

bool tkz_normalizer(tkz_normalizer_t type, uint8_t* in, uint8_t* out) {
    switch (type) {
        case NFC:
            utf8proc_uint8_t* norm = utf8proc_NFC(in);
            strncpy((char*)out, (char*)norm, strlen((char*)norm));
            utf8proc_free(norm);
            return true;
        break;
        default:
            return false;
    }
}
