#ifndef OLMOE_TEST_LAYER_H
#define OLMOE_TEST_LAYER_H

/* Loads the OLMoE model once and runs every layer-module check. Re-loading
 * the ~13 GiB model per check would be wasteful, so the granular checks are
 * static helpers inside test_layer.c; this public entry point dispatches
 * them and reports the combined failure count (per tests/test_main.c). */
int test_layer_loads_and_validates(void);

#endif /* OLMOE_TEST_LAYER_H */