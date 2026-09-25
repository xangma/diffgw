/* Development-only extra orbital-state capture on very fine source grids. */
#include <stddef.h>

static double *diffgw_state_data;
static size_t diffgw_state_capacity;
static size_t diffgw_state_count;

#include "full_v4_source_nqc_inputs_very_fine_adapter.c"

int diffgw_v4_full_source_nqc_states(
    double mass1, double mass2, double chi1, double chi2,
    double requested_omega, double step_M, size_t capacity,
    size_t *count, double *meta, double *data, double *coefficients,
    double *state_data) {
  if (!state_data || !count || capacity < 10 || capacity > 100000) return 1;
  diffgw_state_data = state_data;
  diffgw_state_capacity = capacity;
  diffgw_state_count = 0;
  int result = diffgw_v4_full_source_nqc_inputs(
      mass1, mass2, chi1, chi2, requested_omega, step_M, capacity,
      count, meta, data, coefficients);
  diffgw_state_data = NULL;
  if (!result && diffgw_state_count != *count) result = 5;
  return result;
}
