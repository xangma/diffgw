/* Development-only public LALSuite C API probe; no LAL code is redistributed. */
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <lal/AVFactories.h>
#include <lal/LALConstants.h>
#include <lal/LALDict.h>
#include <lal/LALSimIMR.h>

static int parse_double(const char *text, double *value) {
    char *end;
    errno = 0;
    *value = strtod(text, &end);
    return errno == 0 && end != text && *end == '\0' && isfinite(*value);
}

int main(int argc, char **argv) {
    double m1, m2, chi1, chi2, omega_start, step_m;
    if (argc != 7 || !parse_double(argv[1], &m1)
            || !parse_double(argv[2], &m2)
            || !parse_double(argv[3], &chi1)
            || !parse_double(argv[4], &chi2)
            || !parse_double(argv[5], &omega_start)
            || !parse_double(argv[6], &step_m)
            || m1 <= 0 || m2 <= 0 || omega_start <= 0 || step_m <= 0)
        return 2;

    const double mass_seconds = (m1 + m2) * LAL_MTSUN_SI;
    REAL8Vector *coefficients = XLALCreateREAL8Vector(10);
    LALDict *pa = XLALCreateDict();
    LALDict *tgr = XLALCreateDict();
    SphHarmTimeSeries *modes = NULL;
    REAL8Vector *low = NULL, *high = NULL;
    if (!coefficients || !pa || !tgr) return 3;

    int rc = XLALSimIMRSpinAlignedEOBModes(
        &modes, &low, &high, step_m * mass_seconds,
        m1 * LAL_MSUN_SI, m2 * LAL_MSUN_SI,
        omega_start / (M_PI * mass_seconds), 100e6 * LAL_PC_SI,
        chi1, chi2, 4,
        0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
        coefficients, 1, pa, tgr);
    if (rc != 0 || modes || low || high) return 4;
    printf("%.17g %.17g %.17g %.17g %.17g\n",
           coefficients->data[0], coefficients->data[1],
           coefficients->data[2], coefficients->data[6],
           coefficients->data[7]);
    XLALDestroyREAL8Vector(coefficients);
    XLALDestroyDict(pa);
    XLALDestroyDict(tgr);
    return 0;
}
