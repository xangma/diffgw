/* Development-only macOS interposer for the pinned wheel's GSL spline inputs.
 * The original GSL call is always made; this does not change the fit.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern int gsl_spline_init(void *, const double *, const double *, size_t);

static int capture_init(void *spline, const double *x, const double *y, size_t n)
{
    const char *path = getenv("DIFFGW_GSL_CAPTURE_PATH");
    const char *size_text = getenv("DIFFGW_GSL_CAPTURE_SIZE");
    if (path && size_text && n == (size_t)strtoull(size_text, NULL, 10)) {
        FILE *file = fopen(path, "ab");
        if (file) {
            const uint64_t length = (uint64_t)n;
            fwrite(&length, sizeof(length), 1, file);
            fwrite(x, sizeof(double), n, file);
            fwrite(y, sizeof(double), n, file);
            fclose(file);
        }
    }
    return gsl_spline_init(spline, x, y, n);
}

__attribute__((used)) static struct {
    const void *replacement;
    const void *replacee;
} interpose_init __attribute__((section("__DATA,__interpose"))) = {
    (const void *)capture_init, (const void *)gsl_spline_init
};
