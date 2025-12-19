#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define G 6.67430e-11
#define DT 0.01

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s input.txt tend\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    double tend = atof(argv[2]);
    const double m = 1.0;

    FILE *fin = fopen(filename, "r");
    if (!fin) {
        perror("Cannot open input file");
        return 1;
    }

    int n;
    fscanf(fin, "%d", &n);

    double *x  = malloc(n * sizeof(double));
    double *y  = malloc(n * sizeof(double));
    double *vx = malloc(n * sizeof(double));
    double *vy = malloc(n * sizeof(double));

    for (int i = 0; i < n; i++) {
        double z, vz;
        fscanf(fin, "%lf %lf %lf %lf %lf %lf",
               &x[i], &y[i], &z, &vx[i], &vy[i], &vz);
    }
    fclose(fin);

    double *fx = malloc(n * sizeof(double));
    double *fy = malloc(n * sizeof(double));

    int steps = (int)(tend / DT);
    int threads = omp_get_max_threads();

    double t_start = omp_get_wtime();

    for (int step = 0; step < steps; step++) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            fx[i] = 0.0;
            fy[i] = 0.0;
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            double fx_i = 0.0;
            double fy_i = 0.0;

            for (int j = 0; j < n; j++) {
                if (i == j) continue;

                double dx = x[j] - x[i];
                double dy = y[j] - y[i];
                double r2 = dx*dx + dy*dy + 1e-9;
                double r  = sqrt(r2);
                double f  = G * m * m / (r2 * r);

                fx_i += f * dx;
                fy_i += f * dy;
            }

            fx[i] = fx_i;
            fy[i] = fy_i;
        }

        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            vx[i] += fx[i] / m * DT;
            vy[i] += fy[i] / m * DT;
            x[i]  += vx[i] * DT;
            y[i]  += vy[i] * DT;
        }
    }

    double t_end = omp_get_wtime();
    printf("Threads: %d, Time: %.6f seconds\n", threads, t_end - t_start);

    free(x); free(y); free(vx); free(vy);
    free(fx); free(fy);

    return 0;
}

