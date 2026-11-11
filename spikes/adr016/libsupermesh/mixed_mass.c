/* Spike (ADR 016): the P1 mixed mass matrix B[j_new, i_old] = ∫ φ_new_j φ_old_i over the supermesh,
   2D (triangles) and 3D (tetrahedra), with libsupermesh's R-tree finder and simplex intersections.
   Exact for P1×P1: on a simplex S, ∫ f g = |S|/((d+1)(d+2)) (Σ f_k g_k + Σ f_k Σ g_k). */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "libsupermesh-c.h"

/* barycentric coordinates of point p in simplex v (d+1 vertices, row-major d coords) */
static void bary(int d, const double *v, const double *p, double *lam) {
  double A[9], r[3];
  for (int i = 0; i < d; i++) { r[i] = p[i] - v[i]; for (int j = 0; j < d; j++) A[i*d+j] = v[(j+1)*d+i] - v[i]; }
  double s[3];
  if (d == 2) {
    double det = A[0]*A[3] - A[1]*A[2];
    s[0] = ( A[3]*r[0] - A[1]*r[1]) / det; s[1] = (-A[2]*r[0] + A[0]*r[1]) / det;
  } else {
    double det = A[0]*(A[4]*A[8]-A[5]*A[7]) - A[1]*(A[3]*A[8]-A[5]*A[6]) + A[2]*(A[3]*A[7]-A[4]*A[6]);
    double inv[9] = { A[4]*A[8]-A[5]*A[7], A[2]*A[7]-A[1]*A[8], A[1]*A[5]-A[2]*A[4],
                      A[5]*A[6]-A[3]*A[8], A[0]*A[8]-A[2]*A[6], A[2]*A[3]-A[0]*A[5],
                      A[3]*A[7]-A[4]*A[6], A[1]*A[6]-A[0]*A[7], A[0]*A[4]-A[1]*A[3] };
    for (int i = 0; i < 3; i++) s[i] = (inv[i*3]*r[0] + inv[i*3+1]*r[1] + inv[i*3+2]*r[2]) / det;
  }
  double sum = 0; for (int i = 0; i < d; i++) { lam[i+1] = s[i]; sum += s[i]; } lam[0] = 1.0 - sum;
}

/* Finder: returns the number of candidate pairs; *indices / *indptr are malloc'd (caller frees with lsm_free). */
long lsm_pairs(int d, long nn_old, long ne_old, const double *x_old, const long *en_old,
               long nn_new, long ne_new, const double *x_new, const long *en_new, long **indices, long **indptr) {
  int loc = d + 1; int dd = d;
  libsupermesh_tree_intersection_finder_set_input(&nn_new, &dd, &ne_new, &loc, &nn_old, &dd, &ne_old, &loc,
      (double*)x_new, (long*)en_new, (double*)x_old, (long*)en_old);
  long n; libsupermesh_tree_intersection_finder_query_output(&n);
  *indices = malloc(sizeof(long) * (n > 0 ? n : 1)); *indptr = malloc(sizeof(long) * (ne_new + 1));
  libsupermesh_tree_intersection_finder_get_output(&ne_new, &n, *indices, *indptr);
  return n;
}
void lsm_free(void *p) { free(p); }

/* Assemble B as COO: rows (new nodes), cols (old nodes), vals; (d+1)^2 entries per candidate pair.
   Pieces smaller than rel_tol * |new cell| are dropped; returns the total measure of the kept pieces. */
double lsm_mixed_mass(int d, const double *x_old, const long *en_old, const double *x_new, const long *en_new,
                      long ne_new, const long *indices, const long *indptr, double rel_tol,
                      long *rows, long *cols, double *vals) {
  int loc = d + 1; double total = 0;
  double vn[12], vo[12], out[3 * 4 * 81 * 4]; /* ample for tet∩tet (≤ 3^4 tets) and tri∩tri */
  long k = 0;
  for (long e = 0; e < ne_new; e++) {
    for (int a = 0; a < loc; a++) for (int c = 0; c < d; c++) vn[a*d+c] = x_new[en_new[e*loc+a]*d + c];
    double vol_new;
    if (d == 2) libsupermesh_triangle_area(vn, &vol_new); else libsupermesh_tetrahedron_volume(vn, &vol_new);
    for (long q = indptr[e]; q < indptr[e+1]; q++) {
      long o = indices[q];
      for (int a = 0; a < loc; a++) for (int c = 0; c < d; c++) vo[a*d+c] = x_old[en_old[o*loc+a]*d + c];
      int npieces = 0;
      if (d == 2) libsupermesh_intersect_tris_real(vn, vo, out, &npieces);
      else libsupermesh_intersect_tets_real(vn, vo, out, &npieces);
      double Bloc[16] = {0};
      for (int p = 0; p < npieces; p++) {
        double *piece = out + p * loc * d, vol;
        if (d == 2) libsupermesh_triangle_area(piece, &vol); else libsupermesh_tetrahedron_volume(piece, &vol);
        vol = fabs(vol);
        if (vol < rel_tol * fabs(vol_new)) continue;
        total += vol;
        double ln[4][4], lo[4][4]; /* [piece vertex][basis] */
        for (int m = 0; m < loc; m++) { bary(d, vn, piece + m*d, ln[m]); bary(d, vo, piece + m*d, lo[m]); }
        double w = vol / ((d + 1.0) * (d + 2.0));
        for (int j = 0; j < loc; j++) {
          double sj = 0; for (int m = 0; m < loc; m++) sj += ln[m][j];
          for (int i = 0; i < loc; i++) {
            double si = 0, dot = 0;
            for (int m = 0; m < loc; m++) { si += lo[m][i]; dot += ln[m][j] * lo[m][i]; }
            Bloc[j*loc+i] += w * (dot + sj * si);
          }
        }
      }
      for (int j = 0; j < loc; j++) for (int i = 0; i < loc; i++) {
        rows[k] = en_new[e*loc+j]; cols[k] = en_old[o*loc+i]; vals[k] = Bloc[j*loc+i]; k++;
      }
    }
  }
  return total;
}
