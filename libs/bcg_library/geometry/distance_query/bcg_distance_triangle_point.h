//
// Created by alex on 07.11.20.
//

#ifndef BCG_GRAPHICS_BCG_DISTANCE_TRIANGLE_POINT_H
#define BCG_GRAPHICS_BCG_DISTANCE_TRIANGLE_POINT_H

#include "triangle/bcg_triangle.h"

namespace bcg {

template<int N>
struct distance_point_triangle {
    struct result {
        bcg_scalar_t distance = scalar_max, sqr_distance = scalar_max;
        VectorS <N> closest;
        VectorS<3> barycentric_coords;
    };

    result operator()(const VectorS <N> &point, const triangle <N> &triangle) const {
        VectorS<N> diff = point - triangle.points[0];
        VectorS<N> edge0 = triangle.points[1] - triangle.points[0];
        VectorS<N> edge1 = triangle.points[2] - triangle.points[0];
        auto a00 = edge0.dot(edge0);
        auto a01 = edge0.dot(edge1);
        auto a11 = edge1.dot(edge1);
        auto b0 = -diff.dot(edge0);
        auto b1 = -diff.dot(edge1);

        auto f00 = b0;
        auto f10 = b0 + a00;
        auto f01 = b0 + a01;

        VectorS<2> p0, p1, p;
        bcg_scalar_t dt1, h0, h1;

        // Compute the endpoints p0 and p1 of the segment.  The segment is
        // parameterized by L(z) = (1-z)*p0 + z*p1 for z in [0,1] and the
        // directional derivative of half the quadratic on the segment is
        // H(z) = (p1-p0).dot(gradient[Q](L(z))/2), where gradient[Q]/2 =
        // (F,G).  By design, F(L(z)) = 0 for cases (2), (4), (5), and
        // (6).  Cases (1) and (3) can correspond to no-intersection or
        // intersection of F = 0 with the triangle.
        if (f00 >= 0) {
            if (f01 >= 0) {
                // (1) p0 = (0,0), p1 = (0,1), H(z) = G(L(z))
                get_min_edge02(a11, b1, p);
            } else {
                // (2) p0 = (0,t10), p1 = (t01,1-t01),
                // H(z) = (t11 - t10)*G(L(z))
                p0[0] = 0;
                p0[1] = f00 / (f00 - f01);
                p1[0] = f01 / (f01 - f10);
                p1[1] = 1 - p1[0];
                dt1 = p1[1] - p0[1];
                h0 = dt1 * (a11 * p0[1] + b1);
                if (h0 >= 0) {
                    get_min_edge02(a11, b1, p);
                } else {
                    h1 = dt1 * (a01 * p1[0] + a11 * p1[1] + b1);
                    if (h1 <= 0) {
                        get_min_edge12(a01, a11, b1, f10, f01, p);
                    } else {
                        get_min_interior(p0, h0, p1, h1, p);
                    }
                }
            }
        } else if (f01 <= 0) {
            if (f10 <= 0) {
                // (3) p0 = (1,0), p1 = (0,1), H(z) = G(L(z)) - F(L(z))
                get_min_edge12(a01, a11, b1, f10, f01, p);
            } else {
                // (4) p0 = (t00,0), p1 = (t01,1-t01), H(z) = t11*G(L(z))
                p0[0] = f00 / (f00 - f10);
                p0[1] = 0;
                p1[0] = f01 / (f01 - f10);
                p1[1] = 1 - p1[0];
                h0 = p1[1] * (a01 * p0[0] + b1);
                if (h0 >= 0) {
                    p = p0;  // GetMinEdge01
                } else {
                    h1 = p1[1] * (a01 * p1[0] + a11 * p1[1] + b1);
                    if (h1 <= 0) {
                        get_min_edge12(a01, a11, b1, f10, f01, p);
                    } else {
                        get_min_interior(p0, h0, p1, h1, p);
                    }
                }
            }
        } else if (f10 <= 0) {
            // (5) p0 = (0,t10), p1 = (t01,1-t01),
            // H(z) = (t11 - t10)*G(L(z))
            p0[0] = 0;
            p0[1] = f00 / (f00 - f01);
            p1[0] = f01 / (f01 - f10);
            p1[1] = 1 - p1[0];
            dt1 = p1[1] - p0[1];
            h0 = dt1 * (a11 * p0[1] + b1);
            if (h0 >= 0) {
                get_min_edge02(a11, b1, p);
            } else {
                h1 = dt1 * (a01 * p1[0] + a11 * p1[1] + b1);
                if (h1 <= 0) {
                    get_min_edge12(a01, a11, b1, f10, f01, p);
                } else {
                    get_min_interior(p0, h0, p1, h1, p);
                }
            }
        } else {
            // (6) p0 = (t00,0), p1 = (0,t11), H(z) = t11*G(L(z))
            p0[0] = f00 / (f00 - f10);
            p0[1] = 0;
            p1[0] = 0;
            p1[1] = f00 / (f00 - f01);
            h0 = p1[1] * (a01 * p0[0] + b1);
            if (h0 >= 0) {
                p = p0;  // GetMinEdge01
            } else {
                h1 = p1[1] * (a11 * p1[1] + b1);
                if (h1 <= 0) {
                    get_min_edge02(a11, b1, p);
                } else {
                    get_min_interior(p0, h0, p1, h1, p);
                }
            }
        }

        result res;
        res.barycentric_coords[0] = 1 - p[0] - p[1];
        res.barycentric_coords[1] = p[0];
        res.barycentric_coords[2] = p[1];
        res.closest = triangle.points[0] + p[0] * edge0 + p[1] * edge1;
        diff = point - res.closest;
        res.sqr_distance = diff.dot(diff);
        res.distance = std::sqrt(res.sqr_distance);
        return res;
    }

private:
    inline void get_min_edge02(bcg_scalar_t a11, bcg_scalar_t b1, VectorS<2> &p) const {
        p[0] = 0;
        if (b1 >= 0) {
            p[1] = 0;
        } else if (a11 + b1 <= 0) {
            p[1] = 1;
        } else {
            p[1] = -b1 / a11;
        }
    }

    inline void get_min_edge12(bcg_scalar_t a01, bcg_scalar_t a11, bcg_scalar_t b1, bcg_scalar_t f10, bcg_scalar_t f01,
                               VectorS<2> &p) const {
        auto h0 = a01 + b1 - f10;
        if (h0 >= 0) {
            p[1] = 0;
        } else {
            auto h1 = a11 + b1 - f01;
            if (h1 <= 0) {
                p[1] = 1;
            } else {
                p[1] = h0 / (h0 - h1);
            }
        }
        p[0] = 1 - p[1];
    }

    inline void get_min_interior(const VectorS<2> &p0, bcg_scalar_t h0,
                                 const VectorS<2> &p1, bcg_scalar_t h1, VectorS<2> &p) const {
        auto z = h0 / (h0 - h1);
        p = (1 - z) * p0 + z * p1;
    }
};

using distance_point2_triangle2 = distance_point_triangle<2>;

using distance_point3_triangle3 = distance_point_triangle<3>;
}

#endif //BCG_GRAPHICS_BCG_DISTANCE_TRIANGLE_POINT_H
