#ifndef HUSSAR_GUIDING_BTREE_H
#define HUSSAR_GUIDING_BTREE_H

#include "../internal/tree.h"

namespace guiding {

template<int D>
struct BTreeBase {
    static constexpr int Dimension = D;
    static constexpr int Arity = 1<<D;

    struct ChildData {};

    typedef VectorXf<Dimension> Vector;

    void afterSplit(ChildData &) {}

    void boxForChild(int childIndex, Vector &min, Vector &max, const ChildData &) const {
        for (int dim = 0; dim < Dimension; ++dim) {
            Vector &affected = childIndex & (1 << dim) ? min : max;
            affected[dim] = (min[dim] + max[dim]) / 2;
        }
    }

    int childIndex(Vector &x, const ChildData &) const {
        int childIndex = 0;
        for (int dim = 0; dim < Dimension; ++dim) {
            int slab = x[dim] >= 0.5;
            childIndex |= slab << dim;

            if (slab)
                x[dim] -= 0.5;
            x[dim] *= 2;
        }
        return childIndex;
    }

    template<typename Node>
    int sampleChild(Vector &x, Vector &base, Vector &scale, int index, const std::vector<Node> &nodes) const {
        int childIndex = 0;

        // sample each axis individually to determine sampled child
        for (int dim = 0; dim < Dimension; ++dim) {
            // marginalize over remaining dimensions {dim+1..Dimension-1}
            Float p[2] = { 0, 0 };
            for (int child = 0; child < (1 << (Dimension - dim)); ++child) {
                // we are considering only children that match all our
                // chosen dimensions {0..dim-1} so far.
                // we are collecting the sum of density for children with
                // x[dim] = 0 in p[0], and x[dim] = 1 in p[1].
                int ci = (child << dim) | childIndex;
                p[child & 1] += nodes[nodes[index].children[ci]].value.density;
            }
            
            p[0] /= p[0] + p[1];
            assert(p[0] >= 0 && p[1] >= 0);

            int slab = x[dim] > p[0];
            childIndex |= slab << dim;

            if (slab) {
                base[dim] += 0.5 * scale[dim];
                x[dim] = (x[dim] - p[0]) / (1 - p[0]);
            } else
                x[dim] = x[dim] / p[0];
            scale[dim] /= 2;
        }

        return childIndex;
    }
};

template<int D, typename C>
using BTree = Tree<BTreeBase<D>, C>;

}

#endif
