#ifndef LIBGUIDING_STRUCTURES_KDTREE_H
#define LIBGUIDING_STRUCTURES_KDTREE_H

#include "../internal/tree.h"

namespace guiding {

template<int D>
struct KDTreeBase {
    static constexpr int Dimension = D;
    static constexpr int Arity = 2;

    struct ChildData {
        uint8_t axis = 0;
    };

    typedef VectorXf<Dimension> Vector;

    void afterSplit(ChildData &data) const {
        data.axis = (data.axis + 1) % Dimension;
    }

    GUIDING_CPU_GPU void boxForChild(int childIndex, Vector &min, Vector &max, const ChildData &data) const {
        Vector &affected = childIndex ? min : max;
        affected[data.axis] = (min[data.axis] + max[data.axis]) / 2;
    }

    GUIDING_CPU_GPU int childIndex(Vector &x, const ChildData &data) const {
        int childIndex = 0;
        int slab = x[data.axis] >= 0.5;
        childIndex = slab;

        if (slab)
            x[data.axis] -= 0.5;
        x[data.axis] *= 2;
        return childIndex;
    }

    template<typename Node, typename Allocator>
    GUIDING_CPU_GPU int sampleChild(Vector &x, Vector &base, Vector &scale, int index, const std::vector<Node, Allocator> &nodes) const {
        int childIndex = 0;

        Float p[2] = {
            nodes[nodes[index].children[0]].value.density,
            nodes[nodes[index].children[1]].value.density
        };

        assert(p[0] >= 0 && p[1] >= 0);
        assert((p[0] + p[1]) > 0);
        
        p[0] /= p[0] + p[1];

        int dim = nodes[index].data.axis;
        int slab = x[dim] >= p[0];
        childIndex = slab;

        if (slab) {
            base[dim] += 0.5 * scale[dim];
            x[dim] = (x[dim] - p[0]) / (1 - p[0]);
        } else
            x[dim] = x[dim] / p[0];
        scale[dim] /= 2;

        if (x[dim] >= 1)
            x[dim] = std::nextafterf(1, 0);

        assert(x[dim] >= 0);
        assert(x[dim] < 1);

        return childIndex;
    }
};

template<int D, typename C = Leaf<Empty>, typename A = Empty, template <typename> class Allocator = std::allocator>
using KDTree = Tree<KDTreeBase<D>, C, A, Allocator>;

}

#endif
