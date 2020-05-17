#ifndef HUSSAR_GUIDING_TREE_H
#define HUSSAR_GUIDING_TREE_H

#include <guiding/guiding.h>

#include <array>
#include <vector>
#include <fstream>
#include <cstring>
#include <cassert>

namespace guiding {

template<typename Base, typename C>
class Tree : public Base {
public:
    static constexpr auto Dimension = Base::Dimension;
    static constexpr auto Arity = Base::Arity;

    typedef C Child;
    typedef typename Child::Aux Aux;
    typedef typename Base::Vector Vector;

    struct Settings {
        int maxDepth         = 16;
        Float splitThreshold = 0.002f;
        bool leafReweighting = true;
        bool doFiltering     = true; // box filter [Müller et al.]

        typename Child::Settings child;
    };

private:
    struct TreeNode {
        /**
         * Indexed by a bitstring, where each bit describes the slab for one of the
         * vector dimensions. Bit 0 means lower half [0, 0.5) and bit 1 means upper half
         * [0.5, 1.0).
         * The MSB corresponds to the last dimension of the vector.
         */
        std::array<int, Arity> children;
        Child value; // the accumulation of the estimator (i.e., sum of integrand*weight)
        typename Base::ChildData data;
        
        bool isLeaf() const {
            return children[0] == 0;
        }

        void markAsLeaf() {
            children[0] = 0;
        }

        void reset() {
            value = Child();
        }

        int depth(const std::vector<TreeNode> &nodes) const {
            if (isLeaf())
                return 1;

            int maxDepth = 0;
            for (int i = 0; i < Arity; ++i)
                maxDepth = std::max(maxDepth, nodes[children[i]].depth(nodes));
            return maxDepth + 1;
        }

        void write(std::ostream &os) const {
            write(os, value);
            write(os, children);
        }

        void read(std::istream &is) {
            read(is, value);
            read(is, children);
        }
    };

    std::vector<TreeNode> m_nodes;

public:
    Tree() {
        // haven't learned anything yet, resort to uniform sampling
        setUniform();
    }

    std::string typeId() const {
        return std::string("BTree<") + std::to_string(Dimension) + ", " + typeid(Child).name() + ">";
    }

    // methods for reading from the tree

    const Child &at(const Settings &settings, const Vector &x) const {
        return m_nodes[indexAt(x)].value;
    }

    template<typename ...Args>
    Float pdf(const Settings &settings, const Vector &x, Args&&... params) const {
        return m_nodes[indexAt(x)].value.pdf(
            settings.child,
            std::forward<Args>(params)...
        );
    }

    const Child &sample(const Settings &settings, Vector &x, Float &pdf) const {
        pdf = 1;

        Vector base, scale;
        for (int dim = 0; dim < Dimension; ++dim) {
            base[dim] = 0;
            scale[dim] = 1;
        }

        int index = 0;
        while (!m_nodes[index].isLeaf()) {
            auto newIndex = m_nodes[index].children[this->sampleChild(x, base, scale, index, m_nodes)];
            assert(newIndex > index);
            index = newIndex;
        }

        pdf *= m_nodes[index].value.density;
        assert(m_nodes[index].value.density > 0);
        
        for (int dim = 0; dim < Dimension; ++dim) {
            x[dim] *= scale[dim];
            x[dim] += base[dim];
        }
        
        return m_nodes[index].value;
    }

    // methods for writing to the tree

    template<typename ...Args>
    void splat(const Settings &settings, Float density, const Aux &aux, Float weight, const Vector &x, Args&&... params) {
        if (!settings.doFiltering) {
            m_nodes[indexAt(x)].value.splat(
                settings.child,
                density, aux, weight,
                std::forward<Args>(params)...
            );
            return;
        }

        int depth;
        indexAt(x, depth);
        Float size = 1 / Float(1 << depth);
        
        Vector originMin, originMax, zero, one;
        for (int dim = 0; dim < Dimension; ++dim) {
            originMin[dim] = x[dim] - size/2;
            originMax[dim] = x[dim] + size/2;
            zero[dim] = 0;
            one[dim] = 1;
        }
        
        splatFiltered(
            settings,
            0,
            originMin, originMax,
            zero, one,
            density, aux, weight / (size * size),
            std::forward<Args>(params)...
        );
    }

    /**
     * Rebuilds the entire tree, making sure that leaf nodes that received
     * too few samples are pruned.
     * After building, each leaf node will have a value that is an estimate over
     * the mean value over the leaf node size (i.e., its size has been cancelled out).
     */
    void build(const Settings &settings) {
        std::vector<TreeNode> newNodes;
        newNodes.reserve(m_nodes.size());
        
        build(settings, 0, newNodes);
        if (newNodes[0].value.weight <= 0 || newNodes[0].value.density == 0) {
            // you're building a tree without samples. good luck with that.
            setUniform();
            return;
        }
        
        // normalize density
        m_nodes = newNodes;
        Float norm = m_nodes[0].value.density;

        for (auto &node : m_nodes) {
            node.value.density = node.value.density / norm;
            if (!settings.leafReweighting)
                node.value.aux = node.value.aux / m_nodes[0].value.weight;
        }
    }

    void refine(const Settings &settings) {
        refine(settings, 0);
    }

    // methods that provide statistics

    int depth() const {
        return m_nodes[0].depth(m_nodes);
    }

    size_t nodeCount() const {
        return m_nodes.size();
    }

    const atomic<Aux> &estimate() const {
        return m_nodes[0].value.estimate();
    }

private:
    void setUniform() {
        m_nodes.resize(1);
        m_nodes[0].markAsLeaf();
        m_nodes[0].value.density = 1;
        m_nodes[0].value.aux     = Aux();
        m_nodes[0].value.weight  = 0;
    }

    size_t indexAt(const Vector &y) const {
        int depth;
        return indexAt(y, depth);
    }

    size_t indexAt(const Vector &y, int &depth) const {
        Vector x = y;

        int index = 0;
        depth = 0;
        while (!m_nodes[index].isLeaf()) {
            auto newIndex = m_nodes[index].children[this->childIndex(x, m_nodes[index].data)];
            assert(newIndex > index);
            index = newIndex;

            ++depth;
        }

        return index;
    }

    void refine(const Settings &settings, size_t index, int depth = 0, Float scale = 1) {
        if (m_nodes[index].isLeaf()) {
            Float criterion = m_nodes[index].value.density / scale;
            if (criterion >= settings.splitThreshold && depth < settings.maxDepth)
                split(index);
            else {
                m_nodes[index].reset();
                return;
            }
        }
        
        for (int child = 0; child < Arity; ++child)
            refine(settings, m_nodes[index].children[child], depth + 1, scale * Arity);
    }

    template<typename ...Args>
    void splatFiltered(
        const Settings &settings,
        int index,
        const Vector &originMin, const Vector &originMax,
        const Vector &nodeMin, const Vector &nodeMax,
        Float density, const Aux &aux, Float weight,
        Args&&... params
    ) {
        Float overlap = computeOverlap<Dimension>(originMin, originMax, nodeMin, nodeMax);
        if (overlap > 0) {
            auto &node = m_nodes[index];
            if (node.isLeaf()) {
                node.value.splat(
                    settings.child,
                    density, aux, weight * overlap,
                    std::forward<Args>(params)...
                );
                return;
            }

            for (int child = 0; child < Arity; ++child) {
                Vector childMin = nodeMin;
                Vector childMax = nodeMax;
                this->boxForChild(child, childMin, childMax, node.data);
                
                splatFiltered(
                    settings,
                    node.children[child],
                    originMin, originMax,
                    childMin, childMax,
                    density, aux, weight,
                    std::forward<Args>(params)...
                );
            }
        }
    }

    /**
     * Executes the first pass of building the m_nodes.
     * Parts of the tree that received no samples will be pruned (if requested via settings.leafReweighting).
     * Each node in the tree will receive a value that is the mean over its childrens' values.
     * After this pass, the density of each node will correspond to the average weight within it,
     * i.e., after this pass you must still normalize the densities.
     */
    void build(const Settings &settings, size_t index, std::vector<TreeNode> &newNodes, Float scale = 1) {
        auto &node = m_nodes[index];

        // insert ourself into the tree
        size_t newIndex = newNodes.size();
        newNodes.push_back(node);

        if (node.isLeaf()) {
            auto &newNode = newNodes[newIndex];

            if (settings.leafReweighting && node.value.weight < 1e-3) { // @todo why 1e-3?
                // node received too few samples
                newNode.value.weight = -1;
                return;
            }

            if (!settings.leafReweighting)
                newNode.value.weight = 1 / scale;

            newNode.markAsLeaf();
            newNode.value.build(settings.child);

            if (!settings.leafReweighting)
                newNode.value.weight = node.value.weight;

            return;
        }

        int validCount = 0;
        Float density  = 0;
        Float weight   = 0;
        Aux   aux      = Aux();

        for (int child = 0; child < Arity; ++child) {
            auto newChildIndex = newNodes.size();
            build(settings, node.children[child], newNodes, scale * Arity);
            newNodes[newIndex].children[child] = newChildIndex;

            auto &newChild = newNodes[newChildIndex].value;
            if (newChild.weight >= 0) {
                density += newChild.density;
                aux     += newChild.aux;
                weight  += newChild.weight;

                ++validCount;
            }
        }

        if (!settings.leafReweighting)
            // ignore that children are broken if we are using naive building
            validCount = 4;
        
        if (validCount == 0) {
            // none of the children were valid (received samples)
            // mark this node and its subtree as invalid
            newNodes[newIndex].value.weight = -1;
            return;
        }
        
        // density and value are both normalized according to node area
        newNodes[newIndex].value.density = density / validCount;
        newNodes[newIndex].value.aux     = aux     / validCount;
        newNodes[newIndex].value.weight  = weight;

        if (validCount < Arity) {
            // at least one of the node's children is invalid (has not received enough samples)
            newNodes.resize(newIndex + 1); // remove the subtree of this node...
            newNodes[newIndex].markAsLeaf(); // ...and replace it by a leaf node
        }
    }

    void split(size_t parentIndex) {
        size_t childIndex = m_nodes.size();
        assert(childIndex > parentIndex);
        assert(m_nodes[parentIndex].isLeaf());

        for (int child = 0; child < Arity; ++child) {
            // insert new children
            m_nodes.push_back(m_nodes[parentIndex]);
            this->afterSplit(m_nodes[childIndex + child].data);
        }

        for (int child = 0; child < Arity; ++child)
            // register new children
            m_nodes[parentIndex].children[child] = childIndex + child;
    }

public:
    void write(std::ostream &os) const {
        size_t childCount = m_nodes.size();
        write(os, childCount);
        for (auto &node : m_nodes)
            node.write(os);
    }

    void read(std::istream &is) {
        size_t childCount = m_nodes.size();
        read(is, childCount);
        m_nodes.resize(childCount);

        for (auto &node : m_nodes)
            node.read(is);
    }
};

}

#endif
