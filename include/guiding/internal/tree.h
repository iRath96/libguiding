#ifndef HUSSAR_GUIDING_TREE_H
#define HUSSAR_GUIDING_TREE_H

#include "../guiding.h"

#include <array>
#include <vector>
#include <fstream>
#include <cstring>
#include <cassert>

namespace guiding {

template<typename T>
class Leaf {
public:
    static constexpr auto IsLeaf = true;

    struct Settings {
        bool secondMoment = false;
        Float resetFactor = 0.f;
    };

    typedef T Aux;

    atomic<Aux> aux;
    atomic<Float> weight;
    atomic<Float> density;

    void splat(const Settings &settings, Float density, const Aux &aux, Float weight) {
        if (settings.secondMoment)
            density *= density;
        
        this->aux     += aux     * weight;
        this->density += density * weight;
        this->weight  += weight;
    }

    void build(const Settings &settings) {
        if (weight < 1e-10)
            return;
        
        density = density / weight;
        aux     = aux     / weight;

        if (settings.secondMoment)
            density = std::sqrt(density);
    }

    void build(const Settings &settings, Float scale) {
        density = density * scale;
        aux     = aux * scale;

        if (settings.secondMoment)
            density = std::sqrt(density);
    }

    void refine(const Settings &settings) {
        weight  = weight * settings.resetFactor;
        aux     = aux * weight;
        density = density * weight;
    }

    Float pdf(const Settings &settings) const {
        return density;
    }

    Leaf<T> sample(const Settings &settings) const {
        return *this;
    }

    const atomic<Aux> &estimate() const {
        return aux;
    }

    size_t totalNodeCount() const {
        return 1;
    }

    void dump(const std::string &prefix) const {
        std::cout << prefix << "Leaf (density=" << density << ", weight=" << weight << ")" << std::endl;
    }

    void write(std::ostream &os) const {
        guiding::write(os, aux);
        guiding::write(os, weight);
        guiding::write(os, density);
    }

    void read(std::istream &is) {
        guiding::read(is, aux);
        guiding::read(is, weight);
        guiding::read(is, density);
    }
};

struct TreeFilter {
    enum Enum : uint8_t { // [Müller et al.]
        ENearest    = 0,
        EStochastic = 1,
        EBox        = 2,

        Max         = 3
    };

    static const char *to_string(TreeFilter::Enum value) {
        switch (value) {
        case ENearest: return "nearest";
        case EStochastic: return "stochastic";
        case EBox: return "box filter";
        default: return "(invalid)";
        }
    }
};

struct TreeSplitting {
    enum Enum : uint8_t {
        EDensity = 0,
        EWeight  = 1,

        Max      = 2
    };
};

template<typename Base, typename C>
class Tree : public Base {
public:
    static constexpr auto Dimension = Base::Dimension;
    static constexpr auto Arity = Base::Arity;
    static constexpr auto IsLeaf = false;

    typedef uint16_t Index;

    typedef C Child;
    typedef typename Child::Aux Aux;
    typedef typename Base::Vector Vector;

    struct Settings {
        int minDepth = 0;
        int maxDepth = 16;

        Float splitThreshold = 0.002f;
        bool leafReweighting = true;
        bool mergePartiallyInvalid = false;//Child::IsLeaf;

        TreeSplitting::Enum splitting = TreeSplitting::EDensity;
        TreeFilter::Enum filtering = TreeFilter::ENearest;

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
        std::array<Index, Arity> children;
        Child value; // the accumulation of the estimator (i.e., sum of integrand*weight)
        typename Base::ChildData data;
        
        bool isLeaf() const {
            return children[0] == 0;
        }

        void markAsLeaf() {
            children[0] = 0;
        }

        int depth(const std::vector<TreeNode> &nodes) const {
            if (isLeaf())
                return 0;

            int maxDepth = 0;
            for (int i = 0; i < Arity; ++i)
                maxDepth = std::max(maxDepth, nodes[children[i]].depth(nodes));
            return maxDepth + 1;
        }

        void write(std::ostream &os) const {
            guiding::write(os, children);
            guiding::write(os, value);
            guiding::write(os, data);
        }

        void read(std::istream &is) {
            guiding::read(is, children);
            guiding::read(is, value);
            guiding::read(is, data);
        }
    };

    std::vector<TreeNode> m_nodes;

public:
    Aux   aux;
    Float weight;
    Float density;

    Tree() {
        // haven't learned anything yet, resort to uniform sampling
        setUniform();
    }

    // methods for reading from the tree

    const Child &at(const Settings &settings, const Vector &x) const {
        return m_nodes[indexAt(x)].value;
    }

    template<typename ...Args>
    Float pdf(const Settings &settings, const Vector &x, Args&&... params) const {
        if constexpr (!is_empty<Args...>::value)
            return m_nodes[indexAt(x)].value.pdf(
                settings.child,
                std::forward<Args>(params)...
            );
        
        return m_nodes[indexAt(x)].value.density;
    }

    template<typename ...Args>
    const typename RecurseChild<Child, Args...>::Type &sample(
        const Settings &settings,
        Float &pdf,
        Vector &x,
        Args&&... params
    ) const {
        return m_nodes[indexAt(x)].value.sample(
            settings.child,
            pdf,
            std::forward<Args>(params)...
        );
    }

    const Child &sample(const Settings &settings, Float &pdf, Vector &x) const {
        pdf = 1;

        Vector base, scale;
        for (int dim = 0; dim < Dimension; ++dim) {
            base[dim] = 0;
            scale[dim] = 1;
        }

        Index index = 0;
        while (!m_nodes[index].isLeaf()) {
            auto newIndex = m_nodes[index].children[this->sampleChild(x, base, scale, index, m_nodes)];
            assert(newIndex > index);
            assert(m_nodes[newIndex].value.density > 0);
            index = newIndex;
        }

        pdf *= m_nodes[index].value.density;
        
        for (int dim = 0; dim < Dimension; ++dim) {
            x[dim] *= scale[dim];
            x[dim] += base[dim];
        }
        
        return m_nodes[index].value;
    }

    // methods for writing to the tree

    template<typename ...Args>
    void splat(const Settings &settings, Float density, const Aux &aux, Float weight, const Vector &x, Args&&... params) {
        if (settings.filtering == TreeFilter::ENearest) {
            m_nodes[indexAt(x)].value.splat(
                settings.child,
                density, aux, weight,
                std::forward<Args>(params)...
            );
            return;
        }

        int depth;
        Vector cellMin, cellMax;
        indexAt(x, depth, cellMin, cellMax);
        
        Float volume = 1;
        Vector originMin, originMax, zero, one;
        for (int dim = 0; dim < Dimension; ++dim) {
            Float size = cellMax[dim] - cellMin[dim];
            volume *= size;

            originMin[dim] = x[dim] - size/2;
            originMax[dim] = x[dim] + size/2;
            zero[dim] = 0;
            one[dim] = 1;
        }

        if (settings.filtering == TreeFilter::EStochastic) {
            Vector y;
            for (int dim = 0; dim < Dimension; ++dim) {
                Float alpha = random();
                y[dim] = alpha * originMin[dim] + (1-alpha) * originMax[dim];
            }

            m_nodes[indexAt(y)].value.splat(
                settings.child,
                density, aux, weight,
                std::forward<Args>(params)...
            );

            return;
        }
        
        splatFiltered(
            settings,
            0,
            originMin, originMax,
            zero, one,
            density, aux, weight / volume,
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
        
        bool isValid = build(settings, 0, newNodes);
        if (newNodes[0].value.weight == 0 || newNodes[0].value.density == 0 || !isValid) {
            // you're building a tree without samples. good luck with that.
            //std::cout << "invalid tree: " << newNodes[0].value.weight
            //    << "/ " << newNodes[0].value.density
            //    << "/" << (isValid ? "valid" : "invalid")
            //    << std::endl;
            setUniform(newNodes[0].value.weight);
            return;
        }

        //std::cout << "valid tree: " << newNodes[0].value.weight << std::endl;
        
        // normalize density
        m_nodes = newNodes;
        Float norm = m_nodes[0].value.density;
        assert(!std::isnan(norm));
        assert(norm > 0);

        for (auto &node : m_nodes) {
            assert(!std::isnan(node.value.density));
            node.value.density = node.value.density / norm;
            if (!settings.leafReweighting)
                node.value.aux = node.value.aux / m_nodes[0].value.weight;
        }

        density = norm;
        updateRoot();
    }

    void build(const Settings &settings, Float scale) {
        std::cerr << "you can only disable leaf reweighting for trees that contain leaves" << std::endl;
        assert(false);
    }

    void refine(const Settings &settings) {
        // @todo could use move constructor for performance and refine directly into other tree
        std::vector<TreeNode> newNodes;
        newNodes.reserve(m_nodes.size());
        refine(settings, m_nodes[0], newNodes);

        m_nodes = newNodes;
    }

    // methods that provide statistics

    int depth() const {
        return m_nodes[0].depth(m_nodes);
    }

    int depthAt(const Vector &x) const {
        int depth;
        Vector min, max;
        indexAt(x, depth, min, max);
        return depth;
    }

    size_t nodeCount() const {
        return m_nodes.size();
    }

    size_t totalNodeCount() const { // @todo should be called totalLeafCount()
        size_t count = 0;
        for (auto &node : m_nodes)
            if (node.isLeaf())
                count += node.value.totalNodeCount();
        return count;
    }

    const atomic<Aux> &estimate() const {
        return m_nodes[0].value.estimate();
    }

    void dump(const std::string &prefix) const {
        std::cout << prefix << "Tree (density=" << density << ", weight=" << weight << ")" << std::endl;
        int counter = 16;
        for (auto &node : m_nodes)
            if (node.isLeaf())
                if (counter-- > 0)
                    node.value.dump(prefix + "  ");
        
        if (counter < 0)
            std::cout << prefix << "  ... +" << (-counter) << " more leaves" << std::endl;
    }

    void enumerate(
        std::function<void (const Child &, const Vector &, const Vector &)> callback
    ) const {
        Vector zero, one;
        for (int dim = 0; dim < Dimension; ++dim) {
            zero[dim] = 0;
            one[dim] = 1;
        }

        enumerate(callback, 0, zero, one);
    }

private:
    void enumerate(
        std::function<void (const Child &, const Vector &, const Vector &)> callback,
        Index index,
        const Vector &min, const Vector &max
    ) const {
        auto &node = m_nodes[index];
        if (node.isLeaf()) {
            callback(node.value, min, max);
            return;
        }
        
        for (int childIndex = 0; childIndex < Arity; ++childIndex) {
            Index ci = node.children[childIndex];

            Vector childMin = min;
            Vector childMax = max;
            this->boxForChild(childIndex, childMin, childMax, node.data);

            enumerate(callback, ci, childMin, childMax);
        }
    }

    void updateRoot() {
        aux     = m_nodes[0].value.aux;
        weight  = m_nodes[0].value.weight;
    }

    void setUniform(Float weight = 0) {
        m_nodes.resize(1);
        m_nodes[0].markAsLeaf();
        m_nodes[0].value.density = 1;
        m_nodes[0].value.aux     = Aux();
        m_nodes[0].value.weight  = weight;

        density = 0;
        updateRoot();
    }

    size_t indexAt(const Vector &y) const {
        int depth;
        Vector min, max;
        return indexAt(y, depth, min, max);
    }

    size_t indexAt(const Vector &y, int &depth, Vector &min, Vector &max) const {
        Vector x = y;
        for (int dim = 0; dim < Dimension; ++dim) {
            min[dim] = 0;
            max[dim] = 1;
        }

        Index index = 0;
        depth = 0;
        while (!m_nodes[index].isLeaf()) {
            int childIndex = this->childIndex(x, m_nodes[index].data);
            this->boxForChild(childIndex, min, max, m_nodes[index].data);

            auto newIndex = m_nodes[index].children[childIndex];
            assert(newIndex > index);
            index = newIndex;

            ++depth;
        }

        return index;
    }

    size_t refine(
        const Settings &settings,
        const TreeNode &node, std::vector<TreeNode> &newNodes,
        int depth = 0, Float scale = 1
    ) const {
        assert(newNodes.size() <= std::numeric_limits<Index>::max());

        Index newIndex = newNodes.size();
        newNodes.push_back(node);

        bool canSplit = newNodes.size() < (std::numeric_limits<Index>::max() + Arity);

        Float criterion = node.value.density / scale;
        if (settings.splitting == TreeSplitting::EWeight)
            criterion = node.value.weight;
        if (
            canSplit && (
                (criterion >= settings.splitThreshold && depth < settings.maxDepth) ||
                depth < settings.minDepth
            )
        ) {
            if (node.isLeaf()) {
                // split this node and refine new children recursively
                // note: once we are in this code region, all recursive calls end
                // up in this region or the "criterion not met" region.

                TreeNode childTemplate = node;
                childTemplate.value.weight = childTemplate.value.weight / Arity;
                this->afterSplit(childTemplate.data);

                for (int i = 0; i < Arity; ++i) {
                    auto newChildIndex = refine(
                        settings,
                        childTemplate, newNodes,
                        depth + 1, scale * Arity
                    );
                    newNodes[newIndex].children[i] = newChildIndex;
                }

                // get rid of wasted space
                newNodes[newIndex].value = Child();
            } else {
                // carry over existing children
                for (int i = 0; i < Arity; ++i) {
                    auto newChildIndex = refine(
                        settings,
                        m_nodes[node.children[i]], newNodes,
                        depth + 1, scale * Arity
                    );
                    newNodes[newIndex].children[i] = newChildIndex;
                }
            }
        } else {
            // merge (@todo merge distributions?)
            newNodes[newIndex].markAsLeaf();
            newNodes[newIndex].value.refine(settings.child);
        }

        return newIndex;
    }

    template<typename ...Args>
    void splatFiltered(
        const Settings &settings,
        Index index,
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
    bool build(const Settings &settings, size_t index, std::vector<TreeNode> &newNodes, Float scale = 1) {
        auto &node = m_nodes[index];

        // insert ourself into the tree
        Index newIndex = newNodes.size();
        newNodes.push_back(node);

        if (node.isLeaf()) {
            auto &newNode = newNodes[newIndex];

            if (!settings.leafReweighting)
                newNode.value.build(settings.child, scale);
            else
                newNode.value.build(settings.child);

            if (settings.leafReweighting && newNode.value.weight < 1e-3) { // @todo why 1e-3?
                // node received too few samples
                return false;
            }
     
            return true;
        }

        int validCount = 0;

        {
            // reset parent so we can accumulate children in it
            auto &node = newNodes[newIndex].value;
            node.density = 0;
            node.weight  = 0;
            node.aux     = Aux();
        }

        for (int child = 0; child < Arity; ++child) {
            auto newChildIndex = newNodes.size();
            bool isValid = build(settings, node.children[child], newNodes, scale * Arity);
            newNodes[newIndex].children[child] = newChildIndex;

            if (!isValid && settings.leafReweighting)
                continue;
            
            auto &newParent = newNodes[newIndex].value;
            auto &newChild = newNodes[newChildIndex].value;

            assert(!std::isnan(newChild.density));

            newParent.density += newChild.density;
            newParent.aux     += newChild.aux;
            newParent.weight  += newChild.weight;

            ++validCount;
        }

        if (validCount == 0) {
            // none of the children were valid (received samples)
            // mark this node and its subtree as invalid
            return false;
        }

        {
            // density and value are both normalized according to node area
            auto &node = newNodes[newIndex].value;
            node.density = node.density / validCount;
            node.aux     = node.aux     / validCount;
        }

        if (validCount < Arity && settings.mergePartiallyInvalid) {
            // at least one of the node's children is invalid (has not received enough samples)
            newNodes.resize(newIndex + 1); // remove the subtree of this node...
            newNodes[newIndex].markAsLeaf(); // ...and replace it by a leaf node

            // @todo this will break if our children are distributions!
        }

        return true;
    }

public:
    void write(std::ostream &os) const {
        guiding::write(os, density);
        guiding::write(os, aux);
        guiding::write(os, weight);

        size_t childCount = m_nodes.size();
        guiding::write(os, childCount);
        for (auto &node : m_nodes)
            node.write(os);
    }

    void read(std::istream &is) {
        guiding::read(is, density);
        guiding::read(is, aux);
        guiding::read(is, weight);

        size_t childCount = m_nodes.size();
        guiding::read(is, childCount);
        m_nodes.resize(childCount);

        for (auto &node : m_nodes)
            node.read(is);
    }
};

}

#endif
