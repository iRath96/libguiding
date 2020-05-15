#ifndef HUSSAR_GUIDING_POINTWRAPPER_H
#define HUSSAR_GUIDING_POINTWRAPPER_H

#include <guiding/guiding.h>

#include <array>
#include <vector>
#include <fstream>
#include <cstring>
#include <cassert>

#include <mutex>
#include <shared_mutex>

namespace guiding {

template<int D, typename C>
class PointWrapper {
public:
    static constexpr auto Dimension = D;

    typedef C Distribution;
    typedef VectorXf<Dimension> Vector;

    PointWrapper()
    : m_uniformProb(0.5f) {
        reset();
    }

    PointWrapper(Float uniformProb)
    : m_uniformProb(uniformProb) {
        reset();
    }

    void operator=(const PointWrapper<D, C> &other) {
        m_uniformProb = other.m_uniformProb;
        m_sampling = other.m_sampling;
        m_training = other.m_training;
        
        m_samplesSoFar = other.m_samplesSoFar.load(std::memory_order_relaxed);
        m_nextMilestone = other.m_nextMilestone;
    }

    void reset() {
        std::unique_lock lock(m_mutex);

        m_training = Distribution();
        m_sampling = Distribution();

        m_samplesSoFar = 0;
        m_nextMilestone = 1024;
    }

    Float sample(Vector &x) {
        if (m_uniformProb == 1)
            return 1.f;
        
        std::shared_lock lock(m_mutex);

        Float pdf = 1 - m_uniformProb;
        if (x[0] < m_uniformProb) {
            x[0] /= m_uniformProb;
            pdf *= m_sampling.pdf(x);
        } else {
            x[0] -= m_uniformProb;
            x[0] /= 1 - m_uniformProb;
            m_sampling.sample(x, pdf);
        }

        pdf += m_uniformProb;
        return pdf;
    }

    Float pdf(const Vector &x) {
        if (m_uniformProb == 1)
            return 1.f;
        std::shared_lock lock(m_mutex);
        return m_uniformProb + (1 - m_uniformProb) * m_sampling.pdf(x);
    }

    void splat(const Vector &x, Float v, Float weight) {
        if (m_uniformProb == 1)
            return;
        
        {
            std::shared_lock lock(m_mutex);
            m_training.splat(x, v, weight);
        }

        if (++m_samplesSoFar > m_nextMilestone) {
            // it's wednesday my dudes!
            step();
        }
    }

    size_t samplesSoFar() const { return m_samplesSoFar; }

    void setUniformProb(Float uniformProb) {
        m_uniformProb = uniformProb;
    }

    Distribution &training() { return m_training; }
    const Distribution &training() const { return m_training; }

    Distribution &sampling() { return m_sampling; }
    const Distribution &sampling() const { return m_sampling; }

private:
    void step() {
        std::unique_lock lock(m_mutex);
        if (m_samplesSoFar < m_nextMilestone)
            // someone was here before us!
            return;
        
        m_training.build();
        m_sampling = m_training;
        m_training.refine();

        m_nextMilestone *= 2;
    }

    Distribution m_sampling, m_training;
    Float m_uniformProb;

    std::atomic<size_t> m_samplesSoFar;
    size_t m_nextMilestone;

    mutable std::shared_mutex m_mutex;
};

}

#endif
