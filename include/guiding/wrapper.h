#ifndef LIBGUIDING_WRAPPER_H
#define LIBGUIDING_WRAPPER_H

#include "guiding.h"

#include <array>
#include <vector>
#include <fstream>
#include <cstring>
#include <cassert>
#include <functional>

#include <mutex>
#include <shared_mutex>

namespace guiding {

template<typename T>
Float defaultTarget(const T &x) {
    return Float(x);
}

template<typename C, typename S = Float>
class Wrapper {
public:
    typedef S Sample;
    typedef C Distribution;
    typedef typename Distribution::Vector Vector;
    typedef typename Distribution::AuxWrapper AuxWrapper;

    struct Settings {
        Float uniformProb = 0.5f;

        // @todo could enhance performance by adding template for this
        Float (*target)(const Sample &) = defaultTarget<Sample>;

        typename Distribution::Settings child;
    };

    Settings settings;
    std::function<void ()> onRebuild;

    Wrapper() {
        reset();
    }

    Wrapper(const Settings &settings)
    : settings(settings) {
        reset();
    }

    void operator=(const Wrapper<C, S> &other) {
        settings   = other.settings;
        m_sampling = other.m_sampling;
        m_training = other.m_training;
        
        m_samplesSoFar  = other.m_samplesSoFar.load();
        m_nextMilestone = other.m_nextMilestone;
    }

    void reset() {
        std::unique_lock lock(m_mutex);

        m_training = Distribution();
        m_sampling = Distribution();

        m_samplesSoFar  = 0;
        m_nextMilestone = 1024;
    }

    template<typename ...Args>
    Float sample(Vector &x, Args&&... params) {
        if (settings.uniformProb == 1)
            return 1.f;
        
        std::shared_lock lock(m_mutex);

        Float pdf = 1 - settings.uniformProb; // guiding probability
        if (x[0] < settings.uniformProb) {
            x[0] /= settings.uniformProb;
            pdf *= m_sampling.pdf(
                settings.child,
                x,
                std::forward<Args>(params)...
            );
        } else {
            x[0] -= settings.uniformProb;
            x[0] /= 1 - settings.uniformProb;

            Float gpdf = 1;
            m_sampling.sample(
                settings.child,
                gpdf,
                x,
                std::forward<Args>(params)...
            );
            pdf *= gpdf;
        }

        pdf += settings.uniformProb;
        return pdf;
    }

    template<typename ...Args>
    Float pdf(Args&&... params) const {
        if (settings.uniformProb == 1)
            return 1.f;
        
        std::shared_lock lock(m_mutex);
        return settings.uniformProb + (1 - settings.uniformProb) * m_sampling.pdf(
            settings.child,
            std::forward<Args>(params)...
        );
    }

    template<typename ...Args>
    void splat(const Sample &sample, const AuxWrapper &aux, Float weight, Args&&... params) {
        //if (settings.uniformProb == 1)
        //    return;
        
        {
            Float density = settings.target(sample);
            assert(std::isfinite(density));
            assert(density >= 0);
            assert(std::isfinite(weight));
            assert(weight >= 0);

            std::shared_lock lock(m_mutex);
            m_training.splat(
                settings.child,
                density, aux, weight,
                std::forward<Args>(params)...
            );
        }

        if (++m_samplesSoFar > m_nextMilestone) {
            // it's wednesday my dudes!
            step();
        }
    }

    size_t samplesSoFar() const { return m_samplesSoFar; }

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
        
        m_training.build(settings.child);
        m_sampling = m_training;
        m_training.refine(settings.child);

        if (onRebuild)
            onRebuild();

        //m_sampling.dump("");

        m_nextMilestone *= 2;
    }

    Distribution m_sampling, m_training;

    std::atomic<size_t> m_samplesSoFar;
    size_t m_nextMilestone;

    mutable std::shared_mutex m_mutex;
};

}

#endif
