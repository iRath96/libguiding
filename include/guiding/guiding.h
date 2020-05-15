#ifndef HUSSAR_CORE_GUIDING_H
#define HUSSAR_CORE_GUIDING_H

#include <atomic>
#include <fstream>
#include <iostream>
#include <array>
#include <cassert>
#include <mutex>

namespace guiding {

/**
 * Writes an element to disk.
 * Override this if you store complex objects in your distribution that
 * need special serialization procedures.
 */
template<typename T>
void write(std::ostream &os, const T &t) {
    os.write((const char *)&t, sizeof(T));
}

/**
 * Reads an element to disk.
 * Override this if you store complex objects in your distribution that
 * need special deserialization procedures.
 */
template<typename T>
void read(std::istream &is, T &t) {
    is.read((char *)&t, sizeof(T));
}

/**
 * Assign a target to a distribution value (the density will equal to this, but normalized).
 * For instance, if you want to store Spectrums in your guiding distribution, do this;
 * `Float guiding::target(const Spectrum &x) { return x.average(); }`
 */
template<typename T>
Float target(const T &x) {
    return Float(x);
}

template<typename V>
class atomic {
public:
    void operator+=(const V &v) {
        std::unique_lock lock(m_mutex);
        m_value += v;
    }

    void operator=(const V &value) {
        m_value = value;
    }

    void operator=(const atomic<V> &other) {
        m_value = other.m_value;
    }

    const V &value() const { return m_value; }

private:
    V m_value;
    std::mutex m_mutex;
};

template<>
class atomic<Float> : public std::atomic<Float> {
public:
    atomic() : std::atomic<Float>() {}

    atomic(const atomic<Float> &other) : std::atomic<Float>() {
        *this = other;
    }

    void operator=(const Float &value) {
        store(value, std::memory_order_relaxed);
    }

    void operator=(const atomic<Float> &other) {
        *this = other.load();
    }

    void operator+=(const Float &value) {
        auto current = load();
        while (!compare_exchange_weak(current, current + value));
    }
};

template<int D>
Float computeOverlap(const VectorXf<D> &min1, const VectorXf<D> &max1, const VectorXf<D> &min2, const VectorXf<D> &max2) {
    // @todo this ignores the fact that a hypervolume can extend beyond the [0,1) interval
    // using this directly will give you a bias if you are not using leaf reweighting
    // (directions at the corners will have smaller weights)

    Float overlap = 1;
    for (int i = 0; i < D; ++i)
        overlap *= std::max(std::min(max1[i], max2[i]) - std::max(min1[i], min2[i]), 0.0f);
    return overlap;
}

}

#endif
