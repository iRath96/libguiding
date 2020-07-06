namespace guiding {
    using Float = mitsuba::Float;

    template<int D> struct MitsubaVector { typedef void Type; };
    template<> struct MitsubaVector<2> { typedef mitsuba::Vector2f Type; };
    template<> struct MitsubaVector<3> { typedef mitsuba::Vector3f Type; };
    
    template<int D>
    using VectorXf = typename MitsubaVector<D>::Type;
}

#include <guiding/structures/btree.h>
#include <guiding/structures/kdtree.h>

using namespace guiding;

class GuidingCache {
public:
    struct Sample {
        Spectrum radiance;
        Spectrum bsdf;
        Spectrum contribution;

        Float pdf;
    };

    using GuidingTree = KDTree<3, BTree<2, Leaf<Empty>, Spectrum>>;

    GuidingTree sampling, training;
    GuidingTree::Settings settings;

    AABB aabb;

    GuidingCache(const AABB &aabb) : aabb(aabb) {
        settings = {
            .maxDepth = 20,
            .splitThreshold = splitThreshold(0),
            .splitting = TreeSplitting::EWeight,
            .filtering = TreeFilter::ENearest,

            .child = {
                //.minDepth = 4,
                .maxDepth = 20,
                .splitThreshold = 0.01f,
                .filtering = TreeFilter::EBox,

                .child = {
                    .secondMoment = true
                }
            }
        };
    }

    Float pdf(Vector3f x, const Vector3f &d) const {
        pointToLocal(x);

        return sampling.pdf(
            settings,
            x,
            dirToCanonical(d)
        ) / (4 * M_PI);
    }

    Float sample(Vector3f x, Vector3f &d, Vector2f &rnd) const {
        pointToLocal(x);

        Float pdf;
        sampling.sample(
            settings,
            pdf,
            x,
            rnd
        );

        pdf /= 4 * M_PI;
        d = canonicalToDir(rnd);

        return pdf;
    }

    void splat(Vector3f x, const Vector3f &d, Sample sample) {
        if (sample.pdf < 1e-3)
            return;
        
        pointToLocal(x);
        sample.pdf *= 4 * M_PI;

        Float target = sample.radiance.average();
        if (settings.child.child.secondMoment)
            target = sample.contribution.average();

        training.splat(
            settings,
            target,
            (GuidingTree::AuxWrapper){
                .child = {
                    .value = sample.radiance * sample.bsdf
                }
            },
            1/sample.pdf,
            x,
            dirToCanonical(d)
        );
    }

    void step(int iter) {
        training.build(settings);
        sampling = training;
        training.refine(settings);

        settings.splitThreshold = splitThreshold(iter);
    }

    Spectrum estimate(Vector3f x) const {
        pointToLocal(x);
        return sampling.at(settings, x).aux;
    }

private:
    Float splitThreshold(int iter) const {
        return std::sqrt(std::pow(2, iter-1) / 4) * 2000;
    }

    void pointToLocal(Vector3f &x) const {
        for (int i = 0; i < 3; ++i)
            x[i] = (x[i] - aabb.min[i]) / (aabb.max[i] - aabb.min[i]);
    }

    static Vector3f canonicalToDir(const Vector2f &p) {
        const Float cosTheta = 2 * p.x - 1;
        const Float phi = 2 * M_PI * p.y;

        const Float sinTheta = sqrt(1 - cosTheta * cosTheta);
        Float sinPhi, cosPhi;
        math::sincos(phi, &sinPhi, &cosPhi);

        return { sinTheta * cosPhi, sinTheta * sinPhi, cosTheta };
    }

    static Vector2f dirToCanonical(const Vector3f &d) {
        if (!std::isfinite(d.x) || !std::isfinite(d.y) || !std::isfinite(d.z))
            return { 0, 0 };
        
        const Float cosTheta = std::min(std::max(d.z, -1.0f), +1.0f);
        Float phi = std::atan2(d.y, d.x);
        while (phi < 0)
            phi += 2 * M_PI;
        
        return { (cosTheta + 1) / 2, phi / (2 * M_PI) };
    }
};
