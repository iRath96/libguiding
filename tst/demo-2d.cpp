#include <imgui.h>
#include "utils.h"

#include <vector>
#include <functional>
#include <cmath>

ImVec4 clear_color = ImVec4(0.16, 0.16, 0.18, 1.0);

class GuidingDemo {
public:
    using GuidingTree = Wrapper<BTree<2, Leaf<Empty>, Float>, Float>;

    GuidingTree guiding;
    RandomSampler rnd;

    float uniformProb = 0.2f;
    float threshold = 0.003f;

    bool badMode      = false;
    bool secondMoment = false;
    bool hardReset    = true;

    TreeFilter::Enum filtering = TreeFilter::ENearest;

    GuidingDemo() {
        reset();
    }

    Float integrand(const ImVec2 &p) const {
        Float v = 1;
        v *= std::exp(-std::pow(10 * p[1] + p[0] - 5, 2));
        v *=
            0.8 * std::exp(-std::pow(10 * p[0] - 5, 2)) +
            0.2 * std::exp(-std::pow( 2 * p[0] + p[1] - 1, 2))
        ;

        return v;
    }

    void step(int Nsamples = 512) {
        if (guiding.samplesSoFar() > 100*1000)
            return;
        
        for (int i = 0; i < Nsamples; ++i) {
            VectorXf<2> x = { rnd.get1D(), rnd.get1D() };
            Float pdf = guiding.sample(x);

            Float f = integrand(ImVec2(x[0], x[1]));
            Float rr = std::min(x[0] * 2 + 0.05f, 1.f);
            f *= rnd.get1D() < rr ? 1/rr : 0;

            guiding.splat(f, { .value = f }, 1/pdf, x);
        }
    }

    void reset() {
        guiding = GuidingTree();
        rnd = RandomSampler();

        bool reweighting = !badMode;
        guiding.settings = {
            .uniformProb = uniformProb,
            .child = {
                .splitThreshold  = threshold / (reweighting ? 2 : 1),
                .leafReweighting = reweighting,
                .filtering       = filtering,
                .child = {
                    .secondMoment = secondMoment,
                    .resetFactor  = hardReset ? 0.f : 0.2f
                }
            }
        };
    }

    Float pdf(const ImVec2 &pos) const {
        return guiding.pdf((VectorXf<2>){ pos.x, pos.y });
    }

    Float variance(const ImVec2 &pos) const {
        return std::pow(integrand(pos) / pdf(pos), 2);
    }
};

void initialize() {
}

void render() {
    static bool showDemoWindow = false;
    if (showDemoWindow)
        ImGui::ShowDemoWindow(&showDemoWindow);
    
    static DebugTexture texIntegrand, texGuiding, texVariance;
    static GuidingDemo demo;
    
    demo.step();

    {
        ImGui::Begin("Guiding");
        ImGui::Text("Samples:  %zu" , demo.guiding.samplesSoFar());
        ImGui::Text("Nodes:    %zu" , demo.guiding.sampling().nodeCount());
        ImGui::Text("Estimate: %.4f", float(demo.guiding.sampling().aux));

        texIntegrand.draw([](ImVec2 pos) { return falseColor(demo.integrand(pos)); });
        ImGui::SameLine();
        texGuiding.draw  ([](ImVec2 pos) { return falseColor(5e-2 * demo.pdf(pos)); });
        //texVariance.draw ([](ImVec2 pos) { return falseColor(10 * demo.variance(pos)); });

        if (
            ImGui::Checkbox("Müller",     &demo.badMode     ) |
            selectTreeFilter(demo.filtering) |
            ImGui::Checkbox("2nd moment", &demo.secondMoment) |
            ImGui::Checkbox("Hard reset", &demo.hardReset)
        )
            demo.reset();
        
        ImGui::End();
    }
}
