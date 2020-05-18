#include <iostream>
#include <imgui.h>
#include "utils.h"
#include "pointcloud.h"

ImVec4 clear_color = ImVec4(0.16, 0.16, 0.18, 1.0);

class GuidingDemo {
public:
    using GuidingTree = Wrapper<Float, KDTree<3, BTree<2, Leaf<Float>>>>;

    DebugTexture leafTex;

    TreeFilter::Enum filtering = TreeFilter::ENearest;

    Pointcloud cloud;
    GuidingTree guiding;
    RandomSampler rnd;

    GuidingDemo() {
        reset();
    }

    Float integrand(const VectorXf<3> &x, const VectorXf<2> &d) const {
        Float p[2] = {
            20 * std::pow(x[0] - d[0], 2.f),
            20 * std::pow(x[1] - d[1], 2.f)
        };

        Float v = 1;
        v *= std::exp(-p[0]);
        v *= std::exp(-p[1]);
        return v;
    }

    void step(int Nsamples = 512) {
        if (guiding.samplesSoFar() > 1000*1000)
            return;
        
        for (int i = 0; i < Nsamples; ++i) {
            VectorXf<3> x = { rnd.get1D(), rnd.get1D(), rnd.get1D() };
            VectorXf<2> d = { rnd.get1D(), rnd.get1D() };
            Float pdf = guiding.sample(x, d);

            Float f = integrand(x, d);
            //Float rr = std::min(x[0] * 2 + 0.05f, 1.f);
            //f *= rnd.get1D() < rr ? 1/rr : 0;

            guiding.splat(f, 1/pdf, x, d);
        }
    }

    void reset() {
        guiding = GuidingTree();
        rnd = RandomSampler();

        guiding.settings = {
            .uniformProb = 0.1f,
            .child = { // kd-tree
                .splitThreshold  = 1000.f,
                .splitting       = TreeSplitting::EWeight,
                .filtering       = filtering,
                .child = { // b-tree
                    .splitThreshold  = 0.01f,
                    .leafReweighting = true,
                    .filtering       = filtering,
                    .child = { // leaf nodes
                        .secondMoment = false
                    }
                }
            }
        };
    }

    void updatePointCloud() {
        using Child = GuidingTree::Distribution::Child;
        using Vector = GuidingTree::Distribution::Vector;

        cloud.points.clear();

        std::vector<const Child *> leaves;
        guiding.sampling().enumerate([&](
            const Child &leaf,
            const Vector &min,
            const Vector &max
        ) {
            leaves.push_back(&leaf);
            cloud.points.push_back({
                (min[0] + max[0]) / 2,
                (min[1] + max[1]) / 2,
                (min[2] + max[2]) / 2
            });
        });

        static int counter = 0;
        if (counter++ > 100) {
            cloud.activeIndex++;
            counter = 0;
        }

        if (cloud.points.empty())
            cloud.activeIndex = -1;
        else
            cloud.activeIndex = cloud.activeIndex % cloud.points.size();

        cloud.update();

        if (cloud.activeIndex >= 0) {
            auto &child = *leaves[cloud.activeIndex];
            leafTex.draw([&](ImVec2 pos) {
                return falseColor(1e-1 * child.pdf(guiding.settings.child.child, {
                    pos.x,
                    pos.y
                }));
            });

            ImGui::Text("Index:  %d" , cloud.activeIndex);
            ImGui::Text("Nodes:  %zu", child.nodeCount());
            ImGui::Text("Weight: %f" , child.weight);
        }
    }
};

GuidingDemo *demoPtr = nullptr;
void drawPoints() {
    static int counter = 0;
    if (demoPtr)
        demoPtr->cloud.draw();
}

void initialize() {
    extern void (*customBackground)();
    customBackground = drawPoints;
}

void render() {
    static bool showDemoWindow = false;
    if (showDemoWindow)
        ImGui::ShowDemoWindow(&showDemoWindow);
    
    static GuidingDemo demo;
    demoPtr = &demo;

    demo.step();

    {
        ImGui::Begin("Guiding");
        ImGui::Text("Samples:  %zu" , demo.guiding.samplesSoFar());
        ImGui::Text("Nodes:    %zu" , demo.guiding.sampling().nodeCount());
        ImGui::Text("T. Nodes: %zu" , demo.guiding.sampling().totalNodeCount());

        demo.updatePointCloud();

        if (
            selectTreeFilter(demo.filtering)
        )
            demo.reset();

        ImGui::End();
    }
}
