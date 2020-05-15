#include <imgui.h>
#include <GL/gl.h>

#include <vector>
#include <functional>
#include <random>
#include <cmath>

namespace guiding {

using Float = float;
template<int D>
using VectorXf = std::array<Float, D>;

}

#include <guiding/guiding.h>
#include <guiding/distributions/btree.h>
#include <guiding/wrappers/point.h>

using namespace guiding;

ImVec4 clear_color = ImVec4(0.16, 0.16, 0.18, 1.0);

ImVec4 falseColor(float v) {
    ImVec4 c(1, 1, 1, 1);
    v = std::max(0.f, std::min(1.f, v));
    
    if (v < 0.25) {
        c.x = 0.0;
        c.y = 4.0 * v;
    } else if (v < 0.5) {
        c.x = 0.0;
        c.z = 1.0 + 4.0 * (0.25 - v);
    } else if (v < 0.75) {
        c.x = 4.0 * (v - 0.5);
        c.z = 0.0;
    } else {
        c.y = 1.0 + 4.0 * (0.75 - v);
        c.z = 0.0;
    }
    return c;
}

class Image {
public:
    int width, height;
    ImVec4 *data;

    Image(int w, int h) : width(w), height(h) {
        data = new ImVec4[w*h];
    }

    ImVec4 &at(int x, int y) {
        return data[x + y * width];
    }

    ~Image() {
        delete[] data;
    }
};

struct DebugTexture {
    GLuint gl_tex;
    Image image;
    
    DebugTexture() : image(128, 128) {
        glGenTextures(1, &gl_tex);
        glBindTexture(GL_TEXTURE_2D, gl_tex);

        // Setup filtering parameters for display
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        // Upload pixels into texture
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    }
    
    void draw(std::function<ImVec4 (ImVec2)> fn) {
        for (int y = 0; y < image.height; ++y)
            for (int x = 0; x < image.width; ++x) {
                image.at(x, y) = fn(ImVec2(
                    (x + Float(0.5)) / image.width,
                    (y + Float(0.5)) / image.height
                ));
            }

        glBindTexture(GL_TEXTURE_2D, gl_tex);
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA,
            image.width, image.height,
            0, GL_RGBA,
            GL_FLOAT,
            image.data
        );
        
        ImGui::Image((void *)(intptr_t)gl_tex, ImVec2(image.width, image.height));
    }
};

class RandomSampler {
public:
    RandomSampler() {}

    Float get1D() {
        return std::generate_canonical<Float,std::numeric_limits<Float>::digits>(generator);
    }

private:
    std::default_random_engine generator;
};

class GuidingDemo {
public:
    using GuidingTree = PointWrapper<2, BTreeDistribution<2, Float>>;

    GuidingTree guiding;
    RandomSampler rnd;

    float uniformProb = 0.2f;
    float threshold = 0.005f;

    bool badMode      = false;
    bool doFiltering  = false;
    bool secondMoment = false;

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

            guiding.splat(x, f, 1/pdf);
        }
    }

    void reset() {
        guiding = GuidingTree();
        rnd = RandomSampler();

        bool reweighting = !badMode;
        guiding.setUniformProb(uniformProb);
        guiding.training().leafReweighting() = reweighting;
        guiding.training().splitThreshold()  = threshold / (reweighting ? 2 : 1);
        guiding.training().doFiltering()     = doFiltering;
        guiding.training().secondMoment()    = secondMoment;
    }

    Float pdf(const ImVec2 &pos) const {
        return guiding.sampling().pdf({ pos.x, pos.y });
    }

    Float variance(const ImVec2 &pos) const {
        return std::pow(integrand(pos) / pdf(pos), 2);
    }
};

void initialize() {
}

void render() {
    static bool showDemoWindow = true;
    if (showDemoWindow)
        ImGui::ShowDemoWindow(&showDemoWindow);
    
    static DebugTexture texIntegrand, texGuiding, texVariance;
    static GuidingDemo demo;
    
    demo.step();

    {
        ImGui::Begin("Guiding");
        ImGui::Text("Samples:  %d"  , demo.guiding.samplesSoFar());
        ImGui::Text("Nodes:    %d"  , demo.guiding.sampling().nodeCount());
        ImGui::Text("Estimate: %.4f", float(demo.guiding.sampling().estimate()));

        texIntegrand.draw([](ImVec2 pos) { return falseColor(demo.integrand(pos)); });
        texGuiding.draw  ([](ImVec2 pos) { return falseColor(5e-2 * demo.pdf(pos)); });
        //texVariance.draw ([](ImVec2 pos) { return falseColor(10 * demo.variance(pos)); });

        if (
            ImGui::Checkbox("Müller",     &demo.badMode     ) |
            ImGui::Checkbox("Box Filter", &demo.doFiltering ) |
            ImGui::Checkbox("2nd moment", &demo.secondMoment)
        )
            demo.reset();

        ImGui::End();
    }
}
