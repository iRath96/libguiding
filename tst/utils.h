#include <gl.h>
#include <random>
#include <algorithm>

namespace guiding {

using Float = float;
template<int D>
using VectorXf = std::array<Float, D>;

}

using namespace guiding;

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
