#include <iostream>
#include <gl.h>
#include <map>

static GLuint createShader_helper(GLint type, std::string shader_string) {
    GLuint id = glCreateShader(type);
    const char *shader_string_const = shader_string.c_str();
    glShaderSource(id, 1, &shader_string_const, nullptr);
    glCompileShader(id);

    GLint status;
    glGetShaderiv(id, GL_COMPILE_STATUS, &status);

    if (status != GL_TRUE) {
        char buffer[512];
        std::cerr << "Error while compiling ";
        if (type == GL_VERTEX_SHADER)
            std::cerr << "vertex shader";
        else if (type == GL_FRAGMENT_SHADER)
            std::cerr << "fragment shader";
        //else if (type == GL_GEOMETRY_SHADER)
        //    std::cerr << "geometry shader";
        std::cerr << ":" << std::endl;
        std::cerr << shader_string << std::endl << std::endl;
        glGetShaderInfoLog(id, 512, nullptr, buffer);
        std::cerr << "Error: " << std::endl << buffer << std::endl;
        throw std::runtime_error("Shader compilation failed!");
    }

    return id;
}

class Shader {
private:
    GLuint m_vertexShader;
    GLuint m_fragmentShader;
    GLuint m_program_shader;
    GLuint m_vao;

    struct Buffer {
        GLuint id;
        GLuint glType;
        GLuint dim;
        GLuint compSize;
        GLuint size;
        int version;
    };

    std::map<std::string, Buffer> m_bufferObjects;
public:
    bool init(const std::string &vertex_str, const std::string &fragment_str) {
        glGenVertexArrays(1, &m_vao);
        m_vertexShader =
            createShader_helper(GL_VERTEX_SHADER, vertex_str);
        m_fragmentShader =
            createShader_helper(GL_FRAGMENT_SHADER, fragment_str);

        if (!m_vertexShader || !m_fragmentShader) {
            throw std::runtime_error("wtf");
        }
        
        m_program_shader = glCreateProgram();
        glAttachShader(m_program_shader, m_vertexShader);
        glAttachShader(m_program_shader, m_fragmentShader);
        glLinkProgram(m_program_shader);

        GLint status;
        glGetProgramiv(m_program_shader, GL_LINK_STATUS, &status);

        if (status != GL_TRUE) {
            char buffer[512];
            glGetProgramInfoLog(m_program_shader, 512, nullptr, buffer);
            std::cerr << "Linker error: " << std::endl << buffer << std::endl;
            m_program_shader = 0;
            throw std::runtime_error("Shader linking failed!");
        }

        return true;
    }

    void uploadAttrib(const std::string &name, size_t count, int dim,
                            uint32_t compSize, GLuint glType, bool integral,
                            const void *data, int version = -1) {
        int attribID = 0;
        if (name != "indices") {
            attribID = attrib(name);
            if (attribID < 0)
                return;
        }

        size_t size = count * dim;
        GLuint bufferID;
        auto it = m_bufferObjects.find(name);
        if (it != m_bufferObjects.end()) {
            Buffer &buffer = it->second;
            bufferID = it->second.id;
            buffer.version = version;
            buffer.size = (GLuint) size;
            buffer.compSize = compSize;
        } else {
            glGenBuffers(1, &bufferID);
            Buffer buffer;
            buffer.id = bufferID;
            buffer.glType = glType;
            buffer.dim = dim;
            buffer.compSize = compSize;
            buffer.size = (GLuint) size;
            buffer.version = version;
            m_bufferObjects[name] = buffer;
        }
        size_t totalSize = size * (size_t) compSize;

        if (name == "indices") {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufferID);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, totalSize, data, GL_DYNAMIC_DRAW);
        } else {
            glBindBuffer(GL_ARRAY_BUFFER, bufferID);
            glBufferData(GL_ARRAY_BUFFER, totalSize, data, GL_DYNAMIC_DRAW);
            if (size == 0) {
                glDisableVertexAttribArray(attribID);
            } else {
                glEnableVertexAttribArray(attribID);
                glVertexAttribPointer(attribID, dim, glType, integral, 0, 0);
            }
        }
    }

    void bind() {
        glUseProgram(m_program_shader);
        glBindVertexArray(m_vao);
    }

    GLint uniform(const std::string &name, bool warn = true) const {
        GLint id = glGetUniformLocation(m_program_shader, name.c_str());
        if (id == -1 && warn)
            std::cerr << "warning: did not find uniform " << name << std::endl;
        return id;
    }

    GLint attrib(const std::string &name, bool warn = true) const {
        GLint id = glGetAttribLocation(m_program_shader, name.c_str());
        if (id == -1 && warn)
            std::cerr << "warning: did not find attrib " << name << std::endl;
        return id;
    }

    void drawIndexed(int type, size_t offset, size_t count) {
        glDrawElements(type, (GLsizei) count, GL_UNSIGNED_INT,
                    (const void *)(offset * sizeof(uint32_t)));
    }
};

class Pointcloud {
public:
    struct Vec3f {
        union {
            struct { float x, y, z; };
            float data[3];
        };

        float &operator[](int i) {
            return data[i];
        }
    };

    int activeIndex = -1;
    std::vector<Vec3f> points;

    Pointcloud() {
        initializeShader();
        update();
    }

    void update() {
        nPoints = points.size();

        std::vector<uint32_t> indices(nPoints);
        std::vector<Vec3f> colors(nPoints);
        std::vector<Vec3f> positions(nPoints);

        for (int i = 0; i < nPoints; ++i) {
            indices[i] = i;
            colors[i] = { 1, 1, 1 };
            if (i == activeIndex) {
                colors[i] = { 1, 0, 0 };
            }

            for (int dim = 0; dim < 3; ++dim)
                positions[i][dim] = 2 * points[i][dim] - 1;
        }

        m_shader.bind();
        m_shader.uploadAttrib("indices", nPoints, 1, sizeof(uint32_t), GL_UNSIGNED_INT, true, indices.data());
        m_shader.uploadAttrib("position", nPoints, 3, sizeof(float), GL_FLOAT, false, positions.data());
        m_shader.uploadAttrib("attrColor", nPoints, 3, sizeof(float), GL_FLOAT, false, colors.data());
    }

    void draw() {
        time += 0.005f;
        if (time > 2*M_PI)
            time -= 2*M_PI;
        
        float c = std::cos(time);
        float s = std::sin(time);

        float mvp[16] = {
            c, 0, -s, -1.5f*s,
            0, 1,  0, 0,
            s, 0,  c, 1.5f*c,
            0, 0,  0, 3
        };

        glPointSize(5.f);
        m_shader.bind();
        glUniformMatrix4fv(
            m_shader.uniform("modelViewProj"),
            1,
            GL_FALSE,
            mvp
        );
        m_shader.drawIndexed(GL_POINTS, 0, nPoints);
        glFlush();
    }

private:
    float time = 0;
    Shader m_shader;
    size_t nPoints;
    
    void initializeShader() {
        m_shader.init(
            /* Vertex shader */
            "#version 330\n"
            "uniform mat4 modelViewProj;\n"
            "in vec3 position;\n"
            "in vec3 attrColor;\n"
            "out vec3 fragColor;\n"
            "void main() {\n"
            "    gl_Position = modelViewProj * vec4(position, 1.0);\n"
            "    fragColor = attrColor;"
            "}",

            /* Fragment shader */
            "#version 330\n"
            "out vec4 color;\n"
            "in vec3 fragColor;\n"
            "void main() {\n"
            "    color = vec4(fragColor, 1.0);\n"
            "}"
        );
    }
};
