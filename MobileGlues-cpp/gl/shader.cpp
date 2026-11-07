// Modified by FCL-Team: shader-source API correctness and error handling.
// MobileGlues - gl/shader.cpp
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

#include <cctype>
#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include "shader.h"

#include <GL/gl.h>
#include "log.h"
#include "program.h"
#include "../gles/loader.h"
#include "../includes.h"
#include "glsl/glsl_for_es.h"
#include "../config/settings.h"
#include "FSR1/FSR1.h"
#include "../egl/context.h"

#define DEBUG 0

struct shader_t shaderInfo;

UnorderedMap<GLuint, bool> shader_map_is_sampler_buffer_emulated;

// The uniform initialisers stripped from each translated shader, applied by
// glLinkProgram once a program using the shader links. A shader without any has
// no entry.
UnorderedMap<GLuint, std::vector<uniform_default_t>> shader_uniform_defaults;

namespace {
thread_local GLenum fallback_shader_source_error = GL_NO_ERROR;

GLenum& shader_source_error_slot() {
    return g_current_ctx ? g_current_ctx->shader_source_error : fallback_shader_source_error;
}

void set_shader_source_error(GLenum error) {
    GLenum& slot = shader_source_error_slot();
    if (slot == GL_NO_ERROR) slot = error;
}

bool valid_shader_source_object(GLuint shader) {
    if (GLES.glIsShader(shader)) return true;
    set_shader_source_error(GLES.glIsProgram(shader) ? GL_INVALID_OPERATION : GL_INVALID_VALUE);
    return false;
}
} // namespace

GLenum mg_take_shader_source_error() {
    GLenum& slot = shader_source_error_slot();
    const GLenum error = slot;
    slot = GL_NO_ERROR;
    return error;
}

bool can_run_essl3(unsigned int esversion, const char* glsl) {
    if (strncmp(glsl, "#version 100", 12) == 0) {
        return true;
    }

    unsigned int glsl_version = 0;
    if (strncmp(glsl, "#version 300 es", 15) == 0) {
        glsl_version = 300;
    } else if (strncmp(glsl, "#version 310 es", 15) == 0) {
        glsl_version = 310;
    } else if (strncmp(glsl, "#version 320 es", 15) == 0) {
        glsl_version = 320;
    } else {
        return false;
    }
    return esversion >= glsl_version;
}

bool is_direct_shader(const char* glsl) {
    bool es3_ability = can_run_essl3(hardware->es_version, glsl);
    return es3_ability;
}

bool check_if_sampler_buffer_used(const std::string& str) {
    return str.find("samplerBuffer") != std::string::npos;
}

void glShaderSource(GLuint shader, GLsizei count, const GLchar* const* string, const GLint* length) {
    LOG()
    // Reject invalid calls before reading caller memory or changing any cached
    // source. glIsShader/glIsProgram do not consume an earlier driver error.
    if (count < 0) {
        set_shader_source_error(GL_INVALID_VALUE);
        return;
    }
    if (!valid_shader_source_object(shader)) return;
    // Positive count with a null array has no defined GL memory-access behavior;
    // reject it defensively instead of crashing inside the translator.
    if (count > 0 && !string) {
        set_shader_source_error(GL_INVALID_VALUE);
        return;
    }

    try {
        std::string glsl_src, essl_src;
        std::vector<uniform_default_t> uniform_defaults;
        // Include room for the terminator in GL_SHADER_SOURCE_LENGTH's GLint.
        // Checked subtraction prevents size_t wrap as well as GLint truncation.
        const size_t maximum = std::min(glsl_src.max_size(),
            static_cast<size_t>(std::numeric_limits<GLint>::max()) - 1);
        size_t total = 0;
        for (GLsizei i = 0; i < count; ++i) {
            if (length && length[i] == 0) continue; // do not even touch string[i]
            if (!string[i]) {
                set_shader_source_error(GL_INVALID_VALUE);
                return;
            }
            const size_t part = length && length[i] >= 0
                ? static_cast<size_t>(length[i]) : strlen(string[i]);
            if (part > maximum - total) {
                set_shader_source_error(GL_OUT_OF_MEMORY);
                return;
            }
            total += part;
        }
        glsl_src.reserve(total);
        for (GLsizei i = 0; i < count; ++i) {
            if (length && length[i] == 0) continue;
            const size_t part = length && length[i] >= 0
                ? static_cast<size_t>(length[i]) : strlen(string[i]);
            glsl_src.append(string[i], part); // no separators or implicit newlines
        }

        const bool sampler_buffer = hardware->emulate_texture_buffer && check_if_sampler_buffer_used(glsl_src);
        const bool embedded_nul = glsl_src.find('\0') != std::string::npos;
        // Empty source is a valid replacement. Explicit lengths can include
        // embedded NUL bytes, but this translator/cache accepts C strings; bypass
        // it rather than silently compiling only the prefix before the NUL.
        if (glsl_src.empty() || embedded_nul || is_direct_shader(glsl_src.c_str())) {
            essl_src = glsl_src;
        } else {
            GLint shader_type = 0;
            GLES.glGetShaderiv(shader, GL_SHADER_TYPE, &shader_type);
            int result = -1;
            try {
                const int version = getGLSLVersion(glsl_src.c_str());
                essl_src = GLSLtoGLSLES(glsl_src.c_str(), shader_type, hardware->es_version, version, result,
                                        &uniform_defaults);
            } catch (const std::bad_alloc&) {
                throw;
            } catch (const std::length_error&) {
                throw;
            } catch (const std::exception& error) {
                LOG_W_FORCE("Shader %u translation failed: %s", shader, error.what())
                result = -1;
            }
            if (result < 0 || essl_src.empty()) {
                // Replace the old source even when translation fails. Retain
                // the original bytes and let the driver's compile status/log
                // describe the new input; never silently keep the old source.
                essl_src = glsl_src;
                uniform_defaults.clear();
            }
        }
        if (essl_src.size() > maximum) {
            set_shader_source_error(GL_OUT_OF_MEMORY);
            return;
        }

        // Reserve a bookkeeping entry before touching driver state so a host
        // allocation failure cannot escape this C ABI after submission.
        auto sampler = shader_map_is_sampler_buffer_emulated.end();
        if (hardware->emulate_texture_buffer)
            sampler = shader_map_is_sampler_buffer_emulated.emplace(shader, false).first;
        auto defaults = shader_uniform_defaults.end();
        if (!uniform_defaults.empty())
            defaults = shader_uniform_defaults.emplace(shader, std::vector<uniform_default_t>()).first;
        const GLchar* source = essl_src.c_str();
        const GLint source_length = static_cast<GLint>(essl_src.size());
        GLES.glShaderSource(shader, 1, &source, &source_length);

        // All fallible preparation is complete. Commit using non-allocating
        // operations; rejected calls above leave every previous cache intact.
        shaderInfo.id = shader;
        // glLinkProgram's optional fragment-output patch uses a C string.
        // Do not let that later path turn this exact byte string into a prefix.
        if (essl_src.find('\0') != std::string::npos) shaderInfo.converted.clear();
        else shaderInfo.converted.swap(essl_src);
        shaderInfo.frag_data_changed_converted.clear();
        shaderInfo.frag_data_changed = 0;
        if (sampler != shader_map_is_sampler_buffer_emulated.end()) sampler->second = sampler_buffer;
        if (defaults != shader_uniform_defaults.end()) defaults->second.swap(uniform_defaults);
        else shader_uniform_defaults.erase(shader);
    } catch (const std::bad_alloc&) {
        set_shader_source_error(GL_OUT_OF_MEMORY);
    } catch (const std::length_error&) {
        set_shader_source_error(GL_OUT_OF_MEMORY);
    }
    // Do not use CHECK_GL_ERROR here: even a debug build must leave pre-existing
    // driver errors available to the caller of glGetError.
}

void glGetShaderiv(GLuint shader, GLenum pname, GLint* params) {
    LOG()
    GLES.glGetShaderiv(shader, pname, params);
    if (global_settings.ignore_error >= IgnoreErrorLevel::Partial && pname == GL_COMPILE_STATUS && !*params) {
        GLchar infoLog[512];
        GLES.glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        LOG_W_FORCE("Shader %d compilation failed: \n%s", shader, infoLog)
        LOG_W_FORCE("Now try to cheat.")
        *params = GL_TRUE;
    }
    CHECK_GL_ERROR
}

GLuint glCreateShader(GLenum shaderType) {
    if (global_settings.fsr1_setting != FSR1_Quality_Preset::Disabled && !fsrInitialized) {
        InitFSRResources();
    }

    LOG()
    LOG_D("glCreateShader(%s)", glEnumToString(shaderType))
    GLuint shader = GLES.glCreateShader(shaderType);
    if (shader != 0 && hardware->emulate_texture_buffer) shader_map_is_sampler_buffer_emulated[shader] = false;
    CHECK_GL_ERROR
    return shader;
}
