"""
Purpose
-------
Provide a GLFW and OpenGL backend for CUDA/OpenGL-interoperable 3D RGBA volume rendering in skyrmion_solver.

Usage
-----
Construct ``GLBackend(width, height, volume_shape, title="...")`` to create a window and a 3D ray-marching backend.
Use ``map_volume_pbo()`` and ``unmap_volume_pbo()`` to expose the OpenGL pixel-unpack buffer to CUDA.
Use ``upload_and_draw()`` to upload the CUDA-written RGBA volume into the OpenGL 3D texture and display the frame.

Output
------
A reusable rendering backend that keeps the whole visualization path on the GPU and ray-marches an OpenGL 3D RGBA texture.
"""

from __future__ import annotations
import ctypes
from dataclasses import dataclass
import glfw
import numpy as np
from OpenGL import GL
from numba import cuda
try:
    from cuda.bindings import runtime as cudart
except Exception:
    from cuda import cudart

def _cuda_err_to_int(err) -> int:
    """
    Convert a CUDA error value to an integer error code.

    Parameters
    ----------
    err
        CUDA error value returned by the runtime bindings.

    Returns
    -------
    int
        Integer error code.

    Examples
    --------
    Use ``code = _cuda_err_to_int(err)`` before comparing against the success code.
    """
    try:
        return int(err)
    except Exception:
        return 1

def _cuda_success_value() -> int:
    """
    Return the integer value corresponding to ``cudaSuccess``.

    Parameters
    ----------
    None

    Returns
    -------
    int
        Integer success code for the active CUDA runtime binding.

    Examples
    --------
    Use ``_cuda_err_to_int(err) == _cuda_success_value()`` to test whether a CUDA call succeeded.
    """
    if hasattr(cudart, "cudaError_t") and hasattr(cudart.cudaError_t, "cudaSuccess"):
        try:
            return int(cudart.cudaError_t.cudaSuccess)
        except Exception:
            return 0
    return 0

def _cuda_check(err, where: str = "CUDA call") -> None:
    """
    Raise an error if a CUDA runtime call did not succeed.

    Parameters
    ----------
    err
        CUDA return value or tuple containing the return value.
    where : str, optional
        Description of the CUDA call.

    Returns
    -------
    None
        The function returns normally when the CUDA call succeeded.

    Raises
    ------
    RuntimeError
        Raised if the CUDA runtime call failed.

    Examples
    --------
    Use ``_cuda_check(out, "cudaGraphicsMapResources")`` after a CUDA runtime call.
    """
    if isinstance(err, tuple) and len(err) >= 1:
        err = err[0]
    if _cuda_err_to_int(err) != _cuda_success_value():
        raise RuntimeError(f"{where} failed with error={err!r}")

def _cuda_gl_write_discard_flag() -> int:
    """
    Return the CUDA OpenGL registration flag for write-discard access.

    Parameters
    ----------
    None

    Returns
    -------
    int
        Integer flag value used with ``cudaGraphicsGLRegisterBuffer``.

    Examples
    --------
    Use ``flags = _cuda_gl_write_discard_flag()`` before registering a PBO with CUDA.
    """
    if hasattr(cudart, "cudaGraphicsRegisterFlagsWriteDiscard"):
        try:
            return int(getattr(cudart, "cudaGraphicsRegisterFlagsWriteDiscard"))
        except Exception:
            pass
    if hasattr(cudart, "cudaGraphicsRegisterFlags"):
        enum = getattr(cudart, "cudaGraphicsRegisterFlags")
        for name in ("cudaGraphicsRegisterFlagsWriteDiscard", "WriteDiscard"):
            if hasattr(enum, name):
                try:
                    return int(getattr(enum, name))
                except Exception:
                    pass
    return 2

def _is_ctypes_instance(x) -> bool:
    """
    Check whether an object is a ctypes scalar or structure instance.

    Parameters
    ----------
    x
        Object to test.

    Returns
    -------
    bool
        ``True`` if the object is a ctypes scalar or structure instance and ``False`` otherwise.

    Examples
    --------
    Use ``_is_ctypes_instance(resource)`` to decide whether a ctypes fallback path is available.
    """
    return isinstance(x, (ctypes._SimpleCData, ctypes.Structure))

def _cuda_register_gl_buffer(pbo_id: int, flags: int):
    """
    Register an OpenGL buffer object with CUDA.

    Parameters
    ----------
    pbo_id : int
        OpenGL buffer object identifier.
    flags : int
        CUDA registration flags.

    Returns
    -------
    object
        CUDA graphics resource handle for the registered buffer.

    Raises
    ------
    RuntimeError
        Raised if the CUDA registration call failed.

    Examples
    --------
    Use ``resource = _cuda_register_gl_buffer(pbo_id, flags)`` to register a PBO with CUDA.
    """
    try:
        out = cudart.cudaGraphicsGLRegisterBuffer(int(pbo_id), int(flags))
        if isinstance(out, tuple) and len(out) >= 2:
            err, resource = out[0], out[1]
            _cuda_check(err, "cudaGraphicsGLRegisterBuffer")
            return resource
    except TypeError:
        pass

    resource = ctypes.c_void_p()
    out = cudart.cudaGraphicsGLRegisterBuffer(ctypes.byref(resource), int(pbo_id), int(flags))
    _cuda_check(out, "cudaGraphicsGLRegisterBuffer")
    return resource

def _cuda_unregister_resource(resource) -> None:
    """
    Unregister a CUDA graphics resource.

    Parameters
    ----------
    resource
        CUDA graphics resource handle.

    Returns
    -------
    None
        The resource is unregistered from CUDA.

    Raises
    ------
    RuntimeError
        Raised if the CUDA unregister call failed.

    Examples
    --------
    Use ``_cuda_unregister_resource(resource)`` during cleanup.
    """
    out = cudart.cudaGraphicsUnregisterResource(resource)
    _cuda_check(out, "cudaGraphicsUnregisterResource")

def _cuda_map_resource(resource) -> tuple[int, int]:
    """
    Map a CUDA graphics resource and return its device pointer and size.

    Parameters
    ----------
    resource
        CUDA graphics resource handle.

    Returns
    -------
    tuple[int, int]
        Pair ``(ptr, nbytes)`` containing the mapped device pointer and mapped size in bytes.

    Raises
    ------
    RuntimeError
        Raised if a CUDA mapping call failed.
    TypeError
        Raised if all supported binding signatures fail.

    Examples
    --------
    Use ``ptr, nbytes = _cuda_map_resource(resource)`` before wrapping the mapped storage as a CUDA array.
    """
    map_attempts = [
        lambda: cudart.cudaGraphicsMapResources(1, [resource], 0),
        lambda: cudart.cudaGraphicsMapResources(1, (resource,), 0),
        lambda: cudart.cudaGraphicsMapResources(1, resource, 0),
    ]

    for attempt in map_attempts:
        try:
            out = attempt()
            _cuda_check(out, "cudaGraphicsMapResources")

            out2 = cudart.cudaGraphicsResourceGetMappedPointer(resource)
            if isinstance(out2, tuple) and len(out2) >= 3:
                err, ptr, size = out2[0], out2[1], out2[2]
                _cuda_check(err, "cudaGraphicsResourceGetMappedPointer")
                return int(ptr), int(size)
        except TypeError:
            continue

    if not _is_ctypes_instance(resource):
        raise TypeError("CUDA-OpenGL interop map() failed for a non-ctypes resource handle.")

    out = cudart.cudaGraphicsMapResources(1, ctypes.byref(resource), 0)
    _cuda_check(out, "cudaGraphicsMapResources")

    dev_ptr = ctypes.c_void_p()
    size = ctypes.c_size_t()
    out2 = cudart.cudaGraphicsResourceGetMappedPointer(ctypes.byref(dev_ptr), ctypes.byref(size), resource)
    _cuda_check(out2, "cudaGraphicsResourceGetMappedPointer")
    return int(dev_ptr.value), int(size.value)

def _cuda_unmap_resource(resource) -> None:
    """
    Unmap a previously mapped CUDA graphics resource.

    Parameters
    ----------
    resource
        CUDA graphics resource handle.

    Returns
    -------
    None
        The resource is unmapped and returned to OpenGL access.

    Raises
    ------
    RuntimeError
        Raised if a CUDA unmap call failed.
    TypeError
        Raised if all supported binding signatures fail.

    Examples
    --------
    Use ``_cuda_unmap_resource(resource)`` after CUDA writes to the mapped buffer have finished.
    """
    unmap_attempts = [
        lambda: cudart.cudaGraphicsUnmapResources(1, [resource], 0),
        lambda: cudart.cudaGraphicsUnmapResources(1, (resource,), 0),
        lambda: cudart.cudaGraphicsUnmapResources(1, resource, 0),
    ]

    for attempt in unmap_attempts:
        try:
            out = attempt()
            _cuda_check(out, "cudaGraphicsUnmapResources")
            return
        except TypeError:
            continue

    if not _is_ctypes_instance(resource):
        raise TypeError("CUDA-OpenGL interop unmap() failed for a non-ctypes resource handle.")

    out = cudart.cudaGraphicsUnmapResources(1, ctypes.byref(resource), 0)
    _cuda_check(out, "cudaGraphicsUnmapResources")

def _compile_shader(src: str, shader_type: int) -> int:
    """
    Compile a GLSL shader stage.

    Parameters
    ----------
    src : str
        GLSL source code.
    shader_type : int
        OpenGL shader type enum.

    Returns
    -------
    int
        OpenGL shader object identifier.

    Raises
    ------
    RuntimeError
        Raised if shader compilation failed.

    Examples
    --------
    Use ``shader = _compile_shader(src, GL.GL_VERTEX_SHADER)`` to compile a shader stage.
    """
    shader = GL.glCreateShader(shader_type)
    GL.glShaderSource(shader, src)
    GL.glCompileShader(shader)

    ok = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
    if not ok:
        log = GL.glGetShaderInfoLog(shader).decode("utf-8", errors="ignore")
        raise RuntimeError(f"Shader compile failed:\n{log}")
    return shader

def _link_program(vs_src: str, fs_src: str) -> int:
    """
    Compile vertex and fragment shaders and link them into an OpenGL program.

    Parameters
    ----------
    vs_src : str
        Vertex shader source code.
    fs_src : str
        Fragment shader source code.

    Returns
    -------
    int
        Linked OpenGL program identifier.

    Raises
    ------
    RuntimeError
        Raised if program linking failed.

    Examples
    --------
    Use ``prog = _link_program(vs_src, fs_src)`` to create a shader program.
    """
    vs = _compile_shader(vs_src, GL.GL_VERTEX_SHADER)
    fs = _compile_shader(fs_src, GL.GL_FRAGMENT_SHADER)

    prog = GL.glCreateProgram()
    GL.glAttachShader(prog, vs)
    GL.glAttachShader(prog, fs)
    GL.glLinkProgram(prog)

    ok = GL.glGetProgramiv(prog, GL.GL_LINK_STATUS)
    if not ok:
        log = GL.glGetProgramInfoLog(prog).decode("utf-8", errors="ignore")
        raise RuntimeError(f"Program link failed:\n{log}")

    GL.glDeleteShader(vs)
    GL.glDeleteShader(fs)
    return prog

def cuda_array_from_ptr(ptr_int: int, shape, dtype):
    """
    Wrap an existing device pointer as a Numba CUDA array view.

    Parameters
    ----------
    ptr_int : int
        Integer device pointer address.
    shape
        Shape used to interpret the raw buffer.
    dtype
        NumPy data type of the array view.

    Returns
    -------
    DeviceNDArray
        Numba CUDA array view over the existing device memory.

    Examples
    --------
    Use ``rgba = cuda_array_from_ptr(mapped.ptr, (H, W, 4), np.uint8)`` to interpret a mapped PBO as an array.
    """
    iface = {"data": (int(ptr_int), False), "shape": tuple(shape), "strides": None, "typestr": np.dtype(dtype).newbyteorder("|").str, "version": 2}

    class _Wrapper:
        __slots__ = ("__cuda_array_interface__",)

        def __init__(self, iface_dict):
            self.__cuda_array_interface__ = iface_dict

    return cuda.as_cuda_array(_Wrapper(iface))

@dataclass
class MappedPBO:
    """
    Container describing a mapped CUDA-registered PBO region.

    Parameters
    ----------
    ptr : int
        Integer device pointer to the mapped buffer.
    nbytes : int
        Size of the mapped region in bytes.

    Returns
    -------
    None
        The dataclass stores the mapped pointer and size.

    Examples
    --------
    Use ``mapped = backend.map_volume_pbo()`` and access ``mapped.ptr`` and ``mapped.nbytes``.
    """

    ptr: int
    nbytes: int

@dataclass
class OrbitCameraState:
    """
    Store the orbit-style camera state used by the backend.

    Parameters
    ----------
    view_rotation : ndarray
        Euler-angle style view rotation in degrees.
    view_translation : ndarray
        View translation vector.
    zoom : float
        Zoom parameter used by the ray construction.

    Returns
    -------
    None
        The dataclass stores the current camera state.

    Examples
    --------
    Use ``state = OrbitCameraState(...)`` to store camera values for the renderer.
    """

    view_rotation: np.ndarray
    view_translation: np.ndarray
    zoom: float

class GLBackend:
    """
    Provide a GLFW window and OpenGL 3D volume renderer using CUDA/OpenGL interoperability.

    Parameters
    ----------
    width : int
        Window width in pixels.
    height : int
        Window height in pixels.
    volume_shape : tuple[int, int, int]
        Logical volume shape given as ``(xlen, ylen, zlen)``.
    title : str, optional
        Initial window title.

    Examples
    --------
    Use ``backend = GLBackend(width, height, (xlen, ylen, zlen))`` to create the rendering backend.
    Use ``backend.map_volume_pbo()`` and ``backend.unmap_volume_pbo()`` to expose the 3D upload buffer to CUDA.
    Use ``backend.upload_and_draw()`` to upload the GPU-written volume and display the ray-marched frame.
    """

    _FULLSCREEN_VERTEX_SHADER = r"""
    #version 330 core
    layout (location = 0) in vec2 a_pos;
    out vec2 v_uv;
    void main() {
        v_uv = 0.5 * (a_pos + vec2(1.0));
        gl_Position = vec4(a_pos, 0.0, 1.0);
    }
    """

    _RAYMARCH_FRAGMENT_SHADER = r"""
    #version 330 core

    in vec2 v_uv;
    out vec4 frag_color;

    uniform sampler3D u_volume_tex;

    uniform vec4 u_inv_row0;
    uniform vec4 u_inv_row1;
    uniform vec4 u_inv_row2;

    uniform sampler1D u_transfer_tex;

    uniform vec3 u_box_scale;
    uniform vec3 u_single_color;

    uniform float u_zoom;
    uniform float u_brightness;
    uniform float u_levelset;
    uniform float u_density_min;
    uniform float u_density_max;
    uniform float u_transfer_scale;
    uniform float u_opacity_scale;
    uniform int u_max_steps;
    uniform float u_tstep;
    uniform int u_color_mode;

    const float PI = 3.14159265359;
    const float opacityThreshold = 0.95;

    struct Ray {
        vec3 o;
        vec3 d;
    };

    bool intersectBox(Ray r, vec3 bmin, vec3 bmax, out float tnear, out float tfar) {
        vec3 invR = 1.0 / r.d;
        vec3 tbot = invR * (bmin - r.o);
        vec3 ttop = invR * (bmax - r.o);
        vec3 tmin = min(ttop, tbot);
        vec3 tmax = max(ttop, tbot);

        float largest_tmin = max(max(tmin.x, tmin.y), tmin.z);
        float smallest_tmax = min(min(tmax.x, tmax.y), tmax.z);

        tnear = largest_tmin;
        tfar = smallest_tmax;
        return smallest_tmax > largest_tmin;
    }

    vec3 computeDirection(vec3 v) {
        return vec3(dot(v, u_inv_row0.xyz), dot(v, u_inv_row1.xyz), dot(v, u_inv_row2.xyz));
    }

    vec3 pionVectorToRGB(vec3 n) {
        float s = 1.0;
        float l = 0.5 * (n.z + 1.0);
        float c = (1.0 - abs(2.0 * l - 1.0)) * s;
        float h = atan(n.y, n.x);
        if (h < 0.0) {
            h += 2.0 * PI;
        }
        float x = c * (1.0 - abs(mod(3.0 * h / PI, 2.0) - 1.0));
        float m = l - 0.5 * c;
        float hp = 3.0 * h / PI;

        if (hp >= 0.0 && hp < 1.0) return vec3(c + m, x + m, m);
        else if (hp >= 1.0 && hp < 2.0) return vec3(x + m, c + m, m);
        else if (hp >= 2.0 && hp < 3.0) return vec3(m, c + m, x + m);
        else if (hp >= 3.0 && hp < 4.0) return vec3(m, x + m, c + m);
        else if (hp >= 4.0 && hp < 5.0) return vec3(x + m, m, c + m);
        else if (hp >= 5.0 && hp < 6.0) return vec3(c + m, m, x + m);
        else return vec3(0.0);
    }

    float scalarField(vec3 texcoord) {
        return texture(u_volume_tex, texcoord).x;
    }

    vec3 densityNormal(vec3 texcoord) {
        vec3 texel = 1.0 / vec3(textureSize(u_volume_tex, 0));
        float xm = scalarField(texcoord - vec3(texel.x, 0.0, 0.0));
        float xp = scalarField(texcoord + vec3(texel.x, 0.0, 0.0));
        float ym = scalarField(texcoord - vec3(0.0, texel.y, 0.0));
        float yp = scalarField(texcoord + vec3(0.0, texel.y, 0.0));
        float zm = scalarField(texcoord - vec3(0.0, 0.0, texel.z));
        float zp = scalarField(texcoord + vec3(0.0, 0.0, texel.z));

        vec3 g = vec3(xp - xm, yp - ym, zp - zm);
        float gn = length(g);
        if (gn < 1.0e-10) {
            return vec3(0.0, 0.0, 1.0);
        }
        return g / gn;
    }

    float surfaceShade(vec3 texcoord, vec3 ray_dir) {
        vec3 n = densityNormal(texcoord);
        vec3 light_dir = normalize(vec3(0.6, 0.5, 1.0));
        float diffuse = max(dot(n, light_dir), 0.0);
        vec3 view_dir = normalize(-ray_dir);
        vec3 half_vec = normalize(light_dir + view_dir);
        float specular = pow(max(dot(n, half_vec), 0.0), 24.0);
        float rim = pow(1.0 - max(dot(n, view_dir), 0.0), 2.5);
        return 0.25 + 0.85 * diffuse + 0.35 * specular + 0.1 * rim;
    }

    float boxEdge(vec3 p, vec3 bmin, vec3 bmax)
    {
        float eps = 0.005;

        vec3 d = min(abs(p - bmin), abs(p - bmax));

        int nearFaces = int(d.x < eps) + int(d.y < eps) + int(d.z < eps);

        // edge = intersection of two faces
        return nearFaces >= 2 ? 1.0 : 0.0;
    }
    
    void main() {
        float u = v_uv.x * 2.0 - 1.0;
        float v = v_uv.y * 2.0 - 1.0;

        Ray eyeRay;
        eyeRay.o = vec3(u_inv_row0.w, u_inv_row1.w, u_inv_row2.w);
        eyeRay.d = normalize(computeDirection(vec3(u, v, -u_zoom)));

        vec3 boxMin = -u_box_scale;
        vec3 boxMax =  u_box_scale;

        float tnear;
        float tfar;
        if (!intersectBox(eyeRay, boxMin, boxMax, tnear, tfar)) {
            discard;
        }

        if (tnear < 0.0) {
            tnear = 0.0;
        }

        vec4 sum = vec4(0.0);
        float t = tnear;
        vec3 pos = eyeRay.o + eyeRay.d * tnear;
        vec3 stepv = eyeRay.d * u_tstep;

        for (int i = 0; i < u_max_steps; ++i) {
            float edge = boxEdge(pos, boxMin, boxMax);
            if (edge > 0.5) {
                frag_color = vec4(0.0, 0.0, 0.0, 1.0);
                return;
            }
            vec3 texcoord = (pos / u_box_scale) * 0.5 + 0.5;
            vec4 sample_val = texture(u_volume_tex, texcoord);

            float denom = max(u_density_max - u_density_min, 1.0e-30);
            float norm = (sample_val.x - u_density_min) / denom;

            float input_val = (norm - u_levelset) * u_transfer_scale;
            if (input_val < 0.0) {
                input_val = 0.0;
            }

            vec3 rgb;
            vec4 col;

            if (u_color_mode == 0) {
                float tf_coord = clamp(norm, 0.0, 1.0);
                col = texture(u_transfer_tex, tf_coord);
                col.a *= u_opacity_scale;
            } else {
                rgb = pionVectorToRGB(sample_val.yzw);
                col = input_val * vec4(rgb, u_opacity_scale);
            }

            col.rgb *= col.a;
            sum = sum + col * (1.0 - sum.a);

            if (sum.a > opacityThreshold) {
                break;
            }

            t += u_tstep;
            if (t > tfar) {
                break;
            }
            pos += stepv;
        }

        sum *= u_brightness;
        frag_color = clamp(sum, 0.0, 1.0);
    }
    """

    def __init__(self, width: int, height: int, volume_shape: tuple[int, int, int], title: str = "Soliton Solver Volume Renderer") -> None:
        """
        Create the window and initialize the OpenGL resources used for ray-marched rendering.

        Parameters
        ----------
        width : int
            Window width in pixels.
        height : int
            Window height in pixels.
        volume_shape : tuple[int, int, int]
            Logical volume shape given as ``(xlen, ylen, zlen)``.
        title : str, optional
            Initial window title.

        Returns
        -------
        None
            The backend is initialized with a window, shaders, textures, buffers, and CUDA interop state.

        Examples
        --------
        Use ``backend = GLBackend(width, height, (xlen, ylen, zlen))`` to create the rendering backend.
        """
        self.width = int(width)
        self.height = int(height)

        if len(volume_shape) != 3:
            raise ValueError("volume_shape must be a length-3 tuple of the form (xlen, ylen, zlen).")
        self.xlen = int(volume_shape[0])
        self.ylen = int(volume_shape[1])
        self.zlen = int(volume_shape[2])

        if not glfw.init():
            raise RuntimeError("GLFW init failed.")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)

        self.window = glfw.create_window(self.width, self.height, str(title), None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed.")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_window_user_pointer(self.window, self)
        glfw.set_framebuffer_size_callback(self.window, self._resize_callback)

        from OpenGL import GLUT
        try:
            GLUT.glutInit()
        except Exception:
            try:
                GLUT.glutInit(["skyrmion_solver"])
            except Exception:
                pass

        self.camera = OrbitCameraState(view_rotation=np.array([0.0, 0.0, 0.0], dtype=np.float32), view_translation=np.array([0.0, 0.0, -4.0], dtype=np.float32), zoom=5.5)

        self.brightness = 6.0
        self.levelset = 0.2
        self.transfer_scale = 4.0
        self.opacity_scale = 1.0
        self.max_steps = 2500
        self.tstep = 0.002
        self.density_min = 0.0
        self.density_max = 1.0
        self.box_scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.single_color = np.array([1.0, 0.58, 0.18], dtype=np.float32)
        self.color_mode = 0

        self.hud_enabled = True
        self.hud_top_text = ""
        self.hud_bottom_text = ""

        self._volume_uploaded = False

        self._init_gl_objects()

    def _init_gl_objects(self) -> None:
        """
        Create shaders, fullscreen geometry, the 3D volume texture, and the CUDA-mapped volume-upload PBO.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The OpenGL program, geometry, texture, and PBO are initialized.

        Examples
        --------
        Use ``self._init_gl_objects()`` internally during backend construction.
        """
        self.prog = _link_program(self._FULLSCREEN_VERTEX_SHADER, self._RAYMARCH_FRAGMENT_SHADER)

        quad = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0], dtype=np.float32)

        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)

        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, quad.nbytes, quad, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))
        GL.glBindVertexArray(0)

        self.volume_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_3D, self.volume_tex)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, GL.GL_RGBA32F, self.xlen, self.ylen, self.zlen, 0, GL.GL_RGBA, GL.GL_FLOAT, None)
        GL.glBindTexture(GL.GL_TEXTURE_3D, 0)

        self.transfer_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_1D, self.transfer_tex)
        GL.glTexParameteri(GL.GL_TEXTURE_1D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_1D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_1D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        transfer_func = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.5, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        GL.glTexImage1D(GL.GL_TEXTURE_1D, 0, GL.GL_RGBA32F, transfer_func.shape[0], 0, GL.GL_RGBA, GL.GL_FLOAT, transfer_func)
        GL.glBindTexture(GL.GL_TEXTURE_1D, 0)

        self.volume_pbo = GL.glGenBuffers(1)
        self._volume_pbo_nbytes = int(self.xlen * self.ylen * self.zlen * 4 * np.dtype(np.float32).itemsize)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self.volume_pbo)
        GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, self._volume_pbo_nbytes, None, GL.GL_STREAM_DRAW)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

        flags = _cuda_gl_write_discard_flag()
        self.cuda_volume_res = _cuda_register_gl_buffer(int(self.volume_pbo), int(flags))

        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

    def _resize_callback(self, window, width: int, height: int) -> None:
        """
        Update the stored framebuffer size after a resize.

        Parameters
        ----------
        window
            GLFW window handle.
        width : int
            New framebuffer width.
        height : int
            New framebuffer height.

        Returns
        -------
        None
            The backend width and height are updated.

        Examples
        --------
        Use ``glfw.set_framebuffer_size_callback(self.window, self._resize_callback)`` to register the resize handler.
        """
        self.width = max(int(width), 1)
        self.height = max(int(height), 1)

    def should_close(self) -> bool:
        """
        Return whether the window has been requested to close.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            ``True`` if the window should close.

        Examples
        --------
        Use ``while not backend.should_close():`` to drive the render loop.
        """
        return glfw.window_should_close(self.window)

    def begin_frame(self) -> None:
        """
        Poll GLFW events at the start of a frame.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Pending GLFW events are processed.

        Examples
        --------
        Use ``backend.begin_frame()`` at the start of each frame.
        """
        glfw.poll_events()

    def end_frame(self) -> None:
        """
        Present the current frame.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The front and back buffers are swapped.

        Examples
        --------
        Use ``backend.end_frame()`` after drawing the current frame.
        """
        glfw.swap_buffers(self.window)

    def set_window_title(self, title: str) -> None:
        """
        Set the window title.

        Parameters
        ----------
        title : str
            New window title.

        Returns
        -------
        None
            The GLFW window title is updated.

        Examples
        --------
        Use ``backend.set_window_title("Energy density")`` to update the window title.
        """
        glfw.set_window_title(self.window, str(title))

    def set_volume_shape(self, volume_shape: tuple[int, int, int]) -> None:
        """
        Replace the logical volume dimensions and reallocate the 3D texture and upload PBO.

        Parameters
        ----------
        volume_shape : tuple[int, int, int]
            New volume shape given as ``(xlen, ylen, zlen)``.

        Returns
        -------
        None
            The volume texture and upload PBO storage are reallocated.

        Examples
        --------
        Use ``backend.set_volume_shape((xlen, ylen, zlen))`` when the simulation lattice size changes.
        """
        if len(volume_shape) != 3:
            raise ValueError("volume_shape must be a length-3 tuple of the form (xlen, ylen, zlen).")

        self.xlen = int(volume_shape[0])
        self.ylen = int(volume_shape[1])
        self.zlen = int(volume_shape[2])

        GL.glBindTexture(GL.GL_TEXTURE_3D, self.volume_tex)
        GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, GL.GL_RGBA32F, self.xlen, self.ylen, self.zlen, 0, GL.GL_RGBA, GL.GL_FLOAT, None)
        GL.glBindTexture(GL.GL_TEXTURE_3D, 0)

        if getattr(self, "cuda_volume_res", None) is not None:
            try:
                _cuda_unregister_resource(self.cuda_volume_res)
            except Exception:
                pass
            self.cuda_volume_res = None

        self._volume_pbo_nbytes = int(self.xlen * self.ylen * self.zlen * 4 * np.dtype(np.float32).itemsize)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self.volume_pbo)
        GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, self._volume_pbo_nbytes, None, GL.GL_STREAM_DRAW)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

        flags = _cuda_gl_write_discard_flag()
        self.cuda_volume_res = _cuda_register_gl_buffer(int(self.volume_pbo), int(flags))
        self._volume_uploaded = False

    def map_volume_pbo(self) -> MappedPBO:
        """
        Map the CUDA-registered pixel-unpack buffer used for 3D RGBA volume uploads.

        Parameters
        ----------
        None

        Returns
        -------
        MappedPBO
            Object containing the mapped device pointer and mapped size in bytes.

        Examples
        --------
        Use ``mapped = backend.map_volume_pbo()`` before writing the RGBA volume from CUDA.
        """
        ptr, nbytes = _cuda_map_resource(self.cuda_volume_res)
        return MappedPBO(ptr=int(ptr), nbytes=int(nbytes))

    def unmap_volume_pbo(self) -> None:
        """
        Unmap the volume upload PBO after CUDA writes complete.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The PBO is returned to OpenGL access.

        Examples
        --------
        Use ``backend.unmap_volume_pbo()`` after the CUDA volume-fill kernel completes.
        """
        _cuda_unmap_resource(self.cuda_volume_res)
        self._volume_uploaded = True

    def set_render_params(
        self,
        *,
        brightness: float | None = None,
        levelset: float | None = None,
        transfer_scale: float | None = None,
        opacity_scale: float | None = None,
        max_steps: int | None = None,
        tstep: float | None = None,
        density_min: float | None = None,
        density_max: float | None = None,
        box_scale: tuple[float, float, float] | np.ndarray | None = None,
        single_color: tuple[float, float, float] | np.ndarray | None = None,
        color_mode: int | None = None,
    ) -> None:
        """
        Update the rendering controls used by the ray-march shader.

        Parameters
        ----------
        brightness : float or None, optional
            Brightness multiplier.
        levelset : float or None, optional
            Lower density threshold in normalized units.
        transfer_scale : float or None, optional
            Transfer-function scaling factor.
        opacity_scale : float or None, optional
            Additional opacity multiplier.
        max_steps : int or None, optional
            Maximum ray-march step count.
        tstep : float or None, optional
            Ray-march step size.
        density_min : float or None, optional
            Density normalization minimum.
        density_max : float or None, optional
            Density normalization maximum.
        box_scale : tuple[float, float, float] or ndarray or None, optional
            Box half-extent aspect ratio scaling.
        single_color : tuple[float, float, float] or ndarray or None, optional
            Fixed RGB color for single-color mode.
        color_mode : int or None, optional
            Color mode selector where ``0`` is fixed single-color mode and ``1`` is Runge-vector mode.

        Returns
        -------
        None
            The backend render parameters are updated.

        Examples
        --------
        Use ``backend.set_render_params(levelset=0.05, transfer_scale=8.0)`` to adjust rendering.
        """
        if brightness is not None:
            self.brightness = float(brightness)
        if levelset is not None:
            self.levelset = float(levelset)
        if transfer_scale is not None:
            self.transfer_scale = float(transfer_scale)
        if opacity_scale is not None:
            self.opacity_scale = float(opacity_scale)
        if max_steps is not None:
            self.max_steps = int(max_steps)
        if tstep is not None:
            self.tstep = float(tstep)
        if density_min is not None:
            self.density_min = float(density_min)
        if density_max is not None:
            self.density_max = float(density_max)
        if box_scale is not None:
            arr = np.asarray(box_scale, dtype=np.float32)
            if arr.shape != (3,):
                raise ValueError("box_scale must have shape (3,).")
            self.box_scale = np.maximum(arr, np.array([1.0e-8, 1.0e-8, 1.0e-8], dtype=np.float32))
        if single_color is not None:
            arr = np.asarray(single_color, dtype=np.float32)
            if arr.shape != (3,):
                raise ValueError("single_color must have shape (3,).")
            self.single_color = np.clip(arr, 0.0, 1.0)
        if color_mode is not None:
            self.color_mode = int(color_mode)

    def set_camera_pose(
        self,
        *,
        view_rotation: tuple[float, float, float] | np.ndarray | None = None,
        view_translation: tuple[float, float, float] | np.ndarray | None = None,
        zoom: float | None = None,
    ) -> None:
        """
        Update the orbit-style camera pose used by the backend.

        Parameters
        ----------
        view_rotation : tuple[float, float, float] or ndarray or None, optional
            Euler-style rotation angles in degrees.
        view_translation : tuple[float, float, float] or ndarray or None, optional
            View translation vector.
        zoom : float or None, optional
            Zoom parameter used by the ray construction.

        Returns
        -------
        None
            The stored camera state is updated.

        Examples
        --------
        Use ``backend.set_camera_pose(view_rotation=rot, view_translation=tr, zoom=5.0)`` to update the camera.
        """
        if view_rotation is not None:
            arr = np.asarray(view_rotation, dtype=np.float32)
            if arr.shape != (3,):
                raise ValueError("view_rotation must have shape (3,).")
            self.camera.view_rotation = arr.copy()
        if view_translation is not None:
            arr = np.asarray(view_translation, dtype=np.float32)
            if arr.shape != (3,):
                raise ValueError("view_translation must have shape (3,).")
            self.camera.view_translation = arr.copy()
        if zoom is not None:
            self.camera.zoom = float(zoom)

    def _rotation_matrix(self) -> np.ndarray:
        """
        Construct the current camera rotation matrix from the stored Euler angles.

        Parameters
        ----------
        None

        Returns
        -------
        ndarray
            Float32 array of shape ``(3, 3)`` containing the camera rotation matrix.

        Examples
        --------
        Use ``rot = self._rotation_matrix()`` internally when building the inverse-view rows.
        """
        rx = np.deg2rad(-float(self.camera.view_rotation[0]))
        ry = np.deg2rad(-float(self.camera.view_rotation[1]))

        cx = np.cos(rx)
        sx = np.sin(rx)
        cy = np.cos(ry)
        sy = np.sin(ry)

        rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
        rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)

        return rot_x @ rot_y

    def make_inv_view_rows(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the inverse-view rows used by the fragment shader ray construction.

        Parameters
        ----------
        None

        Returns
        -------
        tuple[ndarray, ndarray, ndarray]
            Three length-4 float32 row vectors.

        Examples
        --------
        Use ``row0, row1, row2 = backend.make_inv_view_rows()`` before drawing.
        """
        rot3 = self._rotation_matrix()

        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = rot3

        tx = -float(self.camera.view_translation[0])
        ty = -float(self.camera.view_translation[1])
        tz = -float(self.camera.view_translation[2])

        trans = np.array([[1.0, 0.0, 0.0, tx], [0.0, 1.0, 0.0, ty], [0.0, 0.0, 1.0, tz], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        model_view = rot @ trans

        row0 = model_view[0, :4].astype(np.float32).copy()
        row1 = model_view[1, :4].astype(np.float32).copy()
        row2 = model_view[2, :4].astype(np.float32).copy()
        return row0, row1, row2

    def set_hud_text(self, *, top: str = "", bottom: str = "") -> None:
        """
        Store HUD text strings.

        Parameters
        ----------
        top : str, optional
            Top HUD text line.
        bottom : str, optional
            Bottom HUD text line.

        Returns
        -------
        None
            The HUD text strings are stored.

        Examples
        --------
        Use ``backend.set_hud_text(top="...", bottom="...")`` to update the stored HUD text.
        """
        self.hud_top_text = str(top or "")
        self.hud_bottom_text = str(bottom or "")

    def _draw_text(self, x: int, y: int, s: str) -> None:
        """
        Draw a text string using a GLUT bitmap font.

        Parameters
        ----------
        x : int
            Pixel x coordinate.
        y : int
            Pixel y coordinate.
        s : str
            Text string to draw.

        Returns
        -------
        None
            The text is drawn in the current OpenGL context.
        """
        from OpenGL import GLUT

        GL.glRasterPos2i(int(x), int(y))
        for ch in (s or ""):
            GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_8_BY_13, ord(ch))

    def _draw_hud_bar(self, *, y0: int, height: int, text_y: int, text: str) -> None:
        """
        Draw a translucent HUD bar with text.

        Parameters
        ----------
        y0 : int
            Top edge of the bar in pixels.
        height : int
            Height of the bar in pixels.
        text_y : int
            Baseline y coordinate for the text.
        text : str
            Text to draw inside the bar.

        Returns
        -------
        None
            The HUD bar and its text are drawn.
        """
        if height <= 0:
            return

        GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_COLOR_BUFFER_BIT | GL.GL_TRANSFORM_BIT)
        GL.glDisable(GL.GL_TEXTURE_3D)
        GL.glDisable(GL.GL_BLEND)

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPushMatrix()
        GL.glLoadIdentity()
        GL.glOrtho(0, self.width, self.height, 0, -1, 1)

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        GL.glLoadIdentity()

        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        GL.glColor4f(0.15, 0.15, 0.15, 0.75)
        GL.glBegin(GL.GL_QUADS)
        GL.glVertex2i(0, y0)
        GL.glVertex2i(self.width, y0)
        GL.glVertex2i(self.width, y0 + height)
        GL.glVertex2i(0, y0 + height)
        GL.glEnd()

        GL.glColor3f(1.0, 1.0, 1.0)
        self._draw_text(8, text_y, text)

        GL.glPopMatrix()
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPopMatrix()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopAttrib()

    def _draw_hud(self) -> None:
        """
        Draw the top and bottom HUD overlays if enabled.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The HUD overlays are drawn when enabled and text is present.
        """
        if not self.hud_enabled:
            return

        top = self.hud_top_text
        bottom = self.hud_bottom_text
        if not top and not bottom:
            return

        bar_h = 22
        if top:
            self._draw_hud_bar(y0=0, height=bar_h, text_y=15, text=top)
        if bottom:
            self._draw_hud_bar(y0=self.height - bar_h, height=bar_h, text_y=self.height - 7, text=bottom)

    def _upload_volume_pbo_to_texture(self) -> None:
        """
        Upload the current pixel-unpack buffer contents into the OpenGL 3D RGBA texture.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The texture contents are updated from the PBO.

        Examples
        --------
        Use ``backend._upload_volume_pbo_to_texture()`` internally after CUDA writes to the volume PBO.
        """
        if not self._volume_uploaded:
            return
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self.volume_pbo)
        GL.glBindTexture(GL.GL_TEXTURE_3D, self.volume_tex)
        GL.glTexSubImage3D(GL.GL_TEXTURE_3D, 0, 0, 0, 0, self.xlen, self.ylen, self.zlen, GL.GL_RGBA, GL.GL_FLOAT, ctypes.c_void_p(0))
        GL.glBindTexture(GL.GL_TEXTURE_3D, 0)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

    def _draw_fullscreen_quad(self) -> None:
        """
        Draw the ray-marched volume to the window.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The framebuffer is cleared and the current volume is ray-marched to the screen.

        Examples
        --------
        Use ``self._draw_fullscreen_quad()`` internally after updating the 3D volume texture.
        """
        if not self._volume_uploaded:
            return

        row0, row1, row2 = self.make_inv_view_rows()

        GL.glViewport(0, 0, self.width, self.height)
        GL.glClearColor(1.0, 1.0, 1.0, 1.0)  # white background
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        GL.glUseProgram(self.prog)

        GL.glUniform4f(GL.glGetUniformLocation(self.prog, "u_inv_row0"), float(row0[0]), float(row0[1]), float(row0[2]), float(row0[3]))
        GL.glUniform4f(GL.glGetUniformLocation(self.prog, "u_inv_row1"), float(row1[0]), float(row1[1]), float(row1[2]), float(row1[3]))
        GL.glUniform4f(GL.glGetUniformLocation(self.prog, "u_inv_row2"), float(row2[0]), float(row2[1]), float(row2[2]), float(row2[3]))

        GL.glUniform3f(GL.glGetUniformLocation(self.prog, "u_box_scale"), float(self.box_scale[0]), float(self.box_scale[1]), float(self.box_scale[2]))
        GL.glUniform3f(GL.glGetUniformLocation(self.prog, "u_single_color"), float(self.single_color[0]), float(self.single_color[1]), float(self.single_color[2]))
        GL.glUniform1f(GL.glGetUniformLocation(self.prog, "u_zoom"), float(self.camera.zoom))
        GL.glUniform1f(GL.glGetUniformLocation(self.prog, "u_brightness"), float(self.brightness))
        GL.glUniform1f(GL.glGetUniformLocation(self.prog, "u_levelset"), float(self.levelset))
        GL.glUniform1f(GL.glGetUniformLocation(self.prog, "u_density_min"), float(self.density_min))
        GL.glUniform1f(GL.glGetUniformLocation(self.prog, "u_density_max"), float(self.density_max))
        GL.glUniform1f(GL.glGetUniformLocation(self.prog, "u_transfer_scale"), float(self.transfer_scale))
        GL.glUniform1f(GL.glGetUniformLocation(self.prog, "u_opacity_scale"), float(self.opacity_scale))
        GL.glUniform1i(GL.glGetUniformLocation(self.prog, "u_max_steps"), int(self.max_steps))
        GL.glUniform1f(GL.glGetUniformLocation(self.prog, "u_tstep"), float(self.tstep))
        GL.glUniform1i(GL.glGetUniformLocation(self.prog, "u_color_mode"), int(self.color_mode))

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_3D, self.volume_tex)
        GL.glUniform1i(GL.glGetUniformLocation(self.prog, "u_volume_tex"), 0)

        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_1D, self.transfer_tex)
        GL.glUniform1i(GL.glGetUniformLocation(self.prog, "u_transfer_tex"), 1)

        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        self._draw_hud()

    def upload_and_draw(self) -> None:
        """
        Upload the current PBO-backed RGBA volume into the 3D texture and draw the frame.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The current GPU-written volume is displayed in the window.

        Examples
        --------
        Use ``backend.upload_and_draw()`` after unmapping the volume PBO.
        """
        self._upload_volume_pbo_to_texture()
        self._draw_fullscreen_quad()

    def close(self) -> None:
        """
        Release CUDA and OpenGL resources and destroy the window.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Registered resources, buffers, textures, shaders, and the window are released.

        Examples
        --------
        Use ``backend.close()`` when the render loop finishes.
        """
        if getattr(self, "cuda_volume_res", None) is not None:
            try:
                _cuda_unregister_resource(self.cuda_volume_res)
            except Exception:
                pass
            self.cuda_volume_res = None

        try:
            if getattr(self, "prog", None):
                GL.glDeleteProgram(self.prog)
            if getattr(self, "vao", None):
                GL.glDeleteVertexArrays(1, [self.vao])
            if getattr(self, "vbo", None):
                GL.glDeleteBuffers(1, [self.vbo])
            if getattr(self, "volume_tex", None):
                GL.glDeleteTextures(1, [self.volume_tex])
            if getattr(self, "transfer_tex", None):
                GL.glDeleteTextures(1, [self.transfer_tex])
            if getattr(self, "volume_pbo", None):
                GL.glDeleteBuffers(1, [self.volume_pbo])
        except Exception:
            pass

        try:
            if getattr(self, "window", None):
                glfw.destroy_window(self.window)
        finally:
            glfw.terminate()