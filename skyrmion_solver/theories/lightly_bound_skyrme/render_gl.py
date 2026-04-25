"""
Purpose
-------
Provide the lightly bound Skyrme CUDA/OpenGL ray-marching renderer that keeps the visualization volume on the GPU.

Usage
-----
Use ``GLRenderer`` to render the lightly bound Skyrme simulation interactively.
Use ``advance_solver`` to step the simulation between rendered frames.
Use ``run_viewer`` to start the interactive viewer loop.

Output
------
A two-mode 3D viewer where energy density controls opacity and either a fixed color or Runge pion-vector coloring controls the rendered RGB output.
"""

from __future__ import annotations
import math
import glfw
import numpy as np
from numba import cuda, float64
import os
from datetime import datetime
from PIL import Image
from OpenGL.GL import glPixelStorei, glReadBuffer, glReadPixels, GL_PACK_ALIGNMENT, GL_FRONT, GL_RGB, GL_UNSIGNED_BYTE
from skyrmion_solver.visualization.gl_backend import GLBackend, cuda_array_from_ptr
from skyrmion_solver.theories.lightly_bound_skyrme.observables import compute_skyrmion_number

DISPLAY_ENERGY_SINGLE = 1
DISPLAY_ENERGY_RUNGE = 2

DISPLAY_TITLES = {
    DISPLAY_ENERGY_SINGLE: "Lightly bound Skyrme: Energy density",
    DISPLAY_ENERGY_RUNGE: "Lightly bound Skyrme: Energy density (Runge)",
}

_REDUCE_BLOCK_SIZE = 256

def _print_matrix(name: str, mat: np.ndarray) -> None:
    """
    Print a 3x3 tensor in a compact fixed-width format.

    Parameters
    ----------
    name : str
        Tensor name.
    mat : ndarray
        Tensor with shape ``(3, 3)``.

    Returns
    -------
    None
        The tensor is printed to standard output.
    """
    print(f"{name} =")
    for i in range(3):
        print("  [" + ", ".join(f"{float(mat[i, j]): .2f}" for j in range(3)) + "]")

@cuda.jit
def reduce_max_kernel(inp, out, n: int):
    """
    Reduce a flattened device array to blockwise maxima on the GPU.

    Parameters
    ----------
    inp : device array
        Input flattened array.
    out : device array
        Output array storing one maximum per CUDA block.
    n : int
        Number of valid entries in ``inp``.

    Returns
    -------
    None
        The blockwise maxima are written into ``out``.

    Examples
    --------
    Launch ``reduce_max_kernel[grid, block](inp, out, n)`` to compute blockwise maxima.
    """
    smem = cuda.shared.array(shape=_REDUCE_BLOCK_SIZE, dtype=float64)

    tid = cuda.threadIdx.x
    block = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    i = block * block_size + tid

    if i < n:
        smem[tid] = inp[i]
    else:
        smem[tid] = -1.0e300

    cuda.syncthreads()

    stride = block_size // 2
    while stride > 0:
        if tid < stride:
            a = smem[tid]
            b = smem[tid + stride]
            smem[tid] = a if a > b else b
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        out[block] = smem[0]

@cuda.jit
def reduce_min_kernel(inp, out, n: int):
    """
    Reduce a flattened device array to blockwise minima on the GPU.

    Parameters
    ----------
    inp : device array
        Input flattened array.
    out : device array
        Output array storing one minimum per CUDA block.
    n : int
        Number of valid entries in ``inp``.

    Returns
    -------
    None
        The blockwise minima are written into ``out``.

    Examples
    --------
    Launch ``reduce_min_kernel[grid, block](inp, out, n)`` to compute blockwise minima.
    """
    smem = cuda.shared.array(shape=_REDUCE_BLOCK_SIZE, dtype=float64)

    tid = cuda.threadIdx.x
    block = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    i = block * block_size + tid

    if i < n:
        smem[tid] = inp[i]
    else:
        smem[tid] = 1.0e300

    cuda.syncthreads()

    stride = block_size // 2
    while stride > 0:
        if tid < stride:
            a = smem[tid]
            b = smem[tid + stride]
            smem[tid] = a if a < b else b
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        out[block] = smem[0]

@cuda.jit
def build_rgba_volume_kernel(volume_zyxc, density_flat, field_flat, density_min_arr, density_max_arr, xlen: int, ylen: int, zlen: int):
    """
    Build the RGBA visualization volume directly on the GPU.

    Parameters
    ----------
    volume_zyxc : device array
        Output RGBA volume with shape ``(zlen, ylen, xlen, 4)``.
    density_flat : device array
        Flattened scalar energy-density volume.
    field_flat : device array
        Flattened Skyrme field with component ordering ``(sigma, pi1, pi2, pi3)``.
    density_min_arr : device array
        Length-1 device array containing the global density minimum.
    density_max_arr : device array
        Length-1 device array containing the global density maximum.
    xlen : int
        Number of lattice points along the x direction.
    ylen : int
        Number of lattice points along the y direction.
    zlen : int
        Number of lattice points along the z direction.

    Returns
    -------
    None
        Channel 0 is filled with normalized energy density and channels 1:4 are filled with the normalized pion vector.

    Examples
    --------
    Launch ``build_rgba_volume_kernel[grid3d, block3d](volume_zyxc, density_flat, field_flat, density_min_arr, density_max_arr, xlen, ylen, zlen)`` to build the render volume.
    """
    x, y, z = cuda.grid(3)
    if x >= xlen or y >= ylen or z >= zlen:
        return

    site = z + y * zlen + x * ylen * zlen
    plane = xlen * ylen * zlen

    density = density_flat[site]
    vmin = density_min_arr[0]
    vmax = density_max_arr[0]
    density_norm = (density - vmin) / (vmax - vmin + 1.0e-30)

    pi1 = field_flat[site + 1 * plane]
    pi2 = field_flat[site + 2 * plane]
    pi3 = field_flat[site + 3 * plane]

    norm2 = pi1 * pi1 + pi2 * pi2 + pi3 * pi3
    if norm2 <= 1.0e-20:
        nx = 0.0
        ny = 0.0
        nz = 1.0
    else:
        invn = 1.0 / math.sqrt(norm2)
        nx = pi1 * invn
        ny = pi2 * invn
        nz = pi3 * invn

    volume_zyxc[z, y, x, 0] = density_norm
    volume_zyxc[z, y, x, 1] = nx
    volume_zyxc[z, y, x, 2] = ny
    volume_zyxc[z, y, x, 3] = nz


class GLRenderer:
    """
    Interactive renderer for the nuclear Skyrme simulation.

    Parameters
    ----------
    width : int
        Window width in pixels.
    height : int
        Window height in pixels.
    depth : int
        Volume depth in lattice points.
    title : str, optional
        Window title.

    Examples
    --------
    Use ``renderer = GLRenderer(width=1200, height=900, depth=params.zlen)`` to create the renderer.
    Use ``renderer.bind_sim(sim)`` to attach a simulation.
    Use ``renderer.render(...)`` inside the display loop to draw a frame.
    """

    def __init__(self, width: int, height: int, depth: int, title: str = "lightly_bound_skyrme (ray-march)") -> None:
        """
        Create the renderer and its GLFW/OpenGL backend.

        Parameters
        ----------
        width : int
            Window width in pixels.
        height : int
            Window height in pixels.
        depth : int
            Logical volume depth in lattice points.
        title : str, optional
            Window title.

        Returns
        -------
        None
            The renderer and backend are initialized.

        Examples
        --------
        Use ``renderer = GLRenderer(1200, 900, params.zlen)`` to create the viewer.
        """
        self.backend = GLBackend(int(width), int(height), (1, 1, int(depth)), title=title)

        self.width = int(width)
        self.height = int(height)
        self.depth = int(depth)

        self.display_mode = DISPLAY_ENERGY_SINGLE
        self._sim = None
        self.request_save_output = False
        self.is_running = False
        self.request_screenshot = False

        self.left_dragging = False
        self.right_dragging = False
        self.last_cursor = (0.0, 0.0)

        self.view_rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.view_translation = np.array([0.0, 0.0, -4.0], dtype=np.float32)

        self.zoom = 5.5
        self.brightness = 6.0
        self.levelset = 0.2
        self.transfer_scale = 4.0
        self.opacity_scale = 1.0
        self.max_steps = 2500
        self.tstep = 0.002
        self.single_color = np.array([0.3, 0.3, 0.3], dtype=np.float32)

        self._volume_dev = None
        self._max_partial = None
        self._max_scratch = None
        self._min_partial = None
        self._min_scratch = None

        glfw.set_key_callback(self.backend.window, self._on_key)
        glfw.set_cursor_pos_callback(self.backend.window, self._on_cursor_pos)
        glfw.set_mouse_button_callback(self.backend.window, self._on_mouse_button)
        glfw.set_scroll_callback(self.backend.window, self._on_scroll)

        self._update_window_title()
        self._sync_camera()

    def bind_sim(self, sim) -> None:
        """
        Attach a simulation instance to the renderer.

        Parameters
        ----------
        sim
            Simulation instance providing fields and parameter arrays.

        Returns
        -------
        None
            The simulation reference is stored by the renderer and the GPU RGBA volume is allocated.

        Examples
        --------
        Use ``renderer.bind_sim(sim)`` to attach the active simulation.
        """
        self._sim = sim

        xlen = int(sim.p_i_h[0])
        ylen = int(sim.p_i_h[1])
        zlen = int(sim.p_i_h[2])
        nsites = xlen * ylen * zlen
        nblocks0 = max(1, (nsites + _REDUCE_BLOCK_SIZE - 1) // _REDUCE_BLOCK_SIZE)

        self.backend.set_volume_shape((xlen, ylen, zlen))
        self._volume_dev = cuda.device_array((zlen, ylen, xlen, 4), dtype=np.float32)

        self._max_partial = cuda.device_array(nblocks0, dtype=np.float64)
        self._max_scratch = cuda.device_array(nblocks0, dtype=np.float64)
        self._min_partial = cuda.device_array(nblocks0, dtype=np.float64)
        self._min_scratch = cuda.device_array(nblocks0, dtype=np.float64)

        self._sync_box_scale()

    def _camera_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the current camera right, up, and forward directions in world coordinates.

        Parameters
        ----------
        None

        Returns
        -------
        tuple[ndarray, ndarray, ndarray]
            Right, up, and forward camera directions.

        Examples
        --------
        Use ``right, up, forward = self._camera_axes()`` when applying camera-relative translation.
        """
        row0, row1, row2 = self.backend.make_inv_view_rows()
        right = row0[:3].copy()
        up = row1[:3].copy()
        forward = row2[:3].copy()
        return right, up, forward

    def _sync_camera(self) -> None:
        """
        Push the current camera state into the OpenGL backend.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The backend camera pose is updated.

        Examples
        --------
        Use ``self._sync_camera()`` after changing the local camera state.
        """
        self.backend.set_camera_pose(view_rotation=self.view_rotation, view_translation=self.view_translation, zoom=self.zoom)

    def _sync_box_scale(self) -> None:
        """
        Set the ray-march box aspect ratio from the simulation domain extents.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The backend box scale is updated.

        Examples
        --------
        Use ``self._sync_box_scale()`` after binding a simulation or changing the domain.
        """
        if self._sim is None:
            return

        xsize = float(self._sim.p_f_h[0])
        ysize = float(self._sim.p_f_h[1])
        zsize = float(self._sim.p_f_h[2])
        scale = max(xsize, ysize, zsize, 1.0e-8)
        box_scale = np.array([xsize / scale, ysize / scale, zsize / scale], dtype=np.float32)
        self.backend.set_render_params(box_scale=box_scale)

    def _sync_render_params(self) -> None:
        """
        Push the current render controls into the OpenGL backend.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The backend render parameters are updated.

        Examples
        --------
        Use ``self._sync_render_params()`` before drawing a frame.
        """
        color_mode = 0 if self.display_mode == DISPLAY_ENERGY_SINGLE else 1
        kwargs = dict(
            brightness=self.brightness,
            levelset=self.levelset,
            transfer_scale=self.transfer_scale,
            opacity_scale=self.opacity_scale,
            max_steps=self.max_steps,
            tstep=self.tstep,
            single_color=self.single_color,
            color_mode=color_mode,
            density_max=1.0,
        )
        self.backend.set_render_params(**kwargs)

    def _reduce_density_extrema(self, density_flat):
        """
        Compute the global density minimum and maximum entirely on the GPU.

        Parameters
        ----------
        density_flat : device array
            Flattened scalar density field.

        Returns
        -------
        tuple[device array, device array]
            Device arrays of length 1 containing ``(density_min, density_max)``.

        Examples
        --------
        Use ``density_min_arr, density_max_arr = self._reduce_density_extrema(sim.en)`` before building the render volume.
        """
        n = int(density_flat.size)
        cur_max_in = density_flat
        cur_min_in = density_flat
        cur_n = n

        cur_max_out = self._max_partial
        cur_min_out = self._min_partial
        next_max_out = self._max_scratch
        next_min_out = self._min_scratch

        while True:
            blocks = max(1, (cur_n + _REDUCE_BLOCK_SIZE - 1) // _REDUCE_BLOCK_SIZE)
            reduce_max_kernel[blocks, _REDUCE_BLOCK_SIZE](cur_max_in, cur_max_out, cur_n)
            reduce_min_kernel[blocks, _REDUCE_BLOCK_SIZE](cur_min_in, cur_min_out, cur_n)

            if blocks == 1:
                return cur_min_out, cur_max_out

            cur_n = blocks
            cur_max_in, cur_max_out, next_max_out = cur_max_out, next_max_out, cur_max_in if cur_max_in is self._max_partial or cur_max_in is self._max_scratch else next_max_out
            cur_min_in, cur_min_out, next_min_out = cur_min_out, next_min_out, cur_min_in if cur_min_in is self._min_partial or cur_min_in is self._min_scratch else next_min_out

            if cur_max_in is density_flat:
                cur_max_out = self._max_scratch
                next_max_out = self._max_partial
            elif cur_max_in is self._max_partial:
                cur_max_out = self._max_scratch
                next_max_out = self._max_partial
            else:
                cur_max_out = self._max_partial
                next_max_out = self._max_scratch

            if cur_min_in is density_flat:
                cur_min_out = self._min_scratch
                next_min_out = self._min_partial
            elif cur_min_in is self._min_partial:
                cur_min_out = self._min_scratch
                next_min_out = self._min_partial
            else:
                cur_min_out = self._min_partial
                next_min_out = self._min_scratch

    def _on_key(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        """
        Handle GLFW key press events.

        Parameters
        ----------
        window
            GLFW window handle.
        key : int
            GLFW key code.
        scancode : int
            Platform specific scan code.
        action : int
            GLFW action code.
        mods : int
            GLFW modifier flags.

        Returns
        -------
        None
            The renderer state is updated in response to the key event.
        """
        _ = window
        _ = scancode
        _ = mods

        if action != glfw.PRESS:
            return

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(self.backend.window, True)
            return

        if key == glfw.KEY_F1:
            self.set_display_mode(DISPLAY_ENERGY_SINGLE)
            return

        if key == glfw.KEY_F2:
            self.set_display_mode(DISPLAY_ENERGY_RUNGE)
            return

        if key == glfw.KEY_O:
            self.request_save_output = True
            return

        if key == glfw.KEY_N:
            self.is_running = not self.is_running
            return

        if key == glfw.KEY_LEFT_BRACKET:
            self.levelset = max(self.levelset - 0.02, 0.0)
            return

        if key == glfw.KEY_RIGHT_BRACKET:
            self.levelset = min(self.levelset + 0.02, 2.0)
            return

        if key == glfw.KEY_K:
            self.transfer_scale = max(self.transfer_scale - 0.2, 0.1)
            return

        if key == glfw.KEY_L:
            self.transfer_scale += 0.2
            return

        if key == glfw.KEY_COMMA:
            self.brightness = max(self.brightness - 0.25, 0.1)
            return

        if key == glfw.KEY_PERIOD:
            self.brightness += 0.25
            return

        if self._sim is None:
            return

        if key == glfw.KEY_R:
            com = self._sim.compute_center_of_mass()
            rms = self._sim.compute_rms_radius()
            print(f"Centre of mass = ({com[0]: .2f}, {com[1]: .2f}, {com[2]: .2f})")
            print(f"RMS radius = {rms: .2f}")
            return
        
        if key == glfw.KEY_P:
            self.request_screenshot = True
            return

    def _on_cursor_pos(self, window, xpos: float, ypos: float) -> None:
        """
        Handle GLFW cursor motion events.

        Parameters
        ----------
        window
            GLFW window handle.
        xpos : float
            Cursor x position.
        ypos : float
            Cursor y position.

        Returns
        -------
        None
            The camera state is updated when dragging is active.

        Examples
        --------
        Use ``glfw.set_cursor_pos_callback(self.backend.window, self._on_cursor_pos)`` to register the cursor handler.
        """
        _ = window

        dx = float(xpos) - float(self.last_cursor[0])
        dy = float(ypos) - float(self.last_cursor[1])
        self.last_cursor = (float(xpos), float(ypos))

        if self.left_dragging:
            self.view_rotation[1] += np.float32(0.35 * dx)
            self.view_rotation[0] += np.float32(0.35 * dy)
            self.view_rotation[0] = np.float32(max(-89.0, min(89.0, float(self.view_rotation[0]))))
            self._sync_camera()
            return

        if self.right_dragging:
            right, up, _forward = self._camera_axes()
            scale = 0.0035 * max(float(np.linalg.norm(self.view_translation)), 1.0)
            delta = (-dx * right + dy * up) * np.float32(scale)
            self.view_translation[:3] += delta.astype(np.float32)
            self._sync_camera()

    def _on_mouse_button(self, window, button: int, action: int, mods: int) -> None:
        """
        Handle GLFW mouse button events.

        Parameters
        ----------
        window
            GLFW window handle.
        button : int
            GLFW mouse button code.
        action : int
            GLFW action code.
        mods : int
            GLFW modifier flags.

        Returns
        -------
        None
            The dragging state is updated.

        Examples
        --------
        Use ``glfw.set_mouse_button_callback(self.backend.window, self._on_mouse_button)`` to register the mouse handler.
        """
        _ = window
        _ = mods

        if action == glfw.PRESS:
            self.last_cursor = glfw.get_cursor_pos(self.backend.window)

        if button == glfw.MOUSE_BUTTON_LEFT:
            self.left_dragging = action == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.right_dragging = action == glfw.PRESS

    def _on_scroll(self, window, xoffset: float, yoffset: float) -> None:
        """
        Handle GLFW scroll-wheel events.

        Parameters
        ----------
        window
            GLFW window handle.
        xoffset : float
            Horizontal scroll amount.
        yoffset : float
            Vertical scroll amount.

        Returns
        -------
        None
            The camera zoom is updated.

        Examples
        --------
        Use ``glfw.set_scroll_callback(self.backend.window, self._on_scroll)`` to register the scroll handler.
        """
        _ = window
        _ = xoffset

        self.zoom = max(0.25, float(self.zoom) - 0.25 * float(yoffset))
        self._sync_camera()

    def set_display_mode(self, mode: int) -> None:
        """
        Set the active display mode.

        Parameters
        ----------
        mode : int
            Display mode identifier.

        Returns
        -------
        None
            The renderer display mode is updated.

        Examples
        --------
        Use ``renderer.set_display_mode(DISPLAY_ENERGY_RUNGE)`` to switch the display mode.
        """
        if mode not in DISPLAY_TITLES:
            return
        self.display_mode = mode
        self._update_window_title()

    def _update_window_title(self) -> None:
        """
        Update the window title to match the active display mode.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The backend window title is updated.

        Examples
        --------
        Use ``self._update_window_title()`` after changing the display mode.
        """
        self.backend.set_window_title(DISPLAY_TITLES.get(self.display_mode, "lightly_bound_skyrme"))

    def close(self) -> None:
        """
        Destroy the OpenGL resources and close the window.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The renderer backend is closed.

        Examples
        --------
        Use ``renderer.close()`` when the viewer loop exits.
        """
        self.backend.close()

    def should_close(self) -> bool:
        """
        Return whether the window should close.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            ``True`` if the window should close.

        Examples
        --------
        Use ``while not renderer.should_close():`` as the main viewer loop condition.
        """
        return self.backend.should_close()

    def begin_frame(self) -> None:
        """
        Poll events at the beginning of a frame.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Frame setup is delegated to the backend.

        Examples
        --------
        Use ``renderer.begin_frame()`` at the start of each frame.
        """
        self.backend.begin_frame()

    def end_frame(self) -> None:
        """
        Swap buffers at the end of a frame.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Frame presentation is delegated to the backend.

        Examples
        --------
        Use ``renderer.end_frame()`` after rendering each frame.
        """
        self.backend.end_frame()

    def set_hud_text(self, *, top: str = "", bottom: str = "") -> None:
        """
        Store HUD text in the backend.

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
        Use ``renderer.set_hud_text(top="...", bottom="...")`` to update the stored HUD strings.
        """
        self.backend.set_hud_text(top=top, bottom=bottom)

    def save_screenshot(self, directory: str = "screenshots") -> str:
        """
        Save the current framebuffer to a PNG image.

        Parameters
        ----------
        directory : str, optional
            Output directory for screenshots.

        Returns
        -------
        str
            Path to the written screenshot file.

        Examples
        --------
        Use ``path = renderer.save_screenshot()`` after drawing a frame.
        """
        width, height = glfw.get_framebuffer_size(self.backend.window)

        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        glReadBuffer(GL_FRONT)
        pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)

        image = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 3))
        image = np.flipud(image)

        os.makedirs(directory, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(directory, f"skyrmion_solver_{stamp}.png")
        Image.fromarray(image).save(path)
        return path

    def render(self, *, density_flat, field_flat) -> None:
        """
        Render one frame to the OpenGL window without any host round-trip.

        Parameters
        ----------
        density_flat : device array
            Device scalar volume used for opacity.
        field_flat : device array
            Device Skyrme field used for Runge coloring.

        Returns
        -------
        None
            The current RGBA volume is built in CUDA into the mapped OpenGL upload buffer and then ray-marched by OpenGL.

        Examples
        --------
        Use ``renderer.render(density_flat=sim.en, field_flat=sim.Field)`` to draw a frame.
        """
        if self._sim is None:
            return

        self._sync_render_params()

        xlen = int(self._sim.p_i_h[0])
        ylen = int(self._sim.p_i_h[1])
        zlen = int(self._sim.p_i_h[2])

        density_min_arr, density_max_arr = self._reduce_density_extrema(density_flat)

        mapped = self.backend.map_volume_pbo()
        try:
            volume_pbo_view = cuda_array_from_ptr(mapped.ptr, (zlen, ylen, xlen, 4), np.float32)

            block3d = (8, 8, 4)
            grid3d = ((xlen + block3d[0] - 1) // block3d[0], (ylen + block3d[1] - 1) // block3d[1], (zlen + block3d[2] - 1) // block3d[2])

            build_rgba_volume_kernel[grid3d, block3d](volume_pbo_view, density_flat, field_flat, density_min_arr, density_max_arr, xlen, ylen, zlen)
            cuda.synchronize()
        finally:
            self.backend.unmap_volume_pbo()

        self.backend.upload_and_draw()


def advance_solver(sim, steps_per_frame: int, state: dict) -> None:
    """
    Advance the simulation by a fixed number of solver steps.

    Parameters
    ----------
    sim
        Simulation instance.
    steps_per_frame : int
        Number of solver steps per rendered frame.
    state : dict
        Mutable state dictionary storing energy, error, and epoch count.

    Returns
    -------
    None
        The state dictionary is updated in place.

    Examples
    --------
    Use ``advance_solver(sim, steps_per_frame, state)`` once per rendered frame.
    """
    if state.get("energy") is None:
        obs = sim.observables()
        state["energy"] = float(obs["energy"])

    if sim.rp.newtonflow:
        for _ in range(int(steps_per_frame)):
            state["energy"], state["error"] = sim.step(state["energy"])
            state["epochs"] += 1


def _compute_density_for_mode(sim, mode: int) -> None:
    """
    Compute the scalar density required by the current display mode.

    Parameters
    ----------
    sim
        Simulation instance.
    mode : int
        Active display mode.

    Returns
    -------
    None
        The shared density buffer is updated in place when needed.

    Examples
    --------
    Use ``_compute_density_for_mode(sim, renderer.display_mode)`` before rendering a frame.
    """
    _ = mode
    if hasattr(sim, "compute_energy_density"):
        sim.compute_energy_density()


def run_viewer(sim, params, *, steps_per_frame: int = 5) -> None:
    """
    Run the interactive OpenGL ray-marching viewer loop.

    Parameters
    ----------
    sim
        Simulation instance.
    params
        Parameter object providing the lattice dimensions.
    steps_per_frame : int, optional
        Number of solver steps taken per rendered frame.

    Returns
    -------
    None
        The interactive viewer runs until the window is closed.

    Examples
    --------
    Use ``run_viewer(sim, sim.rp, steps_per_frame=5)`` to start the interactive viewer.
    """
    renderer = GLRenderer(1200, 900, params.zlen)
    renderer.bind_sim(sim)

    state = {"energy": None, "error": None, "epochs": 0, "baryon": None}

    try:
        while not renderer.should_close():
            renderer.begin_frame()

            if renderer.request_save_output:
                renderer.request_save_output = False
                if hasattr(sim, "save_output"):
                    sim.save_output(precision=32)

            if renderer.is_running:
                advance_solver(sim, steps_per_frame, state)

            try:
                obs = sim.observables()
                state["energy"] = float(obs["energy"])
                if "skyrmion_number" in obs:
                    state["baryon"] = float(obs["skyrmion_number"])
                else:
                    state["baryon"] = float(compute_skyrmion_number(sim.Field, sim.d1fd1x, sim.en, sim.entmp, sim.gridsum_partial, sim.p_i_d, sim.p_f_d, sim.p_i_h))
            except Exception:
                pass

            _compute_density_for_mode(sim, renderer.display_mode)

            dt = float(sim.p_f_h[7])
            epochs = int(state.get("epochs", 0))
            t = epochs * dt
            err = state.get("error", None)

            top_text = f"t={t:.3f} ({epochs} epochs)"
            if err is not None:
                top_text += f", err={float(err):.3e}"

            bottom_parts = []
            if state.get("energy") is not None:
                bottom_parts.append(f"E={float(state['energy']):.6f}")
            if state.get("baryon") is not None:
                bottom_parts.append(f"B={float(state['baryon']):.6f}")
            bottom_parts.append("running" if renderer.is_running else "paused")
            renderer.set_hud_text(top=top_text, bottom=", ".join(bottom_parts))

            renderer.render(density_flat=sim.en, field_flat=sim.Field)
            if renderer.request_screenshot:
                path = renderer.save_screenshot()
                print(f"Saved screenshot to {path}")
                renderer.request_screenshot = False
            renderer.end_frame()

    finally:
        renderer.close()