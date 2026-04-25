"""
Instructions for the omega-Skyrme interactive controls.

Examples
--------
Use ``print_instructions()`` to display the interactive control summary.
"""

def print_instructions():
    """
    Print the interactive control summary for the omega-Skyrme simulation.

    Returns
    -------
    None
        The control summary is printed to the terminal.

    Examples
    --------
    Use ``print_instructions()`` to display the available keyboard and mouse controls.
    """
    print(
        "\nControls:\n"
        "   Energy density display:                 F1\n"
        "   Energy density (Runge) display:         F2\n"
        "   Omega-meson density display:            F3\n"
        "   Toggle arrested Newton flow:            n\n"
        "   Zoom in/out:                            Scroll-wheel\n"
        "   Rotate camera:                          Left-click drag\n"
        "   Increase/decrease level set:            ]/[\n"
        "   Increase/decrease brightness:           ./,\n"
        "   Increase/decrease transfer scale:       l/k\n"
        "   Print RMS radius:                       r\n"
        "   Save screenshot:                        p\n"
        "   Exit simulation:                        Esc\n"
    )