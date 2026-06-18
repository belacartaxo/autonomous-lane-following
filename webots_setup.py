r"""
Portable Webots setup helper.

This module avoids user-specific hardcoded paths such as:
    C:\Users\<USERNAME>\AppData\Local\Programs\Webots

It first checks WEBOTS_HOME. If WEBOTS_HOME is not defined, it tries common
installation paths for Windows 10 and Ubuntu Linux 20.04.6 LTS.

Use this in scripts that import the Webots controller API:

    from webots_setup import setup_webots_path
    setup_webots_path()
"""

import os
import sys
import platform


def find_webots_home():
    """
    Locate the Webots installation folder.

    Priority:
        1. WEBOTS_HOME environment variable
        2. Common Windows installation paths
        3. Common Linux installation paths

    Returns:
        str: Path to the Webots installation folder.

    Raises:
        FileNotFoundError: If Webots cannot be located.
    """

    env_path = os.environ.get("WEBOTS_HOME")

    if env_path and os.path.exists(env_path):
        return env_path

    system = platform.system()

    if system == "Windows":
        candidates = [
            r"C:\Program Files\Webots",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Webots"),
        ]

    elif system == "Linux":
        candidates = [
            "/usr/local/webots",
            "/snap/webots/current/usr/share/webots",
        ]

    else:
        # The project is only required to support Windows 10 and Ubuntu 20.04.6.
        # Unsupported systems are not targeted by the assignment.
        candidates = []

    for path in candidates:
        if path and os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Webots installation was not found.\n"
        "Please install Webots and/or set the WEBOTS_HOME environment variable.\n\n"
        "Examples:\n"
        "  Windows CMD:\n"
        "    set WEBOTS_HOME=C:\\Program Files\\Webots\n\n"
        "  Windows PowerShell:\n"
        "    $env:WEBOTS_HOME='C:\\Program Files\\Webots'\n\n"
        "  Linux terminal:\n"
        "    export WEBOTS_HOME=/usr/local/webots\n"
    )


def setup_webots_path():
    """
    Configure Python and system library paths so that Webots controller modules
    can be imported from external Python scripts.

    This function is safe to call multiple times.
    """

    webots_home = find_webots_home()

    os.environ["WEBOTS_HOME"] = webots_home

    webots_python_path = os.path.join(
        webots_home,
        "lib",
        "controller",
        "python",
    )

    webots_controller_path = os.path.join(
        webots_home,
        "lib",
        "controller",
    )

    if webots_python_path not in sys.path:
        sys.path.insert(0, webots_python_path)

    system = platform.system()

    if system == "Windows":
        current_path = os.environ.get("PATH", "")

        if webots_controller_path not in current_path:
            os.environ["PATH"] = (
                webots_controller_path
                + os.pathsep
                + current_path
            )

    elif system == "Linux":
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")

        if webots_controller_path not in current_ld_path:
            os.environ["LD_LIBRARY_PATH"] = (
                webots_controller_path
                + os.pathsep
                + current_ld_path
            )

    return webots_home


if __name__ == "__main__":
    detected_path = setup_webots_path()
    print(f"Webots found at: {detected_path}")

    try:
        from controller import Supervisor
        print("Webots controller OK")
    except Exception as exc:
        print("Webots path was detected, but the controller module could not be imported.")
        print(f"Error: {exc}")
        raise