"""
DesktopSandbox - OpenSandbox wrapper for RL training with desktop environments.

Based on official OpenSandbox examples:
- examples/desktop/main.py
- examples/playwright/main.py

Provides:
- Screenshot capture (observations)
- Mouse/keyboard actions
- File operations
- Parallel sandbox creation for distributed training
"""

import asyncio
from datetime import timedelta
from typing import Optional, Tuple, List
from pathlib import Path

from opensandbox import Sandbox
from opensandbox.config import ConnectionConfig
from opensandbox.models.execd import RunCommandOpts

from .config import SandboxConfig


async def _print_logs(label: str, execution) -> None:
    """Print execution logs (following official example pattern)."""
    for msg in execution.logs.stdout:
        print(f"[{label} stdout] {msg.text}")
    for msg in execution.logs.stderr:
        print(f"[{label} stderr] {msg.text}")
    if execution.error:
        print(f"[{label} error] {execution.error.name}: {execution.error.value}")


class DesktopSandbox:
    """A desktop sandbox environment for RL training.

    Usage:
        # Method 1: Manual management
        sandbox = DesktopSandbox()
        await sandbox.create()
        try:
            obs = await sandbox.screenshot()
        finally:
            await sandbox.close()

        # Method 2: Context manager (recommended)
        async with DesktopSandbox() as sandbox:
            obs = await sandbox.screenshot()
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self._sandbox: Optional[Sandbox] = None
        self._connection_config = ConnectionConfig(
            domain=self.config.domain,
            request_timeout=timedelta(seconds=self.config.request_timeout_seconds),
        )
        self._tools_installed = False

    async def __aenter__(self) -> "DesktopSandbox":
        """Async context manager entry."""
        await self.create()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    @property
    def sandbox_id(self) -> Optional[str]:
        """Get the sandbox ID."""
        return self._sandbox.id if self._sandbox else None

    @property
    def is_connected(self) -> bool:
        """Check if connected to a sandbox."""
        return self._sandbox is not None

    async def create(self) -> str:
        """Create a new sandbox with desktop environment.

        Returns:
            Sandbox ID
        """
        self._sandbox = await Sandbox.create(
            self.config.image,
            connection_config=self._connection_config,
            env={
                "VNC_PASSWORD": self.config.vnc_password,
                "DISPLAY": ":0",
            },
            resource={
                "cpu": self.config.cpu,
                "memory": self.config.memory,
            },
        )

        # Start desktop environment
        await self._start_desktop()

        # Install required tools
        await self._install_tools()

        return self._sandbox.id

    async def connect(self, sandbox_id: str) -> None:
        """Connect to an existing sandbox.

        Args:
            sandbox_id: The sandbox ID to connect to
        """
        self._sandbox = await Sandbox.connect(
            sandbox_id, connection_config=self._connection_config
        )

    async def close(self) -> None:
        """Close and cleanup the sandbox."""
        if self._sandbox:
            await self._sandbox.kill()
            self._sandbox = None
            self._tools_installed = False

    async def _start_desktop(self) -> None:
        """Start Xvfb and XFCE desktop environment."""
        # Start Xvfb
        await self._sandbox.commands.run(
            f"Xvfb :0 -screen 0 {self.config.screen_geometry} &",
            opts=RunCommandOpts(background=True),
        )
        await asyncio.sleep(2)

        # Start XFCE desktop
        await self._sandbox.commands.run(
            "DISPLAY=:0 dbus-launch startxfce4 &",
            opts=RunCommandOpts(background=True),
        )
        await asyncio.sleep(4)

    async def _install_tools(self) -> None:
        """Install screenshot and automation tools."""
        if self._tools_installed:
            return

        # Install mss for screenshots and pyautogui for actions
        await self._sandbox.commands.run(
            "pip3 install -q mss pyautogui 2>/dev/null",
            opts=RunCommandOpts(timeout=120),
        )
        self._tools_installed = True

    async def screenshot(self) -> bytes:
        """Capture a screenshot of the desktop.

        Returns:
            PNG image data as bytes
        """
        screenshot_code = """
import os
os.environ["DISPLAY"] = ":0"
import mss
with mss.mss() as sct:
    sct.shot(output="/tmp/screen.png")
"""
        await self._sandbox.commands.run(
            f"python3 -c '{screenshot_code}'",
            opts=RunCommandOpts(timeout=10),
        )

        return await self._sandbox.files.read_bytes("/tmp/screen.png")

    async def click(self, x: int, y: int) -> None:
        """Click at the specified coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
        """
        action_code = f"""
import os
os.environ["DISPLAY"] = ":0"
import pyautogui
pyautogui._pyautogui_x11._display = None
pyautogui.click({x}, {y})
"""
        await self._sandbox.commands.run(
            f"python3 -c '{action_code}'",
            opts=RunCommandOpts(timeout=10),
        )

    async def double_click(self, x: int, y: int) -> None:
        """Double-click at the specified coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
        """
        action_code = f"""
import os
os.environ["DISPLAY"] = ":0"
import pyautogui
pyautogui._pyautogui_x11._display = None
pyautogui.doubleClick({x}, {y})
"""
        await self._sandbox.commands.run(
            f"python3 -c '{action_code}'",
            opts=RunCommandOpts(timeout=10),
        )

    async def right_click(self, x: int, y: int) -> None:
        """Right-click at the specified coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
        """
        action_code = f"""
import os
os.environ["DISPLAY"] = ":0"
import pyautogui
pyautogui._pyautogui_x11._display = None
pyautogui.rightClick({x}, {y})
"""
        await self._sandbox.commands.run(
            f"python3 -c '{action_code}'",
            opts=RunCommandOpts(timeout=10),
        )

    async def type_text(self, text: str, interval: float = 0.05) -> None:
        """Type text using the keyboard.

        Args:
            text: Text to type
            interval: Interval between keystrokes in seconds
        """
        # Escape special characters for Python string
        escaped_text = text.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')

        action_code = f"""
import os
os.environ["DISPLAY"] = ":0"
import pyautogui
pyautogui._pyautogui_x11._display = None
pyautogui.typewrite('{escaped_text}', interval={interval})
"""
        await self._sandbox.commands.run(
            f"python3 -c '{action_code}'",
            opts=RunCommandOpts(timeout=60),
        )

    async def press_key(self, key: str) -> None:
        """Press a single key.

        Args:
            key: Key name (e.g., 'enter', 'tab', 'escape', 'f1')
        """
        action_code = f"""
import os
os.environ["DISPLAY"] = ":0"
import pyautogui
pyautogui._pyautogui_x11._display = None
pyautogui.press('{key}')
"""
        await self._sandbox.commands.run(
            f"python3 -c '{action_code}'",
            opts=RunCommandOpts(timeout=10),
        )

    async def hotkey(self, *keys: str) -> None:
        """Press a keyboard shortcut.

        Args:
            keys: Keys to press together (e.g., 'ctrl', 'c')
        """
        keys_str = ", ".join(f"'{k}'" for k in keys)
        action_code = f"""
import os
os.environ["DISPLAY"] = ":0"
import pyautogui
pyautogui._pyautogui_x11._display = None
pyautogui.hotkey({keys_str})
"""
        await self._sandbox.commands.run(
            f"python3 -c '{action_code}'",
            opts=RunCommandOpts(timeout=10),
        )

    async def move_mouse(self, x: int, y: int) -> None:
        """Move mouse to the specified coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
        """
        action_code = f"""
import os
os.environ["DISPLAY"] = ":0"
import pyautogui
pyautogui._pyautogui_x11._display = None
pyautogui.moveTo({x}, {y})
"""
        await self._sandbox.commands.run(
            f"python3 -c '{action_code}'",
            opts=RunCommandOpts(timeout=10),
        )

    async def drag(self, start_x: int, start_y: int, end_x: int, end_y: int) -> None:
        """Drag from one position to another.

        Args:
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            end_x: Ending X coordinate
            end_y: Ending Y coordinate
        """
        action_code = f"""
import os
os.environ["DISPLAY"] = ":0"
import pyautogui
pyautogui._pyautogui_x11._display = None
pyautogui.moveTo({start_x}, {start_y})
pyautogui.drag({end_x - start_x}, {end_y - start_y})
"""
        await self._sandbox.commands.run(
            f"python3 -c '{action_code}'",
            opts=RunCommandOpts(timeout=10),
        )

    async def scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """Scroll the mouse wheel.

        Args:
            clicks: Number of scroll clicks (positive=up, negative=down)
            x: Optional X coordinate to scroll at
            y: Optional Y coordinate to scroll at
        """
        if x is not None and y is not None:
            action_code = f"""
import os
os.environ["DISPLAY"] = ":0"
import pyautogui
pyautogui._pyautogui_x11._display = None
pyautogui.scroll({clicks}, x={x}, y={y})
"""
        else:
            action_code = f"""
import os
os.environ["DISPLAY"] = ":0"
import pyautogui
pyautogui._pyautogui_x11._display = None
pyautogui.scroll({clicks})
"""
        await self._sandbox.commands.run(
            f"python3 -c '{action_code}'",
            opts=RunCommandOpts(timeout=10),
        )

    async def run_command(self, command: str, timeout: int = 60) -> Tuple[int, str]:
        """Run a shell command in the sandbox.

        Args:
            command: Command to run
            timeout: Timeout in seconds

        Returns:
            Tuple of (exit_code, output_text)
        """
        result = await self._sandbox.commands.run(
            command, opts=RunCommandOpts(timeout=timeout)
        )
        return result.exit_code, result.text

    async def read_file(self, path: str) -> bytes:
        """Read a file from the sandbox.

        Args:
            path: Path to the file

        Returns:
            File contents as bytes
        """
        return await self._sandbox.files.read_bytes(path)

    async def write_file(self, path: str, content: bytes) -> None:
        """Write a file to the sandbox.

        Args:
            path: Path to the file
            content: File contents as bytes
        """
        await self._sandbox.files.write_file(path, content)

    async def get_screen_size(self) -> Tuple[int, int]:
        """Get the screen dimensions.

        Returns:
            Tuple of (width, height)
        """
        return self.config.screen_width, self.config.screen_height

    async def start_vnc(self, verbose: bool = False) -> str:
        """Start VNC server for human debugging.

        Based on official desktop example.

        Args:
            verbose: If True, print logs

        Returns:
            noVNC URL for browser access
        """
        # Start x11vnc
        vnc_exec = await self._sandbox.commands.run(
            f'x11vnc -display :0 -passwd "{self.config.vnc_password}" '
            "-forever -shared -rfbport 5900",
            opts=RunCommandOpts(background=True),
        )
        if verbose:
            await _print_logs("x11vnc", vnc_exec)

        # Start noVNC/websockify
        novnc_exec = await self._sandbox.commands.run(
            "/usr/bin/websockify --web=/usr/share/novnc 6080 localhost:5900",
            opts=RunCommandOpts(background=True),
        )
        if verbose:
            await _print_logs("novnc", novnc_exec)

        # Get endpoint
        endpoint = await self._sandbox.get_endpoint(6080)

        # Build noVNC URL
        host_port, path = endpoint.endpoint.split("/", 1)
        host, port = host_port.split(":")
        novnc_url = (
            f"http://{endpoint.endpoint}/vnc.html"
            f"?host={host}&port={port}&path={path}"
        )

        return novnc_url

    async def get_vnc_endpoint(self) -> Tuple[str, str]:
        """Get VNC connection info.

        Returns:
            Tuple of (endpoint, password)
        """
        endpoint = await self._sandbox.get_endpoint(5900)
        return endpoint.endpoint, self.config.vnc_password


async def create_parallel_sandboxes(
    count: int,
    config: Optional[SandboxConfig] = None,
) -> List[DesktopSandbox]:
    """Create multiple sandboxes in parallel.

    Args:
        count: Number of sandboxes to create
        config: Optional configuration (same for all sandboxes)

    Returns:
        List of DesktopSandbox instances
    """
    sandboxes = [DesktopSandbox(config) for _ in range(count)]

    # Create all sandboxes concurrently
    await asyncio.gather(*[sb.create() for sb in sandboxes])

    return sandboxes
