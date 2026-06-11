from dataclasses import dataclass
from typing import Optional


@dataclass
class SandboxConfig:
    """Configuration for OpenSandbox desktop environment."""

    # OpenSandbox server settings
    server_host: str = "localhost"
    server_port: int = 8080
    request_timeout_seconds: int = 300

    # Container resource settings
    cpu: str = "4"
    memory: str = "8Gi"

    # Desktop settings
    screen_width: int = 1280
    screen_height: int = 800
    screen_depth: int = 24

    # VNC settings (for human debugging)
    vnc_password: str = "opensandbox"

    # Image settings
    image: str = "pythonsonggeek/proactivegym-desktop:latest"

    @property
    def domain(self) -> str:
        return f"{self.server_host}:{self.server_port}"

    @property
    def screen_geometry(self) -> str:
        return f"{self.screen_width}x{self.screen_height}x{self.screen_depth}"
