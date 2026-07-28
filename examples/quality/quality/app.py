from pathlib import Path

from quality.config import QualityConfig
from switchplane import Application

app = Application(
    name="quality",
    default_config=Path(__file__).parent / "config.toml",
    config_class=QualityConfig,
)
app.discover_agents("quality.agents")


def main():
    app.run()
