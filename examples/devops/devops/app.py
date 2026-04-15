from pathlib import Path

from switchplane import Application

app = Application(name="devops", default_config=Path(__file__).parent / "config.toml")
app.discover_agents("devops.agents")


def main():
    app.run()
