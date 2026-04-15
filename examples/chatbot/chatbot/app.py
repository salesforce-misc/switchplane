from pathlib import Path

from switchplane import Application

app = Application(name="chatbot", default_config=Path(__file__).parent / "config.toml")
app.discover_agents("chatbot.agents")


def main():
    app.run()
