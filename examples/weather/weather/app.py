from switchplane import Application

app = Application(name="weather")
app.discover_agents("weather.agents")


def main():
    app.run()
