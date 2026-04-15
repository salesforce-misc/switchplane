from switchplane import Application

app = Application(name="hello")
app.discover_agents("hello.agents")


def main():
    app.run()
