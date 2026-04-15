"""Weather watch task - Long-running task that polls weather and reports changes."""

from typing import Any, Protocol, TypedDict

from langgraph.graph import StateGraph
from langgraph.types import Command, interrupt

from switchplane import Field, Task, command
from switchplane.agent_runtime import AgentContext

# -- Weather provider protocol --


class WeatherProvider(Protocol):
    async def get_weather(self, latitude: float, longitude: float) -> dict[str, Any]: ...


# -- OpenMeteo implementation --


class OpenMeteoProvider:
    def __init__(self, base_url: str = "https://api.open-meteo.com/v1"):
        self.base_url = base_url

    async def get_weather(self, latitude: float, longitude: float) -> dict[str, Any]:
        """Fetch current weather from Open-Meteo API."""
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/forecast",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                    "timezone": "auto",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            current = data.get("current", {})
            return {
                "temperature": current.get("temperature_2m"),
                "humidity": current.get("relative_humidity_2m"),
                "wind_speed": current.get("wind_speed_10m"),
                "weather_code": current.get("weather_code"),
                "time": current.get("time"),
            }


# -- Weather code descriptions --


def describe_weather_code(code: int | None) -> str:
    codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    return codes.get(code, f"Unknown ({code})")


# -- Change detection --


def detect_changes(prev: dict, current: dict) -> list[str]:
    changes = []
    if prev.get("weather_code") != current.get("weather_code"):
        changes.append(
            f"Conditions changed: {describe_weather_code(prev.get('weather_code'))} → {describe_weather_code(current.get('weather_code'))}"
        )
    temp_prev = prev.get("temperature")
    temp_curr = current.get("temperature")
    if temp_prev is not None and temp_curr is not None and abs(temp_prev - temp_curr) >= 0.5:
        changes.append(f"Temperature: {temp_prev}°C → {temp_curr}°C")
    wind_prev = prev.get("wind_speed")
    wind_curr = current.get("wind_speed")
    if wind_prev is not None and wind_curr is not None and abs(wind_prev - wind_curr) >= 2.0:
        changes.append(f"Wind speed: {wind_prev} → {wind_curr} km/h")
    return changes


# -- Graph state --


class WatchState(TypedDict):
    latitude: float
    longitude: float
    provider: str
    base_url: str
    poll_interval: int
    current_weather: dict | None
    previous_weather: dict | None
    changes: list[str]
    iteration: int


# -- Graph nodes --


async def fetch_weather(state: WatchState) -> WatchState:
    """Fetch current weather data."""
    provider = OpenMeteoProvider(base_url=state.get("base_url", "https://api.open-meteo.com/v1"))
    weather = await provider.get_weather(state["latitude"], state["longitude"])
    return {
        **state,
        "previous_weather": state.get("current_weather"),
        "current_weather": weather,
        "iteration": state.get("iteration", 0) + 1,
    }


def report_changes(state: WatchState) -> WatchState:
    """Detect and report weather changes."""
    prev = state.get("previous_weather")
    current = state.get("current_weather")
    if prev and current:
        changes = detect_changes(prev, current)
    else:
        changes = []
    return {**state, "changes": changes}


def wait_for_next(state: WatchState) -> WatchState:
    """Pause the graph until the next poll interval.

    Resume data carries updated coordinates (may have changed via the
    ``coordinates`` command while waiting).
    """
    data = interrupt("Waiting for next weather check")
    return {
        **state,
        "latitude": data.get("latitude", state["latitude"]),
        "longitude": data.get("longitude", state["longitude"]),
    }


# -- Build the graph --


def build_graph() -> StateGraph:
    graph = StateGraph(WatchState)
    graph.add_node("fetch_weather", fetch_weather)
    graph.add_node("report_changes", report_changes)
    graph.add_node("wait_for_next", wait_for_next)
    graph.set_entry_point("fetch_weather")
    graph.add_edge("fetch_weather", "report_changes")
    graph.add_edge("report_changes", "wait_for_next")
    graph.add_edge("wait_for_next", "fetch_weather")
    return graph


def _report_progress(ctx: AgentContext, state: WatchState) -> None:
    """Emit progress based on the current iteration's results."""
    weather = state["current_weather"]
    changes = state["changes"]
    iteration = state["iteration"]

    if iteration == 1:
        ctx.progress(
            f"Current weather: {describe_weather_code(weather.get('weather_code'))}, "
            f"{weather.get('temperature')}°C, "
            f"humidity {weather.get('humidity')}%, "
            f"wind {weather.get('wind_speed')} km/h"
        )
    elif changes:
        for change in changes:
            ctx.progress(f"Weather change detected: {change}")
    else:
        ctx.progress(f"No changes (check #{iteration})")


# -- Task implementation --


class WatchTask(Task):
    name = "watch"
    description = "Monitor weather changes at a location"
    mode = "long_running"

    latitude: float = Field(default=49.2827, description="Latitude to monitor")
    longitude: float = Field(default=-123.1207, description="Longitude to monitor")
    interval: int = Field(default=60, description="Poll interval in seconds")

    @command
    def coordinates(self, ctx: AgentContext, lat: float | None = None, lon: float | None = None):
        if lat is not None:
            self.latitude = lat
        if lon is not None:
            self.longitude = lon
        ctx.progress(f"Coordinates updated to ({self.latitude}, {self.longitude})")
        return {"latitude": self.latitude, "longitude": self.longitude}

    async def run(self, ctx: AgentContext) -> None:
        """Long-running weather watch task."""
        poll_interval = int(ctx.config.get("poll_interval", self.interval))
        base_url = ctx.config.get("open_meteo", {}).get("base_url", "https://api.open-meteo.com/v1")

        graph = build_graph().compile(checkpointer=ctx.checkpointer)
        config = {"configurable": {"thread_id": ctx.task_id}}

        initial_state: WatchState = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "provider": "open_meteo",
            "base_url": base_url,
            "poll_interval": poll_interval,
            "current_weather": None,
            "previous_weather": None,
            "changes": [],
            "iteration": 0,
        }

        # On resume, the checkpoint is at the wait_for_next interrupt.
        # Skip straight to the poll loop so the next interval resumes the graph.
        resumed = False
        if ctx.checkpointer:
            checkpoint = await graph.aget_state(config)
            if checkpoint and checkpoint.values:
                result = checkpoint.values
                resumed = True

        if resumed:
            ctx.progress(
                f"Resumed weather watch at check #{result['iteration']} "
                f"for ({result['latitude']}, {result['longitude']}), polling every {poll_interval}s"
            )
        else:
            ctx.progress(
                f"Starting weather watch for ({self.latitude}, {self.longitude}), polling every {poll_interval}s"
            )
            result = await graph.ainvoke(initial_state, config)
            _report_progress(ctx, result)

        def poll_check():
            return {"latitude": self.latitude, "longitude": self.longitude}

        while True:
            data = await ctx.poll_until(poll_check, interval=poll_interval, task=self)
            if data is None:
                break

            result = await graph.ainvoke(Command(resume=data), config)
            _report_progress(ctx, result)

        ctx.complete({"final_weather": result.get("current_weather"), "total_checks": result["iteration"]})
