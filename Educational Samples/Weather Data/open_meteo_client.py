"""
Open-Meteo API client for Python 'requests', driven by open_meteo.toml.

Design goals:
- Simple: it's just query parameters over HTTP.
- Config-driven: endpoints, defaults, and variable sets come from TOML.
- Predictable merging of parameters:
    defaults -> endpoint preset -> profile coords -> per-call overrides
- Basic retry handling for transient failures (429/5xx), controlled by TOML.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Mapping
import time

import requests

@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int
    backoff_seconds: float
    retry_on_http_status: tuple[int, ...]


@dataclass(frozen=True)
class Timeouts:
    connect: float
    read: float


class OpenMeteoClient:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self._cfg = config
        self._om = config["open_meteo"]

        # Session keeps TCP connections alive (faster, fewer handshakes).
        self._session = requests.Session()

        # Basic headers; Open-Meteo doesn't need auth for the free endpoints.
        self._session.headers.update(
            {"User-Agent": self._om.get("user_agent", "OpenMeteoClient/1.0")}
        )

        self._timeouts = Timeouts(
            connect=float(self._om.get("connect_timeout_seconds", 5)),
            read=float(self._om.get("timeout_seconds", 20)),
        )

        self._retry = RetryPolicy(
            max_retries=int(self._om.get("max_retries", 0)),
            backoff_seconds=float(self._om.get("retry_backoff_seconds", 0.0)),
            retry_on_http_status=tuple(self._om.get("retry_on_http_status", [])),
        )

    # ----------------------------
    # Public convenience methods
    # ----------------------------
    def forecast(
        self,
        *,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        profile: Optional[str] = None,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Call the Forecast API."""
        url = self._om["endpoints"]["forecast_url"]
        params = self._build_params("forecast", latitude, longitude, profile, overrides)
        return self._get_json(url, params)

    def air_quality(
        self,
        *,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        profile: Optional[str] = None,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Call the Air Quality API."""
        url = self._om["endpoints"]["air_quality_url"]
        params = self._build_params("air_quality", latitude, longitude, profile, overrides)
        return self._get_json(url, params)

    def geocode(
        self,
        *,
        name: str,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Call the Geocoding API (no lat/lon)."""
        url = self._om["endpoints"]["geocoding_url"]
        params: Dict[str, Any] = {}
        params.update(self._dict_or_empty(self._om.get("geocoding")))
        params.update({"name": name})
        params.update(overrides)
        return self._get_json(url, params)

    def close(self) -> None:
        """Close the underlying requests.Session."""
        self._session.close()

    # ----------------------------
    # Internals
    # ----------------------------
    def _build_params(
        self,
        endpoint_key: str,
        latitude: Optional[float],
        longitude: Optional[float],
        profile: Optional[str],
        overrides: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge params with precedence:
          defaults -> endpoint preset -> profile coords -> explicit lat/lon -> overrides
        Also converts list-valued variables to comma-separated strings, as Open-Meteo expects.
        """
        params: Dict[str, Any] = {}

        # 1) Global defaults
        params.update(self._dict_or_empty(self._om.get("defaults")))

        # 2) Endpoint preset (forecast / air_quality), e.g. hourly=[...]
        params.update(self._dict_or_empty(self._om.get(endpoint_key)))

        # 3) Profile coords (optional)
        if profile:
            prof = self._cfg.get("profiles", {}).get(profile)
            if not prof:
                raise KeyError(f"Unknown profile '{profile}' in TOML (section [profiles.{profile}]).")
            params["latitude"] = prof["latitude"]
            params["longitude"] = prof["longitude"]

        # 4) Explicit coords override profile if provided
        if latitude is not None:
            params["latitude"] = latitude
        if longitude is not None:
            params["longitude"] = longitude

        # 5) Per-call overrides
        params.update(overrides)

        # Convert list variables -> comma-separated strings (Open-Meteo format).
        # Example: hourly=["temperature_2m","precipitation"] -> "temperature_2m,precipitation"
        for key, value in list(params.items()):
            if isinstance(value, (list, tuple)):
                params[key] = ",".join(str(v) for v in value)

        # Clean up: drop None values
        params = {k: v for k, v in params.items() if v is not None}

        return params

    def _get_json(self, url: str, params: Mapping[str, Any]) -> Dict[str, Any]:
        """
        GET with basic retries for transient server/ratelimit responses.
        Raises requests.HTTPError with context on final failure.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(self._retry.max_retries + 1):
            try:
                resp = self._session.get(
                    url,
                    params=params,
                    timeout=(self._timeouts.connect, self._timeouts.read),
                )

                # Retry on configured status codes
                if resp.status_code in self._retry.retry_on_http_status and attempt < self._retry.max_retries:
                    self._sleep_backoff(attempt)
                    continue

                # Raise for non-2xx
                resp.raise_for_status()

                return resp.json()

            except (requests.Timeout, requests.ConnectionError) as e:
                last_exc = e
                if attempt < self._retry.max_retries:
                    self._sleep_backoff(attempt)
                    continue
                raise

            except requests.HTTPError as e:
                # If it's an HTTP error and not eligible for retry (or out of retries), re-raise.
                last_exc = e
                raise

        # Should never get here, but just in case:
        if last_exc:
            raise last_exc
        raise RuntimeError("Request failed without an exception (unexpected).")

    def _sleep_backoff(self, attempt: int) -> None:
        # Exponential-ish backoff: base * (attempt+1)
        time.sleep(self._retry.backoff_seconds * (attempt + 1))

    @staticmethod
    def _dict_or_empty(value: Any) -> Dict[str, Any]:
        return dict(value) if isinstance(value, Mapping) else {}


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    import tomllib

    with open("open_meteo.toml", "rb") as f:
        cfg = tomllib.load(f)

    client = OpenMeteoClient(cfg)

    # 1) Forecast for Copenhagen profile, override forecast_days
    data = client.forecast(profile="copenhagen", forecast_days=3)
    print("Forecast keys:", list(data.keys()))
    print("Forecast:", data.values())

    # 2) Air quality for explicit lat/lon
    aq = client.air_quality(latitude=55.6761, longitude=12.5683)
    print("Air quality keys:", list(aq.keys()))

    # 3) Geocode by name
    geo = client.geocode(name="Copenhagen", count=5, language="en")
    print("Geocode results:", len(geo.get("results", []) or []))

    client.close()
