"""Tests for the four deterministic lookup tools (STEP 3 spec)."""

from agri_agent.agent.tools import make_tools


class TestLookupTools:
    def _tools(self, bundle):
        by_name = {t.name: t for t in make_tools(bundle)}
        assert set(by_name) == {
            "get_vegetation_indices",
            "get_weather_forecast",
            "get_soil_moisture_forecast",
            "get_agronomic_thresholds",
        }
        return by_name

    def test_vegetation_tool_returns_bundle_field_unchanged(self, bundle):
        assert self._tools(bundle)["get_vegetation_indices"].invoke({}) is bundle.vegetation

    def test_weather_tool_returns_bundle_field_unchanged(self, bundle):
        assert self._tools(bundle)["get_weather_forecast"].invoke({}) is bundle.weather_forecast

    def test_soil_moisture_tool_returns_bundle_field_unchanged(self, bundle):
        assert self._tools(bundle)["get_soil_moisture_forecast"].invoke({}) is bundle.soil_moisture_forecast

    def test_thresholds_tool_returns_bundle_field_unchanged(self, bundle):
        assert self._tools(bundle)["get_agronomic_thresholds"].invoke({}) is bundle.thresholds

    def test_tools_accept_extra_arguments_gracefully(self, bundle):
        by_name = self._tools(bundle)
        assert by_name["get_weather_forecast"].invoke({"anything": 1}) is bundle.weather_forecast

    def test_all_tools_are_bound_to_the_same_bundle(self, bundle):
        tools = make_tools(bundle)
        vegetation = tools[0].invoke({})
        assert vegetation is bundle.vegetation
