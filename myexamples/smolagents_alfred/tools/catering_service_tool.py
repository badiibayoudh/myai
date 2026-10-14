from smolagents import Tool
from typing import Any, Optional

class SimpleTool(Tool):
    name = "catering_service_tool"
    description = "This tool returns the highest-rated catering service in Gotham City."
    inputs = {"query":{"type":"string","description":"A search term for finding catering services."}}
    output_type = "string"

    def forward(self, query: str) -> str:
        """
        This tool returns the highest-rated catering service in Gotham City.

        Args:
            query: A search term for finding catering services.
        """
        # Example list of catering services and their ratings
        services = {
            "Gotham Catering Co.": 4.9,
            "Wayne Manor Catering": 4.8,
            "Gotham City Events": 4.7,
        }

        # Find the highest rated catering service (simulating search query filtering)
        best_service = max(services, key=services.get)

        return best_service