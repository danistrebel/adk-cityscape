from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams, StdioConnectionParams
from mcp import StdioServerParameters
from google.adk.tools import google_search
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from a2a.client import ClientFactory, ClientConfig
from google.auth.transport.requests import Request
from google.oauth2 import id_token
import httpx

import datetime
from google.genai import types
import os
from urllib.parse import urlparse

DEFAULT_MODEL='gemini-3-flash-preview'
NANO_BANANA_MODEL='gemini-3-pro-image-preview'

get_weather = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="https://mapstools.googleapis.com/mcp",
        headers={"X-Goog-Api-Key": os.environ["MAPS_API_KEY"] }
    ),
)

nano_banana = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="mcp-gemini-go",
            env=dict(os.environ, PROJECT_ID=os.environ["GOOGLE_CLOUD_PROJECT"]),
        ),
        timeout=60,
    ),
)

async def display_image_with_adk(image_path: str, tool_context: ToolContext):
    """Reads an image file from the local disk and displays it in the chat as an artifact."""

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        await tool_context.save_artifact(
            os.path.basename(image_path),
            types.Part.from_bytes(data=image_bytes, mime_type='image/png'),
        )
        return {
            'status': 'success',
            'detail': f'Image "{os.path.basename(image_path)}" displayed successfully.',
        }
    except FileNotFoundError:
        return {"status": "failed", "detail": f"Image file not found at path: {image_path}"}
    except Exception as e:
        return {"status": "failed", "detail": f"An error occurred: {e}"}

city_profile = LlmAgent(
    model=DEFAULT_MODEL,
    name='city_researcher',
    description="Find most iconic city attributes.",
    instruction="Use the Google search tool to figure out the most iconic landmark and immediate geographical attributes (lakes, major rivers, hills etc.) in in a given city and return a ordered list starting with the most important landmarks.",
    tools=[google_search],
    output_key="city_profile"
)

city_current_weather = LlmAgent(
    model=DEFAULT_MODEL,
    name='city_current_weather',
    description="Looks up the current weather to be used in the city image.",
    instruction="Use the available tool to get a summary of current weather conditions in a city to provide the image with up to date information.",
    tools=[get_weather],
    output_key="city_weather"
)

city_info = ParallelAgent(
    name="city_info",
    sub_agents=[city_profile, city_current_weather]
)

city_drawer = LlmAgent(
    model=DEFAULT_MODEL,
    name='city_drawer',
    description="Draws the cityscape picture.",
    instruction=f"""
    Image Context:
    - Current Date: {datetime.date.today().strftime("%A, %B %d, %Y")}
    - Current Weather
    - Most Prominent Landmarks in that City

    Image Model: {NANO_BANANA_MODEL}

    Instructions:
    1. Come up with an absolute file path for the cityscape of the current city 
        and make sure it's added to the current folders 'generated' folder 
        e.g. {os.getcwd()}/generated/zurich/ for a cityscape of Zurich.
    2. Use the `nano_banana` tool with the specified image model to create the image
        in the above path by following these instructions carefully: 
        
        Present a clear, 45° top-down isometric miniature 3D cartoon scene of [CITY], 
        featuring its most iconic landmarks and architectural elements with a numer of
        cute details to make it look interesting and recognizable. Use soft, 
        refined textures with realistic PBR materials and gentle, lifelike 
        lighting and shadows. Integrate the current weather conditions directly 
        into the city environment to create an immersive atmospheric mood.
        Use a clean, minimalistic composition with a soft, solid-colored background.
        At the top-center, place the title “[CITY]” in large bold text, a prominent
        weather icon beneath it, then the current date and temperature (medium text).
        All text must be centered with consistent spacing, and may subtly overlap the 
        tops of the buildings.
        Square 1080x1080 dimension.
        
    3. Use the `display_image_with_adk` tool with the absolute file path of the generated image.
    """,
    tools=[nano_banana, display_image_with_adk]
)

class GoogleIdTokenAuth(httpx.Auth):
    def __init__(self, audience: str):
        self.audience = audience
        self._tokens = {}

    def auth_flow(self, request):        
        token = id_token.fetch_id_token(Request(), audience=self.audience)
        request.headers["Authorization"] = f"Bearer {token}"
        yield request

def get_cloud_run_client_factory(agent_path: str):
    parsed_url = urlparse(agent_path)
    service_uri = f"{parsed_url.scheme}://{parsed_url.netloc}"

    async_client = httpx.AsyncClient(
        timeout=httpx.Timeout(timeout=30),
        auth=GoogleIdTokenAuth(service_uri),
        headers={"Content-Type": "application/json"}
    )
    client_config = ClientConfig(httpx_client=async_client)
    return ClientFactory(client_config)

cityscape_agent = SequentialAgent(
    name='cityscape_agent',
    description="Creates AI-generated pictures of cities based on the current weather and their unique properties.",
    sub_agents=[city_info, city_drawer],
)

sub_agents = [cityscape_agent]
trip_instruction = ""

if "A2A_CITY_TRIP_URL" in os.environ:
    city_trip_agent = RemoteA2aAgent(
        name="city_trip_agent",
        description="Agent that can recommend city trips.",
        a2a_client_factory=get_cloud_run_client_factory(os.environ["A2A_CITY_TRIP_URL"]),
        agent_card=os.environ["A2A_CITY_TRIP_URL"]+AGENT_CARD_WELL_KNOWN_PATH,
    )
    sub_agents.append(city_trip_agent)
    trip_instruction = "* The user is asking for general travel advice use the city_trip_agent to help them figure out where to go."

root_agent = LlmAgent(
    model=DEFAULT_MODEL,
    name="root_agent",
    instruction=f"""
      <You are a helpful assistant that can motivate people to travel more
      
      If
      * The user is asking for a illustrated cityscape picture use the city_scape_agent to create a city scape with the current weather for a given city
      {trip_instruction}
      * If the user asks something else explain your skills.
      >
    """,
    sub_agents=sub_agents
)
