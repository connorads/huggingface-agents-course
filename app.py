from smolagents import (
    CodeAgent,
    VisitWebpageTool,
    DuckDuckGoSearchTool,
    FinalAnswerTool,
    HfApiModel,
    load_tool,
    tool,
)
from markitdown import MarkItDown
import datetime
import pytz
import yaml

from Gradio_UI import GradioUI

import os
import requests
from urllib.parse import urlparse


@tool
def download_file(url: str, filename: str = None) -> str:
    """
    Downloads a file from an HTTP URL to the './temp' directory.
    The directory is created if it does not already exist.

    Args:
        url: The HTTP URL of the file to download.
        filename: (Optional) The name to save the file as. If not provided, the filename is inferred from the URL.

    Returns:
        The local path to the downloaded file.
    """
    # Define the target directory and create it if it doesn't exist
    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)

    # If no filename is provided, attempt to extract it from the URL
    if not filename:
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            # Use a generic filename with a timestamp if extraction fails
            import time

            filename = f"downloaded_{int(time.time())}"

    local_path = os.path.join(temp_dir, filename)

    try:
        # Stream the download to handle large files efficiently
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad responses

        with open(local_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    file.write(chunk)
        return local_path

    except Exception as e:
        return f"Error downloading file: {str(e)}"


@tool
def to_markdown(file_path: str) -> str:
    """
    MarkItDown is a utility for converting various files to Markdown
    (e.g., for indexing, text analysis, etc). It supports:
    PDF, PowerPoint, Word, Excel, Images (EXIF metadata and OCR),
    Audio (EXIF metadata and speech transcription), HTML,
    Text-based formats (CSV, JSON, XML),
    ZIP files (iterates over contents)

    Args:
        file_path: The path to the file.
    Returns:
        A string containing the content of the file.
    """
    md = MarkItDown()
    result = md.convert(file_path)
    return result.text_content


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


model = HfApiModel(
    max_tokens=2096,
    temperature=0.5,
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    custom_role_conversions=None,
)


# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)


final_answer = FinalAnswerTool()
with open("prompts.yaml", "r") as stream:
    prompt_templates = yaml.safe_load(stream)

agent = CodeAgent(
    model=model,
    tools=[
        image_generation_tool,
        get_current_time_in_timezone,
        to_markdown,
        download_file,
        VisitWebpageTool(),
        DuckDuckGoSearchTool(),
        final_answer,
    ],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates,
)


GradioUI(agent).launch()
