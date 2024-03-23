from typing import Optional, AsyncGenerator

import numpy as np
import tiktoken
import instructor  # type: ignore
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessageParam


client = instructor.patch(OpenAI(api_key="settings.OPEN_AI_TOKEN"))
async_client = instructor.patch(AsyncOpenAI(api_key="settings.OPEN_AI_TOKEN"))


def num_tokens(string: Optional[str]) -> int:
    if not string:
        return 0
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))
