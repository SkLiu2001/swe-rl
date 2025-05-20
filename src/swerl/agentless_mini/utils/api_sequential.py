import time
from typing import Any, Literal
from tqdm.auto import tqdm
import openai
import tenacity
from openai.types.chat import ChatCompletion
from whale import TextGeneration
from whale.util import Timeout
from .envs import API_TYPE, WHALE_API_KEY, WHALE_BASE_URL, OPENAI_API_KEY

# 初始化 API
if API_TYPE == "whale":
    TextGeneration.set_api_key(WHALE_API_KEY, base_url=WHALE_BASE_URL)
else:
    openai.api_key = OPENAI_API_KEY

def retry(errors: Any, max_attempts: int = 5):
    return tenacity.retry(
        retry=tenacity.retry_if_exception_type(errors),
        wait=tenacity.wait_exponential(multiplier=1, min=5, max=20),
        stop=tenacity.stop_after_attempt(max_attempts),
        before_sleep=print,
    )

ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

class APIClient:
    def __init__(self, api_type: str = API_TYPE):
        self.api_type = api_type
        if api_type == "openai":
            self.client = openai.OpenAI()
        else:
            self.client = TextGeneration
        self.default_timeout = 180  # 3分钟超时，使用单个浮点数

    @retry(ERRORS)
    def chat_completions_with_backoff(self, *args, **kwargs):
        if self.api_type == "openai":
            return self.client.chat.completions.create(*args, **kwargs)
        else:
            try:
                return TextGeneration.chat(
                    model=kwargs.get('model'),
                    messages=kwargs.get('messages'),
                    stream=False,
                    temperature=kwargs.get('temperature', 0.6),
                    max_tokens=kwargs.get('max_tokens', 2000),
                    top_p=kwargs.get('top_p', 0.1),
                    timeout=kwargs.get('timeout', self.default_timeout)  # 使用单个浮点数作为超时值
                )
            except Exception as e:
                print(f"Error in API call: {str(e)}")
                raise

    def safe_chat_completion(self, request: dict):
        try:
            return self.chat_completions_with_backoff(**request)
        except Exception as e:
            print(f"Error in safe_chat_completion: {str(e)}")
            if "Read timed out" in str(e):
                print("Request timed out, consider increasing timeout or checking network connection")
            return None

    def delayed_request(
        self,
        request: dict[str, Any],
        mode: Literal["chat", "completion"],
        delay: float | None,
    ):
        if delay is not None:
            time.sleep(delay)
        return self.chat_completions_with_backoff(**request)

def run_with_retry(
    client: APIClient,
    request: dict,
    idx: int,
    retries: int = 3,
    delay: float = 2,
    timeout: float = 180
) -> tuple[int, Any]:
    for attempt in range(retries):
        try:
            response = client.delayed_request(request, "chat", delay)
            return (idx, response)
        except Exception as e:
            print(f"Error in request {idx}: {str(e)}")
            if attempt < retries - 1:
                print(f"Retrying request {idx} (attempt {attempt + 1}/{retries})")
                time.sleep(delay)
            else:
                print(f"Request {idx} failed after {retries} attempts")
                return idx, None

def collect_responses(
    client: APIClient,
    all_requests: list[dict],
    retries: int = 3,
    delay: float = 2,
    timeout: float = 180
) -> list[tuple[int, Any]]:
    idx_and_responses = []
    for idx, request in enumerate(all_requests):
        idx, response = run_with_retry(
            client,
            request,
            idx,
            retries,
            delay,
            timeout
        )
        idx_and_responses.append((idx, response))
    
    return sorted(idx_and_responses, key=lambda x: x[0])

def parse_thinking_output(output: str) -> str:
    return output.strip()

# 使用示例
def main():
    # 创建客户端实例
    client = APIClient(api_type="whale")
    
    # 准备请求
    requests = [
        {
            "model": "qwen25_32_instruct_ac",
            "messages": [
                {"role": "system", "content": "你是一个人工智能。回答用中文。"},
                {"role": "user", "content": "你好"}
            ],
            "temperature": 0.6,
            "max_tokens": 2000,
            "top_p": 0.8,
            "extend_fields": {"top_k": 1}
        }
    ]
    
    # 收集响应
    responses = collect_responses(
        client,
        requests,
        retries=3,
        delay=2,
        timeout=180
    )
    
    # 处理响应
    for idx, response in responses:
        if response:
            print(f"Response {idx}:", response.choices[0].message.content)
        else:
            print(f"Request {idx} failed")

if __name__ == "__main__":
    main() 