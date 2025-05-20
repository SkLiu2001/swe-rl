# # Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

# import asyncio
# import time
# from typing import Any, Awaitable, Literal, TypeVar

# import openai
# import tenacity
# from openai.types.chat import ChatCompletion
# from tqdm.auto import tqdm

#from .envs import ANSWER_END_TAG, ANSWER_START_TAG, THINKING


# def retry(errors: Any, max_attempts: int = 5):
#     return tenacity.retry(
#         retry=tenacity.retry_if_exception_type(errors),
#         wait=tenacity.wait_exponential(multiplier=1, min=5, max=20),
#         stop=tenacity.stop_after_attempt(max_attempts),
#         before_sleep=print,
#     )


# ERRORS = (
#     openai.RateLimitError,
#     openai.APIConnectionError,
#     openai.InternalServerError,
# )


# class OpenAIClient:
#     def __init__(self):
#         self.client = openai.OpenAI()
#         self.async_client = openai.AsyncClient()

#     @retry(ERRORS)
#     def chat_completions_with_backoff(self, *args, **kwargs):
#         return self.client.chat.completions.create(*args, **kwargs)

#     @retry(ERRORS)
#     def completions_with_backoff(self, *args, **kwargs):
#         return self.client.completions.create(*args, **kwargs)

#     @retry(ERRORS)
#     async def chat_completions_with_backoff_async(self, *args, **kwargs):
#         return await self.async_client.chat.completions.create(*args, **kwargs)

#     @retry(ERRORS)
#     async def completions_with_backoff_async(self, *args, **kwargs):
#         return await self.async_client.completions.create(*args, **kwargs)

#     async def safe_chat_completion(self, request: dict):
#         try:
#             return await self.chat_completions_with_backoff_async(**request)
#         except openai.BadRequestError as e:
#             print("Error request:", str(e))
#             return None

#     async def delayed_request(
#         self,
#         request: dict[str, Any],
#         mode: Literal["chat", "completion"],
#         delay: float | None,
#     ):
#         """Prevent quantized rate limit:
#         https://help.openai.com/en/articles/6891753-rate-limit-advice"""
#         if delay is not None:
#             # synchronized sleep
#             time.sleep(delay)
#         if mode == "chat":
#             func = self.chat_completions_with_backoff_async
#         else:
#             func = self.completions_with_backoff_async
#         return await func(**request)

#     def dispatch_chat_completions(
#         self,
#         requests: list[dict[str, Any]],
#         delay: float | None = None,
#     ):
#         return asyncio.run(self._dispatch_chat_completions(requests, delay))

#     def dispatch_completions(
#         self,
#         requests: list[dict[str, Any]],
#         delay: float | None = None,
#     ):
#         return asyncio.run(self._dispatch_completions(requests, delay))

#     async def _dispatch_chat_completions(
#         self,
#         requests: list[dict[str, Any]],
#         delay: float | None = None,
#     ):
#         """Dispatch chat completions requests asynchronously.
#         Args:
#             requests: a list of API argument names to values.
#             delay: interval between requests.
#         """

#         tasks = [self.delayed_request(request, "chat", delay) for request in requests]
#         return await asyncio.gather(*tasks, return_exceptions=False)

#     async def _dispatch_completions(
#         self,
#         requests: list[dict[str, Any]],
#         delay: float | None = None,
#     ):
#         """Dispatch completions requests asynchronously.
#         Args:
#             requests: a list of API argument names to values.
#             delay: interval between requests.
#         """

#         tasks = [
#             self.delayed_request(request, "completion", delay) for request in requests
#         ]
#         return await asyncio.gather(*tasks, return_exceptions=False)


# T = TypeVar("T")


# async def run_with_semaphore(
#     semaphore: asyncio.Semaphore, task: Awaitable[T], index: int
# ):
#     async with semaphore:
#         return (index, await task)


# async def run_with_retry(semaphore, coro, idx, retries=3, delay=2):
#     for attempt in range(retries):
#         try:
#             return await run_with_semaphore(semaphore, coro, idx)
#         except Exception as e:
#             if attempt < retries - 1:
#                 await asyncio.sleep(delay)
#             else:
#                 return idx, None  # 或 raise e，根据需求


# async def collect_responses_async(
#     client: OpenAIClient,
#     semaphore: asyncio.Semaphore,
#     all_requests: list[dict],
#     retries: int = 3,
#     delay: int = 2,
# ):
#     all_tasks = [
#         #run_with_semaphore(semaphore, client.safe_chat_completion(request), idx)
#         run_with_retry(semaphore, client.safe_chat_completion(request), idx, retries, delay)
#         for idx, request in enumerate(all_requests)
#     ]
#     idx_and_responses = list[tuple[int, ChatCompletion | None]]()
#     pbar = tqdm(total=len(all_tasks), desc="Process each instance", leave=False)
#     for completion in asyncio.as_completed(all_tasks):
#         idx, response = await completion
#         idx_and_responses.append((idx, response))
#         pbar.update(1)
#     pbar.close()
#     return idx_and_responses


# def parse_thinking_output(output: str) -> str:
#     """Extract the <solution> part for thinking models"""
#     if THINKING:
#         output = output.split(ANSWER_START_TAG, 1)[-1]
#         output = output.split(ANSWER_END_TAG, 1)[0]
#     return output.strip()

import asyncio
import time
from typing import Any, Awaitable, Literal, TypeVar
from tqdm.auto import tqdm
import openai
import tenacity
from openai.types.chat import ChatCompletion
from whale import TextGeneration, VipServerLocator
from whale.util import Timeout
from .envs import ANSWER_END_TAG, ANSWER_START_TAG, THINKING, API_TYPE, WHALE_API_KEY, WHALE_BASE_URL, OPENAI_API_KEY

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

class MonitoredSemaphore(asyncio.Semaphore):
    def __init__(self, value=1):
        super().__init__(value)
        self._initial_value = value
        self._acquired = 0
        
    async def acquire(self):
        await super().acquire()
        self._acquired += 1
        
    def release(self):
        super().release()
        self._acquired -= 1
        
    @property
    def available(self):
        return self._value
        
    @property
    def used(self):
        return self._acquired

class APIClient:
    def __init__(self, api_type: str = API_TYPE):
        self.api_type = api_type
        if api_type == "openai":
            self.client = openai.OpenAI()
            self.async_client = openai.AsyncClient()
        else:
            self.client = TextGeneration
        self.default_timeout = 240  # 3分钟超时

    @retry(ERRORS)
    async def chat_completions_with_backoff_async(self, *args, **kwargs):
        if self.api_type == "openai":
            return await self.async_client.chat.completions.create(*args, **kwargs)
        else:
            #     timeout = kwargs.pop('timeout', self.default_timeout)
            #     return await asyncio.wait_for(
            #         asyncio.to_thread(
            #             self.client.chat,
            #             model=kwargs.get('model'),
            #             messages=kwargs.get('messages'),
            #             stream=False,
            #             temperature=kwargs.get('temperature', 0.6),
            #             max_tokens=kwargs.get('max_tokens', 2000),
            #             timeout=timeout,
            #             top_p=kwargs.get('top_p', 0.1),
            #             extend_fields=kwargs.get('extend_fields', {})
            #         ),
            #         timeout=120
            #     )
            # except asyncio.TimeoutError:
            #     print("API call timed out after 120 seconds")
            #     raise
            # whale 非异步chat
                # 使用 asyncio.to_thread 将同步调用转换为异步
            return await asyncio.to_thread(
                self.client.chat,
                model=kwargs.get('model'),
                messages=kwargs.get('messages'),
                stream=False,
                temperature=kwargs.get('temperature', 0.6),
                max_tokens=kwargs.get('max_tokens', 2000),
                top_p=kwargs.get('top_p', 0.1),
                timeout=kwargs.get('timeout', self.default_timeout)
            )                 

    async def safe_chat_completion(self, request: dict):
        try:
            return await self.chat_completions_with_backoff_async(**request)
        except Exception as e:
            print(f"Error in safe_chat_completion: {str(e)}")
            if "Read timed out" in str(e):
                print("Request timed out, consider increasing timeout or checking network connection")
            return None

    async def delayed_request(
        self,
        request: dict[str, Any],
        mode: Literal["chat", "completion"],
        delay: float | None,
    ):
        if delay is not None:
            await asyncio.sleep(delay)
        return await self.chat_completions_with_backoff_async(**request)

    async def _dispatch_chat_completions(
        self,
        requests: list[dict[str, Any]],
        delay: float | None = None,
    ):
        tasks = [self.delayed_request(request, "chat", delay) for request in requests]
        return await asyncio.gather(*tasks, return_exceptions=False)

T = TypeVar("T")

async def run_with_retry(
    semaphore: MonitoredSemaphore,
    coro: Awaitable,
    idx: int,
    retries: int = 3,
    delay: float = 2,
    timeout: float = 180
) -> tuple[int, Any]:
    for attempt in range(retries):
        try:
            async with semaphore:
                return (idx, await coro)
        except Exception as e:
            print(f"Error in request {idx}: {str(e)}")
            if attempt < retries - 1:
                print(f"Retrying request {idx} (attempt {attempt + 1}/{retries})")
                await asyncio.sleep(delay)
            else:
                print(f"Request {idx} failed after {retries} attempts")
                return idx, None

async def collect_responses_async(
    client: APIClient,
    semaphore: MonitoredSemaphore,
    all_requests: list[dict],
    retries: int = 3,
    delay: float = 2,
    timeout: float = 180
) -> list[tuple[int, Any]]:
    all_tasks = [
        run_with_retry(
            semaphore,
            client.chat_completions_with_backoff_async(**request),
            idx,
            retries,
            delay,
            timeout
        )
        for idx, request in enumerate(all_requests)
    ]
    
    idx_and_responses = []
    #with tqdm(total=len(all_tasks), desc="Processing requests") as pbar:
    for completion in asyncio.as_completed(all_tasks):
        idx, response = await completion
        idx_and_responses.append((idx, response))
                # pbar.update(1)
                # pbar.set_postfix({
                #     'available': semaphore.available,
                #     'used': semaphore.used
                # })
    
    return sorted(idx_and_responses, key=lambda x: x[0])

def parse_thinking_output(output: str) -> str:
    return output.strip()

# 使用示例
async def main():
    # 创建客户端实例
    client = APIClient(api_type="whale")
    
    # 创建带监控的信号量
    semaphore = MonitoredSemaphore(5)  # 最多5个并发请求
    
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
    responses = await collect_responses_async(
        client,
        semaphore,
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
    asyncio.run(main())