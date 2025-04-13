"""

Implementation of a GPT Manager class to handle the OpenAI API restrictions.

"""

from collections import deque, namedtuple
from typing import List, Union, Literal

from openai.types.chat import ChatCompletionDeveloperMessageParam, ChatCompletionSystemMessageParam, \
    ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam, ChatCompletionToolMessageParam, \
    ChatCompletionFunctionMessageParam

from evalia.llm.gpt import GPTManager
from src.evalia.logs import get_logger

import time
import copy
import tiktoken
import datetime
import os
from dataclasses import dataclass

from src.evalia.llm import LanguageModelResponse

ONE_MINUTE = 60  # One minute in seconds

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Tier level of OpenAI account
# Used to set limits for tokens and requests per minute
OPENAI_TIER = os.getenv('OPENAI_TIER', 'tier 1')

# Logging
logger = get_logger(__name__)

HistoryRecord = namedtuple('HistoryRecord', ['request', 'time', 'tokens_in_last_minute', 'requests_in_last_minute'])

# Dataclass to declare tier-N limits:
# requests per minute (rpm) and tokens per minute (tpm)
@dataclass(frozen=True)
class OpenAILimits:
    rpm: int
    tpm: int

# Values taken from https://platform.openai.com/docs/guides/rate-limits/usage-tiers?context=tier-one
# updated: 2024-10-08
OPENAI_LIMITS = {
    'tier 1': {
        'gpt-4o': OpenAILimits(rpm=500, tpm=30_000),
        'gpt-4o-mini': OpenAILimits(rpm=500, tpm=200_000),
        'gpt-4-turbo': OpenAILimits(rpm=500, tpm=30_000),
        'gpt-4': OpenAILimits(rpm=500, tpm=10_000),
        'gpt-3.5-turbo': OpenAILimits(rpm=3500, tpm=200_000),
    },
    'tier 2': {
        'gpt-4o': OpenAILimits(rpm=5000, tpm=450_000),
        'gpt-4o-mini': OpenAILimits(rpm=5000, tpm=2_000_000),
        'gpt-4-turbo': OpenAILimits(rpm=5000, tpm=450_000),
        'gpt-4': OpenAILimits(rpm=5000, tpm=40_000),
        'gpt-3.5-turbo': OpenAILimits(rpm=3500, tpm=2_000_000),
    },
    'tier 3': {
        'gpt-4o': OpenAILimits(rpm=5000, tpm=800_000),
        'gpt-4o-mini': OpenAILimits(rpm=5000, tpm=4_000_000),
        'gpt-4-turbo': OpenAILimits(rpm=5000, tpm=600_000),
        'gpt-4': OpenAILimits(rpm=5000, tpm=80_000),
        'gpt-3.5-turbo': OpenAILimits(rpm=3500, tpm=4_000_000),
    },
    'tier 4': {
        'gpt-4o': OpenAILimits(rpm=10_000, tpm=2_000_000),
        'gpt-4o-mini': OpenAILimits(rpm=10_000, tpm=10_000_000),
        'gpt-4-turbo': OpenAILimits(rpm=10_000, tpm=800_000),
        'gpt-4': OpenAILimits(rpm=10_000, tpm=300_000),
        'gpt-3.5-turbo': OpenAILimits(rpm=10_000, tpm=10_000_000),
    },
    'tier 5': {
        'gpt-4o': OpenAILimits(rpm=10_000, tpm=30_000_000),
        'gpt-4o-mini': OpenAILimits(rpm=30_000, tpm=150_000_000),
        'gpt-4-turbo': OpenAILimits(rpm=10_000, tpm=2_000_000),
        'gpt-4': OpenAILimits(rpm=10_000, tpm=1_000_000),
        'gpt-3.5-turbo': OpenAILimits(rpm=10_000, tpm=50_000_000),
    },
}


class Request:
    """Class to store number of tokens and time of a request."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.time = time.time()

    def __repr__(self):
        time_formatted = datetime.datetime.fromtimestamp(self.time).strftime('%Y-%m-%d %H:%M:%S')
        return f"Request(tokens={self.tokens}, time={time_formatted})"


class RequestQueue:
    """Class to store a queue of Requests instances."""

    def __init__(self, model):
        self.queue = deque()
        self.current_minute_tokens = 0
        self.current_minute_requests = 0
        self.history = []
        self.start_time = None
        self.model = model

        OFFSET = 0.95  # 5% offset
        def cut_limit(nominal_limit):
            return int(nominal_limit * OFFSET)
        limits = OPENAI_LIMITS[OPENAI_TIER][model]

        self.rpm = cut_limit(limits.rpm)
        self.tpm = cut_limit(limits.tpm)

        logger.info( (
            f"RequestQueue created for model {model}, "
            f"{OPENAI_TIER}. Limits: {limits}."
        ))

    def __repr__(self):
        return f"RequestQueue(tokens={self.current_minute_tokens}, queue={self.queue})"

    def __len__(self):
        return len(self.queue)

    def add(self, request):
        """Add a request to the queue."""
        if request.tokens > self.tpm:
            raise Exception(f"Request with {request.tokens} tokens exceeds the maximum of {self.tpm} tokens per minute.")

        if self.start_time is None:
            self.start_time = time.time()  # Set the start time of the queue.

        while (self.tokens_in_last_minute() + request.tokens) > self.tpm or (self.requests_in_last_minute()) > self.rpm:
            print("TPM:", self.tokens_in_last_minute() + request.tokens)
            print("RPM:", self.requests_in_last_minute())
            print("\u2615 Waiting...", round(ONE_MINUTE - (time.time() - self.queue[0].time), 2), "seconds.")
            print("------------------------------------------------------------------------")
            logger.warning((
                f"Some restriction has been reached. Waiting... "
                f"Current total tokens: {self.current_minute_tokens}. "
                f"Current total requests: {self.current_minute_requests}. "
                f"We will wait {ONE_MINUTE - (time.time() - self.queue[0].time)} seconds."
                ))
            # Wait until the first request is more than one minute old.
            time.sleep(max(ONE_MINUTE - (time.time() - self.queue[0].time), 0))

        self.queue.append(request)
        self.current_minute_tokens += request.tokens
        self.current_minute_requests += 1
        logger.info((
            f"Request added: {request}, "
            f"Tokens in last minute: {self.current_minute_tokens}, "
            f"Requests in last minute: {self.current_minute_requests}"
            ))

        hr = HistoryRecord(
            request=copy.deepcopy(request),
            time=time.time() - self.start_time,
            tokens_in_last_minute=self.current_minute_tokens,
            requests_in_last_minute=self.current_minute_requests
        )
        self.history.append(hr)


    def remove_more_than_one_minute_old(self):
        while len(self.queue) > 0 and self.queue[0].time < (time.time() - ONE_MINUTE):
            self.current_minute_tokens -= self.queue[0].tokens
            self.current_minute_requests -= 1
            self.queue.popleft()

    def tokens_in_last_minute(self):
        self.remove_more_than_one_minute_old()
        return self.current_minute_tokens

    def requests_in_last_minute(self):
        self.remove_more_than_one_minute_old()
        return self.current_minute_requests

    def sent_tokens(self):
        return sum([hr.request.tokens for hr in self.history])

    def modify_last_request(self, new_tokens):
        self.current_minute_tokens += new_tokens - self.queue[-1].tokens
        self.queue[-1].tokens = new_tokens



class GPTSmartManager(GPTManager):
    """Class to handle the OpenAI API restrictions."""

    def __init__(self, model="gpt-3.5-turbo"):
        super().__init__(model)
        self.request_queue = RequestQueue(model)
        logger.info("-----------------------------------")
        logger.info(f"GPTSmartManager started. Model: {self.model}")

    def generate_text(self, query_id: str, query_list: List[str], system_context: str = "", temperature: float = 0.0) -> List[LanguageModelResponse]:
        responses: List[LanguageModelResponse] = []
        for prompt in query_list:
            response = self.query(prompt, system_context, temperature)
            responses.append(LanguageModelResponse(response=response[0].choices[0].message["content"],
                                                   elapsed_time=response[1]["elapsed_time"],
                                                   input_tokens=response[1]["input_tokens"],
                                                   output_tokens=response[1]["output_tokens"]
                                                   )
                             )
        return responses

    def _initialize_encoding(self):
        if self.model.startswith("gpt-4"):
            self.encoding = tiktoken.get_encoding("cl100k_base")
        else:
            self.encoding = tiktoken.encoding_for_model(self.model)

    def count_tokens(self, messages):
        """Returns the number of tokens used by a list of messages."""
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(self.encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens


    def query(self, message: str, system_context: str = "", temperature: float = 0.0):
        """
        Send a query to the OpenAI API.
        A query is a list of messages, each message is a dictionary with the following keys:
        - name: name of the role
        - content: content of the message
        """

        # Before sending the messages, check if the restrictions are met int the last minute
        nt = self.count_tokens(message)  # Calculate the number of tokens in the prompt
        self.request_queue.add(Request(nt + 6))  # Request creation with 6 extra tokens from the answer prompt

        chat_successful = False
        while not chat_successful:
            try:
                chat_completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[GPTSmartManager.create_openai_message("system", system_context),
                              GPTSmartManager.create_openai_message("user", message)
                              ],
                    temperature=temperature,
                    )
                chat_successful = True
            except Exception as e:
                log_message = f"Ha ocurrido un error: {e}"
                print(log_message)
                logger.error(log_message)
                chat_successful = False
                time.sleep(5)

        elapsed_time = round(time.time() - self.request_queue.queue[-1].time, 3)
        print(f"Time: {elapsed_time} seconds")

        self.request_queue.modify_last_request(chat_completion.usage.total_tokens)
        logger.info(f'"Tokens processed: {chat_completion.usage.total_tokens}')
        logger.info(f'"Tokens in this current minute: {self.request_queue.current_minute_tokens}')

        response = chat_completion
        input_tokens = chat_completion.usage.prompt_tokens
        output_tokens = chat_completion.usage.completion_tokens
        logger.info((
            f'"Finished. '
            f'Input tokens: {input_tokens} '
            f'Output tokens: {output_tokens}'
            ))
        return response, {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "elapsed_time": elapsed_time
            }

    @staticmethod
    def create_openai_message(role: Literal['user', 'system'], prompt: str) -> Union[
        ChatCompletionDeveloperMessageParam,
        ChatCompletionSystemMessageParam,
        ChatCompletionUserMessageParam,
        ChatCompletionAssistantMessageParam,
        ChatCompletionToolMessageParam,
        ChatCompletionFunctionMessageParam,
    ]:
        return {
            "role": role,
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }

if __name__ == "__main__":
    import os
    # Read file "query-4ESO-8.1-deunaenuna.txt"
    script_dir = os.path.dirname(__file__)
    test_query_file = script_dir + "/tests/query-4ESO-8.1-deunaenuna.txt"
    with open(test_query_file, "r", encoding="utf8") as f:
        query = f.read()

    messages_list = eval(query)

    gm = GPTSmartManager(model="gpt-3.5-turbo")
    # for _ in range(1):
    #    for m in messages:
    #        print(gm.query(m))

    print(gm.send_queries(1, messages_list[0], 0.9))


