"""

Implementation of a GPT Manager class to handle the OpenAI API restrictions.

"""

from collections import deque, namedtuple
from typing import Tuple
from . import GPTManager, GPTTask
from .gpt_manager import accepts_temperature
from ..logs import get_logger

from openai import OpenAI
from openai import APITimeoutError, APIConnectionError, RateLimitError, BadRequestError
import time
import copy
import tiktoken
import datetime
import os

ONE_MINUTE = 60  # One minute in seconds
RETRY_TIMEOUT_IF_ERROR = 5

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Logging
logger = get_logger(__name__)

HistoryRecord = namedtuple('HistoryRecord', ['request', 'time', 'tokens_in_last_minute', 'requests_in_last_minute'])


class Request:
    """Class to store number of tokens and time of a request."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.time = time.time()

    def __repr__(self):
        time_formatted = datetime.datetime.fromtimestamp(self.time).strftime('%Y-%m-%d %H:%M:%S')
        return f"Request(tokens={self.tokens}, time={time_formatted})"


class RequestQueue:
    """
    This object stores a queue of Request instances.
    The RequestQueue is responsible for discovering and enforcing 
    the real limits (RPM/TPM) of the OpenAI account for a specific model.

    The RPM/TPM limits are obtained by sending a minimal request to OpenAI and
    reading the headers `x-ratelimit-limit-requests` / `x-ratelimit-limit-tokens`
    from the response.
    """

    def __init__(self, model, client):
        self.queue = deque()
        self.current_minute_tokens = 0
        self.current_minute_requests = 0
        self.history = []
        self.start_time = None
        self.model = model

        self.rpm, self.tpm = self._discover_limits(client)

        logger.info((
            f"RequestQueue created for model {model}. "
            f"Limits: rpm={self.rpm}, tpm={self.tpm}."
        ))

    def _discover_limits(self, client):
        """Discover the RPM/TPM limits by sending a minimal request to the model.

        Only the rate limit headers of the response matter, and OpenAI also
        sends them in 400 responses. So a 400 is accepted if it carries them,
        e.g. when a reasoning model (gpt-5) spends the whole token budget
        reasoning and the API answers "Could not finish the message because
        max_tokens or model output limit was reached" (it happens randomly).
        """

        _DISCOVERY_MESSAGES = [{"role": "user", "content": "ping"}]
        _DISCOVERY_MAX_TOKENS = 32
        _LIMIT_HEADERS = ("x-ratelimit-limit-requests", "x-ratelimit-limit-tokens")

        # Security margin over the nominal limit reported by OpenAI
        OFFSET = 0.95

        try:
            raw_response = client.chat.completions.with_raw_response.create(
                model=self.model,
                messages=_DISCOVERY_MESSAGES,
                max_completion_tokens=_DISCOVERY_MAX_TOKENS,
            )
            headers = raw_response.headers
            used_tokens = raw_response.parse().usage.total_tokens
        except BadRequestError as e:
            if not all(h in e.response.headers for h in _LIMIT_HEADERS):
                raise
            logger.info(f"Limits of model {self.model} read from a 400 response: {e.message}")
            headers = e.response.headers
            # The error response has no usage: assume the whole budget was spent
            used_tokens = _DISCOVERY_MAX_TOKENS

        rpm = int(int(headers["x-ratelimit-limit-requests"]) * OFFSET)
        tpm = int(int(headers["x-ratelimit-limit-tokens"]) * OFFSET)

        """
        The discovery request consumes resources (one request and some tokens), so
        it is recorded in the queue like any other, so that subsequent contention
        does not overlook it.
        """
        self.start_time = time.time()
        self._record(Request(used_tokens))

        return rpm, tpm

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

        self._record(request)

    def _record(self, request):
        """Contabiliza una petición ya aceptada: cuenta y guarda su historial."""
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
        super().__init__()
        self.model = model
        self.temperature = 0.0
        self.query_id = None
        self.message = []
        self.prelude = None
        self._initialize_encoding()

        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.request_queue = RequestQueue(model, self.client)

        logger.info(f"-----------------------------------")
        logger.info(f"GPTSmartManager started. Model: {self.model}")

    def _initialize_encoding(self):
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # Model unknown to tiktoken (e.g. newer than the installed
            # version): o200k_base, the tokenizer of the gpt-5 family, is
            # a good enough estimate to enforce the rate limits
            logger.info(f"tiktoken does not know model {self.model}: using o200k_base")
            self.encoding = tiktoken.get_encoding("o200k_base")

    def initialize(self):
        pass

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
    
    def send_queries(self, query_id, query_list, temperature):
        self.temperature = temperature
        self.query_id = query_id
        if isinstance(query_list, list) and all(isinstance(elem, list) for elem in query_list):
            if not accepts_temperature(self.model):
                logger.info((
                    f'"{query_id}" Model {self.model} only supports the default '
                    f'temperature: temperature={temperature} is not sent'
                    ))
            input_tokens = 0
            output_tokens = 0
            elapsed_time = 0
            responses = []
            for messages in query_list:
                response = self.query(messages)
                responses.append(response[0])
                input_tokens += response[1]["input_tokens"]
                output_tokens += response[1]["output_tokens"]
                elapsed_time += response[1]["elapsed_time"]
            return responses, {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "elapsed_time": elapsed_time
                }
        else:
            return self.send_queries(query_id,[query_list],temperature)
            
    
    def query(self, messages):
        """
        Send a query to the OpenAI API.
        A query is a list of messages, each message is a dictionary with the following keys:
        - name: name of the role
        - content: content of the message
        """
            
        # Before sending the messages, check if the restrictions are met int the last minute
        nt = self.count_tokens(messages)  # Calculate the number of tokens in the message
        self.request_queue.add(Request(nt + 6))  # Request creation with 6 extra tokens from the answer prompt

        # Temperature only for the models that accept it (see accepts_temperature)
        optional_params = {}
        if accepts_temperature(self.model):
            optional_params["temperature"] = self.temperature

        chat_successful = False
        while not chat_successful:
            try:
                chat_completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **optional_params,
                    )
                chat_successful = True
            except (APITimeoutError, APIConnectionError, RateLimitError) as e:
                log_message = f"{self.query_id} Retryable error in OpenAI service. Retrying. Error: {e}"
                print(log_message)
                logger.error(log_message)
                chat_successful = False
                time.sleep(RETRY_TIMEOUT_IF_ERROR)
            except:
                raise

        
        elapsed_time = round(time.time() - self.request_queue.queue[-1].time, 3)
        print(f"{self.query_id} Time: {elapsed_time} seconds")

        self.request_queue.modify_last_request(chat_completion.usage.total_tokens)
        logger.info(f'"{self.query_id}" Tokens processed: {chat_completion.usage.total_tokens}')
        logger.info(f'"{self.query_id}" Tokens in this current minute: {self.request_queue.current_minute_tokens}')

        response = chat_completion
        input_tokens = chat_completion.usage.prompt_tokens
        output_tokens = chat_completion.usage.completion_tokens
        logger.info((
            f'"{self.query_id}" Finished. '
            f'Input tokens: {input_tokens} '
            f'Output tokens: {output_tokens}'
            ))
        return response, {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "elapsed_time": elapsed_time
            }
        
    class GPTSmartTask(GPTTask):
        '''Dataclass to store the task information'''
        def __init__(self, responses, stats):
            self.responses = responses
            self.stats = stats
    

    def start_task(self, query_id, query_list, temperature) -> GPTTask:
        responses, stats = self.send_queries(query_id, query_list, temperature)
        return self.GPTSmartTask(responses, stats)

    def cancel_task(self, task: GPTTask):
        pass

    def save_task(self, task: GPTTask, filename: str):
        raise NotImplementedError("Esta clase no permite persistir el estado de una tarea.")
    
    def load_task(self, filename: str) -> GPTTask:
        raise NotImplementedError("Esta clase no permite persistir el estado de una tarea.")
    
    def get_response(self, task: GPTSmartTask, timeout=0, retry=0) -> Tuple[list, dict]:
        return task.responses, task.stats
    
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

    
