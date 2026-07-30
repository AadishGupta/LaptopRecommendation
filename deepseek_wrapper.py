"""
deepseek_wrapper.py - Custom DeepSeek-R1 wrapper using Ollama generate API
Works locally with GPU acceleration
"""

import requests
import json
import re
import logging
from typing import Optional, List, Dict, Any
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

logger = logging.getLogger(__name__)

# =============================================================================
# DEEPSEEK RESPONSE CLEANER
# =============================================================================

def clean_deepseek_response(response: str) -> str:
    """
    Clean DeepSeek-R1 responses by removing thinking tags and artifacts.
    
    DeepSeek-R1 often outputs responses with:
    - "Thinking..." and "...done thinking." markers
    - Alright, I'll... intro text
    - <think> tags
    - Extra whitespace and newlines
    
    This function removes all of these to get just the clean response.
    """
    if not response:
        return response
    
    # Remove thinking section
    if "Thinking..." in response:
        parts = response.split("...done thinking.")
        if len(parts) > 1:
            response = parts[-1].strip()
    
    # Remove any remaining tags
    response = response.replace("Thinking...", "").replace("...done thinking.", "")
    response = response.replace("Alright,", "").strip()
    response = response.replace("I'll", "I will").strip()
    
    # Remove <think>...</think> blocks (closed case)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

    # Handle a TRUNCATED think block: the model ran out of tokens mid-reasoning,
    # so there's an opening <think> with no closing tag. Since everything after
    # it is unfinished reasoning, not an answer, strip from <think> to the end.
    response = re.sub(r'<think>.*$', '', response, flags=re.DOTALL)

    response = re.sub(r'Thinking\.\.\.\s*', '', response)
    
    # Remove excessive newlines
    response = re.sub(r'\n{3,}', '\n\n', response)
    
    # Remove leading/trailing whitespace
    response = response.strip()
    
    return response


# =============================================================================
# DEEPSEEK LLM - LangChain Compatible Wrapper
# =============================================================================

# Prefix used to mark responses that are actually transport/connection
# failures, not model output. Callers should check `resp.startswith(ERROR_PREFIX)`
# and retry / skip rather than treating it as generated text — this is what
# was missing before, letting timeout strings get banked as real questions.
ERROR_PREFIX = "Error:"


class DeepSeekLLM(LLM):
    """
    Custom LangChain LLM for DeepSeek-R1 using the generate API.
    Works locally with Ollama and GPU acceleration.
    
    This wrapper bypasses the chat API which has issues with DeepSeek-R1
    and uses the generate API directly.
    
    Example:
        llm = DeepSeekLLM(temperature=0.3, num_predict=500)
        response = llm.invoke("What is the best laptop for gaming?")
        print(response)
    """
    
    # Model configuration
    model: str = "deepseek-r1:7b"
    temperature: float = 0.3
    num_predict: int = 500
    num_ctx: int = 4096
    base_url: str = "http://localhost:11434"
    system_prompt: Optional[str] = None
    num_gpu: int = 1
    num_thread: int = 4
    think: bool = True
    keep_alive: str = "30m"
    # Was hardcoded to 120 inline at every requests.post() call. Now a real
    # field so callers doing longer/heavier generations (e.g. 30-40 line
    # eval-question batches with think=True) can raise it without editing
    # this file every time. 120s was frequently too short for a cold-loaded
    # 7B model doing a full <think> pass on a 6GB card.
    timeout: int = 180
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "deepseek-r1-local"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Call the DeepSeek-R1 model using the generate API.
        
        Args:
            prompt: The prompt to send to the model
            stop: Optional list of stop sequences
            run_manager: Callback manager for LLM runs
            **kwargs: Additional arguments
            
        Returns:
            str: The cleaned response from the model
        """
        
        # Build the full prompt with system if provided
        full_prompt = prompt
        if self.system_prompt:
            full_prompt = f"{self.system_prompt}\n\n{prompt}"
        
        # Prepare the payload for Ollama
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            # NOTE: "think" must be top-level (NOT under "options") — it's the
            # Ollama flag that actually disables the model's reasoning pass for
            # models like deepseek-r1. Telling the model "don't think" via the
            # system prompt does NOT reliably work; reasoning models are
            # trained to emit <think>...</think> regardless of instructions.
            # Without this, short-token-budget classification calls (20-40
            # tokens for moderation/intent/orchestrator) reliably burn their
            # entire budget on unfinished reasoning and return raw length=0.
            "think": self.think,
            # Keep the model resident in VRAM for this many minutes after the
            # call completes, instead of Ollama's default eviction behavior.
            # Without this, switching between models (e.g. deepseek-r1 for
            # chat, nomic-embed-text for search embeddings, back to
            # deepseek-r1 for compare) can force a reload from disk each time
            # — measured at ~6s per reload on this hardware, on top of actual
            # inference time.
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
                "num_ctx": self.num_ctx,
                "num_gpu": self.num_gpu,
                "num_thread": self.num_thread,
                "stop": stop or [],
            }
        }
        
        # Log the request for debugging
        logger.debug(f"DeepSeekLLM calling with prompt length: {len(prompt)}")
        
        try:
            # Make the API call
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            # Fail-safe: older Ollama builds don't recognize the top-level
            # "think" field and may reject the request. Retry once without it
            # rather than breaking every call on an older install.
            if response.status_code == 400 and "think" in payload:
                logger.warning("Ollama rejected 'think' param (likely older version) — retrying without it")
                payload_retry = {k: v for k, v in payload.items() if k != "think"}
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload_retry,
                    timeout=self.timeout
                )
            
            # Check for successful response
            if response.status_code == 200:
                result = response.json()
                raw = result.get("response", "")
                
                # Clean the response
                cleaned = clean_deepseek_response(raw)

                # If cleaning resulted in empty, return "" and let the CALLER
                # decide the fallback message. Previously this returned a
                # generic apology string here, but since that string is >10
                # characters, it slipped past callers' own "empty response"
                # checks (e.g. conversation_node's `len(response) < 10` test)
                # and reached the user instead of the caller's more useful,
                # context-specific fallback message.
                if not cleaned:
                    logger.warning(
                        "DeepSeek returned empty response after cleaning "
                        f"(raw length={len(raw)}, num_predict={self.num_predict}) — "
                        "likely ran out of tokens inside <think> before reaching "
                        "an answer; consider raising num_predict."
                    )
                    return ""

                return cleaned
            else:
                error_msg = f"{ERROR_PREFIX} {response.status_code} - {response.text}"
                logger.error(error_msg)
                return error_msg
                
        except requests.exceptions.Timeout:
            error_msg = f"{ERROR_PREFIX} Request timed out. The model is taking too long to respond."
            logger.error(error_msg)
            return error_msg
            
        except requests.exceptions.ConnectionError:
            error_msg = f"{ERROR_PREFIX} Cannot connect to Ollama. Please make sure Ollama is running with: ollama serve"
            logger.error(error_msg)
            return error_msg
            
        except Exception as e:
            error_msg = f"{ERROR_PREFIX} {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """
        Make the instance callable for easier use.
        
        Example:
            llm = DeepSeekLLM()
            response = llm("What is the best laptop?")
        """
        return self._call(prompt, **kwargs)
    
    def invoke(self, input: Any, config: Optional[Any] = None, **kwargs) -> str:
        """
        Invoke the model with a prompt (LangChain Runnable-compatible).

        IMPORTANT: LangChain's LCEL chains (e.g. `prompt | llm | StrOutputParser()`)
        call each step as `step.invoke(input, config)`, passing `config`
        POSITIONALLY. The previous signature `invoke(self, prompt, **kwargs)` had
        no `config` parameter, so every chain built with `create_chain()` in
        agent_functions.py (_moderation_chain, _intent_chain, _extraction_chain,
        _req_string_chain, _hyde_chain, etc.) raised
        "DeepSeekLLM.invoke() takes 2 positional arguments but 3 were given" on
        EVERY call and silently fell into the caller's `except` block. This
        means moderation, intent detection, and requirement extraction were
        never actually running the LLM — they were always hitting their
        exception-handler defaults.

        `input` may be a plain string (direct calls like `llm.invoke(prompt)`)
        or a PromptValue (when invoked as part of an LCEL chain) — handle both.
        """
        if hasattr(input, "to_string"):
            prompt = input.to_string()
        else:
            prompt = str(input)
        return self._call(prompt, **kwargs)

    async def ainvoke(self, input: Any, config: Optional[Any] = None, **kwargs) -> str:
        """Async counterpart, same signature fix as invoke()."""
        return self.invoke(input, config, **kwargs)
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """
        Get the identifying parameters for the LLM.
        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "num_predict": self.num_predict,
            "num_ctx": self.num_ctx,
            "base_url": self.base_url,
            "system_prompt": self.system_prompt,
            "num_gpu": self.num_gpu,
            "num_thread": self.num_thread,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "timeout": self.timeout,
        }
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Chat interface compatible with LangChain's chat format.
        Converts chat messages to a prompt for the generate API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            str: The model's response
        """
        # Convert chat messages to a prompt
        prompt_parts = []
        
        # Extract system message if present
        system_content = None
        user_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_content = content
            elif role == "user":
                user_messages.append(f"User: {content}")
            elif role == "assistant":
                user_messages.append(f"Assistant: {content}")
        
        # Build the prompt
        if system_content:
            prompt_parts.append(f"System: {system_content}")
        
        prompt_parts.extend(user_messages)
        prompt = "\n\n".join(prompt_parts)
        
        # Call the model
        return self._call(prompt)
    
    def generate_response(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Generate a response with optional system prompt.
        
        Args:
            prompt: The user prompt
            system: Optional system prompt
            
        Returns:
            str: The model's response
        """
        # Temporarily override system prompt if provided
        original_system = self.system_prompt
        if system:
            self.system_prompt = system
        
        try:
            response = self._call(prompt)
            return response
        finally:
            # Restore original system prompt
            self.system_prompt = original_system


# =============================================================================
# DEEPSEEK CHAT - Simple Chat Interface
# =============================================================================

class DeepSeekChat:
    """
    Simple chat interface for DeepSeek-R1 with conversation history.
    
    Example:
        chat = DeepSeekChat()
        response = chat.send("Hello!")
        print(response)
        response = chat.send("What can you help me with?")
        print(response)
    """
    
    def __init__(
        self,
        model: str = "deepseek-r1:7b",
        temperature: float = 0.3,
        num_predict: int = 500,
        num_ctx: int = 4096,
        base_url: str = "http://localhost:11434",
        system_prompt: Optional[str] = None,
        num_gpu: int = 1,
        num_thread: int = 4,
    ):
        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.num_gpu = num_gpu
        self.num_thread = num_thread
        self.history: List[Dict[str, str]] = []
        
        # Create the LLM
        self.llm = DeepSeekLLM(
            model=model,
            temperature=temperature,
            num_predict=num_predict,
            num_ctx=num_ctx,
            base_url=base_url,
            system_prompt=system_prompt,
            num_gpu=num_gpu,
            num_thread=num_thread,
        )
    
    def send(self, message: str) -> str:
        """
        Send a message and get a response, maintaining conversation history.
        
        Args:
            message: The user's message
            
        Returns:
            str: The model's response
        """
        # Add user message to history
        self.history.append({"role": "user", "content": message})
        
        # Build messages with history
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.history)
        
        # Get response
        response = self.llm.chat(messages)
        
        # Add assistant response to history
        self.history.append({"role": "assistant", "content": response})
        
        return response
    
    def clear_history(self):
        """Clear the conversation history."""
        self.history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.history
    
    def set_system_prompt(self, system_prompt: str):
        """
        Set or update the system prompt.
        
        Args:
            system_prompt: The new system prompt
        """
        self.system_prompt = system_prompt
        self.llm.system_prompt = system_prompt


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_deepseek_wrapper():
    """
    Test the DeepSeek wrapper to ensure it's working properly.
    """
    print("=" * 60)
    print("Testing DeepSeek-R1 Local Wrapper")
    print("=" * 60)
    
    # Test 1: Basic generation using invoke()
    print("\n1. Testing basic generation with invoke()...")
    try:
        llm = DeepSeekLLM(temperature=0.3, num_predict=100)
        response = llm.invoke("Say 'Hello, I am working properly!'")
        print(f"Response: {response}")
        print(f"Length: {len(response)}")
        if len(response) > 5:
            print("✅ Basic generation successful!")
        else:
            print("⚠️  Response too short")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Basic generation using __call__
    print("\n2. Testing basic generation with __call__()...")
    try:
        llm = DeepSeekLLM(temperature=0.3, num_predict=100)
        response = llm("Say 'Hello, I am working properly!'")
        print(f"Response: {response}")
        print(f"Length: {len(response)}")
        if len(response) > 5:
            print("✅ __call__ successful!")
        else:
            print("⚠️  Response too short")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: With system prompt
    print("\n3. Testing with system prompt...")
    try:
        llm_with_system = DeepSeekLLM(
            temperature=0.3,
            num_predict=200,
            system_prompt="You are a helpful laptop shopping assistant. Keep responses concise."
        )
        response = llm_with_system.invoke("What is the best laptop for gaming?")
        print(f"Response: {response[:200]}...")
        print(f"Length: {len(response)}")
        if len(response) > 10:
            print("✅ System prompt test successful!")
        else:
            print("⚠️  Response too short")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Chat interface
    print("\n4. Testing chat interface...")
    try:
        chat = DeepSeekChat(system_prompt="You are a helpful assistant.")
        response = chat.send("Say 'Hello, I am working properly!'")
        print(f"Response: {response}")
        print(f"Length: {len(response)}")
        if len(response) > 5:
            print("✅ Chat interface successful!")
        else:
            print("⚠️  Response too short")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 5: Multi-turn conversation
    print("\n5. Testing multi-turn conversation...")
    try:
        chat = DeepSeekChat()
        response1 = chat.send("Hi! My name is Alice.")
        print(f"Response 1: {response1[:100]}...")
        response2 = chat.send("What is my name?")
        print(f"Response 2: {response2[:100]}...")
        
        if "Alice" in response2:
            print("✅ Multi-turn conversation successful! (Remembered name)")
        else:
            print("⚠️  Multi-turn conversation may have failed to remember context")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ All tests complete!")
    print("=" * 60)
    
    return True


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    test_deepseek_wrapper()