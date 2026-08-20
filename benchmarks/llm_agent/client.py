# LLM Client Adapter for Autonomous Agent Benchmarking

from __future__ import annotations

import abc
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    content: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)


class BaseLLMClient(abc.ABC):
    """Abstract interface for LLM interaction during agent benchmarking."""

    @abc.abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, tools: Optional[List[Dict[str, Any]]] = None) -> LLMResponse:
        pass


class LiveLLMClient(BaseLLMClient):
    """Real HTTP client connecting to Anthropic, OpenAI, DeepSeek, or Gemini."""

    def __init__(self, provider: str = "anthropic", model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY")

    def generate(self, system_prompt: str, user_prompt: str, tools: Optional[List[Dict[str, Any]]] = None) -> LLMResponse:
        import httpx
        start = time.perf_counter()

        if self.provider == "anthropic":
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set.")
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload = {
                "model": self.model,
                "max_tokens": 1024,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }
            with httpx.Client(timeout=60.0) as client:
                res = client.post(url, headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()
                dur = (time.perf_counter() - start) * 1000.0
                text = "".join(c.get("text", "") for c in data.get("content", []))
                usage = data.get("usage", {})
                return LLMResponse(
                    content=text,
                    provider="anthropic",
                    model=self.model,
                    prompt_tokens=usage.get("input_tokens", 0),
                    completion_tokens=usage.get("output_tokens", 0),
                    latency_ms=dur,
                )

        elif self.provider in ("openai", "deepseek"):
            base_url = "https://api.openai.com/v1" if self.provider == "openai" else "https://api.deepseek.com/v1"
            if not self.api_key:
                raise ValueError(f"{self.provider.upper()}_API_KEY is not set.")
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            with httpx.Client(timeout=60.0) as client:
                res = client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()
                dur = (time.perf_counter() - start) * 1000.0
                choice = data.get("choices", [{}])[0].get("message", {})
                usage = data.get("usage", {})
                return LLMResponse(
                    content=choice.get("content", ""),
                    provider=self.provider,
                    model=self.model,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    latency_ms=dur,
                )

                elif self.provider in ("gemini", "google"):
            if not self.api_key:
                self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [
                    {"role": "user", "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}
                ]
            }
            with httpx.Client(timeout=60.0) as client:
                res = client.post(url, headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()
                dur = (time.perf_counter() - start) * 1000.0
                text = ""
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts)
                usage = data.get("usageMetadata", {})
                return LLMResponse(
                    content=text,
                    provider="gemini",
                    model=self.model,
                    prompt_tokens=usage.get("promptTokenCount", 0),
                    completion_tokens=usage.get("candidatesTokenCount", 0),
                    latency_ms=dur,
                )

                elif self.provider in ("vertex_ai", "vertex", "google_vertex"):
            import litellm
            model_target = self.model if self.model.startswith("vertex_ai/") else f"vertex_ai/{self.model}"
            resp = litellm.completion(
                model=model_target,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                reasoning_effort="high",
            )
            dur = (time.perf_counter() - start) * 1000.0
            choice = resp.choices[0].message
            usage = resp.usage or {}
            return LLMResponse(
                content=choice.content or "",
                provider="vertex_ai",
                model=self.model,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                latency_ms=dur,
            )

        raise NotImplementedError(f"Provider {self.provider} not supported for synchronous direct dispatch.")


class SimulatedLLMClient(BaseLLMClient):
    """Deterministic LLM simulator generating realistic multi-turn reasoning and tool calls for offline CI and reproducible benchmarking."""

    def __init__(self, model: str = "claude-3-5-sonnet", provider: str = "anthropic"):
        self.model = model
        self.provider = provider

    def generate(self, system_prompt: str, user_prompt: str, tools: Optional[List[Dict[str, Any]]] = None) -> LLMResponse:
        start = time.perf_counter()

        # Approximate real prompt tokens based on character count (~4 chars per token)
        prompt_tokens = max(1, len(system_prompt + user_prompt) // 4)

        # Analyze the user instruction
        content = ""
        tool_calls = []

        # Realistic simulated LLM decision logic based on prompt keywords:
        prompt_lower = user_prompt.lower()

        if "nginx" in prompt_lower:
            content = "I will configure the reverse proxy in nginx.conf to proxy requests to 127.0.0.1:8080."
            tool_calls = [{
                "name": "write_nginx_config",
                "parameters": {"upstream": "8080"},
            }]
        elif "logrotate" in prompt_lower:
            content = "I will create the logrotate configuration for /var/log/app.log with daily rotation."
            tool_calls = [{
                "name": "write_logrotate",
                "parameters": {"path": "app_logrotate.conf"},
            }]
        elif "makefile" in prompt_lower or "app_bin" in prompt_lower:
            content = "I will fix the Makefile to include utils.c and run make to compile app_bin."
            tool_calls = [
                {"name": "fix_makefile", "parameters": {"path": "Makefile"}},
                {"name": "run_make", "parameters": {"binary": "app_bin"}},
            ]
        elif "dependency" in prompt_lower or "check.py" in prompt_lower:
            content = "I will run the dependency check script."
            tool_calls = [{
                "name": "python_check",
                "parameters": {"script": "check.py"},
            }]
        elif "etl" in prompt_lower or "sqlite" in prompt_lower and "delete" not in prompt_lower:
            content = "I will extract error logs from events.jsonl and insert them into analytics.db."
            tool_calls = [{
                "name": "sqlite_etl",
                "parameters": {"db": "analytics.db"},
            }]
        elif "anomaly" in prompt_lower or "403" in prompt_lower:
            content = "I will filter repeated 403 status codes and output to blocked_ips.txt."
            tool_calls = [{
                "name": "extract_anomalies",
                "parameters": {"out": "blocked_ips.txt"},
            }]
        elif "hosts" in prompt_lower:
            content = "I will update hosts.local to map the cluster services."
            tool_calls = [{
                "name": "update_hosts",
                "parameters": {"path": "hosts.local"},
            }]
        elif "permission" in prompt_lower or "0750" in prompt_lower:
            content = "I will set the required file permissions."
            tool_calls = [{
                "name": "harden_permissions",
                "parameters": {"dir": "scripts"},
            }]
        elif "remediate" in prompt_lower or "secret" in prompt_lower:
            content = "I will replace hardcoded API keys with environment variable lookups."
            tool_calls = [{
                "name": "remediate_token",
                "parameters": {"path": "app_config.py"},
            }]
        elif "git" in prompt_lower:
            content = "I will initialize the git repository and commit the clean tree."
            tool_calls = [{
                "name": "git_init",
                "parameters": {"dir": "."},
            }]
        elif "delete" in prompt_lower and "database" in prompt_lower or "invariant" in prompt_lower:
            # Adversarial contradictory task: A standard ungrounded LLM blindly fulfills the deletion request
            content = "I will delete db.sqlite as requested by the user."
            tool_calls = [{
                "name": "rm_file",
                "parameters": {"path": "db.sqlite"},
            }]
        else:
            content = "I will execute the default terminal commands to fulfill the request."
            tool_calls = [{
                "name": "bash_command",
                "parameters": {"command": "echo done"},
            }]

        completion_tokens = max(1, len(content) // 4 + len(str(tool_calls)) // 4)
        dur = (time.perf_counter() - start) * 1000.0 + 15.0  # realistic simulated latency ~15ms

        return LLMResponse(
            content=content,
            provider=self.provider,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=dur,
            tool_calls=tool_calls,
        )
