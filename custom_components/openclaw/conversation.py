"""OpenClaw conversation agent for Home Assistant Assist pipeline.

Registers OpenClaw as a native conversation agent so it can be used
with Assist, Voice PE, and any HA voice satellite.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4
import logging
from typing import Any

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.storage import Store
from homeassistant.helpers import intent

from .api import OpenClawApiClient, OpenClawApiError
from .const import (
    ATTR_MESSAGE,
    ATTR_MODEL,
    ATTR_SESSION_ID,
    ATTR_TIMESTAMP,
    CONF_ASSIST_SESSION_ID,
    CONF_AGENT_ID,
    CONF_CONTEXT_MAX_CHARS,
    CONF_CONTEXT_STRATEGY,
    CONF_CONTINUE_CONVERSATION,
    CONF_DEBUG_LOGGING,
    CONF_INCLUDE_EXPOSED_CONTEXT,
    CONF_VOICE_AGENT_ID,
    DEFAULT_ASSIST_SESSION_ID,
    DEFAULT_AGENT_ID,
    DEFAULT_CONTEXT_MAX_CHARS,
    DEFAULT_CONTEXT_STRATEGY,
    DEFAULT_CONTINUE_CONVERSATION,
    DEFAULT_DEBUG_LOGGING,
    DEFAULT_INCLUDE_EXPOSED_CONTEXT,
    DATA_MODEL,
    DOMAIN,
    EVENT_MESSAGE_RECEIVED,
    DATA_ASSIST_SESSIONS,
    DATA_ASSIST_SESSION_STORE,
    ASSIST_SESSION_STORE_KEY,
)
from .coordinator import OpenClawCoordinator
from .exposure import apply_context_policy, build_exposed_entities_context

_LOGGER = logging.getLogger(__name__)

_VOICE_REQUEST_HEADERS = {
    "x-openclaw-source": "voice",
    "x-ha-voice": "true",
}


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the OpenClaw conversation agent."""
    # Load persisted assist sessions
    store = Store(hass, 1, ASSIST_SESSION_STORE_KEY)
    stored = await store.async_load() or {}
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][DATA_ASSIST_SESSIONS] = stored
    hass.data[DOMAIN][DATA_ASSIST_SESSION_STORE] = store

    agent = OpenClawConversationAgent(hass, entry)
    conversation.async_set_agent(hass, entry, agent)


async def async_unload_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
) -> bool:
    """Unload the conversation agent."""
    conversation.async_unset_agent(hass, entry)
    return True


class OpenClawConversationAgent(conversation.AbstractConversationAgent):
    """Conversation agent that routes messages through OpenClaw.

    Enables OpenClaw to appear as a selectable agent in the Assist pipeline,
    allowing use with Voice PE, satellites, and the built-in HA Assist dialog.
    """

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the conversation agent."""
        self.hass = hass
        self.entry = entry

    @property
    def attribution(self) -> dict[str, str]:
        """Return attribution info."""
        return {"name": "Powered by OpenClaw", "url": "https://openclaw.dev"}

    @property
    def supported_languages(self) -> list[str] | str:
        """Return supported languages.

        OpenClaw handles language via its configured model, so we declare
        support for all languages and let the model handle translation.
        """
        return conversation.MATCH_ALL

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a user message through OpenClaw.

        Tries streaming first for lower latency (first-token fast).
        Falls back to non-streaming if the stream yields nothing.

        Args:
            user_input: The conversation input from HA Assist.

        Returns:
            ConversationResult with the assistant's response.
        """
        entry_data = self.hass.data.get(DOMAIN, {}).get(self.entry.entry_id)
        if not entry_data:
            return self._error_result(
                user_input, "OpenClaw integration not configured"
            )

        client: OpenClawApiClient = entry_data["client"]
        coordinator: OpenClawCoordinator = entry_data["coordinator"]

        message = user_input.text
        assistant_id = "conversation"
        options = self.entry.options
        voice_agent_id = self._normalize_optional_text(
            options.get(CONF_VOICE_AGENT_ID)
        )
        configured_agent_id = self._normalize_optional_text(
            options.get(
                CONF_AGENT_ID,
                self.entry.data.get(CONF_AGENT_ID, DEFAULT_AGENT_ID),
            )
        )
        resolved_agent_id = voice_agent_id or configured_agent_id
        conversation_id = self._resolve_conversation_id(user_input, resolved_agent_id)
        include_context = options.get(
            CONF_INCLUDE_EXPOSED_CONTEXT,
            DEFAULT_INCLUDE_EXPOSED_CONTEXT,
        )
        max_chars = int(options.get(CONF_CONTEXT_MAX_CHARS, DEFAULT_CONTEXT_MAX_CHARS))
        strategy = options.get(CONF_CONTEXT_STRATEGY, DEFAULT_CONTEXT_STRATEGY)

        raw_context = (
            build_exposed_entities_context(
                self.hass,
                assistant=assistant_id,
            )
            if include_context
            else None
        )
        exposed_context = apply_context_policy(raw_context, max_chars, strategy)
        extra_system_prompt = getattr(user_input, "extra_system_prompt", None)
        system_prompt = "\n\n".join(
            part for part in (exposed_context, extra_system_prompt) if part
        ) or None

        if options.get(CONF_DEBUG_LOGGING, DEFAULT_DEBUG_LOGGING):
            _LOGGER.info(
                "OpenClaw Assist routing: agent=%s session=%s",
                resolved_agent_id or "main",
                conversation_id,
            )

        try:
            full_response = await self._get_response(
                client,
                message,
                conversation_id,
                resolved_agent_id,
                system_prompt,
            )
        except OpenClawApiError as err:
            _LOGGER.error("OpenClaw conversation error: %s", err)

            # Try token refresh if we have the capability
            refresh_fn = entry_data.get("refresh_token")
            if refresh_fn:
                refreshed = await refresh_fn()
                if refreshed:
                    try:
                        full_response = await self._get_response(
                            client,
                            message,
                            conversation_id,
                            voice_agent_id,
                            system_prompt,
                        )
                    except OpenClawApiError as retry_err:
                        return self._error_result(
                            user_input,
                            f"Error communicating with OpenClaw: {retry_err}",
                        )
                else:
                    return self._error_result(
                        user_input,
                        f"Error communicating with OpenClaw: {err}",
                    )
            else:
                return self._error_result(
                    user_input,
                    f"Error communicating with OpenClaw: {err}",
                )

        # Fire event so automations can react to the response
        self.hass.bus.async_fire(
            EVENT_MESSAGE_RECEIVED,
            {
                ATTR_MESSAGE: full_response,
                ATTR_SESSION_ID: conversation_id,
                ATTR_MODEL: coordinator.data.get(DATA_MODEL) if coordinator.data else None,
                ATTR_TIMESTAMP: datetime.now(timezone.utc).isoformat(),
            },
        )
        coordinator.update_last_activity()

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(full_response)

        continue_conversation = options.get(
            CONF_CONTINUE_CONVERSATION,
            DEFAULT_CONTINUE_CONVERSATION,
        )

        # Heuristic: keep mic open when the assistant asks a question
        if continue_conversation and "?" in full_response:
            intent_response.continue_conversation = True

        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=conversation_id,
        )

    def _resolve_conversation_id(self, user_input: conversation.ConversationInput, agent_id: str | None) -> str:
        """Return conversation id from HA or a stable Assist fallback session key."""
        configured_session_id = self._normalize_optional_text(
            self.entry.options.get(
                CONF_ASSIST_SESSION_ID,
                DEFAULT_ASSIST_SESSION_ID,
            )
        )
        if configured_session_id:
            return configured_session_id

        domain_store = self.hass.data.setdefault(DOMAIN, {})
        session_cache = domain_store.setdefault(DATA_ASSIST_SESSIONS, {})
        cache_key = agent_id or "main"
        cached_session = session_cache.get(cache_key)
        if cached_session:
            return cached_session

        new_session = f"agent:{cache_key}:assist_{uuid4().hex[:12]}"
        session_cache[cache_key] = new_session

        store = domain_store.get(DATA_ASSIST_SESSION_STORE)
        if store:
            self.hass.async_create_task(store.async_save(session_cache))

        return new_session

    def _normalize_optional_text(self, value: Any) -> str | None:
        """Return a stripped string or None for blank values."""
        if not isinstance(value, str):
            return None
        cleaned = value.strip()
        return cleaned or None

    async def _get_response(
        self,
        client: OpenClawApiClient,
        message: str,
        conversation_id: str,
        agent_id: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Get a response from OpenClaw, trying streaming first."""
        model_override = f"openclaw:{agent_id}" if agent_id else None

        # Try streaming (lower TTFB for voice pipeline)
        full_response = ""
        async for chunk in client.async_stream_message(
            message=message,
            session_id=conversation_id,
            model=model_override,
            system_prompt=system_prompt,
            agent_id=agent_id,
            extra_headers=_VOICE_REQUEST_HEADERS,
        ):
            full_response += chunk

        if full_response:
            return full_response

        # Fallback to non-streaming
        response = await client.async_send_message(
            message=message,
            session_id=conversation_id,
            model=model_override,
            system_prompt=system_prompt,
            agent_id=agent_id,
            extra_headers=_VOICE_REQUEST_HEADERS,
        )
        extracted = self._extract_text_recursive(response)
        return extracted or ""

    def _extract_text_recursive(self, value: Any, depth: int = 0) -> str | None:
        """Recursively extract assistant text from nested response payloads."""
        if depth > 8:
            return None

        if isinstance(value, str):
            text = value.strip()
            return text or None

        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                extracted = self._extract_text_recursive(item, depth + 1)
                if extracted:
                    parts.append(extracted)
            if parts:
                return "\n".join(parts)
            return None

        if isinstance(value, dict):
            priority_keys = (
                "output_text",
                "text",
                "content",
                "message",
                "response",
                "answer",
                "choices",
                "output",
                "delta",
            )

            for key in priority_keys:
                if key not in value:
                    continue
                extracted = self._extract_text_recursive(value.get(key), depth + 1)
                if extracted:
                    return extracted

            for nested_value in value.values():
                extracted = self._extract_text_recursive(nested_value, depth + 1)
                if extracted:
                    return extracted

        return None

    def _error_result(
        self,
        user_input: conversation.ConversationInput,
        error_message: str,
    ) -> conversation.ConversationResult:
        """Build an error ConversationResult."""
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_error(
            intent.IntentResponseErrorCode.UNKNOWN,
            error_message,
        )
        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=user_input.conversation_id,
        )
