# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
WebSocket handler for the Workflow Builder chat interface.

This module provides WebSocket-based communication for running workflows
directly from the Workflow Builder UI.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any

from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from pydantic import BaseModel
from pydantic import Field

from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import Message
from nat.data_models.api_server import ResponseIntermediateStep
from nat.data_models.api_server import ResponsePayloadOutput
from nat.data_models.api_server import ResponseSerializable
from nat.data_models.api_server import TextContent
from nat.data_models.api_server import UserMessageContentRoleType
from nat.front_ends.fastapi.response_helpers import generate_streaming_response
from nat.front_ends.fastapi.step_adaptor import StepAdaptor
from nat.front_ends.fastapi.step_adaptor import StepAdaptorConfig
from nat.workflow_builder_api.session_registry import WorkflowSession

logger = logging.getLogger(__name__)


# WebSocket Message Types
class WebSocketMessageType:
    """WebSocket message type constants."""

    USER_MESSAGE = "user_message"
    SYSTEM_RESPONSE = "system_response_message"
    SYSTEM_INTERMEDIATE = "system_intermediate_message"
    SYSTEM_INTERACTION = "system_interaction_message"
    ERROR = "error"


class WebSocketMessageStatus:
    """WebSocket message status constants."""

    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"


# Inbound message models
class UserMessageContent(BaseModel):
    """Content of a user message."""

    messages: list[dict[str, Any]] = Field(default_factory=list)


class InboundUserMessage(BaseModel):
    """User message received from the WebSocket."""

    type: str = WebSocketMessageType.USER_MESSAGE
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    content: UserMessageContent
    timestamp: str | None = None


# Outbound message models
class SystemResponseContent(BaseModel):
    """Content of a system response."""

    text: str = ""


class OutboundSystemResponse(BaseModel):
    """System response sent to the WebSocket."""

    type: str = WebSocketMessageType.SYSTEM_RESPONSE
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    parent_id: str
    status: str = WebSocketMessageStatus.IN_PROGRESS
    content: SystemResponseContent = Field(default_factory=SystemResponseContent)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class OutboundIntermediateStep(BaseModel):
    """Intermediate step message sent to the WebSocket."""

    type: str = WebSocketMessageType.SYSTEM_INTERMEDIATE
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    parent_id: str
    status: str = WebSocketMessageStatus.IN_PROGRESS
    content: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class OutboundError(BaseModel):
    """Error message sent to the WebSocket."""

    type: str = WebSocketMessageType.ERROR
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    parent_id: str | None = None
    content: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class WorkflowBuilderWebSocketHandler:
    """
    Handles WebSocket connections for the Workflow Builder chat interface.

    This handler manages the communication between the UI and the workflow,
    processing user messages and streaming responses back.
    """

    def __init__(self, websocket: WebSocket, session: WorkflowSession):
        self._websocket = websocket
        self._session = session
        self._step_adaptor = StepAdaptor(StepAdaptorConfig())
        self._running_task: asyncio.Task | None = None
        self._current_message_id: str | None = None

    async def __aenter__(self) -> WorkflowBuilderWebSocketHandler:
        await self._websocket.accept()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._running_task and not self._running_task.done():
            self._running_task.cancel()

    async def run(self) -> None:
        """Main loop to process WebSocket messages."""
        try:
            while True:
                message = await self._websocket.receive_json()
                await self._handle_message(message)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected for session %s", self._session.session_id)
        except asyncio.CancelledError:
            logger.info("WebSocket handler cancelled for session %s", self._session.session_id)
        except Exception as e:
            logger.exception("WebSocket error for session %s: %s", self._session.session_id, e)
            await self._send_error(str(e))

    async def _handle_message(self, message: dict[str, Any]) -> None:
        """Route incoming messages to appropriate handlers."""
        message_type = message.get("type", "")

        if message_type == WebSocketMessageType.USER_MESSAGE:
            await self._handle_user_message(message)
        else:
            logger.warning("Unknown message type: %s", message_type)

    async def _handle_user_message(self, raw_message: dict[str, Any]) -> None:
        """Process a user message and run the workflow."""
        try:
            message = InboundUserMessage(**raw_message)
            self._current_message_id = message.id

            # Extract user text from messages
            user_text = self._extract_user_text(message.content.messages)
            if not user_text:
                await self._send_error("No user message content found", message.id)
                return

            # Build chat request
            chat_messages = self._build_chat_messages(message.content.messages)
            chat_request = ChatRequest(messages=chat_messages)

            # Run workflow in background task
            async def run_workflow():
                try:
                    logger.info("Starting workflow for message: %s", message.id)
                    logger.info("Session manager: %s", self._session.session_manager)
                    logger.info("Workflow: %s", self._session.session_manager.workflow)

                    async with self._session.session_manager.session(
                            user_message_id=message.id,
                            conversation_id=message.conversation_id,
                    ) as session:
                        logger.info("Session created, starting to stream responses")
                        response_count = 0
                        async for response in generate_streaming_response(
                                chat_request,
                                session=session,
                                streaming=True,
                                step_adaptor=self._step_adaptor,
                                result_type=ChatResponseChunk,
                                output_type=ChatResponseChunk,
                        ):
                            response_count += 1
                            logger.info("Received response %d: %s", response_count, type(response).__name__)
                            await self._send_response(response, message.id, message.conversation_id)

                        logger.info("Workflow completed with %d responses", response_count)

                    # Send completion message
                    await self._send_complete(message.id, message.conversation_id)

                except Exception as e:
                    logger.exception("Workflow error: %s", e)
                    await self._send_error(str(e), message.id, message.conversation_id)
                finally:
                    logger.info("run_workflow task finished for message: %s", message.id)

            self._running_task = asyncio.create_task(run_workflow())

        except Exception as e:
            logger.exception("Error handling user message: %s", e)
            await self._send_error(str(e))

    def _extract_user_text(self, messages: list[dict[str, Any]]) -> str | None:
        """Extract text content from user messages."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", [])
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            return item.get("text", "")
                        if isinstance(item, str):
                            return item
        return None

    def _build_chat_messages(self, messages: list[dict[str, Any]]) -> list[Message]:
        """Convert raw messages to Message format for ChatRequest."""
        result: list[Message] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", [])

            # Normalize content to list of TextContent
            if isinstance(content, str):
                content_list = [TextContent(type="text", text=content)]
            elif isinstance(content, list):
                content_list = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        content_list.append(TextContent(type="text", text=item.get("text", "")))
                    elif isinstance(item, str):
                        content_list.append(TextContent(type="text", text=item))
            else:
                content_list = [TextContent(type="text", text=str(content))]

            role_type = (UserMessageContentRoleType.USER if role == "user" else UserMessageContentRoleType.ASSISTANT)
            result.append(Message(role=role_type, content=content_list))

        return result

    def _extract_final_answer(self, payload: str) -> str | None:
        """
        Extract the final answer text from a workflow completion payload.

        The payload contains a ChatResponseChunk with the final answer in
        choices[0].delta.content or choices[0].message.content.
        """
        import re

        # Try to extract content from delta.content or message.content patterns
        # Pattern: content='...' or content="..."
        content_match = re.search(r"content='([^']*)'|content=\"([^\"]*)\"", payload)
        if content_match:
            return content_match.group(1) or content_match.group(2)

        # Try ChoiceDelta pattern
        delta_match = re.search(r"ChoiceDelta\(content='([^']*)'", payload)
        if delta_match:
            return delta_match.group(1)

        # Try to find any quoted text after content=
        alt_match = re.search(r"delta=ChoiceDelta\(content='(.+?)', role=", payload)
        if alt_match:
            return alt_match.group(1)

        return None

    async def _send_response(
        self,
        response: ResponseSerializable,
        parent_id: str,
        conversation_id: str,
    ) -> None:
        """Send a response message to the WebSocket."""
        logger.debug("Processing response type: %s", type(response).__name__)

        if isinstance(response, ResponsePayloadOutput):
            # Extract text from response payload
            text = ""
            payload = response.payload
            logger.debug("ResponsePayloadOutput payload type: %s", type(payload).__name__)
            if hasattr(payload, "choices") and payload.choices:
                choice = payload.choices[0]
                if hasattr(choice, "delta") and choice.delta:
                    text = getattr(choice.delta, "content", "") or ""
                elif hasattr(choice, "message") and choice.message:
                    text = getattr(choice.message, "content", "") or ""

            if text:
                logger.debug("Sending text response: %s", text[:100] if len(text) > 100 else text)
                msg = OutboundSystemResponse(
                    conversation_id=conversation_id,
                    parent_id=parent_id,
                    content=SystemResponseContent(text=text),
                )
                await self._websocket.send_json(msg.model_dump())
            else:
                logger.debug("ResponsePayloadOutput had no text content")
        elif isinstance(response, ResponseIntermediateStep):
            # Check if this is the final workflow completion with the answer
            if response.name == "Function Complete: <workflow>":
                # Extract the final answer from the payload
                final_text = self._extract_final_answer(response.payload)
                if final_text:
                    logger.debug("Extracted final answer: %s",
                                 final_text[:100] if len(final_text) > 100 else final_text)
                    msg = OutboundSystemResponse(
                        conversation_id=conversation_id,
                        parent_id=parent_id,
                        content=SystemResponseContent(text=final_text),
                    )
                    await self._websocket.send_json(msg.model_dump())
                    return

            # Handle intermediate steps - send properly formatted content
            logger.debug("Sending intermediate step: %s", response.name)
            msg = OutboundIntermediateStep(
                conversation_id=conversation_id,
                parent_id=parent_id,
                content={
                    "id": response.id,
                    "parent_id": response.parent_id,
                    "type": response.type,
                    "name": response.name,
                    "payload": response.payload,
                },
            )
            await self._websocket.send_json(msg.model_dump())
        else:
            # Handle other response types - try to serialize as dict
            logger.debug("Sending other response type: %s", type(response).__name__)
            try:
                if hasattr(response, "model_dump"):
                    content = response.model_dump()
                else:
                    content = {"data": str(response)}
            except Exception:
                content = {"data": str(response)}

            msg = OutboundIntermediateStep(
                conversation_id=conversation_id,
                parent_id=parent_id,
                content=content,
            )
            await self._websocket.send_json(msg.model_dump())

    async def _send_complete(self, parent_id: str, conversation_id: str) -> None:
        """Send a completion message to the WebSocket."""
        msg = OutboundSystemResponse(
            conversation_id=conversation_id,
            parent_id=parent_id,
            status=WebSocketMessageStatus.COMPLETE,
            content=SystemResponseContent(text=""),
        )
        await self._websocket.send_json(msg.model_dump())

    async def _send_error(
        self,
        error_message: str,
        parent_id: str | None = None,
        conversation_id: str | None = None,
    ) -> None:
        """Send an error message to the WebSocket."""
        msg = OutboundError(
            conversation_id=conversation_id or "unknown",
            parent_id=parent_id,
            content={
                "error": error_message, "text": error_message
            },
        )
        try:
            await self._websocket.send_json(msg.model_dump())
        except Exception:
            logger.exception("Failed to send error message")
