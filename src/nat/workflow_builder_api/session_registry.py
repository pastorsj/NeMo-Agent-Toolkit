# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Session registry for managing active workflow sessions in the Workflow Builder.

This module provides a centralized registry for tracking active workflow sessions,
each associated with a validated Config and SessionManager.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nat.data_models.config import Config
    from nat.runtime.session import SessionManager

logger = logging.getLogger(__name__)


@dataclass
class WorkflowSession:
    """Represents an active workflow session."""

    session_id: str
    config: Config
    session_manager: SessionManager
    created_at: datetime
    last_activity: datetime


class WorkflowSessionRegistry:
    """
    Registry for managing active workflow sessions.

    This is a singleton that tracks all active sessions and provides
    methods for creating, retrieving, and destroying sessions.
    """

    _instance: WorkflowSessionRegistry | None = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __new__(cls) -> WorkflowSessionRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._sessions = {}
            cls._instance._cleanup_task = None
        return cls._instance

    @classmethod
    def get(cls) -> WorkflowSessionRegistry:
        """Get the singleton instance of the registry."""
        return cls()

    @property
    def sessions(self) -> dict[str, WorkflowSession]:
        """Get all active sessions."""
        return self._sessions

    async def create_session(self, config: Config) -> WorkflowSession:
        """
        Create a new workflow session from a validated Config.

        Args:
            config: A validated NAT Config object.

        Returns:
            WorkflowSession with the session ID and manager.
        """
        from nat.builder.workflow_builder import WorkflowBuilder
        from nat.runtime.session import SessionManager

        session_id = str(uuid.uuid4())

        # Build the workflow and create session manager
        workflow_builder = await WorkflowBuilder.from_config(config=config).__aenter__()
        session_manager = await SessionManager.create(
            config=config,
            shared_builder=workflow_builder,
        )

        now = datetime.utcnow()
        session = WorkflowSession(
            session_id=session_id,
            config=config,
            session_manager=session_manager,
            created_at=now,
            last_activity=now,
        )

        async with self._lock:
            self._sessions[session_id] = session

        logger.info("Created workflow session: %s", session_id)
        return session

    async def get_session(self, session_id: str) -> WorkflowSession | None:
        """
        Get a session by ID and update its last activity time.

        Args:
            session_id: The session ID to look up.

        Returns:
            The WorkflowSession if found, None otherwise.
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.last_activity = datetime.utcnow()
            return session

    async def destroy_session(self, session_id: str) -> bool:
        """
        Destroy a session and clean up its resources.

        Args:
            session_id: The session ID to destroy.

        Returns:
            True if the session was found and destroyed, False otherwise.
        """
        async with self._lock:
            session = self._sessions.pop(session_id, None)

        if session:
            try:
                await session.session_manager.shutdown()
                logger.info("Destroyed workflow session: %s", session_id)
                return True
            except Exception as e:
                logger.exception("Error shutting down session %s: %s", session_id, e)
                return True  # Still consider it destroyed

        return False

    async def cleanup_stale_sessions(self, max_idle_minutes: int = 30) -> int:
        """
        Clean up sessions that have been idle for too long.

        Args:
            max_idle_minutes: Maximum idle time in minutes before a session is cleaned up.

        Returns:
            Number of sessions cleaned up.
        """
        cutoff = datetime.utcnow() - timedelta(minutes=max_idle_minutes)
        stale_sessions: list[str] = []

        async with self._lock:
            for session_id, session in self._sessions.items():
                if session.last_activity < cutoff:
                    stale_sessions.append(session_id)

        cleaned = 0
        for session_id in stale_sessions:
            if await self.destroy_session(session_id):
                cleaned += 1

        if cleaned > 0:
            logger.info("Cleaned up %d stale sessions", cleaned)

        return cleaned

    async def start_cleanup_task(self, interval_minutes: int = 5) -> None:
        """Start a background task to periodically clean up stale sessions."""
        if self._cleanup_task is not None:
            return

        async def cleanup_loop():
            while True:
                await asyncio.sleep(interval_minutes * 60)
                try:
                    await self.cleanup_stale_sessions()
                except Exception as e:
                    logger.exception("Error in session cleanup: %s", e)

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def stop_cleanup_task(self) -> None:
        """Stop the background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None

    async def shutdown_all(self) -> None:
        """Shutdown all active sessions."""
        await self.stop_cleanup_task()

        session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            await self.destroy_session(session_id)

        logger.info("Shutdown all workflow sessions")
