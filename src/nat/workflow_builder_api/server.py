# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
CLI entry point for running the Workflow Builder API server.
"""

import argparse
import logging
import sys


def main():
    """Run the Workflow Builder API server."""
    parser = argparse.ArgumentParser(
        description="NAT Workflow Builder API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8100,
        help="Port to bind to (default: 8100)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level (default: info)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required. Install it with: pip install uvicorn")
        sys.exit(1)

    print(f"Starting NAT Workflow Builder API on http://{args.host}:{args.port}")
    print(f"API Documentation: http://{args.host}:{args.port}/docs")
    print(f"Registry Endpoint: http://{args.host}:{args.port}/api/v1/registry")

    uvicorn.run(
        "nat.workflow_builder_api.app:get_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        factory=True,
    )


if __name__ == "__main__":
    main()
