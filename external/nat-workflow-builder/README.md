# NAT Workflow Builder

A visual drag-and-drop interface for building NeMo Agent Toolkit (NAT) agent workflows.

## Overview

This UI allows users to visually compose agent workflows by dragging and dropping NAT components onto a canvas. When connected to the NAT Workflow Builder API, components can be configured with specific implementation types (such as which LLM provider to use) and their parameters. Components can be connected together when one component requires a reference to another (such as a Function that needs an LLM).

## Technology Stack

Matches the NAT-UI technology stack:

- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **React 18** - UI library
- **@dnd-kit** - Drag and drop functionality
- **Lucide React** - Icons

## Quick Start

### 1. Start the NAT Workflow Builder API

From the NAT repository root:

```bash
# Make sure you have NAT installed
pip install -e .

# Start the API server
python -m nat.workflow_builder_api.server --port 8100
```

The API will be available at `http://localhost:8100` with documentation at `http://localhost:8100/docs`.

### 2. Start the UI

```bash
cd external/nat-workflow-builder
npm install
npm run dev
```

The UI will be available at `http://localhost:3100`.

## NAT Component Types

The following component types are supported for agent building:

| Component | Description | Category |
|-----------|-------------|----------|
| **Agent** | AI agent that reasons, uses tools, and completes tasks | Core |
| **Function** | Custom function or tool for agent actions | Core |
| **Function Group** | Group of related functions | Core |
| **LLM** | Large Language Model for reasoning | AI & ML |
| **Embedder** | Generate text embeddings for semantic search | AI & ML |
| **Retriever** | Retrieve relevant documents for RAG | AI & ML |
| **Memory** | Persistent memory storage for context | Data |
| **Object Store** | Store and retrieve objects (S3, and so on) | Data |
| **Authentication** | API authentication provider | Infrastructure |
| **Middleware** | Request/response middleware | Infrastructure |

## Project Structure

```text
nat-workflow-builder/
├── components/
│   ├── Canvas/              # Main canvas for placing components
│   │   ├── Canvas.tsx       # Canvas with connection drawing
│   │   └── PlacedComponentNode.tsx  # Component with ports
│   ├── ConfigModal/         # Component configuration modal
│   │   ├── ConfigModal.tsx  # Modal with type dropdown
│   │   └── SchemaFormField.tsx  # Dynamic form fields
│   ├── ConnectionLines/     # SVG connection line rendering
│   │   └── ConnectionLine.tsx
│   ├── ConnectionPorts/     # Visual connection port components
│   │   └── ConnectionPort.tsx
│   ├── NATComponents/       # Modular NAT component library
│   │   └── ComponentIcon.tsx
│   └── Sidebar/             # Component palette sidebar
│       ├── Sidebar.tsx      # With API connection status
│       └── DraggablePaletteItem.tsx
├── contexts/
│   ├── WorkflowContext.tsx  # State management for workflow
│   └── RegistryContext.tsx  # State for NAT registry data
├── lib/
│   └── api.ts               # API client for NAT registry
├── pages/
│   ├── _app.tsx
│   ├── _document.tsx
│   └── index.tsx
├── styles/
│   └── globals.css
├── types/
│   ├── index.ts             # Component types
│   └── registry.ts          # API response types with ports
└── README.md
```

## Features

### Drag and Drop Interface

- Drag components from the sidebar palette onto the canvas
- Components organized by category (Core, AI and ML, Data, Infrastructure)

### Visual Nodes

- Each component type has a distinct color and icon
- Shows configuration status (configured versus unconfigured)

### Dynamic Configuration

When the NAT API is running:

- **Type Selection**: Dropdown shows all registered implementations (such as `nvidia/nim_llm`, `openai/gpt-4`)
- **Schema-Based Forms**: Configuration fields are generated from Pydantic model schemas
- **Field Types**: Supports strings, numbers, booleans, enums, arrays, and objects
- **Connection Ports**: Shows which components can connect to which (such as Functions that need LLMs)

### Component Connections

Components can be connected when one requires a reference to another:

- **Connection Ports**: Each component shows its input and output connection points
- **Visual Feedback**: Ports are color-coded by type (LLM, Embedder, Function, and so on)
- **Required and Optional**: Required connections are clearly marked
- **Drag to Connect**: Drag from an output port to a compatible input port
- **Validation**: Only compatible connections are allowed (LLM output to LLM input)

Connection types supported:

| Type | Description | Color |
|------|-------------|-------|
| LLM | Large Language Model reference | Green |
| Embedder | Text embedding model reference | Blue |
| Function | Custom function reference (agents also provide this) | Purple |
| Function Group | Function group reference | Violet |
| Retriever | Document retriever reference | Cyan |
| Memory | Memory storage reference | Orange |
| Object Store | Object storage reference | Yellow |
| Authentication | Auth provider reference | Red |
| Middleware | Middleware reference | Pink |

### Offline Mode

- Works without the API for basic workflow design
- Shows "Offline Mode" indicator
- Configuration is disabled until API is available

### Workflow Export

- Export workflow configuration as JSON
- Includes all component positions, configurations, and connections

## API Endpoints

The NAT Workflow Builder API provides these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check with registry status |
| `/api/v1/registry` | GET | All registered component types with connection ports |
| `/api/v1/registry/{category}` | GET | Types for a specific category |
| `/api/v1/registry/{category}/{full_type}` | GET | Detailed type info with input ports |
| `/api/v1/schema/{category}/{full_type}` | GET | JSON Schema only |

## Development

### Running Both Services

In separate terminals:

```bash
# Terminal 1: NAT API
python -m nat.workflow_builder_api.server --reload

# Terminal 2: UI
cd external/nat-workflow-builder
npm run dev
```

### Building for Production

```bash
npm run build
npm start
```

## License

Apache-2.0
