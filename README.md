# ABD V1 — Advanced Brain Desktop Assistant

> A JARVIS-style personal AI assistant for Windows, powered by **Google Gemini**.  
> Understands natural-language commands and uses tools to manage files, launch apps, search the web, read PDFs, and assist with code.

---

## ✨ Features

| Category | Capabilities |
|---|---|
| **AI Conversation** | Multi-turn chat with session memory, powered by Gemini |
| **File Management** | List, search, read, create, edit files in your workspace |
| **PDF Assistant** | Read & summarise research papers, answer follow-up questions |
| **App Launcher** | Open 20+ Windows apps by name (Calculator, WhatsApp, VS Code, …) |
| **Web & YouTube** | Google search, open URLs, YouTube search — all in your browser |
| **Coding Assistant** | Read, explain, create, and suggest improvements for code files |
| **Safety** | Workspace restriction, path traversal prevention, action logging |

---

## 🗂️ Project Structure

```
ABD/
├── main.py                 ← Entry point — run this to start ABD
├── config.py               ← Loads .env settings
├── requirements.txt
├── .env.example            ← Template — copy to .env and fill in
├── .gitignore
├── README.md
│
├── brain/
│   ├── agent.py            ← Core agent loop (tool-call orchestration)
│   ├── router.py           ← Tool registry + Gemini function declarations
│   └── prompts.py          ← ABD's personality & system prompt
│
├── llm/
│   └── gemini_client.py    ← Google Gemini API wrapper
│
├── tools/
│   ├── file_tools.py       ← File management (list/read/create/edit)
│   ├── pdf_tools.py        ← PDF reading & chunking (pypdf)
│   ├── app_tools.py        ← Windows application launcher (whitelist)
│   ├── browser_tools.py    ← Open URLs & Google search
│   ├── youtube_tools.py    ← YouTube search
│   └── code_tools.py       ← Code reading, creation, analysis
│
├── safety/
│   └── permissions.py      ← Path validation, confirmations, logging
│
├── workspace/              ← ABD's sandbox (files created here)
└── tests/
    ├── test_safety.py
    └── test_file_tools.py
```

---

## 🚀 Installation

### 1. Prerequisites

- **Python 3.11+** — [Download](https://www.python.org/downloads/)
- A **Google Gemini API key** — [Get one free](https://aistudio.google.com/app/apikey)

### 2. Clone / Download the project

```bash
# If using git:
git clone <your-repo-url>
cd "ABD ASSISTANT"

# Or just open the project folder in your terminal.
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure your API key

```bash
# Copy the template
copy .env.example .env       # Windows
cp .env.example .env         # macOS/Linux

# Open .env in any text editor and set your key:
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```

> ⚠️ **Never share or commit your `.env` file.** It is listed in `.gitignore`.

---

## ▶️ Running ABD

```bash
python main.py
```

### Optional flags

```bash
python main.py --debug    # Verbose logging for troubleshooting
python main.py --version  # Print version and exit
```

---

## 💬 Example Commands

Once ABD is running, try these:

### General
```
Hello ABD
What can you do?
help
```

### File Management
```
List my workspace files
Search for Python files
Read hello.py
Create a file called notes.txt with content "Meeting at 3pm"
```

### Coding
```
Create hello.py that prints "Hello ABD"
Read main.py and explain what it does
List all code files in the workspace
Create a Python calculator with add, subtract, multiply, divide
```

### Application Launcher
```
Open Calculator
Open Notepad
Open WhatsApp
Open VS Code
Open File Explorer
```

### Web & YouTube Search
```
Search the web for Docker networking
Search YouTube for Python FastAPI tutorials
Open https://github.com
```

### PDF Assistant
```
Read and summarise this PDF: C:\Users\me\Documents\paper.pdf
What is the methodology section in the loaded PDF?
What are the main conclusions?
```

### Built-in Commands
```
help       — Show example commands
reset      — Clear conversation history and start fresh
exit       — Quit ABD
version    — Show version info
```

---

## 🏗️ Architecture

```
User Input
  └─► Agent (brain/agent.py)
        └─► GeminiClient (llm/gemini_client.py)
              └─► Gemini API → FunctionCall returned
        └─► Router (brain/router.py)
              └─► Safety Check (safety/permissions.py)
                    └─► Tool Execution (tools/*.py)
                          └─► Result returned to Agent
        └─► GeminiClient (result sent back to Gemini)
              └─► Final text response
  └─► Printed to terminal (via Rich)
```

The agent supports up to **10 tool-call iterations per turn** to handle multi-step tasks (e.g. "search YouTube for X and then open calculator") without entering infinite loops.

---

## 🛡️ Safety Features

- **Workspace restriction** — ABD can only create/edit files inside the `workspace/` directory
- **Path traversal prevention** — `../` and absolute paths outside workspace are blocked
- **No arbitrary shell execution** — The app launcher uses a whitelist; no raw terminal commands
- **Overwrite confirmation** — Creating a file that already exists requires `overwrite=True`, which the agent only sets after asking you
- **Action logging** — All important actions are written to `abd_actions.log`

---

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run only safety tests
python -m pytest tests/test_safety.py -v

# Run only file tool tests
python -m pytest tests/test_file_tools.py -v
```

> Tests do **not** require a Gemini API key — they mock the configuration.

---

## ⚙️ Configuration Reference

All settings are in your `.env` file:

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Your Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model to use |
| `WORKSPACE_DIR` | `workspace` | Directory for file operations |
| `LOG_FILE` | `abd_actions.log` | Action log file path |
| `MAX_TOOL_ITERATIONS` | `10` | Max tool calls per agent turn |
| `PDF_CHUNK_SIZE` | `8000` | Characters per PDF chunk |

---

## 🔮 Future (V2+)

ABD V1 is designed to grow. Planned future features:

- **V2** — Voice input/output (speech-to-text, TTS)
- **V3** — Code execution sandbox (safe Python REPL)
- **V4** — Screen vision (screenshot analysis)
- **V5** — Long-term memory (vector DB)
- **V6** — Multi-agent DevOps automation

The modular architecture (tools → router → agent) means new tools can be added by:
1. Creating a function in `tools/`
2. Adding its `FunctionDeclaration` in `brain/router.py`
3. Registering it in `TOOL_REGISTRY`

No core agent code needs to change.

---

## 📄 License

MIT — free to use, modify, and distribute.

---

*Built with ❤️ using Python + Google Gemini*
