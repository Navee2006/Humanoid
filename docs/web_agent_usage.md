# Web Agent Usage Guide

## Overview

The Web Agent is an AI-powered browser automation system that uses Google's Gemini 2.5 Computer Use model to control a headless Chromium browser through natural language commands.

## Quick Start

### Using with A.D.A Voice Interface

1. **Start the backend server:**
   ```powershell
   cd c:\Users\arunk\Downloads\ada_v2\ada_v2
   .\venv\Scripts\python.exe backend\server.py
   ```

2. **Start the frontend:**
   ```powershell
   npm run dev
   ```

3. **Use voice commands:**
   - Click the power button to start
   - Say: "Open a browser and go to Google"
   - Say: "Search for Python tutorials"
   - Say: "Click on the first result"

### Direct Python Usage

```python
import asyncio
from backend.web_agent import WebAgent

async def main():
    agent = WebAgent()
    
    # Define a callback to receive updates
    async def update_callback(screenshot_b64, log_text):
        print(f"Update: {log_text}")
    
    # Run a task
    result = await agent.run_task(
        prompt="Go to example.com and tell me the page title",
        update_callback=update_callback
    )
    
    print(f"Result: {result}")

asyncio.run(main())
```

## Features

### Supported Actions

The web agent can perform these browser actions:

- **Navigation**
  - `navigate` - Go to a URL
  - `go_back` - Navigate back
  - `go_forward` - Navigate forward
  - `search` - Go to Google

- **Mouse Actions**
  - `click_at` - Click at coordinates
  - `hover_at` - Hover at coordinates
  - `drag_and_drop` - Drag from one point to another

- **Keyboard Actions**
  - `type_text_at` - Type text at coordinates
  - `key_combination` - Press key combinations

- **Scrolling**
  - `scroll_document` - Scroll the page
  - `scroll_at` - Scroll at specific coordinates

- **Utility**
  - `wait_5_seconds` - Wait for page to load

### Example Commands

**Simple Navigation:**
```
"Go to wikipedia.org"
"Navigate to github.com"
```

**Search Tasks:**
```
"Go to Google and search for 'Python asyncio'"
"Find the latest news about AI"
```

**Complex Tasks:**
```
"Go to Amazon and find USB-C cables under $10"
"Navigate to YouTube and search for cooking tutorials"
```

## Configuration

### Environment Variables

Required in `.env`:
```
GEMINI_API_KEY=your_api_key_here
```

### Browser Settings

Configured in `backend/web_agent.py`:
- **Screen Size**: 1440x900 (default)
- **Headless Mode**: True (runs in background)
- **Model**: gemini-2.5-computer-use-preview-10-2025

## Testing

### Run Simple Tests (No API calls)

```powershell
.\venv\Scripts\python.exe test_web_agent_simple.py
```

Tests:
- ✅ Web agent initialization
- ✅ Coordinate denormalization
- ✅ Playwright browser launch

### Run Full Test Suite

```powershell
.\venv\Scripts\python.exe -m pytest tests/test_web_agent.py -v
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'playwright'"

**Solution:**
```powershell
.\venv\Scripts\pip.exe install playwright
.\venv\Scripts\python.exe -m playwright install chromium
```

### Issue: "Playwright browsers not installed"

**Solution:**
```powershell
.\venv\Scripts\python.exe -m playwright install chromium
```

### Issue: "Too Many Requests" / API Rate Limit

**Solution:**
- Wait 30-60 minutes for quota to reset
- Check your API usage at: https://aistudio.google.com/app/apikey
- Consider upgrading to a paid tier for higher limits

### Issue: Web agent not responding in A.D.A

**Checklist:**
1. ✅ Backend server running
2. ✅ Frontend connected (check browser console)
3. ✅ Power button clicked (audio session started)
4. ✅ GEMINI_API_KEY set in `.env`
5. ✅ No API rate limit errors in backend logs

## Architecture

```
User Voice Command
    ↓
A.D.A (ada.py)
    ↓
Web Agent (web_agent.py)
    ↓
Gemini Computer Use Model
    ↓
Playwright Browser (Chromium)
    ↓
Screenshots sent back to frontend
```

## API Limits

**Free Tier:**
- Rate limits apply
- Requests per minute: Limited
- Requests per day: Limited

**Best Practices:**
- Avoid rapid consecutive requests
- Use simple tests for development
- Monitor API usage in Google AI Studio

## Advanced Usage

### Custom Update Callbacks

```python
async def my_callback(screenshot_b64, log_text):
    # Save screenshots
    if screenshot_b64:
        with open(f"screenshot_{time.time()}.png", "wb") as f:
            f.write(base64.b64decode(screenshot_b64))
    
    # Log actions
    print(f"[{time.time()}] {log_text}")

result = await agent.run_task(
    prompt="Your task here",
    update_callback=my_callback
)
```

### Headless vs Headed Mode

To see the browser (for debugging):

```python
# In web_agent.py, line 203:
self.browser = await p.chromium.launch(headless=False)  # Change to False
```

## Support

For issues or questions:
1. Check backend logs for errors
2. Verify Playwright installation
3. Check API key configuration
4. Review rate limit status

## Version Info

- **Playwright**: v1.57.0
- **Gemini Model**: gemini-2.5-computer-use-preview-10-2025
- **Python**: 3.13.7
