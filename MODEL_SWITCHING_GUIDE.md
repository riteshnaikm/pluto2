# 🤖 Model Switching Guide - Gemini ↔️ ChatGPT

This guide explains how to switch between Google Gemini and OpenAI ChatGPT models for resume evaluation in the HR Assistant Suite.

## 📋 Overview

The application now supports **two AI providers** for resume evaluation:

1. **Google Gemini** (Default) - Free tier available, fast, good quality
2. **OpenAI ChatGPT** (Optional) - Paid service, potentially higher quality

You can easily switch between them using environment variables in your `.env` file.

---

## 🚀 Quick Start

### Option A: Use Google Gemini (Default, Free)

1. In your `.env` file, set:
```env
MODEL_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash
GOOGLE_API_KEY=your_google_api_key_here
```

2. Restart the application

### Option B: Use OpenAI ChatGPT (Paid)

1. Get your OpenAI API key from: https://platform.openai.com/api-keys

2. In your `.env` file, set:
```env
MODEL_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_openai_api_key_here
```

3. Restart the application

---

## 🔧 Configuration Details

### Environment Variables

Add these to your `.env` file:

```env
# Choose provider: "gemini" or "openai"
MODEL_PROVIDER=gemini

# Gemini configuration (if using Gemini)
GOOGLE_API_KEY=your_google_api_key
GEMINI_MODEL=gemini-2.5-flash

# OpenAI configuration (if using OpenAI)
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
```

### Available Models

**Gemini Models:**
- `gemini-2.5-flash` ⚡ (Recommended - Fast, good quality, free tier)
- `gemini-2.5-pro` 🚀 (More powerful, slower)
- `gemini-2.0-flash` ⚡ (Previous version)

**OpenAI Models:**
- `gpt-4o-mini` ⚡ (Recommended - Fastest, cheapest, good quality)
- `gpt-4o` 🚀 (Most capable, expensive)
- `gpt-4-turbo` 🔥 (Powerful, expensive)
- `gpt-3.5-turbo` 💨 (Fast, cheap, lower quality)

---

## 💰 Cost Comparison

### Google Gemini
- **Free Tier**: 15 requests per minute
- **Paid Tier**: Very affordable
- **Best for**: Development, testing, small-scale production

### OpenAI ChatGPT
- **Pricing** (as of Dec 2024):
  - GPT-4o-mini: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
  - GPT-4o: ~$2.50 per 1M input tokens, ~$10 per 1M output tokens
- **Best for**: High-quality production evaluations

**Estimate**: Average resume evaluation uses ~5,000-8,000 tokens total
- Gemini: Free or ~$0.00001 per evaluation
- GPT-4o-mini: ~$0.003-0.005 per evaluation
- GPT-4o: ~$0.05-0.08 per evaluation

---

## 🎯 When to Use Which Model?

### Use **Gemini** when:
- ✅ You want free or very cheap evaluations
- ✅ You're in development/testing phase
- ✅ Speed is important
- ✅ You have budget constraints
- ✅ You're handling high volume of resumes

### Use **ChatGPT** when:
- ✅ You need highest quality analysis
- ✅ You're doing critical hiring decisions
- ✅ You have API credits available
- ✅ You want more detailed reasoning
- ✅ Accuracy is more important than cost

---

## 🔄 How to Switch

### During Development

1. Open your `.env` file
2. Change `MODEL_PROVIDER=gemini` to `MODEL_PROVIDER=openai` (or vice versa)
3. Restart the Flask application:
   ```bash
   # Stop the running application (Ctrl+C)
   # Then restart:
   python run.py
   ```

### In Production (Linux Server)

1. SSH into your server
2. Edit the `.env` file:
   ```bash
   cd /path/to/hr-assistant
   nano .env
   ```
3. Change `MODEL_PROVIDER` value
4. Restart the service:
   ```bash
   sudo systemctl restart hr-assistant
   ```

---

## 🧪 Testing the Implementation

### Test 1: Check Logs on Startup

When you start the application, you should see logs indicating which model is active:

```
🤖 Model Provider: GEMINI
📦 Using Gemini Model: gemini-2.5-flash
```

or

```
🤖 Model Provider: OPENAI
📦 Using OpenAI Model: gpt-4o-mini
```

### Test 2: Evaluate a Resume

1. Upload a resume through the web interface
2. Check the evaluation results
3. The model used will be logged in the application logs

---

## 🐛 Troubleshooting

### Error: "OpenAI API key not found"

**Solution**: Make sure you've added `OPENAI_API_KEY` to your `.env` file and restarted the app.

### Error: "Rate limit exceeded"

**Solution**: 
- For Gemini: Wait a minute or upgrade to paid tier
- For OpenAI: Check your API usage limits and credits

### Falling Back to Gemini

The application automatically falls back to Gemini if OpenAI fails. Check the logs for:
```
OpenAI failed, falling back to Gemini
```

### Model Not Switching

**Solution**:
1. Verify your `.env` file has the correct values
2. Make sure you **restarted** the application after changing `.env`
3. Check application logs for any errors

---

## 📊 Feature Coverage

The model switching applies to:

✅ **Resume Evaluation** - Main JD matching analysis  
✅ **Job Stability Analysis** - Tenure and job-hopping assessment  
✅ **Career Progression** - Career trajectory analysis  
✅ **Interview Questions** - Technical and behavioral questions  
✅ **Recruiter Handbook** - Detailed candidate analysis  

The model switching does **NOT** apply to:
- HR Document Chatbot (uses Groq/Llama models via RAG)

---

## 📝 Technical Details

### How It Works

1. **Abstraction Layer**: `generate_content_unified()` function routes calls to the appropriate API
2. **Response Normalization**: Both APIs return a standardized response object with `.text` attribute
3. **Error Handling**: Automatic fallback to Gemini if OpenAI fails
4. **Streaming Support**: Both models support streaming for real-time responses

### Code Changes Made

- Added `openai` package to `requirements.txt`
- Created `UnifiedModelResponse` wrapper class
- Implemented `generate_content_unified()` function
- Updated all model calls to use the unified interface
- Added environment variables for model selection

---

## 🔐 Security Notes

- Never commit your `.env` file to version control
- Keep your API keys secure
- Rotate API keys periodically
- Monitor API usage and set spending limits (especially for OpenAI)

---

## 📞 Support

If you encounter any issues:

1. Check the application logs: `sudo journalctl -u hr-assistant -f` (Linux) or terminal output (Windows)
2. Verify API keys are correct
3. Check your API provider's status page:
   - Gemini: https://status.cloud.google.com/
   - OpenAI: https://status.openai.com/

---

## 🎉 Summary

You now have the flexibility to choose between:
- **Gemini** for cost-effective, fast evaluations
- **ChatGPT** for premium quality analysis

Simply change one environment variable and restart - no code changes needed!

**Default recommendation**: Start with `gemini-2.5-flash`, switch to OpenAI only if you need higher quality or have specific requirements.

---

**Last Updated**: December 2024  
**Version**: 1.0

