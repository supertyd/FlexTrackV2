export ANTHROPIC_BASE_URL="http://localhost:11211/api/openai/v1"
export ANTHROPIC_AUTH_TOKEN="ollama"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export DISABLE_NON_ESSENTIAL_MODEL_CALLS=1
claude --model "gemini-3.1-pro-preview" -p "Hello"
