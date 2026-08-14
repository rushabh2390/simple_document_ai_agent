#!/bin/sh

# 1. Start Ollama server in background
ollama serve &

# 2. Wait until Ollama engine is responsive
echo "Waiting for Ollama service boot sequence..."
until ollama list > /dev/null 2>&1; do
  sleep 2
done
echo "Ollama core engine online!"

# 3. Download model only if it does NOT exist locally
if ! ollama list | grep -q "qwen2.5-coder:3b"; then
  echo "Pulling qwen2.5-coder:3b model (this may take a few minutes)..."
  ollama pull qwen2.5-coder:3b
  echo "Model initialization complete!"
else
  echo "Model qwen2.5-coder:3b already downloaded and ready."
fi

# 4. Keep container running
wait