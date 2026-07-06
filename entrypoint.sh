#!/bin/sh

# Start Ollama server process in the background
ollama serve &

# Wait for the Ollama daemon port to open and respond
echo "Waiting for Ollama service boot sequence..."
while ! nc -z localhost 11434; do   
  sleep 1
done
echo "Ollama core engine online!"

# Auto-download the target model
echo "Pulling llama3.2 manifest layers..."
ollama pull llama3.2
echo "Model initialization complete!"

# Bring the background server process to the foreground so the container stays alive
wait