FROM python:3.11-slim

# 1. Create the mandatory Hugging Face user (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 2. Set the working directory
WORKDIR $HOME/app

# 3. Copy requirements and install (giving ownership to the new user)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your app files
COPY --chown=user . .

# 5. Expose the correct port
EXPOSE 7860

# 6. Start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
