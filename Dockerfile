# Final stage
FROM python:3.12-slim-bookworm

# Set the working directory
WORKDIR /workspace

# Copy the application from the builder
#COPY --from=builder --chown=workspaces:workspace /workspace /workspace
COPY . .

# Place executables in the environment at the front of the path
#ENV PATH="/workspace/.venv/bin:$PATH"
RUN pip install uv
#RUN uv sync --extra-index-url https://download.pytorch.org/whl/cpu --system
RUN uv pip install -r pyproject.toml --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu --system

# Add PYTHONPATH
ENV PYTHONPATH="/workspace"

# CMD ["python", "src/train.py"]
CMD ["tail", "-f", "/dev/null"]
