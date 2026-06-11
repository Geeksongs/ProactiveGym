# ProactiveGym Desktop Environment
# Based on opensandbox/desktop with pre-installed applications
#
# Docker Hub: pythonsonggeek/proactivegym-desktop:latest
# Pull: docker pull pythonsonggeek/proactivegym-desktop:latest

FROM opensandbox/desktop:latest

USER root

# Install Chrome
RUN apt-get update && apt-get install -y wget gnupg2 \
    && wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Install LibreOffice (Writer, Calc, Impress)
RUN apt-get update && apt-get install -y \
    libreoffice-writer \
    libreoffice-calc \
    libreoffice-impress \
    && rm -rf /var/lib/apt/lists/*

# Install PDF reader (Evince)
RUN apt-get update && apt-get install -y evince \
    && rm -rf /var/lib/apt/lists/*

# Install VSCode
RUN apt-get update && apt-get install -y wget gpg \
    && wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /usr/share/keyrings/packages.microsoft.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list \
    && apt-get update \
    && apt-get install -y code \
    && rm -rf /var/lib/apt/lists/*

# Install Python tools for RL (mss for screenshots, pyautogui for automation)
RUN pip3 install --no-cache-dir mss pyautogui

# Switch back to desktop user
USER desktop

# Set display environment
ENV DISPLAY=:0
